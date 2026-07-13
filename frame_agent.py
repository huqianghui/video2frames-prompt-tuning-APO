"""Frame-analysis agent for Agent-Lightning APO on the video2frames task.

The agent receives a task with pre-extracted video frames stored in Azure Blob
Storage and the tunable instruction prompt (a [PromptTemplate][agentlightning.types.PromptTemplate]
resource). It appends per-video frame placeholders of the form `<frame n | Xs>`
(n starting at 1, one frame roughly every 3 seconds) after the fixed
instruction, sends the frames to an Azure OpenAI multimodal deployment, and
scores the structured JSON output against the ground truth with a hybrid
reward:

    reward = 0.2 * scene_type exact match
           + 0.2 * is_courier_action exact match
           + 0.6 * LLM-judge semantic score over english_detail/brief/title

Invalid JSON output scores 0. Only the fixed instruction prompt is tuned by
APO; the frame placeholder section is rebuilt per task at runtime.

Usage (debug a few rollouts with the baseline prompt):
    python frame_agent.py --limit 2 [--base64] [--split val]

Requires Azure OpenAI settings in the environment (see README.md).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, cast

from dotenv import load_dotenv
from openai import AzureOpenAI, BadRequestError
from openai.types.chat import ChatCompletionContentPartParam, ChatCompletionMessageParam
from pydantic import BaseModel, Field

from agentlightning.litagent import rollout
from agentlightning.types import Dataset, PromptTemplate
from blob_utils import PROJECT_ROOT, BlobConfig, blob_config_from_env, blob_sas_url, load_env

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
BASELINE_PROMPT_PATH = DATA_DIR / "baseline_prompt.txt"

SCENE_WEIGHT = 0.2
COURIER_WEIGHT = 0.2
JUDGE_WEIGHT = 0.6


class FrameTask(TypedDict):
    """One task record produced by `prepare_data.py`."""

    id: str
    video: str
    family: str
    frame_blobs: List[str]
    num_frames: int
    seconds_per_frame: int
    solution: Dict[str, Any]


class JudgeResponse(BaseModel):
    reason: str = Field(description="The reason for the score. No more than 100 characters.")
    score: float = Field(description="The score for the semantic match on a 0-1 scale. Be critical.")


def task_model() -> str:
    return os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")


def judge_model() -> str:
    return os.environ.get("JUDGE_MODEL", "gpt-4.1-mini")


def prompt_template_baseline(path: Path = BASELINE_PROMPT_PATH) -> PromptTemplate:
    """Load the fixed instruction prompt extracted by `prepare_data.py`."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python prepare_data.py` first.")
    return PromptTemplate(template=path.read_text(encoding="utf-8"), engine="f-string")


def frame_placeholder(index: int, seconds_per_frame: int) -> str:
    """Placeholder for frame `index` (1-based): `<frame n | Xs>` with X = (n-1)*step."""
    return f"<frame {index} | {(index - 1) * seconds_per_frame}s>"


def build_frame_section(num_frames: int, seconds_per_frame: int) -> str:
    """Textual frame placeholder section appended after the fixed instruction."""
    placeholders = " ".join(frame_placeholder(i, seconds_per_frame) for i in range(1, num_frames + 1))
    return f"### FRAMES\n{placeholders}"


def download_as_data_uri(url: str) -> str:
    """Download an image and encode it as a base64 data URI (chartqa pattern)."""
    import httpx

    response = httpx.get(url, timeout=60.0)
    response.raise_for_status()
    encoded = base64.b64encode(response.content).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def build_multimodal_content(
    fixed_prompt: str,
    task: FrameTask,
    config: BlobConfig,
    use_base64: bool = False,
) -> List[ChatCompletionContentPartParam]:
    """Build OpenAI content parts: fixed prompt, then `<frame n | Xs>` labels interleaved with images."""
    parts: List[ChatCompletionContentPartParam] = [
        {"type": "text", "text": f"{fixed_prompt}\n\n### FRAMES"},
    ]
    for index, blob_path in enumerate(task["frame_blobs"], start=1):
        url = blob_sas_url(config, blob_path)
        if use_base64:
            url = download_as_data_uri(url)
        parts.append({"type": "text", "text": frame_placeholder(index, task["seconds_per_frame"])})
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


def parse_model_output(raw: str) -> Dict[str, Any]:
    """Parse the model's JSON output, tolerating markdown code fences.

    Raises `ValueError` when the output is not a JSON object.
    """
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model output is not valid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise ValueError(f"Model output is not a JSON object: {type(parsed)}")
    return cast(Dict[str, Any], parsed)


def judge_text_fields(client: AzureOpenAI, generated: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """LLM-judge semantic similarity of english_detail/brief/title on a 0-1 scale."""
    judge_prompt = (
        "You are a strict grader of video event descriptions.\n"
        "Compare the generated fields against the expected ground truth. "
        "Judge whether they describe the same subjects and actions; wording may differ.\n\n"
        f"Generated:\n"
        f"- english_detail: {generated.get('english_detail')}\n"
        f"- brief: {generated.get('brief')}\n"
        f"- title: {generated.get('title')}\n\n"
        f"Expected:\n"
        f"- english_detail: {expected['english_detail']}\n"
        f"- brief: {expected['brief']}\n"
        f"- title: {expected['title']}\n\n"
        "Score the semantic match on a 0-1 scale. Be critical; partial credit is allowed."
    )
    completion = client.chat.completions.parse(
        model=judge_model(),
        messages=[{"role": "user", "content": judge_prompt}],
        response_format=JudgeResponse,
        temperature=0.0,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        logger.warning("Judge returned no parsed response; scoring 0.")
        return 0.0
    logger.info("Judge score %.2f: %s", parsed.score, parsed.reason)
    return max(0.0, min(1.0, parsed.score))


def compute_reward(generated: Dict[str, Any], expected: Dict[str, Any], judge_score: float) -> float:
    """Hybrid reward from exact-match fields and the judge score."""
    scene_match = str(generated.get("scene_type", "")).strip().lower() == expected["scene_type"]
    courier_value = generated.get("is_courier_action")
    if isinstance(courier_value, str):
        courier_value = courier_value.strip().lower() == "true"
    courier_match = bool(courier_value) == expected["is_courier_action"]
    return SCENE_WEIGHT * scene_match + COURIER_WEIGHT * courier_match + JUDGE_WEIGHT * judge_score


@rollout
def frame_analyzer(task: FrameTask, prompt_template: PromptTemplate) -> float:
    """Analyze video frames with the tunable instruction prompt and score the output.

    The `prompt_template` resource holds only the fixed instruction text tuned
    by APO; the `<frame n | Xs>` placeholder section and the frame images are
    appended at runtime and are not part of the optimization target.
    """
    load_env()
    config = blob_config_from_env()
    client = AzureOpenAI()
    use_base64 = os.environ.get("FRAMES_AS_BASE64", "").lower() in ("1", "true", "yes")

    fixed_prompt = prompt_template.template
    content = build_multimodal_content(fixed_prompt, task, config, use_base64=use_base64)
    messages: List[ChatCompletionMessageParam] = [{"role": "user", "content": content}]

    logger.info(
        "Task %s (%s): sending %d frames to %s", task["id"], task["family"], task["num_frames"], task_model()
    )
    try:
        response = client.chat.completions.create(
            model=task_model(),
            messages=messages,
            temperature=0.0,
        )
    except BadRequestError as e:
        # Some customer videos (e.g. ucf_crime) are rejected by the Azure OpenAI
        # content safety filter. This is a per-task data issue, identical for every
        # candidate prompt, so score it 0 instead of failing the rollout.
        logger.warning("Task %s: request rejected (%s), reward 0.", task["id"], e)
        return 0.0
    raw_output = response.choices[0].message.content or ""
    logger.debug("Task %s raw output: %s", task["id"], raw_output)

    try:
        generated = parse_model_output(raw_output)
    except ValueError as e:
        logger.warning("Task %s: invalid JSON output, reward 0. %s", task["id"], e)
        return 0.0

    judge_score = judge_text_fields(client, generated, task["solution"])
    reward = compute_reward(generated, task["solution"], judge_score)
    logger.info("Task %s reward: %.3f", task["id"], reward)
    return reward


def load_tasks(split: str, limit: Optional[int] = None) -> Dataset[FrameTask]:
    """Load a dataset split written by `prepare_data.py`, optionally truncated to `limit` tasks."""
    path = DATA_DIR / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python prepare_data.py` first.")
    tasks: List[FrameTask] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            tasks.append(cast(FrameTask, json.loads(line)))
    if limit is not None:
        tasks = tasks[:limit]
    return cast(Dataset[FrameTask], tasks)


async def debug_frame_analyzer(limit: int, split: str) -> None:
    """Run a few rollouts locally and print the spans and rewards."""
    from agentlightning.adapter import TraceToMessages
    from agentlightning.reward import find_final_reward
    from agentlightning.runner import LitAgentRunner
    from agentlightning.store import InMemoryLightningStore
    from agentlightning.tracer.agentops import AgentOpsTracer

    runner = LitAgentRunner[FrameTask](AgentOpsTracer())
    store = InMemoryLightningStore()
    prompt_template = prompt_template_baseline()
    tasks = load_tasks(split)
    with runner.run_context(agent=frame_analyzer, store=store):
        for task in cast(List[FrameTask], tasks)[:limit]:
            logger.info("=== Task %s (%s frames) ===", task["id"], task["num_frames"])
            rollout_obj = await runner.step(task, resources={"prompt_template": prompt_template})
            spans = await store.query_spans(rollout_obj.rollout_id)
            messages = TraceToMessages().adapt(spans)
            logger.info("Adapted %d message groups from trace", len(messages))
            logger.info("Final reward: %s", find_final_reward(spans))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug the frame analyzer agent on a few tasks.")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--base64", action="store_true", help="Send frames as base64 data URIs instead of SAS URLs.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv()
    if args.base64:
        os.environ["FRAMES_AS_BASE64"] = "true"
    asyncio.run(debug_frame_analyzer(args.limit, args.split))
