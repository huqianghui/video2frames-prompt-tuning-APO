"""Evaluate a prompt (baseline or APO-tuned) on the held-out test split.

Runs every task in `data/test.jsonl` through the frame-analysis agent with the
given prompt file and reports the mean reward. Results are written to
`results/eval_<name>.json`.

Usage:
    python evaluate.py                          # baseline prompt on test split
    python evaluate.py --prompt results/best_prompt.txt --name tuned
    python evaluate.py --split val --limit 5
    # Same prompt scored under both rewards, then: python compare_rewards.py <a> <b>
    python evaluate.py --name baseline_v1 --reward-version v1
    python evaluate.py --name baseline_v2 --reward-version v2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from dotenv import load_dotenv

from agentlightning.reward import find_final_reward, find_reward_spans, get_rewards_from_span
from agentlightning.runner import LitAgentRunner
from agentlightning.store import InMemoryLightningStore
from agentlightning.tracer.agentops import AgentOpsTracer
from agentlightning.types import PromptTemplate
from blob_utils import PROJECT_ROOT, load_env
from frame_agent import BASELINE_PROMPT_PATH, FrameTask, frame_analyzer, load_tasks
from reward import REWARD_VERSION_ENV, resolve_version

logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"


def extract_components(spans: List[Any]) -> Optional[Dict[str, float]]:
    """Recover per-component reward scores from the trace.

    `frame_analyzer` emits one multi-dimensional reward span (components plus the
    `total` primary dimension) before returning; the runner then auto-emits the
    float return value as a single-dimension span. Scan backwards for the last
    multi-dimensional span and drop the redundant `total` dimension.
    """
    for span in reversed(find_reward_spans(spans)):
        dims = get_rewards_from_span(span)
        if dims is not None and len(dims) > 1:
            return {d.name: d.value for d in dims if d.name != "total" and d.value is not None}
    return None


async def evaluate_prompt(prompt_path: Path, split: str, limit: int, name: str, reward_version: str) -> Dict[str, Any]:
    """Run the agent on a dataset split and aggregate the rewards."""
    prompt_template = PromptTemplate(template=prompt_path.read_text(encoding="utf-8"), engine="f-string")
    tasks = cast(List[FrameTask], load_tasks(split))
    if limit > 0:
        tasks = tasks[:limit]

    runner = LitAgentRunner[FrameTask](AgentOpsTracer())
    store = InMemoryLightningStore()
    details: List[Dict[str, Any]] = []
    with runner.run_context(agent=frame_analyzer, store=store):
        for index, task in enumerate(tasks, start=1):
            logger.info("[%d/%d] Evaluating task %s (%s)", index, len(tasks), task["id"], task["family"])
            try:
                rollout_obj = await runner.step(task, resources={"prompt_template": prompt_template})
                spans = await store.query_spans(rollout_obj.rollout_id)
                reward = find_final_reward(spans)
                components = extract_components(cast(List[Any], spans))
            except Exception:
                logger.exception("Task %s failed; recording reward 0.", task["id"])
                reward = 0.0
                components = None
            details.append({"id": task["id"], "family": task["family"], "reward": reward, "components": components})

    rewards = [d["reward"] for d in details if d["reward"] is not None]
    component_values: Dict[str, List[float]] = {}
    for d in details:
        for key, value in (d.get("components") or {}).items():
            component_values.setdefault(key, []).append(value)
    component_stats = {
        key: {"mean": sum(values) / len(values), "n": len(values)}
        for key, values in sorted(component_values.items())
    }
    summary: Dict[str, Any] = {
        "name": name,
        "prompt_file": str(prompt_path),
        "split": split,
        "reward_version": reward_version,
        "num_tasks": len(details),
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "component_stats": component_stats,
        "details": details,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"eval_{name}.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Mean reward %.3f over %d tasks. Details in %s", summary["mean_reward"], len(details), output_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a prompt on the test split.")
    parser.add_argument("--prompt", type=Path, default=BASELINE_PROMPT_PATH, help="Prompt file to evaluate.")
    parser.add_argument("--name", default="baseline", help="Label used in the results file name.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N tasks (0 = all).")
    parser.add_argument(
        "--reward-version",
        default=None,
        help="Reward version from reward/ (default: REWARD_VERSION env var or v1). "
        "Use distinct --name labels when scoring the same prompt under different versions.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv()
    load_env()
    reward_version = resolve_version(args.reward_version)
    os.environ[REWARD_VERSION_ENV] = reward_version
    logger.info("Reward version: %s", reward_version)
    summary = asyncio.run(evaluate_prompt(args.prompt, args.split, args.limit, args.name, reward_version))
    print(json.dumps({k: v for k, v in summary.items() if k != "details"}, indent=2))


if __name__ == "__main__":
    main()
