"""Probe which tasks are rejected by the Azure OpenAI content safety filter.

The input-image content filter rejects some videos (mostly `ucf_crime`) with a
400 `content_policy_violation` error. The rejection depends only on the frames,
not on the prompt, so blocked tasks score 0 for every candidate prompt and only
add noise to APO. This script sends each task's frames once with a minimal
prompt (`detail: low`, one output token) to find the blocked tasks, reports the
ratio per split and family, and can remove them from the dataset files.

Usage:
    python probe_content_filter.py [--splits train val test] [--apply] [--from-report]

Without `--apply` the script only writes the probe report to
`data/content_filter_probe.json`. With `--apply` it also rewrites
`data/<split>.jsonl` without the blocked tasks (the removed ids are kept in the
report; regenerate the splits with `prepare_data.py` if you need refills).
`--from-report` skips probing and applies an existing report (implies `--apply`).

Probe results are cached per task id in `data/content_filter_cache.json` (the
filter decision depends only on the frames, so it is stable per video): each
video is probed at most once across all runs of this script and of
`prepare_data.py --probe-content-filter`. Delete the cache file to force
re-probing.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, cast

from dotenv import load_dotenv
from openai import AzureOpenAI, BadRequestError
from openai.types.chat import ChatCompletionContentPartParam

from blob_utils import blob_config_from_env, blob_sas_url, load_env
from frame_agent import DATA_DIR, FrameTask, task_model

logger = logging.getLogger(__name__)

REPORT_PATH = DATA_DIR / "content_filter_probe.json"
CACHE_PATH = DATA_DIR / "content_filter_cache.json"


def load_probe_cache(path: Path = CACHE_PATH) -> Dict[str, bool]:
    """Load the persistent task-id → blocked cache (empty when missing)."""
    if not path.exists():
        return {}
    cache = cast(Dict[str, bool], json.loads(path.read_text(encoding="utf-8")))
    logger.info("Loaded content-filter probe cache with %d entries from %s", len(cache), path)
    return cache


def save_probe_cache(cache: Dict[str, bool], path: Path = CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def probe_task_cached(client: AzureOpenAI, task: FrameTask, config: Any, cache: Dict[str, bool]) -> bool:
    """`probe_task` with a persistent cache: each video is probed at most once across runs.

    The filter decision depends only on the input frames, so the result is stable
    per video (task id). New results are appended to the cache file immediately,
    so an interrupted run loses nothing.
    """
    task_id = task["id"]
    if task_id in cache:
        logger.info("Task %s (%s): cache hit (blocked=%s)", task_id, task["family"], cache[task_id])
        return cache[task_id]
    blocked = probe_task(client, task, config)
    cache[task_id] = blocked
    save_probe_cache(cache)
    return blocked


def probe_task(client: AzureOpenAI, task: FrameTask, config: Any) -> bool:
    """Send the task's frames with a minimal prompt. Returns True when blocked."""
    parts: List[ChatCompletionContentPartParam] = [{"type": "text", "text": "Reply with the single word: ok"}]
    for blob_path in task["frame_blobs"]:
        parts.append(
            {"type": "image_url", "image_url": {"url": blob_sas_url(config, blob_path), "detail": "low"}}
        )
    try:
        client.chat.completions.create(
            model=task_model(),
            messages=[{"role": "user", "content": parts}],
            max_tokens=1,
            temperature=0.0,
        )
    except BadRequestError as e:
        if e.code == "content_policy_violation" or "content" in str(e).lower():
            logger.warning("Task %s (%s): BLOCKED (%s)", task["id"], task["family"], e.code)
            return True
        raise
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe tasks against the Azure content safety filter.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--apply", action="store_true", help="Rewrite the jsonl files without blocked tasks.")
    parser.add_argument("--from-report", action="store_true",
                        help="Skip probing; apply the blocked task list from an existing report.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv()
    load_env()

    cache: Dict[str, bool] = {}
    if args.from_report:
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        args.apply = True
        config = client = None
    else:
        config = blob_config_from_env()
        client = AzureOpenAI()
        report = {"model": task_model(), "splits": {}}
        cache = load_probe_cache()

    for split in args.splits:
        path = DATA_DIR / f"{split}.jsonl"
        tasks = [cast(FrameTask, json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if args.from_report:
            blocked = cast(List[Dict[str, str]], report["splits"][split]["blocked_tasks"])
        else:
            assert client is not None and config is not None
            blocked = []
            for i, task in enumerate(tasks, start=1):
                logger.info("[%s %d/%d] probing task %s (%s, %d frames)",
                            split, i, len(tasks), task["id"], task["family"], task["num_frames"])
                if probe_task_cached(client, task, config, cache):
                    blocked.append({"id": task["id"], "family": task["family"], "video": task["video"]})
            report["splits"][split] = {
                "total": len(tasks),
                "blocked": len(blocked),
                "blocked_ratio": round(len(blocked) / len(tasks), 4) if tasks else 0.0,
                "blocked_tasks": blocked,
            }
        logger.info("%s: %d/%d blocked (%.1f%%)", split, len(blocked), len(tasks),
                    100 * len(blocked) / len(tasks) if tasks else 0.0)

        if args.apply and blocked:
            blocked_ids = {b["id"] for b in blocked}
            kept = [t for t in tasks if t["id"] not in blocked_ids]
            with open(path, "w", encoding="utf-8") as f:
                for t in kept:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
            logger.info("%s: removed %d blocked tasks, %d remain in %s", split, len(blocked), len(kept), path)

    if not args.from_report:
        REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Report written to %s", REPORT_PATH)
    print(json.dumps({s: {k: v for k, v in r.items() if k != "blocked_tasks"} for s, r in report["splits"].items()},
                     indent=2))


if __name__ == "__main__":
    main()
