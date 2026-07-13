"""Convert the customer-provided `qwen_0318_swift_task.json` into frame-based APO datasets.

The source file is a pandas `DataFrame.to_json()` dump (column-oriented, not
records), with columns `messages`, `solution`, `videos`, and `task`. Every row
shares one identical prompt that starts with a `<video>` placeholder. This
script:

1. Re-assembles the column-oriented JSON into row records.
2. Extracts the shared instruction prompt, strips the `<video>` placeholder,
   and stores the fixed instruction (the part APO will tune) in
   `data/baseline_prompt.txt`.
3. Parses each `solution` JSON string into a normalized ground-truth dict.
4. Stratified-samples train/val/test subsets by dataset family, resolving the
   pre-extracted frame blobs for each sampled video from Azure Blob Storage.
5. Writes `data/train.jsonl`, `data/val.jsonl`, and `data/test.jsonl`.

Usage:
    python prepare_data.py [--train-size 40] [--val-size 24] [--test-size 30]
                           [--seed 42] [--source original_data/qwen_0318_swift_task.json]
                           [--output-dir data] [--full] [--probe-content-filter]

`--full` additionally writes `data/full.jsonl` with all records (without frame
listings, for later large-scale runs). `--probe-content-filter` probes every
candidate against the Azure OpenAI content safety filter during sampling and
skips blocked videos (~3% of the data), so the splits reach their target sizes
with tasks that are guaranteed to pass the filter (requires Azure OpenAI
credentials; one cheap low-detail request per candidate).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, TypedDict, cast

from blob_utils import PROJECT_ROOT, BlobConfig, blob_config_from_env, list_frame_blobs

logger = logging.getLogger(__name__)

VIDEO_PLACEHOLDER = "<video>"
SECONDS_PER_FRAME = 3

SOLUTION_TEXT_FIELDS = ("english_detail", "brief", "title")
SOLUTION_FIELDS = SOLUTION_TEXT_FIELDS + ("scene_type", "is_courier_action")


class SourceRecord(TypedDict):
    """One row of the customer dataset after re-assembling the pandas dump."""

    id: str
    prompt: str
    video: str
    family: str
    solution: Dict[str, Any]


def load_pandas_column_json(path: Path) -> List[SourceRecord]:
    """Load the pandas column-oriented JSON dump and re-assemble row records."""
    logger.info("Loading source dataset from %s", path)
    with open(path, encoding="utf-8") as f:
        columns = json.load(f)
    for required in ("messages", "solution", "videos"):
        if required not in columns:
            raise ValueError(f"Source file is missing the {required!r} column.")

    records: List[SourceRecord] = []
    for row_key in sorted(columns["messages"], key=int):
        messages = columns["messages"][row_key]
        if len(messages) != 1 or messages[0].get("role") != "user":
            raise ValueError(f"Row {row_key}: expected a single user message, got {messages!r}")
        videos = columns["videos"][row_key]
        if len(videos) != 1:
            raise ValueError(f"Row {row_key}: expected exactly one video, got {videos!r}")
        video = videos[0]
        records.append(
            SourceRecord(
                id=row_key,
                prompt=messages[0]["content"],
                video=video,
                family=video_family(video),
                solution=normalize_solution(columns["solution"][row_key], row_key),
            )
        )
    logger.info("Loaded %d records", len(records))
    return records


def video_family(video_path: str) -> str:
    """Dataset family of a video, i.e. the first path segment after `videos/`."""
    marker = "/videos/"
    index = video_path.find(marker)
    if index < 0:
        raise ValueError(f"Video path does not contain {marker!r}: {video_path}")
    return video_path[index + len(marker) :].split("/", 1)[0]


def normalize_solution(solution: str, row_key: str) -> Dict[str, Any]:
    """Parse a `solution` JSON string into a normalized ground-truth dict."""
    parsed_any: Any = json.loads(solution)
    if not isinstance(parsed_any, dict):
        raise ValueError(f"Row {row_key}: solution is not a JSON object: {solution!r}")
    parsed = cast(Dict[str, Any], parsed_any)
    missing = [field for field in SOLUTION_FIELDS if field not in parsed]
    if missing:
        raise ValueError(f"Row {row_key}: solution is missing fields {missing}")
    normalized: Dict[str, Any] = {field: str(parsed[field]).strip() for field in SOLUTION_TEXT_FIELDS}
    scene_type = str(parsed["scene_type"]).strip().lower()
    if scene_type not in ("indoor", "outdoor"):
        logger.warning("Row %s: unexpected scene_type %r", row_key, parsed["scene_type"])
    normalized["scene_type"] = scene_type
    is_courier = parsed["is_courier_action"]
    if isinstance(is_courier, str):
        is_courier = is_courier.strip().lower() == "true"
    normalized["is_courier_action"] = bool(is_courier)
    return normalized


def extract_fixed_prompt(records: Sequence[SourceRecord]) -> str:
    """Extract the shared instruction prompt and strip the `<video>` placeholder.

    The returned text is the fixed part of the prompt that APO tunes; the
    per-video frame placeholders are appended by the agent at runtime.
    """
    prompts = {record["prompt"] for record in records}
    if len(prompts) != 1:
        raise ValueError(f"Expected a single shared prompt, found {len(prompts)} distinct prompts.")
    prompt = prompts.pop()
    if prompt.count(VIDEO_PLACEHOLDER) != 1:
        raise ValueError(f"Expected exactly one {VIDEO_PLACEHOLDER!r} placeholder in the shared prompt.")
    return prompt.replace(VIDEO_PLACEHOLDER, "").strip()


def stratified_sample(records: Sequence[SourceRecord], total: int, rng: random.Random) -> List[SourceRecord]:
    """Sample `total` records proportionally across dataset families (at least 1 each)."""
    by_family: Dict[str, List[SourceRecord]] = defaultdict(list)
    for record in records:
        by_family[record["family"]].append(record)

    quotas: Dict[str, int] = {}
    remaining = total
    for family, members in sorted(by_family.items(), key=lambda item: len(item[1])):
        quota = max(1, round(total * len(members) / len(records)))
        quota = min(quota, len(members), remaining)
        quotas[family] = quota
        remaining -= quota
    # Distribute any leftover quota to the largest families.
    for family, members in sorted(by_family.items(), key=lambda item: -len(item[1])):
        if remaining <= 0:
            break
        extra = min(remaining, len(members) - quotas[family])
        quotas[family] += extra
        remaining -= extra

    sampled: List[SourceRecord] = []
    for family, quota in quotas.items():
        sampled.extend(rng.sample(by_family[family], quota))
    rng.shuffle(sampled)
    return sampled


def resolve_frames(
    records: List[SourceRecord],
    needed: int,
    config: BlobConfig,
    is_blocked: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> List[Dict[str, Any]]:
    """Attach frame blob listings to records, skipping videos with no frames.

    Consumes records from the front of `records` until `needed` tasks have
    frames resolved (records without frames are skipped with a warning).
    When `is_blocked` is given, tasks it flags (e.g. rejected by the Azure
    content safety filter) are skipped as well, so the returned tasks all pass.
    """
    tasks: List[Dict[str, Any]] = []
    while records and len(tasks) < needed:
        record = records.pop(0)
        frame_blobs = list_frame_blobs(config, record["video"])
        if not frame_blobs:
            logger.warning("Skipping record %s (%s): no frames in blob storage", record["id"], record["video"])
            continue
        task: Dict[str, Any] = {
            "id": record["id"],
            "video": record["video"],
            "family": record["family"],
            "frame_blobs": frame_blobs,
            "num_frames": len(frame_blobs),
            "seconds_per_frame": SECONDS_PER_FRAME,
            "solution": record["solution"],
        }
        if is_blocked is not None and is_blocked(task):
            logger.warning(
                "Skipping record %s (%s): blocked by the content safety filter", record["id"], record["video"]
            )
            continue
        tasks.append(task)
        if len(tasks) % 10 == 0:
            logger.info("Resolved frames for %d/%d tasks", len(tasks), needed)
    if len(tasks) < needed:
        logger.warning("Only resolved %d of %d requested tasks (ran out of candidates)", len(tasks), needed)
    return tasks


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %d rows to %s", len(rows), path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "original_data" / "qwen_0318_swift_task.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--train-size", type=int, default=40)
    parser.add_argument("--val-size", type=int, default=24)
    parser.add_argument("--test-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", action="store_true", help="Also write full.jsonl with all records (no frame lists).")
    parser.add_argument(
        "--probe-content-filter",
        action="store_true",
        help="Probe each candidate against the Azure content safety filter and skip blocked videos.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    records = load_pandas_column_json(args.source)
    fixed_prompt = extract_fixed_prompt(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = args.output_dir / "baseline_prompt.txt"
    baseline_path.write_text(fixed_prompt, encoding="utf-8")
    logger.info("Wrote fixed instruction prompt (%d chars) to %s", len(fixed_prompt), baseline_path)

    if args.full:
        write_jsonl(
            args.output_dir / "full.jsonl",
            [
                {
                    "id": r["id"],
                    "video": r["video"],
                    "family": r["family"],
                    "seconds_per_frame": SECONDS_PER_FRAME,
                    "solution": r["solution"],
                }
                for r in records
            ],
        )

    rng = random.Random(args.seed)
    total = args.train_size + args.val_size + args.test_size
    # Oversample so that videos without frames can be skipped and backfilled.
    oversampled = stratified_sample(records, min(len(records), total * 2), rng)

    config = blob_config_from_env()
    is_blocked: Optional[Callable[[Dict[str, Any]], bool]] = None
    if args.probe_content_filter:
        from dotenv import load_dotenv
        from openai import AzureOpenAI

        from frame_agent import FrameTask
        from probe_content_filter import load_probe_cache, probe_task_cached

        load_dotenv()
        client = AzureOpenAI()
        probe_cache = load_probe_cache()

        def probe_candidate(task: Dict[str, Any]) -> bool:
            logger.info("Probing task %s (%s, %d frames)", task["id"], task["family"], task["num_frames"])
            return probe_task_cached(client, cast(FrameTask, task), config, probe_cache)

        is_blocked = probe_candidate

    tasks = resolve_frames(oversampled, total, config, is_blocked=is_blocked)
    if len(tasks) < total:
        raise RuntimeError(f"Could not resolve enough tasks with frames: got {len(tasks)}, need {total}.")

    write_jsonl(args.output_dir / "train.jsonl", tasks[: args.train_size])
    write_jsonl(args.output_dir / "val.jsonl", tasks[args.train_size : args.train_size + args.val_size])
    write_jsonl(args.output_dir / "test.jsonl", tasks[args.train_size + args.val_size : total])
    logger.info("Done. Datasets are in %s", args.output_dir)


if __name__ == "__main__":
    main()
