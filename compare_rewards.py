"""Compare two evaluation result files task-by-task (e.g. reward v1 vs v2).

Joins the per-task details of two `results/eval_*.json` files produced by
`evaluate.py` on task id, prints a per-task delta table plus overall and
per-family means, and writes the joined result next to the inputs.

Usage:
    python evaluate.py --name baseline_v1 --reward-version v1
    python evaluate.py --name baseline_v2 --reward-version v2
    python compare_rewards.py results/eval_baseline_v1.json results/eval_baseline_v2.json
    python compare_rewards.py a.json b.json --output results/compare_custom.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def load_eval(path: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if "details" not in summary:
        raise ValueError(f"{path} has no 'details' field; is it an evaluate.py output?")
    return summary


def eval_label(summary: Dict[str, Any]) -> str:
    reward_version = summary.get("reward_version", "?")
    return f"{summary.get('name', 'unnamed')} ({reward_version})"


def compare(summary_a: Dict[str, Any], summary_b: Dict[str, Any]) -> Dict[str, Any]:
    """Join two evaluation summaries on task id and compute per-task/per-family deltas."""
    details_b = {d["id"]: d for d in summary_b["details"]}
    rows: List[Dict[str, Any]] = []
    for detail_a in summary_a["details"]:
        detail_b = details_b.get(detail_a["id"])
        if detail_b is None:
            logger.warning("Task %s only present in the first file; skipping.", detail_a["id"])
            continue
        reward_a, reward_b = detail_a["reward"], detail_b["reward"]
        delta = None if reward_a is None or reward_b is None else reward_b - reward_a
        rows.append(
            {"id": detail_a["id"], "family": detail_a["family"], "reward_a": reward_a, "reward_b": reward_b, "delta": delta}
        )
    only_b = [task_id for task_id in details_b if task_id not in {d["id"] for d in summary_a["details"]}]
    if only_b:
        logger.warning("%d tasks only present in the second file: %s", len(only_b), only_b)

    per_family: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"a": [], "b": []})
    for row in rows:
        if row["reward_a"] is not None:
            per_family[row["family"]]["a"].append(row["reward_a"])
        if row["reward_b"] is not None:
            per_family[row["family"]]["b"].append(row["reward_b"])

    def mean(values: List[float]) -> Optional[float]:
        return sum(values) / len(values) if values else None

    family_means = {
        family: {"mean_a": mean(v["a"]), "mean_b": mean(v["b"])} for family, v in sorted(per_family.items())
    }
    rewards_a = [r["reward_a"] for r in rows if r["reward_a"] is not None]
    rewards_b = [r["reward_b"] for r in rows if r["reward_b"] is not None]
    return {
        "a": {"file": summary_a.get("prompt_file"), "name": summary_a.get("name"), "reward_version": summary_a.get("reward_version")},
        "b": {"file": summary_b.get("prompt_file"), "name": summary_b.get("name"), "reward_version": summary_b.get("reward_version")},
        "num_tasks": len(rows),
        "mean_a": mean(rewards_a),
        "mean_b": mean(rewards_b),
        "family_means": family_means,
        "tasks": rows,
    }


def print_report(result: Dict[str, Any], label_a: str, label_b: str) -> None:
    def fmt(value: Optional[float]) -> str:
        return "  n/a" if value is None else f"{value:5.3f}"

    print(f"\nA = {label_a}\nB = {label_b}\n")
    print(f"{'task':<8} {'family':<14} {'A':>6} {'B':>6} {'B-A':>7}")
    for row in sorted(result["tasks"], key=lambda r: (r["delta"] is None, r["delta"] or 0.0)):
        print(f"{row['id']:<8} {row['family']:<14} {fmt(row['reward_a']):>6} {fmt(row['reward_b']):>6} {fmt(row['delta']):>7}")
    print(f"\n{'family means':<23} {'A':>6} {'B':>6}")
    for family, means in result["family_means"].items():
        print(f"{family:<23} {fmt(means['mean_a']):>6} {fmt(means['mean_b']):>6}")
    print(f"\n{'OVERALL':<23} {fmt(result['mean_a']):>6} {fmt(result['mean_b']):>6}  ({result['num_tasks']} tasks)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two evaluate.py result files task-by-task.")
    parser.add_argument("eval_a", type=Path, help="First eval_*.json (shown as A).")
    parser.add_argument("eval_b", type=Path, help="Second eval_*.json (shown as B).")
    parser.add_argument("--output", type=Path, default=None, help="Where to write the joined JSON (default: alongside inputs).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    summary_a, summary_b = load_eval(args.eval_a), load_eval(args.eval_b)
    result = compare(summary_a, summary_b)
    print_report(result, eval_label(summary_a), eval_label(summary_b))

    output = args.output or args.eval_a.parent / f"compare_{args.eval_a.stem}_vs_{args.eval_b.stem}.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\ncomparison written to {output}")


if __name__ == "__main__":
    main()
