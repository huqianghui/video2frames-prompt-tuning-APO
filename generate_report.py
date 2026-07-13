"""Generate a human-readable report of an APO run from `apo.log`.

The APO algorithm only keeps the history-best prompt in memory; the full
optimization trace (candidate prompts, per-rollout rewards, gradient critiques,
validation scores) lives in the log file written by `setup_apo_logger` in
`apo_train.py`. This script parses that log and writes:

- `results/report.md` — per-round candidates with parent, train-batch rewards,
  gradient critique, validation score, and the full prompt text.
- `results/report.json` — the same data in structured form.

The log may contain several runs (one per process); by default the last run is
reported.

Usage:
    python generate_report.py [--log log/apo.log] [--output-dir results] [--run -1]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from blob_utils import PROJECT_ROOT

logger = logging.getLogger(__name__)

_RECORD_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[(?P<level>\w+)\] "
    r"\(Process-(?P<pid>\d+) (?P<name>[\w.]+)\)\s+(?P<message>.*)$",
    re.DOTALL,
)
_PREFIX_RE = re.compile(r"^\[(?P<parts>[^\]]+)\]\s*(?P<rest>.*)$", re.DOTALL)


@dataclass
class LogRecord:
    timestamp: str
    pid: str
    round_num: Optional[int]
    prompt_version: Optional[str]
    message: str


@dataclass
class Candidate:
    version: str
    parent: Optional[str] = None
    round_num: Optional[int] = None
    prompt_text: Optional[str] = None
    train_rewards: Optional[List[Optional[float]]] = None
    train_average: Optional[float] = None
    gradient: Optional[str] = None
    val_rewards: Optional[List[Optional[float]]] = None
    val_score: Optional[float] = None


@dataclass
class RunReport:
    pid: str
    started: str
    finished: str
    baseline_score: Optional[float] = None
    best_version: Optional[str] = None
    best_score: Optional[float] = None
    best_updated: bool = False
    candidates: Dict[str, Candidate] = field(default_factory=dict)
    rounds: List[int] = field(default_factory=list)


def parse_records(log_path: Path) -> List[LogRecord]:
    """Split the log file into records (continuation lines are folded into the previous record)."""
    records: List[LogRecord] = []
    raw: List[str] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if _RECORD_RE.match(line):
            raw.append(line)
        elif raw:
            raw[-1] += "\n" + line
    for entry in raw:
        match = _RECORD_RE.match(entry)
        assert match is not None
        message = match.group("message")
        round_num: Optional[int] = None
        version: Optional[str] = None
        prefix_match = _PREFIX_RE.match(message)
        if prefix_match:
            for part in prefix_match.group("parts").split("|"):
                part = part.strip()
                if part.startswith("Round "):
                    round_num = int(part.split()[1])
                elif part.startswith("Prompt "):
                    version = part.split()[1]
            message = prefix_match.group("rest")
        records.append(
            LogRecord(
                timestamp=match.group("ts"),
                pid=match.group("pid"),
                round_num=round_num,
                prompt_version=version,
                message=message,
            )
        )
    return records


def _parse_rewards(message: str) -> tuple[List[Optional[float]], float]:
    rewards_match = re.search(r"Rewards: \[(?P<rewards>[^\]]*)\], average is (?P<avg>[\d.eE+-]+)", message)
    assert rewards_match is not None
    rewards: List[Optional[float]] = []
    for token in rewards_match.group("rewards").split(","):
        token = token.strip()
        if token:
            rewards.append(None if token == "None" else float(token))
    return rewards, float(rewards_match.group("avg"))


def build_run_reports(records: List[LogRecord]) -> List[RunReport]:
    """Group records by process id (one APO run per process) and extract the run structure."""
    runs: List[RunReport] = []
    pending_mode: Dict[str, str] = {}
    for record in records:
        if not runs or runs[-1].pid != record.pid:
            runs.append(RunReport(pid=record.pid, started=record.timestamp, finished=record.timestamp))
            pending_mode = {}
        run = runs[-1]
        run.finished = record.timestamp
        version = record.prompt_version
        message = record.message

        if version is not None and version not in run.candidates:
            run.candidates[version] = Candidate(version=version, round_num=record.round_num)
        if record.round_num is not None and record.round_num not in run.rounds:
            run.rounds.append(record.round_num)

        if "Seed prompt baseline score:" in message:
            run.baseline_score = float(message.rsplit(":", 1)[1].strip())
        elif "in train mode" in message and version is not None:
            pending_mode[version] = "train"
        elif "in val mode" in message and version is not None:
            pending_mode[version] = "val"
        elif "Evaluated" in message and "Rewards:" in message and version is not None:
            rewards, average = _parse_rewards(message)
            candidate = run.candidates[version]
            mode = pending_mode.get(version, "val")
            if mode == "train":
                candidate.train_rewards, candidate.train_average = rewards, average
            else:
                candidate.val_rewards = rewards
        elif "Candidate score:" in message and version is not None:
            run.candidates[version].val_score = float(message.rsplit(":", 1)[1].strip())
        elif "Gradient computed" in message and "has result:" in message and version is not None:
            run.candidates[version].gradient = message.split("has result:", 1)[1].strip()
        elif re.match(r"New prompt template created from parent (v\d+): (v\d+)$", message):
            parent, child = re.findall(r"v\d+", message)[-2:]
            run.candidates.setdefault(child, Candidate(version=child, round_num=record.round_num))
            run.candidates[child].parent = parent
            # The gradient/train stats were logged under the parent's prefix while branching.
            source = run.candidates.get(parent)
            if source is not None:
                run.candidates[child].gradient = source.gradient
                run.candidates[child].train_rewards = source.train_rewards
                run.candidates[child].train_average = source.train_average
                source.gradient = source.train_rewards = source.train_average = None
        elif message.startswith("New prompt template created from parent") and version is not None:
            run.candidates[version].prompt_text = message.split(":", 1)[1].strip().strip("`").strip()
        elif "Best prompt updated" in message and version is not None:
            run.best_updated = True
            run.best_version = version
            score_match = re.search(r"New best score: (?P<score>[\d.eE+-]+)", message)
            if score_match:
                run.best_score = float(score_match.group("score"))
        elif "history best:" in message:
            run.best_score = float(message.rsplit("history best:", 1)[1].strip().rstrip(")"))
            if run.best_version is None:
                run.best_version = version
    return runs


def render_markdown(run: RunReport, log_path: Path) -> str:
    lines: List[str] = [
        "# APO Run Report",
        "",
        f"- Log: `{log_path}` (process {run.pid}, {run.started} → {run.finished})",
        f"- Seed prompt (v0) baseline score on val: **{run.baseline_score}**",
        f"- Best prompt: **{run.best_version}** with score **{run.best_score}**"
        + ("" if run.best_updated else " (seed prompt was never beaten)"),
        "",
    ]
    for version, candidate in sorted(run.candidates.items(), key=lambda item: int(item[0][1:])):
        lines.append(f"## Prompt {version}" + (f" (from {candidate.parent})" if candidate.parent else " (seed)"))
        if candidate.round_num is not None:
            lines.append(f"- Round: {candidate.round_num}")
        if candidate.train_rewards is not None:
            lines.append(f"- Train-batch rewards: {candidate.train_rewards} (avg {candidate.train_average})")
        if candidate.val_score is not None:
            lines.append(f"- Validation score: **{candidate.val_score}**" + (f" (rewards {candidate.val_rewards})" if candidate.val_rewards else ""))
        if candidate.gradient:
            lines.extend(["", "<details><summary>Gradient critique</summary>", "", candidate.gradient, "", "</details>"])
        if candidate.prompt_text:
            lines.extend(["", "<details><summary>Full prompt text</summary>", "", "```", candidate.prompt_text, "```", "", "</details>"])
        lines.append("")
    return "\n".join(lines)


def run_to_dict(run: RunReport) -> Dict[str, Any]:
    return {
        "pid": run.pid,
        "started": run.started,
        "finished": run.finished,
        "baseline_score": run.baseline_score,
        "best_version": run.best_version,
        "best_score": run.best_score,
        "best_updated": run.best_updated,
        "candidates": {
            version: {
                "parent": c.parent,
                "round": c.round_num,
                "train_rewards": c.train_rewards,
                "train_average": c.train_average,
                "val_rewards": c.val_rewards,
                "val_score": c.val_score,
                "gradient": c.gradient,
                "prompt_text": c.prompt_text,
            }
            for version, c in run.candidates.items()
        },
    }


def generate_report(
    log_path: Path = PROJECT_ROOT / "log" / "apo.log",
    output_dir: Path = PROJECT_ROOT / "results",
    run_index: int = -1,
) -> Optional[Path]:
    """Parse `log_path` and write report.md/report.json for the selected run.

    Returns the markdown report path, or None when the log contains no runs.
    """
    if not log_path.exists():
        logger.warning("Log file %s not found; skipping report generation.", log_path)
        return None
    runs = build_run_reports(parse_records(log_path))
    if not runs:
        logger.warning("No APO runs found in %s; skipping report generation.", log_path)
        return None
    run = runs[run_index]
    output_dir.mkdir(parents=True, exist_ok=True)
    report_md = output_dir / "report.md"
    report_md.write_text(render_markdown(run, log_path), encoding="utf-8")
    (output_dir / "report.json").write_text(
        json.dumps(run_to_dict(run), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Report for run %s (of %d) written to %s", run.pid, len(runs), report_md)
    return report_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an APO run report from apo.log.")
    parser.add_argument("--log", type=Path, default=PROJECT_ROOT / "log" / "apo.log")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--run", type=int, default=-1, help="Run index in the log (default: last run).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    path = generate_report(args.log, args.output_dir, args.run)
    if path is not None:
        print(path)


if __name__ == "__main__":
    main()
