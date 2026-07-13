"""Offline tests for generate_report (synthetic apo.log fixture, no customer data)."""

import json
from pathlib import Path

from generate_report import build_run_reports, generate_report, generate_tree_from_report, parse_records

LOGGER_NAME = "agentlightning.algorithm.apo.apo"


def line(ts: str, pid: str, prefix: str, message: str, level: str = "INFO") -> str:
    return f"2026-07-13 {ts} [{level}] (Process-{pid} {LOGGER_NAME})   [{prefix}] {message}"


def make_log(tmp_path: Path) -> Path:
    rows = [
        # Run 1: an earlier aborted run under a different process id.
        line("09:00:00,000", "111", "Round 00 | Prompt v0", "Evaluating seed prompt on validation dataset before optimization..."),
        line("09:00:01,000", "111", "Round 00 | Prompt v0", "Seed prompt baseline score: 0.100"),
        # Run 2: the run of interest.
        line("10:00:00,000", "222", "Round 00 | Prompt v0", 'Evaluating prompt "You are..." on 2 tasks in val mode'),
        line("10:00:01,000", "222", "Round 00 | Prompt v0", "Evaluated 2 rollouts. Statuses: Counter({'succeeded': 2}). Rewards: [0.74, 0.0], average is 0.37"),
        line("10:00:02,000", "222", "Round 00 | Prompt v0", "Seed prompt baseline score: 0.370"),
        line("10:00:03,000", "222", "Round 01 | Beam 01 | Branch 01 | Prompt v0", 'Evaluating prompt "You are..." on 2 tasks in train mode'),
        line("10:00:04,000", "222", "Round 01 | Beam 01 | Branch 01 | Prompt v0", "Evaluated 2 rollouts. Statuses: Counter({'succeeded': 2}). Rewards: [0.64, 0.9400000000000001], average is 0.79"),
        line("10:00:05,000", "222", "Round 01 | Beam 01 | Branch 01 | Prompt v0", "Gradient computed with gpt-4.1 has result: ## Critique"),
        "- Point one",
        "- Point two",
        line("10:00:06,000", "222", "Round 01 | Beam 01 | Branch 01 | Prompt v0", "New prompt template created from parent v0: v1"),
        line("10:00:07,000", "222", "Round 01 | Prompt v1", "New prompt template created from parent v0:"),
        "```You are an improved analyzer.",
        "Answer in JSON.```",
        line("10:00:08,000", "222", "Round 01 | Prompt v0", 'Evaluating prompt "You are..." on 2 tasks in val mode'),
        line("10:00:09,000", "222", "Round 01 | Prompt v0", "Evaluated 2 rollouts. Statuses: Counter({'succeeded': 2}). Rewards: [0.74, 0.0], average is 0.37"),
        line("10:00:10,000", "222", "Round 01 | Prompt v0", "Candidate score: 0.370"),
        line("10:00:11,000", "222", "Round 01 | Prompt v1", 'Evaluating prompt "You are an improved'),
        'analyzer..." on 2 tasks in val mode',
        line("10:00:12,000", "222", "Round 01 | Prompt v1", "Evaluated 2 rollouts. Statuses: Counter({'succeeded': 2}). Rewards: [0.9, 0.7], average is 0.8"),
        line("10:00:13,000", "222", "Round 01 | Prompt v1", "Candidate score: 0.800"),
        line("10:00:14,000", "222", "Round 01", "Top 2 candidates on validation dataset: ['v1:0.800', 'v0:0.370']"),
        line("10:00:15,000", "222", "Round 01 | Prompt v1", "Best prompt updated. New best score: 0.800 (prev: 0.370)"),
    ]
    path = tmp_path / "apo.log"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_parse_records_folds_continuations_and_prefixes(tmp_path: Path) -> None:
    records = parse_records(make_log(tmp_path))
    assert len(records) == 18  # continuation lines folded, not separate records
    gradient = next(r for r in records if "Gradient computed" in r.message)
    assert "- Point two" in gradient.message
    assert gradient.round_num == 1
    assert gradient.prompt_version == "v0"
    multiline_eval = next(r for r in records if "You are an improved" in r.message and "Evaluating" in r.message)
    assert "in val mode" in multiline_eval.message


def test_build_run_reports_extracts_run_structure(tmp_path: Path) -> None:
    runs = build_run_reports(parse_records(make_log(tmp_path)))
    assert [run.pid for run in runs] == ["111", "222"]
    run = runs[-1]
    assert run.baseline_score == 0.37
    assert set(run.candidates) == {"v0", "v1"}

    v0, v1 = run.candidates["v0"], run.candidates["v1"]
    assert v1.parent == "v0"
    # Branch train stats and gradient are logged under the parent prefix but belong to the child.
    assert v0.gradient is None and v0.train_average is None
    assert v1.gradient is not None and "- Point one" in v1.gradient
    assert v1.train_rewards == [0.64, 0.9400000000000001]
    assert v1.train_average == 0.79
    assert v1.prompt_text == "You are an improved analyzer.\nAnswer in JSON."

    assert v0.val_score == 0.37
    assert v1.val_score == 0.8
    assert v1.val_rewards == [0.9, 0.7]
    assert run.best_updated is True
    assert run.best_version == "v1"
    assert run.best_score == 0.8
    assert run.beam_history == {1: ["v1", "v0"]}


def test_generate_report_writes_markdown_and_json(tmp_path: Path) -> None:
    log_path = make_log(tmp_path)
    report_md = generate_report(log_path=log_path, output_dir=tmp_path / "results")
    assert report_md is not None and report_md.exists()
    text = report_md.read_text(encoding="utf-8")
    assert "## Prompt v1 (from v0)" in text
    assert "Best prompt: **v1** with score **0.8**" in text
    assert "You are an improved analyzer." in text

    data = json.loads((tmp_path / "results" / "report.json").read_text(encoding="utf-8"))
    assert data["pid"] == "222"
    assert data["candidates"]["v1"]["val_score"] == 0.8
    assert data["beam_history"] == {"1": ["v1", "v0"]}

    tree = (tmp_path / "results" / "tree.md").read_text(encoding="utf-8")
    assert "v0 (seed) | val 0.37 | beam R1" in tree
    assert "└── v1 (R1) | val 0.8 | beam R1 | * BEST 0.8" in tree
    assert "You are an improved analyzer." not in tree  # no prompt texts in the tree view


def test_generate_report_missing_log_returns_none(tmp_path: Path) -> None:
    assert generate_report(log_path=tmp_path / "missing.log", output_dir=tmp_path / "results") is None


def test_generate_tree_from_report(tmp_path: Path) -> None:
    generate_report(log_path=make_log(tmp_path), output_dir=tmp_path / "results")
    report_md = tmp_path / "results" / "report.md"
    tree_md = generate_tree_from_report(report_md)
    assert tree_md == tmp_path / "results" / "tree.md"  # defaults to the report's directory
    tree = tree_md.read_text(encoding="utf-8")
    assert "v0 (seed) | val 0.37" in tree
    assert "└── v1 (R1) | val 0.8 | * BEST 0.8" in tree  # no beam markers in this mode
    assert "beam R1" not in tree
    assert str(report_md) in tree  # source points at the report, not the log


def test_generate_tree_from_report_missing_file_returns_none(tmp_path: Path) -> None:
    assert generate_tree_from_report(tmp_path / "missing.md", output_dir=tmp_path / "results") is None
