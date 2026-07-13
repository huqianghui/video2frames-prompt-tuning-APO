"""Offline tests for prepare_data (uses a small synthetic fixture, not customer data)."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import pytest

import prepare_data
from prepare_data import (
    SourceRecord,
    extract_fixed_prompt,
    load_pandas_column_json,
    normalize_solution,
    resolve_frames,
    stratified_sample,
    video_family,
)

SHARED_PROMPT = "<video> You are an expert video analyzer. Output strictly as a JSON object."


def make_solution(scene: str = "indoor", courier: bool = False) -> str:
    return json.dumps(
        {
            "english_detail": "A person walked across the room.",
            "brief": "A person walked.",
            "title": "Person Walks",
            "scene_type": scene,
            "is_courier_action": courier,
        }
    )


def make_source_file(tmp_path: Path, n_charades: int = 6, n_virat: int = 2) -> Path:
    """Build a pandas column-oriented JSON dump like the customer file."""
    messages: Dict[str, Any] = {}
    solution: Dict[str, Any] = {}
    videos: Dict[str, Any] = {}
    task: Dict[str, Any] = {}
    row = 0
    for _ in range(n_charades):
        messages[str(row)] = [{"role": "user", "content": SHARED_PROMPT}]
        solution[str(row)] = make_solution()
        videos[str(row)] = [f"/workspace/home/azureuser/data/sft_data/videos/Charades/{row:05d}.mp4"]
        task[str(row)] = "main"
        row += 1
    for _ in range(n_virat):
        messages[str(row)] = [{"role": "user", "content": SHARED_PROMPT}]
        solution[str(row)] = make_solution("outdoor", True)
        videos[str(row)] = [f"/workspace/home/azureuser/data/sft_data/videos/VIRAT/clips/{row:05d}.mp4"]
        task[str(row)] = "main"
        row += 1
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"messages": messages, "solution": solution, "videos": videos, "task": task}))
    return path


def test_load_pandas_column_json(tmp_path: Path) -> None:
    records = load_pandas_column_json(make_source_file(tmp_path))
    assert len(records) == 8
    assert records[0]["id"] == "0"
    assert records[0]["family"] == "Charades"
    assert records[6]["family"] == "VIRAT"
    assert records[0]["prompt"] == SHARED_PROMPT
    assert records[0]["solution"]["is_courier_action"] is False
    assert records[7]["solution"]["scene_type"] == "outdoor"


def test_extract_fixed_prompt_strips_video_placeholder(tmp_path: Path) -> None:
    records = load_pandas_column_json(make_source_file(tmp_path))
    fixed = extract_fixed_prompt(records)
    assert "<video>" not in fixed
    assert fixed.startswith("You are an expert video analyzer.")


def test_extract_fixed_prompt_rejects_divergent_prompts(tmp_path: Path) -> None:
    records = load_pandas_column_json(make_source_file(tmp_path))
    records[0]["prompt"] = "<video> A different prompt."
    with pytest.raises(ValueError, match="single shared prompt"):
        extract_fixed_prompt(records)


def test_normalize_solution_coerces_types() -> None:
    raw = json.dumps(
        {
            "english_detail": " detail ",
            "brief": "brief",
            "title": "title",
            "scene_type": " Indoor ",
            "is_courier_action": "True",
        }
    )
    normalized = normalize_solution(raw, "0")
    assert normalized["english_detail"] == "detail"
    assert normalized["scene_type"] == "indoor"
    assert normalized["is_courier_action"] is True


def test_normalize_solution_missing_field_raises() -> None:
    with pytest.raises(ValueError, match="missing fields"):
        normalize_solution(json.dumps({"english_detail": "x"}), "0")


def test_video_family() -> None:
    assert video_family("/workspace/x/videos/ucf_crime/Part-1/Abuse/clip.mp4") == "ucf_crime"


def test_stratified_sample_deterministic_and_covers_families(tmp_path: Path) -> None:
    records = load_pandas_column_json(make_source_file(tmp_path, n_charades=20, n_virat=4))
    sample_a = stratified_sample(records, 6, random.Random(42))
    sample_b = stratified_sample(records, 6, random.Random(42))
    assert [r["id"] for r in sample_a] == [r["id"] for r in sample_b]
    assert len(sample_a) == 6
    families = {r["family"] for r in sample_a}
    assert families == {"Charades", "VIRAT"}
    assert len({r["id"] for r in sample_a}) == 6


def test_resolve_frames_skips_blocked_and_backfills(monkeypatch: pytest.MonkeyPatch) -> None:
    records: List[SourceRecord] = [
        SourceRecord(
            id=str(i),
            prompt=SHARED_PROMPT,
            video=f"/workspace/x/videos/Charades/{i}.mp4",
            family="Charades",
            solution={},
        )
        for i in range(5)
    ]
    monkeypatch.setattr(
        prepare_data, "list_frame_blobs", lambda config, video: [f"training/frame/{video}_frame/0.jpg"]
    )

    probed: List[str] = []

    def is_blocked(task: Dict[str, Any]) -> bool:
        probed.append(task["id"])
        return task["id"] == "1"

    tasks = resolve_frames(records, needed=3, config=None, is_blocked=is_blocked)  # type: ignore[arg-type]
    assert [t["id"] for t in tasks] == ["0", "2", "3"]  # "1" blocked, backfilled by "3"
    assert probed == ["0", "1", "2", "3"]
    assert all(t["num_frames"] == 1 for t in tasks)


def test_stratified_sample_respects_family_sizes() -> None:
    records: List[SourceRecord] = [
        SourceRecord(
            id=str(i),
            prompt=SHARED_PROMPT,
            video=f"/workspace/x/videos/{'Charades' if i < 9 else 'NWPU'}/{i}.mp4",
            family="Charades" if i < 9 else "NWPU",
            solution={},
        )
        for i in range(10)
    ]
    sample = stratified_sample(records, 10, random.Random(0))
    assert len(sample) == 10
