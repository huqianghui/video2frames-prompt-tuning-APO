"""Offline tests for prepare_data (uses a small synthetic fixture, not customer data)."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import pytest

import prepare_data
from prepare_data import (
    SourceRecord,
    allocate_cells,
    extract_fixed_prompt,
    load_frozen_test_videos,
    load_pandas_column_json,
    normalize_solution,
    resolve_frames,
    stratified_sample,
    stratified_split,
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


def make_source_file(tmp_path: Path, n_charades: int = 6, n_virat: int = 2, n_charades_courier: int = 0) -> Path:
    """Build a pandas column-oriented JSON dump like the customer file."""
    messages: Dict[str, Any] = {}
    solution: Dict[str, Any] = {}
    videos: Dict[str, Any] = {}
    task: Dict[str, Any] = {}
    row = 0
    for i in range(n_charades):
        messages[str(row)] = [{"role": "user", "content": SHARED_PROMPT}]
        solution[str(row)] = make_solution(courier=i < n_charades_courier)
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
            solution={"scene_type": "indoor", "is_courier_action": False},
        )
        for i in range(10)
    ]
    sample = stratified_sample(records, 10, random.Random(0))
    assert len(sample) == 10


def test_stratified_sample_covers_joint_cells(tmp_path: Path) -> None:
    records = load_pandas_column_json(make_source_file(tmp_path, n_charades=20, n_virat=4, n_charades_courier=8))
    sample = stratified_sample(records, 12, random.Random(42))
    cells = {(r["family"], r["solution"]["is_courier_action"]) for r in sample}
    assert cells == {("Charades", False), ("Charades", True), ("VIRAT", True)}


def make_tasks(spec: Dict[Any, int]) -> List[Dict[str, Any]]:
    """Build resolved-task dicts from a {(family, is_courier_action): count} spec."""
    tasks: List[Dict[str, Any]] = []
    index = 0
    for (family, courier), count in spec.items():
        for _ in range(count):
            tasks.append(
                {
                    "id": str(index),
                    "video": f"/workspace/x/videos/{family}/{index}.mp4",
                    "family": family,
                    "frame_blobs": ["0.jpg"],
                    "num_frames": 1,
                    "seconds_per_frame": 3,
                    "solution": {
                        "english_detail": "d",
                        "brief": "b",
                        "title": "t",
                        "scene_type": "indoor",
                        "is_courier_action": courier,
                    },
                }
            )
            index += 1
    return tasks


def courier_count(tasks: List[Dict[str, Any]]) -> int:
    return sum(1 for t in tasks if t["solution"]["is_courier_action"])


def test_allocate_cells_exact_sizes() -> None:
    cell_sizes = {"a": 7, "b": 5, "c": 1}
    split_sizes = {"train": 6, "val": 4, "test": 3}
    allocation = allocate_cells(cell_sizes, split_sizes)
    for split, size in split_sizes.items():
        assert sum(counts[split] for counts in allocation.values()) == size
    for cell, size in cell_sizes.items():
        assert sum(allocation[cell].values()) == size
        assert all(count >= 0 for count in allocation[cell].values())


def test_allocate_cells_spreads_singletons() -> None:
    allocation = allocate_cells({"a": 1, "b": 1, "c": 1}, {"x": 1, "y": 1, "z": 1})
    placements = {split for counts in allocation.values() for split, count in counts.items() if count == 1}
    assert placements == {"x", "y", "z"}  # singletons don't all pile into one split


def test_allocate_cells_size_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="sum to"):
        allocate_cells({"a": 3}, {"train": 2})


def test_stratified_split_mirrors_distribution() -> None:
    tasks = make_tasks({("A", False): 14, ("A", True): 6, ("B", False): 10})
    splits = stratified_split(tasks, {"train": 15, "val": 9, "test": 6}, random.Random(42))
    assert [len(splits[name]) for name in ("train", "val", "test")] == [15, 9, 6]
    # Pool is 20% courier-positive; every split stays close to that.
    assert courier_count(splits["train"]) == 3
    assert courier_count(splits["val"]) == 2
    assert courier_count(splits["test"]) == 1
    all_ids = [t["id"] for rows in splits.values() for t in rows]
    assert sorted(all_ids) == sorted(t["id"] for t in tasks)  # disjoint, nothing lost


def test_stratified_split_enforces_val_courier_floor() -> None:
    # Pool is only 10% positive; proportional allocation would give val 1 positive.
    tasks = make_tasks({("A", False): 18, ("A", True): 2})
    splits = stratified_split(tasks, {"train": 10, "val": 10}, random.Random(42), val_courier_min=0.2)
    assert courier_count(splits["val"]) == 2  # ceil(10 * 0.2)
    assert len(splits["val"]) == 10 and len(splits["train"]) == 10


def test_stratified_split_warns_when_pool_lacks_positives(caplog: pytest.LogCaptureFixture) -> None:
    tasks = make_tasks({("A", False): 19, ("A", True): 1})
    with caplog.at_level("WARNING", logger="prepare_data"):
        splits = stratified_split(tasks, {"train": 10, "val": 10}, random.Random(42), val_courier_min=0.3)
    assert courier_count(splits["val"]) == 1  # all the pool has
    assert any("unreachable" in record.message for record in caplog.records)


def test_stratified_split_deterministic_and_size_checked() -> None:
    tasks = make_tasks({("A", False): 8, ("A", True): 4, ("B", False): 4})
    split_a = stratified_split(tasks, {"train": 8, "val": 8}, random.Random(7))
    split_b = stratified_split(tasks, {"train": 8, "val": 8}, random.Random(7))
    assert [t["id"] for t in split_a["val"]] == [t["id"] for t in split_b["val"]]
    with pytest.raises(ValueError, match="split sizes sum to"):
        stratified_split(tasks, {"train": 8, "val": 4}, random.Random(7))


def test_load_frozen_test_videos(tmp_path: Path) -> None:
    test_path = tmp_path / "test.jsonl"
    rows = make_tasks({("A", False): 2, ("B", True): 1})
    test_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    videos = load_frozen_test_videos(test_path)
    assert videos == {row["video"] for row in rows}
    with pytest.raises(FileNotFoundError, match="freeze-test"):
        load_frozen_test_videos(tmp_path / "missing.jsonl")
