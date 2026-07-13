"""Offline tests for the content-filter probe cache (no network, no credentials)."""

from typing import Any, Dict

import probe_content_filter as pcf


def make_task(task_id: str = "42") -> Dict[str, Any]:
    return {
        "id": task_id,
        "video": "/workspace/home/azureuser/data/sft_data/videos/Charades/X.mp4",
        "family": "Charades",
        "frame_blobs": ["training/frame/Charades/X.mp4_frame/0.jpg"],
        "num_frames": 1,
        "seconds_per_frame": 3,
        "solution": {},
    }


def test_load_probe_cache_missing_returns_empty(tmp_path):
    assert pcf.load_probe_cache(tmp_path / "missing.json") == {}


def test_save_and_load_probe_cache_round_trip(tmp_path):
    path = tmp_path / "cache.json"
    pcf.save_probe_cache({"1": True, "2": False}, path)
    assert pcf.load_probe_cache(path) == {"1": True, "2": False}


def test_probe_task_cached_hits_cache_without_probing(monkeypatch):
    def fail_probe(*args: Any) -> bool:
        raise AssertionError("probe_task must not be called on a cache hit")

    monkeypatch.setattr(pcf, "probe_task", fail_probe)
    cache = {"42": True}
    assert pcf.probe_task_cached(None, make_task("42"), None, cache) is True  # type: ignore[arg-type]
    assert cache == {"42": True}


def test_probe_task_cached_probes_once_and_persists(monkeypatch, tmp_path):
    calls: list[str] = []
    saved_path = tmp_path / "cache.json"

    def fake_probe(client: Any, task: Dict[str, Any], config: Any) -> bool:
        calls.append(task["id"])
        return False

    real_save = pcf.save_probe_cache
    monkeypatch.setattr(pcf, "probe_task", fake_probe)
    monkeypatch.setattr(pcf, "save_probe_cache", lambda cache: real_save(cache, saved_path))

    cache: Dict[str, bool] = {}
    assert pcf.probe_task_cached(None, make_task("7"), None, cache) is False  # type: ignore[arg-type]
    assert pcf.probe_task_cached(None, make_task("7"), None, cache) is False  # type: ignore[arg-type]
    assert calls == ["7"]
    assert cache == {"7": False}
    assert pcf.load_probe_cache(saved_path) == {"7": False}
