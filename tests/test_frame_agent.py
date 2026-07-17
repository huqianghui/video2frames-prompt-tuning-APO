"""Offline tests for frame_agent (no network, no credentials)."""

from pathlib import Path
from typing import Any, Dict, List, cast

import pytest

from blob_utils import BlobConfig
from frame_agent import (
    FrameTask,
    build_frame_section,
    build_multimodal_content,
    frame_placeholder,
    prompt_template_baseline,
)

CONFIG = BlobConfig(
    blob_endpoint="https://example.blob.core.windows.net/",
    sas_token="sv=1&sig=abc",
    container_name="process-videos",
    frames_folder="training/frame",
)


def make_task(num_frames: int = 3) -> FrameTask:
    return FrameTask(
        id="0",
        video="/workspace/x/videos/Charades/0A8ZT.mp4",
        family="Charades",
        frame_blobs=[f"training/frame/Charades/0A8ZT.mp4_frame/{i}.jpg" for i in range(num_frames)],
        num_frames=num_frames,
        seconds_per_frame=3,
        solution={
            "english_detail": "A person walked.",
            "brief": "Walked.",
            "title": "Walk",
            "scene_type": "indoor",
            "is_courier_action": False,
        },
    )


def test_frame_placeholder_format() -> None:
    assert frame_placeholder(1, 3) == "<frame 1 | 0s>"
    assert frame_placeholder(2, 3) == "<frame 2 | 3s>"
    assert frame_placeholder(10, 3) == "<frame 10 | 27s>"


def test_build_frame_section() -> None:
    section = build_frame_section(3, 3)
    assert section == "### FRAMES\n<frame 1 | 0s> <frame 2 | 3s> <frame 3 | 6s>"


def test_build_multimodal_content_structure() -> None:
    task = make_task(2)
    parts = cast(List[Dict[str, Any]], build_multimodal_content("INSTRUCTION", task, CONFIG))
    # Leading instruction text + (label text + image) per frame.
    assert len(parts) == 1 + 2 * 2
    assert parts[0]["type"] == "text"
    assert parts[0]["text"].startswith("INSTRUCTION")
    assert parts[1] == {"type": "text", "text": "<frame 1 | 0s>"}
    assert parts[2]["type"] == "image_url"
    assert parts[2]["image_url"]["url"] == (
        "https://example.blob.core.windows.net/process-videos/"
        "training/frame/Charades/0A8ZT.mp4_frame/0.jpg?sv=1&sig=abc"
    )
    assert parts[3] == {"type": "text", "text": "<frame 2 | 3s>"}
    assert "1.jpg" in parts[4]["image_url"]["url"]


def test_prompt_template_baseline_reads_file(tmp_path: Path) -> None:
    prompt_file = tmp_path / "baseline_prompt.txt"
    prompt_file.write_text("You are an expert video analyzer.", encoding="utf-8")
    template = prompt_template_baseline(prompt_file)
    assert template.template == "You are an expert video analyzer."
    assert "<video>" not in template.template


def test_prompt_template_baseline_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        prompt_template_baseline(tmp_path / "missing.txt")
