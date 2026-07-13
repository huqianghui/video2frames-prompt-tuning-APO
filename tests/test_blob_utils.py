"""Offline tests for blob_utils (no network, no credentials)."""

import pytest

from blob_utils import (
    BlobConfig,
    blob_sas_url,
    parse_connection_string,
    sort_frame_blobs,
    video_path_to_frame_prefix,
)

CONFIG = BlobConfig(
    blob_endpoint="https://example.blob.core.windows.net/",
    sas_token="sv=2026-02-06&sig=abc",
    container_name="process-videos",
    frames_folder="training/frame",
)


def test_parse_connection_string() -> None:
    parts = parse_connection_string(
        "BlobEndpoint=https://example.blob.core.windows.net/;QueueEndpoint=https://example.queue.core.windows.net/;"
        "SharedAccessSignature=sv=2026-02-06&sig=abc"
    )
    assert parts["BlobEndpoint"] == "https://example.blob.core.windows.net/"
    assert parts["SharedAccessSignature"] == "sv=2026-02-06&sig=abc"


@pytest.mark.parametrize(
    "video_path,expected",
    [
        (
            "/workspace/home/azureuser/data/sft_data/videos/Charades/0A8ZT.mp4",
            "training/frame/Charades/0A8ZT.mp4_frame/",
        ),
        (
            "/workspace/home/azureuser/data/sft_data/videos/NWPU/video/NWPUCampusDataset/processed/D001_01.mp4",
            "training/frame/NWPU/video/NWPUCampusDataset/processed/D001_01.mp4_frame/",
        ),
        (
            "/workspace/home/azureuser/data/sft_data/videos/VIRAT/Public Dataset/VIRAT Video Dataset Release 1.0/"
            "Training Dataset/videos/VIRAT_S_000200_01_000226_000268.mp4",
            "training/frame/VIRAT/Public Dataset/VIRAT Video Dataset Release 1.0/"
            "Training Dataset/videos/VIRAT_S_000200_01_000226_000268.mp4_frame/",
        ),
    ],
)
def test_video_path_to_frame_prefix(video_path: str, expected: str) -> None:
    assert video_path_to_frame_prefix(video_path, "training/frame") == expected


def test_video_path_without_marker_raises() -> None:
    with pytest.raises(ValueError):
        video_path_to_frame_prefix("/somewhere/else/clip.mp4")


def test_sort_frame_blobs_numeric() -> None:
    names = [f"prefix/{i}.jpg" for i in (0, 1, 10, 11, 2, 3)]
    assert sort_frame_blobs(names) == [f"prefix/{i}.jpg" for i in (0, 1, 2, 3, 10, 11)]


def test_sort_frame_blobs_bad_name_raises() -> None:
    with pytest.raises(ValueError):
        sort_frame_blobs(["prefix/not-a-frame.txt"])


def test_blob_sas_url_quotes_spaces() -> None:
    url = blob_sas_url(CONFIG, "training/frame/VIRAT/Public Dataset/clip.mp4_frame/0.jpg")
    assert url == (
        "https://example.blob.core.windows.net/process-videos/"
        "training/frame/VIRAT/Public%20Dataset/clip.mp4_frame/0.jpg?sv=2026-02-06&sig=abc"
    )
