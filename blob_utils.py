"""Azure Blob Storage helpers for the video2frames prompt tuning project.

Reads the blob connection settings from the repository root `.env` file
(`blob4videodatasets_*` variables) and provides utilities to map the original
video paths found in `qwen_0318_swift_task.json` to their pre-extracted frame
blobs, list those frames in playback order, and build SAS-signed URLs that can
be passed directly to a multimodal LLM.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from urllib.parse import quote

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent

VIDEO_PATH_MARKER = "/videos/"
FRAME_DIR_SUFFIX = "_frame"


# Common Azure OpenAI variable-name variants mapped to the names the openai SDK reads.
_ENV_ALIASES = {
    "AZURE_OPENAI_API_KEY": "AZURE_OPENAI_KEY",
    "OPENAI_API_VERSION": "AZURE_OPENAI_API_VERSION",
}


def load_env() -> None:
    """Load environment variables from the repo root and project `.env` files."""
    for env_path in (REPO_ROOT / ".env", PROJECT_ROOT / ".env"):
        if env_path.exists():
            load_dotenv(env_path, override=False)
    for canonical, alias in _ENV_ALIASES.items():
        if not os.environ.get(canonical) and os.environ.get(alias):
            os.environ[canonical] = os.environ[alias]


@dataclass(frozen=True)
class BlobConfig:
    """Connection settings for the frames blob container."""

    blob_endpoint: str
    sas_token: str
    container_name: str
    frames_folder: str

    @property
    def container_url(self) -> str:
        return f"{self.blob_endpoint.rstrip('/')}/{self.container_name}"


def parse_connection_string(connection_string: str) -> Dict[str, str]:
    """Parse an Azure storage connection string into a key-value dict."""
    parts: Dict[str, str] = {}
    for segment in connection_string.split(";"):
        segment = segment.strip()
        if not segment:
            continue
        key, _, value = segment.partition("=")
        parts[key] = value
    return parts


def blob_config_from_env() -> BlobConfig:
    """Build a [BlobConfig][blob_utils.BlobConfig] from `blob4videodatasets_*` env vars."""
    load_env()
    connection_string = os.environ.get("blob4videodatasets_connection_string", "")
    if not connection_string:
        raise RuntimeError(
            "blob4videodatasets_connection_string is not set. "
            "Ensure the repository root .env file exists and contains the blob settings."
        )
    parts = parse_connection_string(connection_string)
    blob_endpoint = parts.get("BlobEndpoint", "")
    sas_token = parts.get("SharedAccessSignature", "")
    if not blob_endpoint or not sas_token:
        raise RuntimeError("Connection string is missing BlobEndpoint or SharedAccessSignature.")
    return BlobConfig(
        blob_endpoint=blob_endpoint,
        sas_token=sas_token,
        container_name=os.environ.get("blob4videodatasets_container_name", "process-videos"),
        frames_folder=os.environ.get("blob4videodatasets_frames_folder_name", "training/frame"),
    )


def video_path_to_frame_prefix(video_path: str, frames_folder: str = "training/frame") -> str:
    """Map an original video path to its frame blob prefix.

    Example:
        `/workspace/home/azureuser/data/sft_data/videos/Charades/0A8ZT.mp4`
        maps to `training/frame/Charades/0A8ZT.mp4_frame/`.
    """
    marker_index = video_path.find(VIDEO_PATH_MARKER)
    if marker_index < 0:
        raise ValueError(f"Video path does not contain {VIDEO_PATH_MARKER!r}: {video_path}")
    relative = video_path[marker_index + len(VIDEO_PATH_MARKER) :]
    return f"{frames_folder.strip('/')}/{relative}{FRAME_DIR_SUFFIX}/"


_FRAME_INDEX_RE = re.compile(r"(\d+)\.[a-zA-Z]+$")


def sort_frame_blobs(blob_names: List[str]) -> List[str]:
    """Sort frame blob names numerically by frame index (0.jpg, 1.jpg, ..., 10.jpg)."""

    def frame_index(name: str) -> int:
        match = _FRAME_INDEX_RE.search(name.rsplit("/", 1)[-1])
        if match is None:
            raise ValueError(f"Cannot extract frame index from blob name: {name}")
        return int(match.group(1))

    return sorted(blob_names, key=frame_index)


def list_frame_blobs(config: BlobConfig, video_path: str) -> List[str]:
    """List the frame blobs for a video, sorted in playback order.

    Returns an empty list when no frames were extracted for the video.
    """
    from azure.storage.blob import ContainerClient

    prefix = video_path_to_frame_prefix(video_path, config.frames_folder)
    container = ContainerClient.from_container_url(f"{config.container_url}?{config.sas_token}")
    names = [blob.name for blob in container.list_blobs(name_starts_with=prefix)]
    if not names:
        logger.warning("No frames found under blob prefix %s", prefix)
        return []
    return sort_frame_blobs(names)


def blob_sas_url(config: BlobConfig, blob_path: str) -> str:
    """Build a SAS-signed HTTPS URL for a blob path (URL-quoting spaces etc.)."""
    return f"{config.container_url}/{quote(blob_path)}?{config.sas_token}"
