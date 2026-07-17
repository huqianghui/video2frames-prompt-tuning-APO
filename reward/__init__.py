"""Versioned reward registry for the video2frames task.

The reward function is the optimization target of the whole APO pipeline, so it
is packaged for fast iteration: each version is a self-contained subpackage

    reward/
        v1/  config.yaml + reward.py + *.poml  (original hybrid verification reward)
        v2/  config.yaml + reward.py + *.poml  (upgraded hybrid reward: per-field judges + rules + gates)

Adding a `v3` means copying an existing version folder and editing its
`config.yaml`/`reward.py`. Each version owns its `text_gradient_*.poml` — it
states the optimization objective, which is reward-specific — and declares it
in the `apo_meta_prompts` section of its `config.yaml`; the reward-agnostic
apply-edit prompt defaults to the shared `prompts/` file. No other file in the
project needs to change.

Selection order: explicit argument > `REWARD_VERSION` env var > `v1`.
The env var is used so the choice propagates into forked runner processes.
"""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from typing import List, Optional, cast

from reward.base import JudgeResponse, RewardFunction, RewardResult, load_yaml_config, parse_model_output

__all__ = [
    "DEFAULT_VERSION",
    "REWARD_VERSION_ENV",
    "JudgeResponse",
    "RewardFunction",
    "RewardResult",
    "available_versions",
    "load_reward",
    "parse_model_output",
    "poml_override",
    "resolve_version",
]

logger = logging.getLogger(__name__)

REWARD_DIR = Path(__file__).resolve().parent
DEFAULT_VERSION = "v1"
REWARD_VERSION_ENV = "REWARD_VERSION"


def available_versions() -> List[str]:
    """Version subpackages present on disk (any `reward/<name>/__init__.py`)."""
    return sorted(p.name for p in REWARD_DIR.iterdir() if p.is_dir() and (p / "__init__.py").exists())


def resolve_version(version: Optional[str] = None) -> str:
    """Resolve the reward version: explicit argument > `REWARD_VERSION` env var > default."""
    resolved = version or os.environ.get(REWARD_VERSION_ENV) or DEFAULT_VERSION
    if not (REWARD_DIR / resolved / "__init__.py").exists():
        raise ValueError(f"Unknown reward version {resolved!r}. Available: {available_versions()}")
    return resolved


def load_reward(version: Optional[str] = None) -> RewardFunction:
    """Load the reward implementation for `version` (see [resolve_version][reward.resolve_version]).

    Each version package must expose a `get_reward() -> RewardFunction` factory.
    """
    resolved = resolve_version(version)
    module = importlib.import_module(f"reward.{resolved}")
    reward = cast(RewardFunction, module.get_reward())
    if not isinstance(reward, RewardFunction):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"reward.{resolved}.get_reward() returned {type(reward)}, expected RewardFunction")
    return reward


def poml_override(version: str, kind: str) -> Optional[Path]:
    """Per-version APO meta-prompt override declared in the version's `config.yaml`.

    Each version's `config.yaml` has an `apo_meta_prompts` section mapping a
    kind (`text_gradient` / `apply_edit`) to either `null` (use the shared
    default in `prompts/`) or a filename inside `reward/<version>/`.

    Returns the resolved override path, or `None` to fall back to the shared default.
    """
    config = load_yaml_config(REWARD_DIR / resolve_version(version) / "config.yaml")
    filename = (config.get("apo_meta_prompts") or {}).get(kind)
    if filename is None:
        return None
    path = REWARD_DIR / version / filename
    if not path.exists():
        raise FileNotFoundError(
            f"reward/{version}/config.yaml declares apo_meta_prompts.{kind} = {filename!r}, but {path} does not exist"
        )
    return path
