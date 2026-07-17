"""Shared interfaces for versioned reward functions.

Each reward version lives in its own subpackage (`reward/v1/`, `reward/v2/`, ...)
containing a `reward.py` implementation, a `config.yaml` with every tunable
knob (weights, judge model, judge prompts, rules), and optional POML overrides
for the APO meta-prompts. Versions are selected at runtime via the
`REWARD_VERSION` environment variable or the `--reward-version` CLI flag (see
[load_reward][reward.load_reward]).
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, cast

import yaml
from openai import AzureOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class JudgeResponse(BaseModel):
    """Structured output returned by the LLM judge."""

    reason: str = Field(description="The reason for the score. No more than 100 characters.")
    score: float = Field(description="The score for the semantic match on a 0-1 scale. Be critical.")


@dataclass
class RewardResult:
    """Final reward plus a per-component breakdown for logging and analysis."""

    total: float
    components: Dict[str, float] = field(default_factory=dict)


class RewardFunction(ABC):
    """One versioned reward implementation.

    Subclasses score the raw model output of a rollout against the ground-truth
    solution and return both the scalar reward (fed back to APO) and the
    component breakdown (logged for offline analysis).
    """

    version: str

    @abstractmethod
    def score(self, raw_output: str, expected: Dict[str, Any], client: AzureOpenAI) -> RewardResult:
        """Score the raw model output against the ground-truth solution."""

    @abstractmethod
    def config_dump(self) -> Dict[str, Any]:
        """Reward configuration recorded into run summaries for reproducibility."""


def parse_model_output(raw: str) -> Dict[str, Any]:
    """Parse the model's JSON output, tolerating markdown code fences.

    Raises `ValueError` when the output is not a JSON object. Every reward
    version treats that as a broken output contract and scores 0.
    """
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model output is not valid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise ValueError(f"Model output is not a JSON object: {type(parsed)}")
    return cast(Dict[str, Any], parsed)


def resolve_judge_model(config_value: str) -> str:
    """The `JUDGE_MODEL` environment variable wins over the per-version config value."""
    return os.environ.get("JUDGE_MODEL", config_value)


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load a version's `config.yaml` into a plain dict (validated by the version's pydantic model)."""
    with open(path, encoding="utf-8") as f:
        data: Any = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(data)}")
    return cast(Dict[str, Any], data)
