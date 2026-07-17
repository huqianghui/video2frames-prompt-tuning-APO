"""Reward v1: the original hybrid verification reward.

    reward = scene_weight x exact match of scene_type
           + courier_weight x exact match of is_courier_action
           + judge_weight x combined LLM-judge score over english_detail/brief/title

Invalid JSON output scores 0. This version is kept byte-for-byte equivalent to
the pre-modularization implementation so historical runs stay comparable — it
remains the verification/faceoff metric even when training runs on a newer
optimization reward (see `reward/v2/`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from openai import AzureOpenAI
from pydantic import BaseModel

from reward.base import (
    JudgeResponse,
    RewardFunction,
    RewardResult,
    load_yaml_config,
    parse_model_output,
    resolve_judge_model,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


class RewardV1Config(BaseModel):
    """Tunable knobs for reward v1, loaded from `config.yaml`."""

    scene_weight: float
    courier_weight: float
    judge_weight: float
    judge_model: str
    judge_temperature: float
    judge_prompt: str
    apo_meta_prompts: Dict[str, Optional[str]] = {}


class HybridRewardV1(RewardFunction):
    """Weighted sum of two exact-match fields and one combined LLM-judge score."""

    version = "v1"

    def __init__(self, config: RewardV1Config) -> None:
        self.config = config

    def score(self, raw_output: str, expected: Dict[str, Any], client: AzureOpenAI) -> RewardResult:
        try:
            generated = parse_model_output(raw_output)
        except ValueError as e:
            logger.warning("Reward v1: invalid JSON output, reward 0. %s", e)
            return RewardResult(total=0.0, components={"invalid_json": 1.0})
        judge_score = self.judge_text_fields(client, generated, expected)
        return self.combine(generated, expected, judge_score)

    def judge_text_fields(self, client: AzureOpenAI, generated: Dict[str, Any], expected: Dict[str, Any]) -> float:
        """LLM-judge semantic similarity of english_detail/brief/title on a 0-1 scale."""
        judge_prompt = self.config.judge_prompt.format(
            generated_detail=generated.get("english_detail"),
            generated_brief=generated.get("brief"),
            generated_title=generated.get("title"),
            expected_detail=expected["english_detail"],
            expected_brief=expected["brief"],
            expected_title=expected["title"],
        )
        completion = client.chat.completions.parse(
            model=resolve_judge_model(self.config.judge_model),
            messages=[{"role": "user", "content": judge_prompt}],
            response_format=JudgeResponse,
            temperature=self.config.judge_temperature,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            logger.warning("Judge returned no parsed response; scoring 0.")
            return 0.0
        logger.info("Judge score %.2f: %s", parsed.score, parsed.reason)
        return max(0.0, min(1.0, parsed.score))

    def combine(self, generated: Dict[str, Any], expected: Dict[str, Any], judge_score: float) -> RewardResult:
        """Hybrid reward from exact-match fields and the judge score (no network)."""
        scene_match = str(generated.get("scene_type", "")).strip().lower() == expected["scene_type"]
        courier_value = generated.get("is_courier_action")
        if isinstance(courier_value, str):
            courier_value = courier_value.strip().lower() == "true"
        courier_match = bool(courier_value) == expected["is_courier_action"]
        total = (
            self.config.scene_weight * scene_match
            + self.config.courier_weight * courier_match
            + self.config.judge_weight * judge_score
        )
        return RewardResult(
            total=total,
            components={
                "scene_match": float(scene_match),
                "courier_match": float(courier_match),
                "judge": judge_score,
            },
        )

    def config_dump(self) -> Dict[str, Any]:
        return self.config.model_dump()


def create_reward() -> HybridRewardV1:
    return HybridRewardV1(RewardV1Config(**load_yaml_config(CONFIG_PATH)))
