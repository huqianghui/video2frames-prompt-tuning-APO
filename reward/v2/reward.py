"""Reward v2: the upgraded hybrid reward, redesigned for signal-to-noise ratio.

    soft_opt = 0.45 x judge_detail + 0.20 x judge_brief + 0.10 x judge_title
             + 0.25 x rule_compliance

    Multiplicative gates (caps, not weights):
        scene_type wrong        -> soft_opt x= 0.5
        courier false positive  -> soft_opt x= 0.3
        courier false negative  -> soft_opt x= 0.2
        invalid JSON            -> 0

Design decisions (following the SkillOpt-04 analysis article; see `doc/reward-design.md`):

1. The single combined judge is split into three per-field judges, so the
   text-gradient step sees *which* field failed and the averaged judge noise
   shrinks by ~1/sqrt(3).
2. The saturated scene/courier components become multiplicative caps ("do not
   regress") instead of dead 0.2 weights, returning their weight to components
   that still carry signal.
3. Courier misclassification is asymmetric: a missed courier (false negative)
   is penalized harder than a false alarm.
4. `rule_compliance` scores the hard style rules the baseline instruction has
   always demanded (word limits, "person" instead of gendered terms, no
   camera/frame/timestamp mentions, Non-Notable trigger sentence) with
   deterministic zero-noise checks.

Optionally each judge field can be scored `judge_samples` times with the median
taken (set `judge_samples: 3` and a non-zero `judge_temperature` in
`config.yaml`) to trade 3x judge cost for ~sqrt(3) noise reduction.
"""

from __future__ import annotations

import logging
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# judge_prompts key -> output field under judgment.
JUDGED_FIELDS: Dict[str, str] = {"detail": "english_detail", "brief": "brief", "title": "title"}


class GateConfig(BaseModel):
    """Multiplicative caps applied after the weighted sum."""

    scene_error: float
    courier_false_positive: float
    courier_false_negative: float


class RuleConfig(BaseModel):
    """Deterministic rule-compliance checks (regex / word counts, zero noise)."""

    required_keys: List[str]
    max_words: Dict[str, int]
    forbidden_gender_words: List[str]
    forbidden_meta_words: List[str]
    non_notable_prefix: str


class RewardV2Config(BaseModel):
    """Tunable knobs for reward v2, loaded from `config.yaml`."""

    weights: Dict[str, float]
    gates: GateConfig
    judge_model: str
    judge_temperature: float
    judge_samples: int
    judge_prompts: Dict[str, str]
    rules: RuleConfig
    apo_meta_prompts: Dict[str, Optional[str]] = {}


def _word_pattern(words: List[str]) -> Optional[re.Pattern[str]]:
    if not words:
        return None
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE)


class HybridRewardV2(RewardFunction):
    """Split per-field judges + deterministic rule compliance + multiplicative gates."""

    version = "v2"

    def __init__(self, config: RewardV2Config) -> None:
        missing = set(JUDGED_FIELDS) - set(config.judge_prompts)
        if missing:
            raise ValueError(f"config.yaml judge_prompts missing keys: {sorted(missing)}")
        expected_weights = {f"judge_{key}" for key in JUDGED_FIELDS} | {"rule_compliance"}
        if set(config.weights) != expected_weights:
            raise ValueError(f"config.yaml weights must be exactly {sorted(expected_weights)}, got {sorted(config.weights)}")
        self.config = config
        self._gender_pattern = _word_pattern(config.rules.forbidden_gender_words)
        self._meta_pattern = _word_pattern(config.rules.forbidden_meta_words)

    def score(self, raw_output: str, expected: Dict[str, Any], client: AzureOpenAI) -> RewardResult:
        try:
            generated = parse_model_output(raw_output)
        except ValueError as e:
            logger.warning("Reward v2: invalid JSON output, reward 0. %s", e)
            return RewardResult(total=0.0, components={"invalid_json": 1.0})

        components: Dict[str, float] = {}
        for prompt_key, field_name in JUDGED_FIELDS.items():
            components[f"judge_{prompt_key}"] = self._judge_field(
                client, prompt_key, generated.get(field_name), expected[field_name]
            )
        rule_score, rule_items = self.rule_compliance(generated, expected)
        components["rule_compliance"] = rule_score
        components.update(rule_items)

        return self.combine(generated, expected, components)

    def combine(
        self, generated: Dict[str, Any], expected: Dict[str, Any], components: Dict[str, float]
    ) -> RewardResult:
        """Weighted sum of the soft components, then multiplicative gates (no network)."""
        total = sum(self.config.weights[name] * components[name] for name in self.config.weights)

        scene_ok = str(generated.get("scene_type", "")).strip().lower() == expected["scene_type"]
        if not scene_ok:
            total *= self.config.gates.scene_error
            components["gate_scene_error"] = 1.0

        courier_value = generated.get("is_courier_action")
        if isinstance(courier_value, str):
            courier_value = courier_value.strip().lower() == "true"
        generated_courier = bool(courier_value)
        expected_courier = bool(expected["is_courier_action"])
        if generated_courier and not expected_courier:
            total *= self.config.gates.courier_false_positive
            components["gate_courier_false_positive"] = 1.0
        elif not generated_courier and expected_courier:
            total *= self.config.gates.courier_false_negative
            components["gate_courier_false_negative"] = 1.0

        return RewardResult(total=max(0.0, min(1.0, total)), components=components)

    def _judge_field(self, client: AzureOpenAI, prompt_key: str, generated: Any, expected: Any) -> float:
        """Score one text field with the LLM judge; median over `judge_samples` calls."""
        judge_prompt = self.config.judge_prompts[prompt_key].format(generated=generated, expected=expected)
        scores: List[float] = []
        for sample in range(max(1, self.config.judge_samples)):
            completion = client.chat.completions.parse(
                model=resolve_judge_model(self.config.judge_model),
                messages=[{"role": "user", "content": judge_prompt}],
                response_format=JudgeResponse,
                temperature=self.config.judge_temperature,
            )
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                logger.warning("Judge (%s, sample %d) returned no parsed response; scoring 0.", prompt_key, sample)
                scores.append(0.0)
                continue
            logger.info("Judge %s score %.2f: %s", prompt_key, parsed.score, parsed.reason)
            scores.append(max(0.0, min(1.0, parsed.score)))
        return statistics.median(scores)

    def rule_compliance(self, generated: Dict[str, Any], expected: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Deterministic 0/1 checks averaged into one score (no network).

        Every check mirrors a hard rule already present in the baseline
        instruction; the reward finally measures what the prompt demands.
        """
        rules = self.config.rules
        items: Dict[str, float] = {}

        items["rule_exact_keys"] = float(set(generated.keys()) == set(rules.required_keys))

        text_parts: List[str] = []
        for field_name, limit in rules.max_words.items():
            value = generated.get(field_name)
            text = value if isinstance(value, str) else ""
            text_parts.append(text)
            items[f"rule_len_{field_name}"] = float(bool(text) and len(text.split()) <= limit)
        all_text = " ".join(text_parts)

        items["rule_no_gender_words"] = float(self._gender_pattern is None or not self._gender_pattern.search(all_text))
        items["rule_no_meta_words"] = float(self._meta_pattern is None or not self._meta_pattern.search(all_text))

        generated_detail = generated.get("english_detail")
        generated_trigger = isinstance(generated_detail, str) and generated_detail.startswith(rules.non_notable_prefix)
        expected_trigger = str(expected["english_detail"]).startswith(rules.non_notable_prefix)
        items["rule_non_notable_trigger"] = float(generated_trigger == expected_trigger)

        return sum(items.values()) / len(items), items

    def config_dump(self) -> Dict[str, Any]:
        return self.config.model_dump()


def create_reward() -> HybridRewardV2:
    return HybridRewardV2(RewardV2Config(**load_yaml_config(CONFIG_PATH)))
