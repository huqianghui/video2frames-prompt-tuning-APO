"""Offline tests for the versioned reward package (no network, no credentials)."""

import json
from typing import Any, Dict, cast

import pytest
from openai import AzureOpenAI

from reward import available_versions, load_reward, parse_model_output, poml_override, resolve_version
from reward.v1.reward import HybridRewardV1
from reward.v1.reward import create_reward as create_reward_v1
from reward.v2.reward import HybridRewardV2
from reward.v2.reward import create_reward as create_reward_v2

FAKE_CLIENT = cast(AzureOpenAI, object())

SOLUTION: Dict[str, Any] = {
    "english_detail": "A person walked.",
    "brief": "Walked.",
    "title": "Walk",
    "scene_type": "indoor",
    "is_courier_action": False,
}

GOOD_OUTPUT: Dict[str, Any] = dict(SOLUTION)


# --- registry ---------------------------------------------------------------


def test_available_versions_contains_v1_v2() -> None:
    assert {"v1", "v2"} <= set(available_versions())


def test_resolve_version_default_is_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REWARD_VERSION", raising=False)
    assert resolve_version() == "v1"


def test_resolve_version_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REWARD_VERSION", "v2")
    assert resolve_version() == "v2"
    # An explicit argument wins over the env var.
    assert resolve_version("v1") == "v1"


def test_resolve_version_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown reward version"):
        resolve_version("v999")


def test_load_reward_versions() -> None:
    assert load_reward("v1").version == "v1"
    assert load_reward("v2").version == "v2"


def test_poml_override_declared_in_config() -> None:
    # The text-gradient prompt states the objective, so every version owns one.
    for version in ("v1", "v2"):
        override = poml_override(version, "text_gradient")
        assert override is not None and override.exists()
        assert override.parent.name == version
    # apply_edit is reward-agnostic: both versions fall back to the shared prompts/.
    assert poml_override("v1", "apply_edit") is None
    assert poml_override("v2", "apply_edit") is None


# --- shared parsing ----------------------------------------------------------


def test_parse_model_output_plain_and_fenced() -> None:
    payload = {"english_detail": "x", "scene_type": "indoor"}
    assert parse_model_output(json.dumps(payload)) == payload
    fenced = f"```json\n{json.dumps(payload)}\n```"
    assert parse_model_output(fenced) == payload


@pytest.mark.parametrize("raw", ["not json at all", "[1, 2, 3]", "```json\nnope\n```"])
def test_parse_model_output_invalid_raises(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_model_output(raw)


# --- v1 ----------------------------------------------------------------------


@pytest.fixture(name="v1")
def v1_fixture() -> HybridRewardV1:
    return create_reward_v1()


def test_v1_combine_all_correct(v1: HybridRewardV1) -> None:
    assert v1.combine(dict(SOLUTION), SOLUTION, judge_score=1.0).total == pytest.approx(1.0)


def test_v1_combine_partial(v1: HybridRewardV1) -> None:
    generated = {"scene_type": "outdoor", "is_courier_action": False}
    # scene wrong (0), courier right (0.2), judge 0.5 -> 0.2 + 0.3
    assert v1.combine(generated, SOLUTION, judge_score=0.5).total == pytest.approx(0.5)


def test_v1_combine_courier_string_coercion(v1: HybridRewardV1) -> None:
    generated = {"scene_type": "indoor", "is_courier_action": "false"}
    assert v1.combine(generated, SOLUTION, judge_score=0.0).total == pytest.approx(0.4)


def test_v1_score_invalid_json_is_zero(v1: HybridRewardV1) -> None:
    result = v1.score("not json", SOLUTION, FAKE_CLIENT)
    assert result.total == 0.0
    assert result.components == {"invalid_json": 1.0}


def test_v1_score_with_stubbed_judge(v1: HybridRewardV1, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(HybridRewardV1, "judge_text_fields", lambda self, client, generated, expected: 0.5)
    result = v1.score(json.dumps(GOOD_OUTPUT), SOLUTION, FAKE_CLIENT)
    assert result.total == pytest.approx(0.2 + 0.2 + 0.6 * 0.5)
    assert result.components["judge"] == 0.5


# --- v2: rule compliance -----------------------------------------------------


@pytest.fixture(name="v2")
def v2_fixture() -> HybridRewardV2:
    return create_reward_v2()


def test_v2_rule_compliance_perfect(v2: HybridRewardV2) -> None:
    score, items = v2.rule_compliance(dict(GOOD_OUTPUT), SOLUTION)
    assert score == pytest.approx(1.0)
    assert all(value == 1.0 for value in items.values())


def test_v2_rule_word_limits(v2: HybridRewardV2) -> None:
    generated = dict(GOOD_OUTPUT)
    generated["english_detail"] = "word " * 51
    generated["title"] = "one two three four five six seven"
    _, items = v2.rule_compliance(generated, SOLUTION)
    assert items["rule_len_english_detail"] == 0.0
    assert items["rule_len_title"] == 0.0
    assert items["rule_len_brief"] == 1.0


def test_v2_rule_empty_field_fails_length(v2: HybridRewardV2) -> None:
    generated = dict(GOOD_OUTPUT)
    generated["brief"] = ""
    _, items = v2.rule_compliance(generated, SOLUTION)
    assert items["rule_len_brief"] == 0.0


def test_v2_rule_gender_words(v2: HybridRewardV2) -> None:
    generated = dict(GOOD_OUTPUT)
    generated["english_detail"] = "A man walked to the door."
    _, items = v2.rule_compliance(generated, SOLUTION)
    assert items["rule_no_gender_words"] == 0.0
    # Word-boundary match: "mailman" or "humanity" must not trigger it.
    generated["english_detail"] = "A person showed humanity."
    _, items = v2.rule_compliance(generated, SOLUTION)
    assert items["rule_no_gender_words"] == 1.0


def test_v2_rule_meta_words(v2: HybridRewardV2) -> None:
    generated = dict(GOOD_OUTPUT)
    generated["english_detail"] = "The camera shows a person walking."
    _, items = v2.rule_compliance(generated, SOLUTION)
    assert items["rule_no_meta_words"] == 0.0


def test_v2_rule_exact_keys(v2: HybridRewardV2) -> None:
    generated = dict(GOOD_OUTPUT)
    generated["extra"] = "x"
    _, items = v2.rule_compliance(generated, SOLUTION)
    assert items["rule_exact_keys"] == 0.0


def test_v2_rule_non_notable_trigger(v2: HybridRewardV2) -> None:
    prefix = v2.config.rules.non_notable_prefix
    quiet_solution = dict(SOLUTION)
    quiet_solution["english_detail"] = f"{prefix}; the yard stayed empty."
    generated = dict(GOOD_OUTPUT)
    # Ground truth is non-notable but the output describes an event -> fail.
    _, items = v2.rule_compliance(generated, quiet_solution)
    assert items["rule_non_notable_trigger"] == 0.0
    generated["english_detail"] = f"{prefix}, only light changed."
    _, items = v2.rule_compliance(generated, quiet_solution)
    assert items["rule_non_notable_trigger"] == 1.0


# --- v2: formula and gates ---------------------------------------------------


def full_components(**overrides: float) -> Dict[str, float]:
    components = {"judge_detail": 1.0, "judge_brief": 1.0, "judge_title": 1.0, "rule_compliance": 1.0}
    components.update(overrides)
    return components


def test_v2_combine_perfect_is_one(v2: HybridRewardV2) -> None:
    assert v2.combine(dict(GOOD_OUTPUT), SOLUTION, full_components()).total == pytest.approx(1.0)


def test_v2_combine_weighted_sum(v2: HybridRewardV2) -> None:
    components = full_components(judge_detail=0.8, judge_brief=0.5, judge_title=0.4)
    expected = 0.45 * 0.8 + 0.20 * 0.5 + 0.10 * 0.4 + 0.25 * 1.0
    assert v2.combine(dict(GOOD_OUTPUT), SOLUTION, components).total == pytest.approx(expected)


def test_v2_gate_scene_error_halves(v2: HybridRewardV2) -> None:
    generated = dict(GOOD_OUTPUT)
    generated["scene_type"] = "outdoor"
    result = v2.combine(generated, SOLUTION, full_components())
    assert result.total == pytest.approx(0.5)
    assert result.components["gate_scene_error"] == 1.0


def test_v2_gate_courier_false_positive(v2: HybridRewardV2) -> None:
    generated = dict(GOOD_OUTPUT)
    generated["is_courier_action"] = True
    result = v2.combine(generated, SOLUTION, full_components())
    assert result.total == pytest.approx(0.3)
    assert result.components["gate_courier_false_positive"] == 1.0


def test_v2_gate_courier_false_negative(v2: HybridRewardV2) -> None:
    courier_solution = dict(SOLUTION)
    courier_solution["is_courier_action"] = True
    result = v2.combine(dict(GOOD_OUTPUT), courier_solution, full_components())
    assert result.total == pytest.approx(0.2)
    assert result.components["gate_courier_false_negative"] == 1.0


def test_v2_gates_stack(v2: HybridRewardV2) -> None:
    generated = dict(GOOD_OUTPUT)
    generated["scene_type"] = "outdoor"
    generated["is_courier_action"] = True
    assert v2.combine(generated, SOLUTION, full_components()).total == pytest.approx(0.5 * 0.3)


def test_v2_score_invalid_json_is_zero(v2: HybridRewardV2) -> None:
    result = v2.score("not json", SOLUTION, FAKE_CLIENT)
    assert result.total == 0.0
    assert result.components == {"invalid_json": 1.0}


def test_v2_score_with_stubbed_judges(v2: HybridRewardV2, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        HybridRewardV2, "_judge_field", lambda self, client, prompt_key, generated, expected: 1.0
    )
    result = v2.score(json.dumps(GOOD_OUTPUT), SOLUTION, FAKE_CLIENT)
    assert result.total == pytest.approx(1.0)
    assert result.components["rule_compliance"] == pytest.approx(1.0)


def test_v2_config_dump_records_weights_and_gates(v2: HybridRewardV2) -> None:
    dump = v2.config_dump()
    assert dump["weights"]["judge_detail"] == 0.45
    assert dump["gates"]["courier_false_negative"] == 0.2
