"""Offline tests for the project-specific APO meta-prompts (no network, no credentials)."""

import pytest

import apo_train
from reward import available_versions, poml_override

FIVE_FIELDS = ["english_detail", "brief", "title", "scene_type", "is_courier_action"]


def test_custom_poml_files_exist():
    assert apo_train.APPLY_EDIT_POML.is_file()
    for version in available_versions():
        gradient = poml_override(version, "text_gradient")
        assert gradient is not None and gradient.is_file()


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_gradient_poml_keeps_required_slots_and_objective(version: str):
    gradient = poml_override(version, "text_gradient")
    assert gradient is not None
    text = gradient.read_text(encoding="utf-8")
    assert "{{ prompt_template }}" in text
    assert "experiment in experiments" in text
    for field in FIVE_FIELDS:
        assert field in text
    assert "content safety filter" in text
    assert "&lt;video&gt;" in text


def test_apply_edit_poml_keeps_required_slots_and_contract():
    text = apo_train.APPLY_EDIT_POML.read_text(encoding="utf-8")
    assert "{{ prompt_template }}" in text
    assert "{{ critique }}" in text
    for field in FIVE_FIELDS:
        assert field in text
    assert "&lt;video&gt;" in text
