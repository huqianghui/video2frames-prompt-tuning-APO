"""Offline tests for the project-specific APO meta-prompts (no network, no credentials)."""

import apo_train

FIVE_FIELDS = ["english_detail", "brief", "title", "scene_type", "is_courier_action"]


def test_custom_poml_files_exist():
    assert apo_train.GRADIENT_POML.is_file()
    assert apo_train.APPLY_EDIT_POML.is_file()


def test_gradient_poml_keeps_required_slots_and_objective():
    text = apo_train.GRADIENT_POML.read_text(encoding="utf-8")
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
