# APO Meta-Prompt Customization (POML)

**English** | [中文](apo-poml-customization.zh.md)

APO itself is driven by two prompts — *meta-prompts* that operate on the prompt
being tuned. This document explains what they do, why the framework defaults
are not enough for this project, and exactly what the project-specific versions
in [`prompts/`](../prompts) change.

## 1. What the two files do

Each APO beam-search round expands a candidate prompt in two LLM steps
(`textual_gradient_and_apply_edit` in `agentlightning/algorithm/apo/apo.py`):

| Step | Template | Model (env var) | Input | Output |
| --- | --- | --- | --- | --- |
| Text gradient | `text_gradient_*.poml` | `APO_GRADIENT_MODEL` (default `gpt-4.1`) | Current prompt + a batch of rollout traces (messages, rewards) | A **critique**: a bullet list of concrete causes of low reward and testable changes |
| Apply edit | `apply_edit_*.poml` | `APO_APPLY_EDIT_MODEL` (default `gpt-4.1-mini`) | Current prompt + the critique | The **rewritten prompt** for the next candidate |

The critique is the "gradient" and the edit is the "update step" — the quality
of both meta-prompts directly bounds the quality of the search.

By default the framework picks a template *at random per expansion* from three
gradient variants and two edit variants
(`agentlightning/algorithm/apo/prompts/*.poml`). Passing a single file for each
step (as this project does) also makes runs more reproducible.

## 2. Why the defaults are not enough here

The default templates are task-agnostic: they only say "raise reward" and let
the gradient model infer the objective from the traces. For this project that
leaves three real failure modes:

1. **The edit model can silently drop the JSON contract.** Our reward scores
   any output that is not a valid JSON object with exactly the five fields as
   `0`. A rewrite that "simplifies" the format instruction produces a candidate
   whose every rollout scores 0 — a whole expansion of the budget wasted.
2. **The gradient model does not know the reward structure.** It cannot see
   that the three text fields carry 0.6 of the weight while the two
   classification fields carry 0.2 each, so critiques may chase low-value
   fixes. It also does not know that content-safety-filter rejections depend
   only on the input frames — without being told, it will blame the prompt for
   failures the prompt cannot fix.
3. **The edit model may add its own media placeholders.** In this project the
   `<frame n | Xs>` placeholder section and the images are appended by the
   agent at runtime *after* the tuned instruction; a rewrite that reintroduces
   `<video>` or frame markers would duplicate or conflict with that section.

One customization that other APO projects need is deliberately **absent**
here: brace/placeholder protection. The example strict templates forbid
literal `{`/`}` because their tuned prompt is rendered with Python
`str.format`. Our agent uses the template text verbatim
(`frame_agent.py`: `fixed_prompt = prompt_template.template`), so JSON
examples with curly braces in the prompt are safe — and explicitly allowed.

## 3. Final changes vs the framework defaults

Both files start from `*_variant01.poml` and change only what the table lists:

| File | Change | Purpose |
| --- | --- | --- |
| `prompts/text_gradient_video2frames.poml` | Added **Optimization Objective** section: the 5-field valid-JSON contract (else 0), the 0.2/0.2/0.6 reward formula and judge criterion, and "content-filter rejections are not the prompt's fault" | Aim critiques at what actually moves the reward |
| | Added **Critique Constraints** section: never suggest `<video>`/frame placeholders (runtime appends them), never rename/add/remove the five fields, JSON brace examples are allowed | Keep critiques inside the task's contract |
| `prompts/apply_edit_video2frames.poml` | Removed "Preserve placeholder variables inside curly brackets" | Our template has no placeholders; the rule is misleading here |
| | Added three revision rules: keep the 5-field valid-JSON requirement, never add `<video>`/frame placeholders, JSON brace examples allowed | Prevent rewrites from breaking the reward contract |
| | Output format reworded to "return only the improved prompt text" | No placeholder wording |

Everything else — the experiment loop, the `{{ prompt_template }}` /
`{{ critique }}` / `{{ experiments }}` slots the algorithm fills in — is kept
identical to the defaults, so the files stay drop-in compatible with
`APO(gradient_prompt_files=..., apply_edit_prompt_files=...)`.

## 4. Usage

`apo_train.py` uses the project templates **by default**:

```bash
.venv/bin/python apo_train.py                 # project meta-prompts (prompts/)
.venv/bin/python apo_train.py --default-poml  # framework built-in templates
```

`results/summary.json` records which set was used (`"custom_poml"`), so A/B
comparisons between the two remain traceable. Offline tests in
`tests/test_apo_train.py` guard the required template slots and the contract
keywords.

If the reward changes after the customer conversation
([reward-design.md](reward-design.md)) — e.g. new weights or per-field judge
scores — update the **Optimization Objective** section of the gradient template
to match, or the search will optimize against a stale description of the
reward.
