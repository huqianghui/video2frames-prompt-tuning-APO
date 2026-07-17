# Reward Design and Open Questions for the Customer

**English** | [中文](reward-design.zh.md)

The reward function is the optimization target: APO rewrites the prompt in
whatever direction scores higher. If the reward encodes the wrong priorities,
APO will optimize the wrong thing — precisely. This document records how the
current reward is defined, why, which parts are assumptions that only the
customer can confirm, and how to run that conversation before a large-scale
training run.

> **Versioning note:** this document describes **reward v1**, the original
> hybrid reward. The reward now lives in the versioned `reward/` package
> (select with `--reward-version` / `REWARD_VERSION`); v1 is the default and
> stays byte-equivalent to the original implementation so historical runs
> remain comparable. **Reward v2** (`reward/v2/`) upgrades the hybrid design
> per the SkillOpt-04 analysis article — per-field judges, deterministic rule
> compliance, and multiplicative gates for scene/courier errors; several of
> the open questions below (per-field judging, asymmetric courier costs) are
> implemented there as configurable assumptions to confirm. See the README's
> "Reward" section for the v2 formula, version selection, and the
> `compare_rewards.py` comparison workflow.

## 1. Current definition (v1)

Implemented in `reward/v1/` (weights and judge prompt in `config.yaml`,
scoring in `reward.py`):

```
reward = 0.2 × exact match of scene_type          (case-insensitive)
       + 0.2 × exact match of is_courier_action   (tolerates "true"/"false" strings)
       + 0.6 × LLM-judge semantic score over english_detail / brief / title
```

Two hard zero rules:

- Output that is not a valid JSON object → `0`.
- Request rejected by the Azure content safety filter → `0` (the rejection
  depends only on the input frames, identical for every candidate prompt).

The judge (`JUDGE_MODEL`, default `gpt-4.1-mini`, `temperature=0`, structured
output with `reason` + `score`) is instructed to check whether the generated
text "describes the same subjects and actions" as the ground truth, wording may
differ, be critical, partial credit allowed. It returns **one combined 0–1
score** for all three text fields.

## 2. Why it is designed this way

1. **Split by field nature.** Of the five output fields, `scene_type`
   (indoor/outdoor) and `is_courier_action` (bool) are objectively checkable —
   exact match is free, noise-free, and unambiguous. The three free-text fields
   can never match exactly, so semantic comparison by an LLM judge is the only
   practical grader.
2. **APO needs a continuous signal.** With exact matches alone the reward would
   take only five discrete values; the gradient model would have almost nothing
   to critique. The judge's partial credit distinguishes "slightly off
   description" from "completely wrong", which is what the text-gradient step
   feeds on.
3. **Weights follow content share.** The three text fields are the bulk of the
   output (and the part a prompt can influence most), hence `0.6`; the two
   classification fields get `0.2` each.
4. **The zero rules remove non-prompt noise.** Invalid JSON means the format
   contract is broken (downstream cannot consume the output — punish hard).
   Content-filter rejections are independent of the candidate prompt, so
   scoring them 0 (and excluding those videos at sampling time via the probe
   cache) keeps them from polluting comparisons.

## 3. What only the customer can answer

These are assumptions baked into the reward. Getting them wrong means APO
optimizes a precisely wrong target, so confirm them **before** the large run:

| # | Question | Why it matters | If the answer differs |
| --- | --- | --- | --- |
| 1 | Is `is_courier_action` the business-critical signal (this looks like a courier/delivery detection product)? Are false positives and false negatives equally bad? | At weight 0.2 a prompt that fixes courier detection gains little reward; APO will prioritize text quality instead. Misclassification costs are usually asymmetric. | Raise `COURIER_WEIGHT`; replace symmetric exact match with asymmetric scoring (e.g. missed courier costs more than a false alarm). |
| 2 | Which text field is actually consumed downstream — `brief` (user-facing?), `english_detail` (search/archive?), `title`? | The judge currently emits one combined score; a prompt that improves the important field while degrading an unimportant one scores flat. | Split the judge into per-field scores with separate weights. |
| 3 | How was the ground truth produced — human annotation or model-generated (the dataset name suggests SFT distillation)? Known quality issues? | The judge grades *against the GT*. Noisy GT both caps the achievable score and can steer tuning toward reproducing GT artifacts. | Clean or re-annotate a subset for val/test; or instruct the judge to tolerate specific GT quirks. |
| 4 | What improvement is worth shipping (e.g. +0.05 average reward, or +X pp courier accuracy)? | This is the effect size `δ` in [dataset-sizing.md](dataset-sizing.md) — it determines how large val/test must be and when to stop tuning. | Resize the splits with the sizing formula before the run. |
| 5 | Is downstream parsing strict JSON, or tolerant (e.g. strips markdown fences)? | We currently score any non-JSON output 0 — the harshest possible penalty. | Relax the parser / partial credit for recoverable outputs. |
| 6 | Can the customer hand-score 10–20 sample outputs? | Calibrates the LLM judge. If judge scores do not correlate with human judgment, the judge rubric must be fixed *before* tuning — it is the examiner of the whole system. | Iterate on the judge prompt / model until correlation is acceptable. |

Questions 1–4 should be settled before spending on a full run; 5–6 are cheap to
check in parallel.

## 4. Suggested next steps

1. **Send the customer a short brief** (this document works): the reward
   formula, the six questions above, plus 2–3 concrete scored examples from
   `results/eval_baseline.json` so the discussion is grounded in real outputs
   rather than abstractions.
2. **Run the pilot in parallel** (Stage 1 of
   [dataset-sizing.md](dataset-sizing.md), default 40/24/30 splits) — it
   measures reward noise σ and produces the example outputs for step 1, and
   nothing in it is wasted even if the weights change later.
3. **Fold the answers back in.** Weight changes are three lines in
   `reward/v1/config.yaml` (`scene_weight` / `courier_weight` /
   `judge_weight`); per-field judging and asymmetric courier scoring are
   already implemented in `reward/v2/` (weights and gate ratios in
   `reward/v2/config.yaml`), with unit tests in `tests/test_reward.py`.
4. **Only then run the full APO ladder** (Stage 2+). Changing the reward after
   a big run means paying for the run again — the reward conversation is the
   cheapest insurance in the whole project.
