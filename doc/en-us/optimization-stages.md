# APO Optimization Stage Guide: Ordering the Model / Data / Prompt Levers

**English** | [中文](../optimization-stages.md)

When picking up or resuming this project, the key question is not "how do I
tune the prompt" but **"which stage am I in, and which lever should I pull"**.
This document turns the conclusions of the 2026-07 end-to-end experiments
([reward-comparison.md](reward-comparison.md) sections 5–6) into an actionable
stage roadmap: locate first, act second, and avoid pulling the wrong lever at
the wrong stage.

## 0. Core conclusion: the three levers live on different axes

| Lever | Score it can move | Nature | Measured evidence |
| --- | --- | --- | --- |
| **Swap the target model** | +0.04 and up, **the only lever that raises the ceiling** | Gains come from perception (the 0.45-weight judge_detail slice), which prompts cannot reach | gpt-5.4 with zero tuning: +0.041; the untuned prompt already beats the tuned prompt on the old model |
| **APO prompt tuning** | +0.03–0.05, **a one-time dividend** | Only harvests the deterministic rule-compliance slice (0.25 weight); one edit round exhausts it | All gains in both runs came from round 1; 8/8 round-2 children regressed |
| **Data size** | +0, but decides **whether you can see the other two** | Produces no gains, only measurement resolution | Paired SE at test=30 is 0.035 ≥ the 0.03–0.04 effects being measured, so every conclusion stays "not yet conclusive" |

In one line: **the model raises the ceiling, the prompt pays a one-time
dividend, the data sets the ruler's precision.**

**What "perception" means precisely**: the model does receive the frame
images, but *cannot discern* what happened from the sampled frames (actions
too subtle, information lost between frames, objects too small), so the
generated english_detail misses or invents events — the description is
genuinely wrong, not a scoring artifact. Instructions change *how things are
said*, not *what was seen*; that gap is forever out of a prompt's reach, and
the levers for it are a stronger model or denser frames.

## 1. Locate first: which stage are you in

Do not guess — run one baseline evaluation and decompose the reward
components; each weakness maps to one lever:

```bash
.venv/bin/python evaluate.py --name baseline --reward-version v2
# component details in results/eval_baseline.json and the run logs
```

| Observed symptom | Bottleneck | Lever to pull | Stage |
| --- | --- | --- | --- |
| Gaps between candidates/versions < 2×SE; no conclusion possible | Ruler | grow test, `judge_samples`, repeat passes | Stage 1 |
| `judge_detail`/`brief`/`title` low while rule_compliance is fine | Perception | stronger multimodal model, denser frames | Stage 2 |
| `rule_compliance` low (word caps, missing keys, gender words, meta words) | Prompt | one APO run | Stage 3 |
| Gates (scene/courier) fire often and the frames are genuinely ambiguous | Data/labels | manual audit, align with the customer | Stage 4 |

Four-part evidence chain for "it is perception, not an under-tuned prompt"
(all reproducible on your own runs):

1. **Is APO saturated?** All round-2 children below their parents, and the
   critiques only repeating rule-compliance issues (each candidate's gradient
   text is in `results/<run_id>/report.json`);
2. **Cross-scale counter-evidence**: the tuned prompt improves on the v2
   scale but the v1 scale (0.6 weight on the semantic judge) does not move →
   the prompt improved only format/rules, not semantics;
3. **Controlled-variable probe**: same baseline prompt, same test split, same
   judge, only `AZURE_OPENAI_DEPLOYMENT` changed — if the score moves, it is
   perception; if not, it is not;
4. **Component decomposition**: the weakness sits in judge_detail
   (perception) vs rule_compliance (prompt).

## 2. Stage roadmap

### Stage 0 — smoke (once per environment)

Verify the loop runs; ignore scores. See
[dataset-sizing.md](dataset-sizing.md) section 6, Stage 0.

### Stage 1 — sharpen the ruler: the main problem is measurement, not optimization

Without this step, no later change can be judged effective or not. **Note:
this is not a re-balancing of the three splits** — train 80 / val 100 are
already sufficient (measured calibration in
[dataset-sizing.md](dataset-sizing.md) section 8). Only two things change:

1. **Grow test to ~100 tasks once, then freeze** (`prepare_data.py
   --test-size 100 --freeze-test`). Side effect: all old baseline anchors are
   invalidated; re-measure the baseline once.
2. **Reduce single-measurement variance**: set `judge_samples: 3` in
   `reward/v2/config.yaml`; run key comparisons 2–3 times on the same split
   and average.

**Exit criterion**: paired SE on test ≤ half the effect size you need to
resolve (effects here are ~0.03–0.04 → paired SE ≤ 0.02, i.e. ~100 tasks
plus repeat passes).

### Stage 2 — swap the engine: at the current score range the main problem is perception

Confirm the gain with a controlled-variable probe before switching:

```bash
# change only AZURE_OPENAI_DEPLOYMENT, keep everything else fixed;
# run at least 2–3 passes and average
.venv/bin/python evaluate.py --name baseline_<newmodel>_v2 --reward-version v2
```

Reading: confirmed only when the mean paired delta ≥ 2× the paired SE. Also
inspect the component details of the worst regressions — if the new model's
failures are verbosity/rule-type, they are prompt-fixable and Stage 3 gains
extra headroom. To probe the ceiling, one pass with a larger tier (e.g. pro)
serves as a reference.

**Exit criterion**: once confirmed, switch `AZURE_OPENAI_DEPLOYMENT` and move
to Stage 3.

### Stage 3 — harvest the prompt dividend on the new model (once)

A new model makes *different kinds* of mistakes, so the rule-compliance slice
reopens. Configure per the validated conclusions: **width for diversity, no
extra depth**:

```bash
.venv/bin/python apo_train.py --reward-version v2 \
  --branch-factor 4 --beam-rounds 2 --gradient-batch-size 8 \
  --n-runners 12   # pick concurrency per performance-tuning.md on Linux
```

**Termination signal (important)**: in `report.json`, all round-2 children
score below their parents and the critiques start repeating the same issues →
the search is saturated, **stop**. Do not keep running prompt rounds on the
same model — candidate gaps are typically ≤1–2 SE, which is picking coins in
noise.

**Exit criterion**: tuned-vs-baseline paired gap ≥ 2×SE on the frozen test,
or accept the current best after the termination signal fires.

### Stage 4 — the remaining gap belongs to data/input

When judge_detail is still the weakness after growing test, swapping the
model, and one APO run:

- **Measure the judge_detail ceiling first (nearly free).** Feed the
  ground-truth `english_detail` itself to the judge as if it were the model
  output — the score you get is the theoretical maximum for this (frames,
  labels, judge) combination. Customer labels are typically written from the
  *full video*, so events invisible in the sampled frames cap the score no
  matter how strong the model is. If the ceiling is, say, 0.8, the current
  score is much closer to the top than it looks and "low" is a scale
  artifact; only the gap *below the ceiling* is worth chasing.
- **Manually audit the lowest-judge_detail tasks.** `evaluate.py` saves
  per-task components in `results/eval_<name>.json`; sort by `judge_detail`,
  take the bottom ~10, and compare frames vs ground truth vs model output.
  Each outcome names a different culprit: a human can see it in the frames
  but the model missed it → perception (stronger model, Stage 2); a human
  cannot see it in the frames either but the label says it happened → frame
  density / label misalignment (improve the input, or align labels with the
  customer); the output is actually fine but scored low → judge calibration
  (back to Stage 1).
- **Improve the input, not the prompt**: denser frame sampling, higher
  resolution — the model cannot see what the frames don't show; validate the
  change through the Stage-2 probe procedure.
- **Manually audit the tasks that still fire gates**: where scene/courier are
  genuinely ambiguous in the frames, that is a labeling/requirements question
  to align with the customer, not a prompt problem.

## 3. Loop rules

- Changed the target model or the input (frame density/resolution) → return
  to Stage 3 and harvest the prompt dividend once more;
- Changed the reward definition or the judge model → the ruler changed;
  return to Stage 1 and re-measure the baseline anchors;
- Unsure at any point → go back to the locating table in section 1 and
  decompose components before acting.

Pulling the wrong lever at the wrong stage (most commonly: repeatedly tuning
the prompt while perception-bound) only buys noise with money.
