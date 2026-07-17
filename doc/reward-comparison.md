# Reward v1 vs v2 Comparison Log

**English** | [中文](reward-comparison.zh.md)

This document records the measured comparison between the two reward versions,
at two levels:

1. **Scale calibration (done, section 2)** — the same baseline prompt scored
   under v1 and v2. Answers "how do the two reward scales and rankings
   differ". v1 and v2 scores are **not directly comparable** (different
   formulas, different strictness), so any cross-version comparison needs this
   baseline first.
2. **End-to-end comparison (pending, section 4)** — one full APO run per
   reward version, producing `best_prompt_v1` / `best_prompt_v2`, followed by
   a **2×2 evaluation** (both prompts × both rewards). Answers "which reward
   drives APO to a better prompt".

## 1. Run configuration (scale calibration)

| Item | Value |
| --- | --- |
| Date | 2026-07-17 |
| Prompt | `data/baseline_prompt.txt` (untuned seed prompt) |
| Split | `data/test.jsonl`, 30 tasks, sha256 prefix `d4993bcbaf1a` |
| Generation model | `gpt-4.1-mini` (`AZURE_OPENAI_DEPLOYMENT`) |
| Judge model | `gpt-4.1-mini` (same for both versions) |
| Reward configs | `reward/v1/config.yaml`, `reward/v2/config.yaml`, both defaults |
| Execution | macOS serial (~4 min per pass) |

```bash
.venv/bin/python evaluate.py --name baseline_v1 --reward-version v1
.venv/bin/python evaluate.py --name baseline_v2 --reward-version v2
.venv/bin/python compare_rewards.py results/eval_baseline_v1.json results/eval_baseline_v2.json
```

Detail files (including the join) live in the local `results/` (not in git).

## 2. Results

### Overall and per family

| | v1 | v2 |
| --- | --- | --- |
| **Overall mean (30 tasks)** | **0.734** | **0.522** |
| Charades (22) | 0.757 | 0.560 |
| NWPU (2) | 0.691 | 0.518 |
| VIRAT (2) | 0.838 | 0.674 |
| project (4) | 0.577 | **0.239** |

### Per task (sorted by drop v2−v1)

| task | family | v1 | v2 | Δ(v2−v1) |
| --- | --- | --- | --- | --- |
| 5266 | project | 0.808 | 0.311 | −0.497 |
| 375 | Charades | 0.808 | 0.334 | −0.474 |
| 2577 | Charades | 0.512 | 0.181 | −0.331 |
| 5540 | project | 0.452 | 0.138 | −0.314 |
| 3698 | Charades | 0.644 | 0.337 | −0.307 |
| 3108 | Charades | 0.934 | 0.648 | −0.286 |
| 5142 | project | 0.536 | 0.252 | −0.284 |
| 1799 | Charades | 0.922 | 0.652 | −0.270 |
| 5085 | project | 0.512 | 0.255 | −0.257 |
| 2656 | Charades | 0.808 | 0.608 | −0.200 |
| 3059 | Charades | 0.568 | 0.370 | −0.198 |
| 4902 | NWPU | 0.634 | 0.436 | −0.198 |
| 2444 | Charades | 0.652 | 0.454 | −0.198 |
| 2370 | Charades | 0.868 | 0.678 | −0.190 |
| 482 | Charades | 0.904 | 0.720 | −0.184 |
| 4002 | Charades | 0.628 | 0.444 | −0.184 |
| 569 | Charades | 0.676 | 0.502 | −0.174 |
| 584 | Charades | 0.634 | 0.462 | −0.172 |
| 262 | Charades | 0.604 | 0.432 | −0.172 |
| 4417 | Charades | 0.856 | 0.684 | −0.172 |
| 5730 | VIRAT | 0.868 | 0.700 | −0.168 |
| 898 | Charades | 0.772 | 0.609 | −0.163 |
| 1323 | Charades | 0.544 | 0.381 | −0.163 |
| 5717 | VIRAT | 0.808 | 0.648 | −0.160 |
| 4860 | NWPU | 0.748 | 0.599 | −0.149 |
| 2277 | Charades | 0.808 | 0.663 | −0.145 |
| 991 | Charades | 0.832 | 0.697 | −0.135 |
| 4089 | Charades | 0.856 | 0.738 | −0.118 |
| 2152 | Charades | 0.940 | 0.867 | −0.073 |
| 1735 | Charades | 0.892 | 0.855 | −0.037 |

## 3. Interpretation

1. **v2 scoring lower across the board is by design, not a regression.**
   Per-field judges are stricter, rule compliance carries 0.25 weight, and
   scene/courier mistakes multiply the score down (×0.5 / ×0.3 / ×0.2); any
   weak spot directly depresses the total.
2. **v2 discriminates much harder.** The project family drops to 0.239
   (still 0.577 under v1); the best-vs-worst family spread widens from 0.26
   (v1) to 0.44 (v2). For APO this means a steeper gradient signal — "what
   went wrong" is more visible in the score.
3. **The two rewards agree on direction.** All 30 tasks score v2 < v1 with
   broadly the same ranking, so v2 does not distort the objective; it is
   stricter and more granular.
4. **Attribution of the biggest drops (from v2 component logs):**
   - `5540` / `2577`: courier false-positive gate fired (×0.3), on top of a
     low `judge_detail` (0.34 / 0.28).
   - `5266`: scene gate fired (×0.5). Note the v1 pass got the scene right —
     see the noise caveat below.
   - `375`: no gate; `rule_no_meta_words` failed (camera/frame/timestamp-type
     wording) plus very low per-field judge scores (0.18/0.18/0.03).

**Noise caveat (important):** each evaluation pass re-calls the generation
model, so v1 and v2 scored **different generations**. Per-task deltas mix
"reward definition difference" with "generation variance" (e.g. `5266`'s
scene was correct in the v1 pass and wrong in the v2 pass). The mean-level
conclusions (v2 stricter, more discriminative, direction-consistent) are
unaffected, but do not over-read any single task's Δ. To strictly isolate the
reward definitions, cache one set of generations and re-score them offline.

## 4. Next step: end-to-end comparison (Linux, high concurrency)

On a Linux VM (Python ≤ 3.13, `fork` start method) run one APO training per
reward version, then the 2×2 evaluation:

```bash
# Two trainings (pick beam/batch per doc/performance-tuning.md; big-VM example)
.venv/bin/python apo_train.py --reward-version v1 --n-runners 12 --gradient-batch-size 8
.venv/bin/python apo_train.py --reward-version v2 --n-runners 12 --gradient-batch-size 8
# Note each results/<run_id>/ (results/latest only points at the newest run!)

# 2×2 evaluation: each best prompt scored under both rewards
.venv/bin/python evaluate.py --prompt results/<run_v1>/best_prompt.txt --name tuned_v1_under_v1 --reward-version v1
.venv/bin/python evaluate.py --prompt results/<run_v1>/best_prompt.txt --name tuned_v1_under_v2 --reward-version v2
.venv/bin/python evaluate.py --prompt results/<run_v2>/best_prompt.txt --name tuned_v2_under_v1 --reward-version v1
.venv/bin/python evaluate.py --prompt results/<run_v2>/best_prompt.txt --name tuned_v2_under_v2 --reward-version v2

# Compare (only same-reward-scale comparisons are meaningful)
.venv/bin/python compare_rewards.py results/eval_tuned_v1_under_v1.json results/eval_tuned_v2_under_v1.json
.venv/bin/python compare_rewards.py results/eval_tuned_v1_under_v2.json results/eval_tuned_v2_under_v2.json
```

How to read the results:

- Use section 2's baseline numbers as the anchor; measure each tuned prompt's
  improvement over baseline *within the same reward scale*.
- If `tuned_v2` is no worse than `tuned_v1` on the **v1 scale** and clearly
  better on the v2 scale, v2's extra signal (per-field judges, rules, gates)
  delivered real gains rather than overfitting its own formula.
- Classification accuracy (scene/courier) can be tallied from the component
  details in `results/eval_*.json` or the run logs as an objective yardstick
  independent of either reward scale.

Fill in section 5 once the runs finish.

## 5. End-to-end results (to be filled in)

_After the two APO runs and the 2×2 evaluation, record here: run IDs, beam
parameters, mean_reward per cell, improvement over baseline, conclusion._
