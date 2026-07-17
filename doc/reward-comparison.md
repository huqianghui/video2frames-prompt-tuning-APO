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

## 5. End-to-end results (2026-07-17, Linux VM)

### Run configuration

| Item | Value |
| --- | --- |
| Date | 2026-07-17 (Linux VM, parallel runners) |
| Data | `prepare_data.py --train-size 80 --val-size 100 --freeze-test --probe-content-filter` (train 80 / val 100 / test 30) |
| APO run (v1 reward) | `results/20260717_143056/` |
| APO run (v2 reward) | `results/20260717_145308/` |
| Beam / batch parameters | see each run's `summary.json` on the VM (not copied here) |

> **Test-split caveat:** the frozen `test.jsonl` on the VM (sha256 prefix
> `b5065f2d3016`) is **not** the split used for the section-2 scale
> calibration (different task IDs — e.g. 2282/4221/… vs 5266/375/…, and it
> contains a `ucf_crime` task the old split did not). The section-2 baseline
> anchors (v1 0.734 / v2 0.522) do not apply; the baseline prompt was
> therefore **re-evaluated on this split** (2026-07-18) to provide the anchor
> row below.

### 2×3 matrix (mean_reward on test, 30 tasks)

| | scored under v1 | scored under v2 |
| --- | --- | --- |
| `baseline` (seed prompt) | 0.7552 | 0.5686 |
| `tuned_v1` (best of v1-reward run) | 0.7509 (−0.004) | 0.5993 (+0.031) |
| `tuned_v2` (best of v2-reward run) | **0.7609** (+0.006) | **0.6033** (+0.035) |

(Deltas in parentheses are vs baseline on the same scale.)

Training-side numbers (val, 100 tasks): v1 run seed 0.764 → best 0.798;
v2 run seed 0.594 → best 0.643. Beam 2/2/2 in both runs; in **both** runs
every round-2 child scored below its round-1 parent, i.e. all gains came
from the first edit and the search saturated after one round.

Per-task win/loss for `tuned_v2` vs `tuned_v1`:

- v1 scale: 13 win / 11 loss / 6 tie; v2 scale: 14 win / 15 loss / 1 tie.
- Per-family means point in no consistent direction (tuned_v2 better on
  Charades and ucf_crime, worse on NWPU/VIRAT/project under v1 scale; mixed
  under v2 scale).

### Conclusion

With the baseline anchor in place the picture is sharper than "statistical
tie":

- **On the v1 scale nothing improved** — tuned_v1 −0.004, tuned_v2 +0.006,
  both within noise. The coarse semantic judge (0.6 weight in v1) saw no
  difference from either tuned prompt.
- **On the v2 scale both tuned prompts improved by ~+0.03** — including
  `tuned_v1`, which never saw the v2 reward during training. What actually
  improved is the part v2 measures explicitly and v1 barely does: rule
  compliance and field structure (deterministic, prompt-fixable), not the
  semantic content of the descriptions.
- Head-to-head, `tuned_v2` vs `tuned_v1` remains within noise on both scales
  (+0.010 / +0.004; per-task near coin-flip).

### Diagnosis: why the gain is ~0.04 regardless of the "wider" v2 scale

Evidence from `report.json` (per-candidate val reward vectors) and the
gradient critiques:

1. **v2 widened the scale along the task axis, not the prompt axis.** v2's
   lower baseline (0.57 vs 0.76) comes from harsher scoring of *hard tasks*
   (per-field judges, gates on visually ambiguous videos). That headroom is
   claimable by a better vision model, not by instruction wording. What a
   prompt edit can actually move — the deterministic rule-compliance slice
   (0.25 weight) plus marginal judge gains — is worth roughly +0.03–0.05,
   and APO captured almost exactly that (+0.035 on test).
2. **Between-candidate spread ≈ measurement noise.** Across the 9 candidates
   per run, val means span ~0.05 while the SE of a candidate mean (n=100,
   per-task SD ~0.12) is ~0.011–0.013. Most candidates are statistically
   indistinguishable, so beam selection among the mid-pack is noisy.
3. **Per-task churn is huge and mostly cancels.** In the v2 run, best-vs-seed
   per-task deltas range −0.40…+0.65 (26 tasks improve >0.2, 12 regress
   >0.2); the +0.049 val mean gain is a small residual of large opposing
   movements dominated by generation/judge variance.
4. **The search saturated after one edit.** In both runs all four round-2
   children scored below their round-1 parents. More rounds at this
   configuration would not have helped; the first critique already harvested
   the low-hanging fruit (the critiques overwhelmingly target rule
   compliance — word caps, exactly-five-keys, person-not-gendered,
   no-meta-words — "gate" is mentioned once across all gradients).
5. **Gates did respond on val** (seed had 4 val tasks below 0.3, best had 0),
   but gate-triggering is largely content-driven, so the effect is small and
   does not transfer cleanly to test.

### Evidence appendix (numbers behind the diagnosis)

All numbers computed from `results/<run_id>/report.json` (per-candidate
100-task val reward vectors) and the gradient critique texts embedded in it.

**A. Candidate val scores (9 candidates per run).**

| Candidate | v1 run (parent) | v2 run (parent) |
| --- | --- | --- |
| v0 seed | 0.764 | 0.594 |
| v1 (R1, v0) | 0.788 | 0.637 |
| v2 (R1, v0) | 0.774 | 0.615 |
| v3 (R1, v0) | 0.781 | **0.643** |
| v4 (R1, v0) | 0.773 | 0.615 |
| v5 (R2) | 0.746 (← v3) | 0.613 (← v1) |
| v6 (R2) | 0.740 (← v3) | 0.617 (← v1) |
| v7 (R2) | 0.775 (← v1) | 0.627 (← v3) |
| v8 (R2) | 0.784 (← v1) | 0.617 (← v3) |

Every round-2 child scores below its parent in **both** runs (8 of 8
regressions) — the basis for "the search saturated after one edit".

**B. Noise floor vs candidate spread.** Mean per-task reward SD across
candidates: 0.113 (v1 run) / 0.133 (v2 run). With n=100 val tasks, the SE of
a candidate mean is 0.0113 / 0.0133. Candidate mean spread (max−min): 0.057
(v1 run) / 0.049 (v2 run) — i.e. the whole field spans ~4 SE and the
mid-pack differs by ≤1–2 SE, so beam selection among them is noise-driven.

**C. Per-task churn, best vs seed (same val split).**

| | v1 run (v1−v0) | v2 run (v3−v0) |
| --- | --- | --- |
| Mean delta | +0.034 | +0.049 |
| Delta quantiles (min/q1/med/q3/max) | −0.42 / −0.07 / +0.03 / +0.14 / +0.50 | −0.40 / −0.09 / +0.03 / +0.20 / +0.65 |
| Tasks improved > 0.2 | 13 | 26 |
| Tasks regressed > 0.2 | 10 | 12 |

The net mean gain is a small residual of large opposing per-task movements —
generation/judge variance dominates individual tasks.

**D. Gate response on val (v2 run).** Val tasks with reward < 0.3 (gated or
failed): seed v0 = 4, best v3 = 0.

**E. What the critiques actually talk about.** Keyword counts across all
gradient critiques in the v2 run: "rule" 45, "courier" 30, "scene" 29,
"compliance" 9, "judge" 2, "gate" 1. The critic's actionable suggestions
overwhelmingly target the deterministic rule-compliance slice (word caps,
exactly-five-keys, person-not-gendered, no meta words) — consistent with
the cross-scale result that rule compliance is what improved.

**F. Cross-scale transfer.** `tuned_v1` (trained only against v1) gains
+0.031 on the v2 scale but −0.004 on its own v1 scale; `tuned_v2` gains
+0.035 / +0.006 respectively. Both prompts improved the same thing — the
component that only v2 measures explicitly (rules/structure). If the tuned
prompts had improved semantic description quality, the v1 scale (0.6 judge
weight) would have moved too. It did not.

### Follow-ups (in priority order)

1. **Decompose before optimizing further**: from `results/eval_*.json`
   component details, average rule_compliance, per-field judge scores and
   gate rates for baseline vs tuned. This tells how much prompt-fixable
   headroom actually remains (likely little) vs perception-bound headroom.
2. **If judge_detail is the ceiling, improve the input, not the prompt**:
   more/denser frames, higher resolution, or a stronger multimodal
   deployment (`AZURE_OPENAI_DEPLOYMENT`). Prompt tuning cannot make the
   model see what the frames don't show.
3. **Reduce noise so APO can climb finer gradients**: `judge_samples: 3` in
   `reward/v2/config.yaml`; consider caching generations and re-scoring
   offline to separate reward effects from generation variance.
4. **Reshape the beam, don't deepen it**: round 2 regressed in both runs, so
   spend budget on first-round diversity instead — e.g.
   `--branch-factor 4..6 --beam-rounds 2` — and enlarge
   `--gradient-batch-size` so critiques see more failure modes (including
   gate-firing tasks).
5. **Audit remaining gate-firing test tasks by eye**: if scene/courier are
   genuinely ambiguous in the frames, that is a data/labeling question for
   the customer, not a prompt problem.
6. **Test split is small (30)** — mean SE alone is ~0.02–0.03; resolving
   0.01-level effects needs a larger test split (see
   [dataset-sizing.md](dataset-sizing.md)).

## 6. Target-model probe: gpt-5.4 (2026-07-18, preliminary)

Follow-up #2 above, executed: same untuned baseline prompt, same frozen test
split (30 tasks, `b5065f2d3016`), same v2 reward and judge (`gpt-4.1-mini`),
only `AZURE_OPENAI_DEPLOYMENT` switched from `gpt-4.1-mini` to `gpt-5.4`.
Single pass (`results/eval_baseline_gpt54_v2.json`).

| target model (baseline prompt, v2 scale) | mean_reward |
| --- | --- |
| gpt-4.1-mini | 0.5686 |
| **gpt-5.4** | **0.6095 (+0.041)** |
| reference: tuned_v2 @ gpt-4.1-mini | 0.6033 |

Per-task paired stats (gpt-5.4 − gpt-4.1-mini, same task IDs):

- 17 tasks up / 13 down; delta quantiles −0.51 / −0.04 / +0.02 / +0.13 / +0.52.
- Paired delta SD 0.193 → SE of the mean delta (n=30) = 0.035, so
  **+0.041 ≈ 1.2 SE — directionally positive, not yet conclusive.**
- Tasks below 0.3 (gated/failed): 3 → 1.
- Family means: project 0.487→0.553, Charades 0.581→0.606, VIRAT 0.493→0.826;
  biggest regressions `2023` (−0.51, landing at 0.28, likely a gate) and
  `482` (−0.39) — gpt-5.4 appears to make *different* mistakes, worth a
  component-level look.

**Reading:** the model swap alone (+0.041, zero tuning) matches or exceeds
the entire APO gain on the old model (+0.035), and the untuned prompt on
gpt-5.4 already beats the tuned prompt on gpt-4.1-mini — consistent with the
section-5 diagnosis that the remaining headroom is perception-bound, not
prompt-bound.

**Pending before switching:** (1) two repeat passes
(`--name baseline_gpt54_v2_r2/_r3`) to average out generation variance;
(2) component details for `2023`/`482` (which gate or rule fired — if
gpt-5.4's failures are verbosity/rule-type, they are prompt-fixable and a
re-run of APO on the new target has extra headroom); (3) transfer check of
the old tuned prompt
(`evaluate.py --prompt results/20260717_145308/best_prompt.txt --name tuned_v2_gpt54 --reward-version v2`);
(4) if confirmed, re-run APO with target gpt-5.4 (v2 reward,
`--branch-factor 4`, `judge_samples: 3`).
