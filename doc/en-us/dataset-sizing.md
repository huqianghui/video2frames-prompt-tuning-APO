# Choosing Dataset Sizes for APO

**English** | [中文](../dataset-sizing.md)

How large do the `train` / `val` / `test` splits need to be? Large enough that the
score differences you care about are visible above the evaluation noise — and no
larger. This document gives a concrete procedure for this project: estimate the
noise from runs you already have, derive the required sample sizes, and scale up
in stages instead of guessing.

## 1. Why size matters: selection noise

APO selects prompts by their **average reward on the val split**. Every average
over `n` tasks carries a standard error

```
SE = σ / √n
```

where `σ` is the per-task reward standard deviation. When two candidate prompts
are compared, the difference of two such averages is only trustworthy when it
exceeds roughly `2.8 × SE` (a two-sample z-test at 95% confidence,
`√2 × 1.96 ≈ 2.8`). Anything smaller is coin-flipping — APO will "select" prompts
based on noise, and the reported best score will not reproduce.

To detect a true difference of size `δ` between two prompts, you need

```
n ≈ 2 × (1.96 × σ / δ)²    per prompt, on the val split
```

With a typical `σ ≈ 0.20` for this project's hybrid reward:

| val size `n` | SE | smallest reliably detectable gap (≈2.8×SE) |
| --- | --- | --- |
| 24 (default) | ~0.041 | > 0.11 |
| 50 | ~0.028 | > 0.08 |
| 100 | ~0.020 | > 0.055 |
| 200 | ~0.014 | > 0.04 |

Interpretation for this project: with the default `val` of 24, a prompt edit that
improves the true reward by 0.05 is **invisible** — which is consistent with
smoke runs reporting "seed prompt was never beaten" at 0.37 vs 0.37. Expect
useful prompt-tuning gains in the 0.03–0.10 range, so a val split of **64–100**
is a realistic working size.

## 2. Estimate σ from runs you already have (free)

Do not guess `σ` — the per-task rewards are already on disk:

- `results/eval_<name>.json` (written by `evaluate.py`) contains one reward per
  task.
- `results/report.json` (written by `generate_report.py` after each APO run)
  contains `val_rewards` per candidate.

Compute the standard deviation of those rewards; that is your `σ`. Plug it into
the formula above with the `δ` you want to detect, and you have your val size.
Re-estimate after any change to the reward function or the task model — `σ` is a
property of the (model, reward, data) combination, not a constant.

## 3. Different splits need different sizes

### Why not the 8:1:1 of weight training

The traditional train:val:test ≈ 8:1:1 split assumes train is the *fuel* of
gradient descent — the volume of data the model consumes directly determines
how much it learns, so train dominates; val only picks hyperparameters / early
stopping; test is a one-shot acceptance check. In APO the three splits are
consumed in completely different ways:

| Split | Weight training (where 8:1:1 comes from) | APO in this project |
| --- | --- | --- |
| train | Fuel for gradient descent, fully consumed every epoch; more = better learning | Only sampled in small failure batches (`--gradient-batch-size`, 4–8 tasks) for the critic to write text critiques; more rows add sampling diversity, not "learning volume" |
| val | Picks hyperparameters / early stopping; lightly used | Scores **every candidate in full each round** (e.g. 9 candidates × 100 tasks); its SE directly sets the resolution of beam selection — the statistically heaviest split |
| test | Final acceptance | Same — one-shot held-out acceptance, sized by the effect the final claim must resolve |

In one line: 8:1:1 optimizes "how much the model learns"; APO must optimize
"how accurately we measure". Train is a sampling pool, val is the measuring
instrument, test is the final verdict. So APO's size ordering is typically
**val ≥ train, with test sized independently by effect size** — unrelated to
8:1:1 and often inverted (this project actually runs 80/100/30, val > train).
The right approach is not a ratio but deriving each split's size from its
statistical demand:

The three splits play different roles, so they scale differently:

- **`val` — scale this first.** It drives candidate selection inside APO; its
  noise directly causes wrong selections. Target the `n` from the formula
  (typically 64–100 here). Keep `--val-batch-size` equal to the full val size so
  every candidate is scored on the same tasks.
- **`test` — second priority.** The final baseline-vs-tuned comparison is
  **paired**: `evaluate.py` runs both prompts on the same tasks, so use the
  standard deviation of the per-task reward *differences* (`σ_d`, usually much
  smaller than `σ`) in a one-sample version of the formula
  (`n ≈ (1.96 × σ_d / δ)²`). Around 100 tasks is usually enough for a credible
  final claim.
- **`train` — usually fine as is.** Each critique only samples
  `--gradient-batch-size` (default 4) tasks; a pool of 40 already
  provides variety. To improve the gradient signal, increase
  `--gradient-batch-size` or `--beam-rounds` before adding train data.

### Terminology: what a "round" actually is (real train / val consumption)

Phrases like "a few tasks per round" blur three levels — run, round, and
branch. The precise hierarchy and where sampling happens:

```
1 run (one apo_train.py execution)
└── beam_rounds rounds
    └── each round: every surviving prompt in the beam (beam_width of them)
        └── each spawns branch_factor children
            └── each child = 1 critique step:
                sample gradient_batch_size tasks from the train pool → rollout
                → critic reads the failures and writes a critique → rewrite
```

- **Train is sampled per critique** (i.e. per child candidate generated), not
  per round and not per run. Consumption per run:

  ```
  critique steps ≈ beam_rounds × beam_width × branch_factor
  train rollouts = critique steps × gradient_batch_size   (with replacement)
  ```

- **Val is consumed per candidate**: every candidate (seed included) is scored
  on the full val split, so `val rollouts = number of candidates × val_size`.

Concretely, the 2026-07-17 v2 run (beam 2×2×2, gradient batch 8, val 100):
round 1 spawned v1–v4 from the seed v0, round 2 spawned two children each
from the survivors v1 and v3 (v5–v8) — 8 critique steps × 8 tasks =
**64 train rollouts** (a pool of 80 is plenty), versus 9 candidates × 100
tasks = **900 val rollouts**. That is the direct sense in which val is the
statistically heaviest split, and why a train pool far smaller than val is
still sufficient.

## 4. Staged scaling: coarse screening + large-set re-scoring

Evaluating every candidate on a large val split is the expensive part
(`cost ≈ beam_rounds × beam_width × branch_factor × (gradient_batch + val_batch)`
rollouts, each one multimodal call + one judge call). The standard remedy is a
racing / successive-halving ladder — cheap screening for the many, expensive
scoring for the few:

1. **Screen** — run APO with a small-to-medium val (e.g. 24–64). Small samples
   are enough to discard clearly bad candidates; only near-ties are decided by
   noise.
2. **Re-score** — after training, take the top 2–3 candidate prompts from
   `results/report.md` plus the baseline, and re-evaluate each on a larger
   held-out re-scoring split (100–200 tasks, sampled from the full 5850 pool,
   disjoint from train/val/test):

   ```bash
   .venv/bin/python evaluate.py --name baseline
   .venv/bin/python evaluate.py --prompt results/best_prompt.txt --name tuned
   # plus any runner-up prompt saved from report.md
   ```

   Pick the final winner by these scores, not the small-val scores.
3. **Escalate only on evidence** — if the re-scored best-vs-baseline gap is
   smaller than `2 × SE` of the re-scoring set, the effect is not established.
   First search harder (more beam rounds / width, better gradient batches);
   only grow the datasets when a promising-but-unconfirmed gap needs a tighter
   confidence interval.

This ladder needs no changes to APO itself — `prepare_data.py` sizes are CLI
flags and `evaluate.py` accepts any prompt file and split.

## 5. Sampling techniques

- **Keep `val`/`test` stratified-random** (already the default:
  `prepare_data.py` stratifies by dataset family with a fixed seed). These
  splits must represent the deployment distribution; never bias them.
- **Bias `train` toward hard examples if anything.** The gradient step learns
  from failures, so low-reward tasks carry the most information. Score a
  candidate pool with the baseline prompt first, then over-sample the low-reward
  tasks into `train`. This usually beats simply adding more (mostly easy) train
  data.
- **Always sample with `--probe-content-filter`** so content-filter-blocked
  videos (uniform reward 0, pure noise) never enter any split; probe results are
  cached per video in `data/content_filter_cache.json`, so growing the splits
  later re-probes only the new videos.

## 6. Step-by-step playbook: growing data and beam size together

The dataset sizes and the beam hyperparameters are one budget — grow them in
lockstep, one stage at a time, and let each stage's numbers decide the next
move. Total rollout cost per run is roughly:

```
rollouts ≈ val_size                                   (seed-prompt baseline)
         + beam_rounds × beam_width × branch_factor
           × (gradient_batch_size + val_batch_size)
```

**Stage 0 — smoke (once per environment).** Verify the loop, not the science.

```bash
.venv/bin/python prepare_data.py --train-size 2 --val-size 2 --test-size 2 --probe-content-filter
.venv/bin/python apo_train.py --smoke
```

Move on when: run completes, `results/report.md` is generated.

**Stage 1 — pilot: measure the noise.** Default sizes, default beam.

```bash
.venv/bin/python prepare_data.py --train-size 40 --val-size 24 --test-size 30 --seed 42 --probe-content-filter
.venv/bin/python apo_train.py                     # beam 2x2x2, gradient batch 4, val batch 24
.venv/bin/python evaluate.py --name baseline      # per-task rewards -> sigma
```

Read out of this stage: `σ` from `results/eval_baseline.json`, and the spread of
candidate val scores in `results/report.md`. Decision rule:

- Candidate scores all within `2.8 × σ/√24` of each other → val is too small to
  select; go to Stage 2.
- One candidate clearly wins → you may already confirm it on test and stop.

**Stage 2 — widen the search, then sharpen the ruler.** Change one axis at a
time so you can attribute the improvement:

1. *More exploration, same data* — raise `--beam-rounds 3` (depth: more rounds
   of critique-and-rewrite) or `--beam-width 3` / `--branch-factor 3` (breadth:
   more parallel candidates). Prefer depth first: rounds compound, width
   multiplies cost linearly for one-shot diversity.
2. *Better critiques* — raise `--gradient-batch-size 8` so each text-gradient
   sees more failure examples (cheap: train rollouts only).
3. *Sharper selection* — regrow val to the size Section 1 prescribes and match
   the flag:

   ```bash
   .venv/bin/python prepare_data.py --train-size 40 --val-size 64 --test-size 100 --seed 42 --probe-content-filter
   .venv/bin/python apo_train.py --beam-rounds 3 --val-batch-size 64
   ```

   The probe cache makes regrowing cheap: only the newly added videos are
   probed. Note that regrowing with a different size re-deals all splits —
   train/val/test stay disjoint, but individual tasks may move between splits,
   so re-run the baseline eval afterwards.

Decision rule: keep escalating exploration while each round still produces a new
best (`report.md` shows `Best prompt updated` in late rounds). If the last round
never improves, more rounds are wasted money — stop growing the beam.

**Stage 3 — confirm on held-out data.** Re-score the finalists (Section 4,
step 2) on test / a large re-scoring split:

```bash
.venv/bin/python evaluate.py --name baseline
.venv/bin/python evaluate.py --prompt results/best_prompt.txt --name tuned
```

Ship the tuned prompt only if the paired gap exceeds `2 × SE` of the test set.
Otherwise return to Stage 2 with the cheapest untried lever.

Which knob for which symptom:

| Symptom (from report.md / eval) | Knob | Direction |
| --- | --- | --- |
| candidates tie within noise | `--val-batch-size` + val split size | grow val |
| candidates all similar to seed | `--branch-factor`, gradient model | more/stronger edits |
| best improves every round | `--beam-rounds` | add rounds |
| last rounds never improve | `--beam-rounds` | stop growing |
| critiques repeat the same complaint | `--gradient-batch-size`, harder train tasks | richer failures |
| final gap plausible but unconfirmed | test split size | grow test |

## 7. Recommended defaults for this project

| Split | Pilot (smoke) | Working size | When to grow further |
| --- | --- | --- | --- |
| train | 4 | 80 | only if gradient batches look repetitive |
| val | 2 | 100 (from measured σ, see section 8) | ranking of top candidates unstable across runs |
| test | 2 | ~100 (to resolve 0.03–0.04 effects) | final paired gap has `p ≈ 0.05`, need tighter CI |

Procedure in one line: estimate `σ` from `results/eval_*.json`, size val with
`n ≈ 2(1.96σ/δ)²` for the gap `δ` you care about, keep train small but hard,
screen with small val, confirm on a large held-out set.

## 8. Measured calibration and per-stage sizes (updated 2026-07-18)

The end-to-end runs of 2026-07-17/18 (see
[reward-comparison.md](reward-comparison.md) sections 5–6) replace section 1's
estimates with measured numbers:

- **Per-task reward SD (σ)**: 0.113 (v1 reward) / 0.133 (v2 reward), from the
  `val_rewards` of every candidate in `results/<run_id>/report.json` — a bit
  below the 0.20 assumed in section 1.
- **SE of a candidate mean at val=100**: 0.011–0.013. Across both runs the 9
  candidate means span only ~0.05, i.e. the whole field covers ~4 SE and beam
  selection among the mid-pack is noise-driven — val=100 is "sufficient but
  not generous" for this task.
- **Paired delta SD on test (σ_d)**: 0.193 (gpt-5.4 probe, 30 paired tasks).
  At n=30 the paired SE is 0.035, while this project's real effects are all
  in the 0.03–0.04 range (whole-APO +0.035, target-model swap +0.041) —
  **a 30-task test cannot resolve them**, which is why "within noise / not
  yet conclusive" recurs throughout reward-comparison. Getting a 0.04 effect
  to 2 SE needs n ≈ (0.193 / 0.02)² ≈ 90–100 tasks.

Per-stage recommended sizes:

| Stage | train | val | test | Rationale |
| --- | --- | --- | --- | --- |
| Stage 0 smoke | 2 | 2 | 2 | verify the loop, ignore scores |
| Stage 1 pilot | 40 | 24 | 30 | measure σ, not draw conclusions |
| Stage 2 production runs (current) | 80 | 100 | 30 | val SE ≈ 0.012 supports beam selection |
| Stage 3 final (next re-split) | 80 | 100 | **~100** | resolving 0.03–0.04 effects needs paired SE ≤ 0.02 |

The final sizes are 80:100:100; as a general rule of thumb remember
**train : val : test ≈ 2 : 4 : 4** — roughly 8:1:1 inverted. Note this is a
ratio of relative sizes, not a partition of a whole (the pool has 5850 tasks;
only 280 are drawn on demand), and when the budget grows the increment should
go to val/test first while train stays flat, so larger setups drift further
from this starting point. Two operational reminders:

1. **Grow test once, then freeze.** Re-sampling re-deals the tasks, which
   invalidates every recorded baseline anchor (0.5686, 0.6095 etc. in
   reward-comparison sections 5/6) and forces re-measurement. Do not grow
   test in small increments.
2. **Repeat passes complement, not replace, a larger test.** Running the same
   test split 2–3 times and averaging cuts generation/judge variance by
   1/√k at far lower cost than adding tasks — but it cannot reduce the
   task-sampling noise of having only 30 tasks. Key claims should rest on
   both ~100 test tasks and 2–3 repeat passes.
