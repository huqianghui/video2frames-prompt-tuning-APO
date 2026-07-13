# Choosing Dataset Sizes for APO

**English** | [中文](dataset-sizing.zh.md)

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
- **`train` — usually fine as is.** Each APO round only samples
  `--gradient-batch-size` (default 4) tasks to critique; a pool of 40 already
  provides variety. To improve the gradient signal, increase
  `--gradient-batch-size` or `--beam-rounds` before adding train data.

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
| train | 4 | 40 (keep) | only if gradient batches look repetitive |
| val | 2 | 64–100 (from σ) | ranking of top candidates unstable across runs |
| test | 2 | ~100 | final paired gap has `p ≈ 0.05`, need tighter CI |

Procedure in one line: estimate `σ` from `results/eval_*.json`, size val with
`n ≈ 2(1.96σ/δ)²` for the gap `δ` you care about, keep train small but hard,
screen with small val, confirm on a large held-out set.
