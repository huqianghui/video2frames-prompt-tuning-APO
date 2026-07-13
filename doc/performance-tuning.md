# Performance Tuning: Concurrency and Run-Time Estimation

**English** | [中文](performance-tuning.zh.md)

How to make `apo_train.py` run faster on a capable Linux VM with a
high-quota Azure OpenAI deployment, and how to compute the right
parameter values instead of guessing.

## The concurrency model

APO schedules rollouts as **parallel within a batch, sequential across
batches**:

1. Every evaluation (one prompt × one batch of tasks) enqueues the whole
   batch into the store; the `--n-runners` runner processes consume it in
   parallel. Effective parallelism = `min(n_runners, batch size)`.
2. The algorithm loop itself is sequential: each parent's gradient
   evaluation, and each candidate's validation evaluation, run one after
   another.
3. Producing each new candidate costs 2 additional **sequential** LLM
   calls (text gradient + apply edit) that no runner setting can
   parallelize.
4. Inside one rollout, the multimodal call and the judge call are also
   sequential — a rollout occupies one runner for its full duration.

## Formulas

Let:

| Symbol | CLI flag | Default |
| --- | --- | --- |
| `R` | `--beam-rounds` | 2 |
| `W` | `--beam-width` | 2 |
| `B` | `--branch-factor` | 2 |
| `g` | `--gradient-batch-size` | 4 |
| `v` | `--val-batch-size` | 24 |
| `n` | `--n-runners` | 4 |
| `t` | — | measured rollout latency (≈ 6 s in practice) |

**Total rollouts per run** (initial seed validation + per-round gradient
and validation evaluations):

```
N = v + R × W × B × (g + v)
```

**Wall-clock estimate:**

```
T ≈ t × [ ceil(v/n) × (1 + R·W·B)          # seed + candidate validations
        + R·W·B × ceil(g/n) ]              # gradient evaluations
  + R × W × B × (t_gradient + t_edit)      # sequential meta-prompt calls
```

`t_gradient + t_edit` is typically 20–40 s with `gpt-4.1` /
`gpt-4.1-mini`.

**Example (defaults, n = 4):** N = 24 + 2·2·2·(4+24) = 248 rollouts;
T ≈ 6·(6·9 + 8·1) + 8·30 ≈ 10–12 min.

**Example (n = 12, g = 8):** validation waves drop from 6 to 2 per
candidate; T ≈ 6·(2·9 + 8·1) + 8·30 ≈ 6–7 min, and the sequential
meta-prompt calls become the dominant cost.

## Choosing `n_runners`

- **Upper bound by batch size:** runners beyond `min(g, v)` sit idle
  during the corresponding phase. With `v = 24` there is no benefit past
  `n = 24`.
- **Upper bound by Azure OpenAI quota:** each in-flight rollout holds one
  request (frames are token-heavy). Required capacity is roughly

  ```
  RPM ≈ n × 60 / t × 2        # ×2: analysis call + judge call
  TPM ≈ n × 60 / t × tokens_per_rollout
  ```

  A rollout with 10–20 frames costs ~10–20 k input tokens, so `n = 12`
  at `t = 6 s` needs on the order of 1.5–2.5 M TPM on the multimodal
  deployment. Check your deployment quota before raising `n`.
- **CPU is rarely the limit:** runners spend their time waiting on the
  API, so `n` may exceed the VM's core count.

**Recommended starting point on a large VM with high quota:**

```bash
.venv/bin/python apo_train.py --n-runners 12 --gradient-batch-size 8
```

Raising `g` both fills the runners during the gradient phase and gives
the critique model more evidence per parent.

## When runners stop helping

Once `n ≥ v`, the remaining wall time is dominated by the
`R × W × B × 2` sequential gradient/apply-edit calls (~2 min per round
at defaults). Reducing `R`, `W`, or `B` cuts this linearly — at the cost
of search breadth (see the trade-offs in
[dataset-sizing.md](dataset-sizing.md)). Parallelizing the per-branch
meta calls would require a change to the upstream APO implementation
(`agentlightning/algorithm/apo/apo.py`, `_generate_candidate_prompts`),
which loops over branches with `await` in series.

## Quick reference

| Goal | Change |
| --- | --- |
| Faster validation phases | `--n-runners` up to `min(v, quota limit)` |
| No idle runners during gradient phase | `--gradient-batch-size ≈ n_runners` |
| Cheaper/faster run overall | lower `R`, `W`, `B` (fewer candidates) |
| Better statistics, slower | higher `v` (see dataset-sizing.md) |
