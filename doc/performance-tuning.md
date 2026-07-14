# Performance Tuning: Beam Hyperparameters, Concurrency, and Run-Time Estimation

**English** | [中文](performance-tuning.zh.md)

Cost, run time, and search quality are faces of the same coin:
`--beam-rounds` / `--beam-width` / `--branch-factor` define the shape of
the search (and therefore the total rollout count = API cost),
`--n-runners` decides how much of that work is amortized in parallel,
and your Azure OpenAI quota caps the parallelism. This document covers
all three layers together: what each parameter means, how they interact,
the cost formulas, how to derive `n_runners` and the required quota, and
how to estimate wall-clock time. (See
[dataset-sizing.md](dataset-sizing.md) for growing the validation split
together with these hyperparameters.)

## Mental model: beam search over a prompt version tree

Think of prompt optimization as search over a "prompt version tree": the
seed prompt v0 is the root, and every new candidate is a rewrite of some
parent. The three beam hyperparameters control the tree's **depth,
surviving width, and fan-out** respectively.

### `--beam-rounds` (R, depth) — how many iterations

One round = "rewrite the current good prompts → score on the validation
split → prune". More rounds let improvements **compound across rounds**
— e.g. a best-prompt derivation chain v0 → v4 → v7 is two rounds each
advancing one step. R decides how *far* the optimization can travel.

### `--beam-width` (W, width) — how many survivors per round

After each round's scoring, only the W highest-scoring prompts survive
into the next round as "parents". W = 1 degenerates to greedy
hill-climbing (follow only the single best path; easy to get stuck in a
local optimum); larger W keeps more "alternative evolutionary lines"
alive simultaneously.

### `--branch-factor` (B, fan-out) — how many children per survivor

For each parent: run `g` (`--gradient-batch-size`) train rollouts → the
gradient model writes a critique → the apply-edit model produces **B
different** rewrites from it. Larger B tries more diverse edits in a
single direction.

### One full round, and how the parameters interact

Every round runs three fixed steps, each on a different dataset (source:
`agentlightning/algorithm/apo/apo.py`):

```
Round r:
  ① Candidate generation (W parents in the beam, × B each)
     └─ draw g tasks from train (--gradient-batch-size, apo.py:650)
        → run rollouts for traces → gradient model writes a critique
        → apply-edit model produces a new prompt
        [train's role: supply "failure samples" to the critic;
         these scores never enter any ranking]
        (W×B new candidates total; each costs 2 extra sequential meta calls)

  ② Beam scoring / selection
     └─ draw one batch from val (--val-batch-size, apo.py:719)
        → score all candidates → sort → top-W survive into round r+1
        [val's role: decide who stays alive]

  ③ Best re-evaluation
     └─ the FULL val dataset (apo.py:777)
        → this round's beam leader is re-tested → best is updated only
          if the re-test beats the history best
        [role: a noise gate — a candidate must score high twice to
         become best]
```

Three easy-to-miss consequences of ③:

- **Only each round's beam leader gets to challenge the best.** If the
  truly best prompt is pushed to 2nd place by noise in ②, it never even
  gets a re-test.
- **The final best score comes from the re-test, not from ②** — which
  is why tree.md can show a candidate scoring 0.756 in ② yet not being
  best: its re-test came in lower, meaning the first score was partly
  noise.
- **When `--val-batch-size` < the full val size, ② and ③ use different
  data**: ② uses random subsets (shuffled, epoch-based rotation — each
  task appears at most once per epoch), ③ uses everything. When they
  are equal (e.g. 24/24 or 100/100), ③ degenerates to a second run of
  the same tasks — it only guards against LLM randomness and provides
  no fresh-data check.

- **W × B is the per-round "exploration budget".** W=1, B=4 and W=2, B=2
  both produce 4 candidates per round, but the former bets the budget on
  many edits of a single line, while the latter keeps two independent
  lines alive.
- **R vs W×B is a depth-vs-breadth trade-off.** Compounding improvements
  need R; hedging against a wrong line needs W; per-round edit diversity
  needs B.
- Do not raise W above the per-round candidate supply (last round's
  survivors + new candidates), or pruning becomes a no-op.

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

## Cost and run-time formulas

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

Rollout count ≈ API cost, and it is **linear in R, W, and B** — doubling
all three quadruples the cost:

| Quantity | Formula | Defaults (2/2/2) | Deeper (4/2/2) | Aggressive (4/2/3) |
| --- | --- | --- | --- | --- |
| New prompts | R×W×B | 8 | 16 | 24 |
| Sequential meta calls | R×W×B×2 | 16 | 32 | 48 |
| Total rollouts | v + R×W×B×(g+v) | 248 | 536 | 792 |

(With v = 24, g = 8.)

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

Note the wall-clock asymmetry: rollouts are amortized by `--n-runners`,
but the `W×B×2` gradient/apply-edit calls per round are **sequential**
— so in wall-clock terms, **raising R is more expensive than raising
B**.

## Choosing `n_runners`

- **Upper bound by batch size:** runners beyond `min(g, v)` sit idle
  during the corresponding phase. With `v = 24`, `n = 4` needs
  `ceil(24/4) = 6` validation waves per candidate while `n = 12` needs
  only 2; there is no benefit past `n = 24`. Same for the gradient
  phase — raise `--gradient-batch-size` to ≈ n_runners, or runners idle
  there.
- **Upper bound by Azure OpenAI quota:** each in-flight rollout issues 1
  analysis request to the multimodal deployment and 1 judge request to
  the judge deployment. Required capacity is roughly

  ```
  multimodal TPM ≈ n × (60 / t) × input tokens per rollout
  multimodal RPM ≈ n × (60 / t)
  judge TPM      ≈ n × (60 / t) × judge input tokens (text-only, a few k per rollout)
  ```

  A rollout with 10–20 frames costs ~10–20 k input tokens (the frame
  images dominate), i.e. roughly **150 k TPM per runner** on the
  multimodal deployment. Check your deployment quota before raising `n`.
- **CPU is rarely the limit:** runners spend their time waiting on the
  API, so `n` may exceed the VM's core count.

**Worked example (multimodal deployment 2.5 M TPM, judge deployment
3 M TPM):**

| n-runners | Multimodal TPM needed | vs 2.5 M quota |
| --- | --- | --- |
| 4 | ~600 k | uses 1/4 — wasteful |
| **12** | ~1.8 M | **comfort zone, recommended** |
| 16 | ~2.4 M | at the ceiling; frame-heavy videos may hit 429 |
| ≥ 24 | — | exceeds v, no benefit |

The judge calls are text-only — even `n = 16` uses only a few hundred k
TPM, so the judge deployment is rarely the bottleneck. Gradient and
apply-edit are just `W×B` sequential calls per round — negligible.
Occasional 429s are retried with backoff by the SDK (slower, not fatal)
— to squeeze the quota, run once, confirm the log has no `429`, then
raise `n`.

**Recommended starting point on a large VM with high quota:**

```bash
.venv/bin/python apo_train.py \
  --beam-rounds 4 --beam-width 2 --branch-factor 2 \
  --n-runners 12 --gradient-batch-size 8
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

## Tuning decision table

| Observation | Action |
| --- | --- |
| Best prompt appeared in the last round (not converged) | raise R |
| All of a round's candidates score below their parents (edit diversity too low) | raise B |
| Several lines score close together and you fear pruning the wrong one | raise W |
| Validation phases are slow | raise `--n-runners` up to `min(v, quota limit)` |
| Runners idle during the gradient phase | `--gradient-batch-size ≈ n_runners` |
| Cheaper/faster run overall | lower `R`, `W`, `B` (fewer candidates) |
| Score differences are within evaluation noise | grow `v` first (see dataset-sizing.md), don't touch R/W/B yet |
