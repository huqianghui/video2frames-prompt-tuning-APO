# Video2Frames Prompt Tuning (APO)

**English** | [中文](README.zh.md)

Tune the fixed instruction prompt of a video-surveillance frame-analysis task with
[Agent-Lightning](../README.md)'s APO (Automatic Prompt Optimization) algorithm.

The customer dataset `original_data/qwen_0318_swift_task.json` contains 5850 video analysis tasks
(video → structured JSON with `english_detail` / `brief` / `title` / `scene_type` /
`is_courier_action`). This project converts each video task into a **multi-frame image
task**: the original `<video>` placeholder is removed and a frame placeholder section
is appended after the instruction, one placeholder per frame:

```
<frame 1 | 0s> <frame 2 | 3s> ... <frame n | 3(n-1)s>
```

Frames were pre-extracted (roughly one frame every 3 seconds; the frame count varies
per video) and stored in Azure Blob Storage. APO tunes **only the fixed instruction
part** of the prompt; the frame placeholder section is rebuilt per task at runtime.

> **Important:** `original_data/qwen_0318_swift_task.json` is customer-provided data
> and must never be committed or pushed to GitHub. The `original_data/`, `data/`,
> `log/`, and `results/` directories are tracked as empty folders (`.gitkeep` only);
> their contents are git-ignored because they contain or derive from customer data.
> Copy `original_data/`, `data/`, and the repository-root `.env` to the training
> machine separately (e.g. via scp).

## Installation

This project must run against agent-lightning **0.3.1 built from this repository's
source** (the PyPI 0.3.0 release is missing required functionality):

```bash
cd video2frames-prompt-tuning
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import agentlightning; print(agentlightning.__version__)"  # expect 0.3.1
```

## Configuration

Blob storage settings are read from the repository root `.env`
(`blob4videodatasets_connection_string`, `blob4videodatasets_container_name`,
`blob4videodatasets_frames_folder_name`). Additionally, the following Azure OpenAI
variables must be added to the `.env` (or exported):

| Variable | Purpose | Default |
| --- | --- | --- |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | required |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | required |
| `OPENAI_API_VERSION` | Azure OpenAI API version (e.g. `2024-10-21`) | required |
| `AZURE_OPENAI_DEPLOYMENT` | Multimodal deployment analyzed frames are sent to | `gpt-4o` |
| `JUDGE_MODEL` | Deployment used by the LLM judge in the reward | `gpt-4.1-mini` |
| `APO_GRADIENT_MODEL` | Deployment APO uses to critique prompts | `gpt-4.1` |
| `APO_APPLY_EDIT_MODEL` | Deployment APO uses to rewrite prompts | `gpt-4.1-mini` |
| `FRAMES_AS_BASE64` | Set to `true` to send frames as base64 data URIs instead of SAS URLs | unset |

## Workflow

```bash
# 1. Prepare the datasets (stratified sample; resolves frame blobs from Azure).
#    Sampling and splitting stratify jointly by (family, is_courier_action),
#    so every split mirrors the pool's label mix; the val split is guaranteed
#    at least --val-courier-min (default 0.15) courier positives, and each
#    split's courier/scene_type/family distribution is logged. scene_type has
#    no quota (distribution report only).
#    --probe-content-filter checks every candidate against the Azure content
#    safety filter during sampling (~3% of videos are rejected regardless of
#    the prompt) and backfills blocked ones, so the splits reach their target
#    sizes with tasks that are guaranteed to pass. Probe results are cached per
#    video in data/content_filter_cache.json, so re-running the script (or
#    probe_content_filter.py) never re-probes an already-probed video.
.venv/bin/python prepare_data.py --train-size 40 --val-size 24 --test-size 30 --seed 42 --probe-content-filter

# 1b. (Second round) Regrow train/val while keeping the held-out test split
#     frozen: --freeze-test leaves test.jsonl untouched and excludes its videos
#     from resampling (--test-size is ignored). Note: same-seed frozen runs do
#     not reproduce a previous round's train/val (the candidate pool changed).
.venv/bin/python prepare_data.py --train-size 80 --val-size 100 --freeze-test --probe-content-filter

# 2. (Optional) Audit existing splits against the content safety filter.
.venv/bin/python probe_content_filter.py          # report only (data/content_filter_probe.json)
.venv/bin/python probe_content_filter.py --apply  # report + drop blocked tasks (no backfill)
.venv/bin/python probe_content_filter.py --from-report  # re-apply an existing report

# 3. Debug a single rollout with the baseline prompt.
.venv/bin/python frame_agent.py --limit 1

# 4. Smoke-test the APO loop end to end (minimal beam, cheap).
.venv/bin/python apo_train.py --smoke

# 5. Full APO run. Every run gets a timestamped run ID: the APO log goes to
#    log/apo_<run_id>.log and all artifacts to results/<run_id>/ — best prompt
#    (best_prompt.txt), a full optimization report (per-round candidate
#    prompts, rewards, gradient critiques, validation scores) in report.md +
#    report.json, a compact prompt version tree (derivation, scores, winner)
#    in tree.md, and summary.json which records the run parameters plus a
#    fingerprint (row count + hash) of the data/ splits used. When the best
#    prompt beats the seed, diffs.md shows unified diffs for each step of its
#    derivation chain (e.g. v0 → v4 → v7) plus the overall seed → best diff.
#    Runs never overwrite each other; results/latest points at the newest run.
.venv/bin/python apo_train.py

# 5b. (Optional) Re-generate the report from the log of any past run.
.venv/bin/python generate_report.py --log log/apo_<run_id>.log --output-dir results/<run_id>

# 5c. (Optional) Build only the version tree from an existing report.md — no
#     log needed (e.g. a report.md copied from another machine). Beam-survival
#     markers are unavailable in this mode.
.venv/bin/python generate_report.py --from-report results/latest/report.md

# 6. Compare baseline vs tuned prompt on the held-out test split.
.venv/bin/python evaluate.py --name baseline
.venv/bin/python evaluate.py --prompt results/latest/best_prompt.txt --name tuned

# 7. (Optional) Compare reward versions: score the same prompt under v1 and v2,
#    then join the results task-by-task.
.venv/bin/python evaluate.py --name baseline_v1 --reward-version v1
.venv/bin/python evaluate.py --name baseline_v2 --reward-version v2
.venv/bin/python compare_rewards.py results/eval_baseline_v1.json results/eval_baseline_v2.json
```

AgentOps SaaS upload is **disabled by default** (spans are still traced
locally, which is all APO needs). Pass `--enable-agentops-service` to
`apo_train.py` only if you want session replays on app.agentops.ai.

The default split sizes (40/24/30) are a pilot configuration. See
[doc/dataset-sizing.md](doc/dataset-sizing.md) for how to estimate the split
sizes your target effect size actually requires, and for a stage-by-stage
playbook for growing the datasets and the beam hyperparameters together.

## Test Split: Role and Usage Conditions

The three splits play different roles in APO; **test never participates in any
optimization decision**:

| Split | Role | Consumer |
| --- | --- | --- |
| train | Rollouts for the critic's textual gradients (`--gradient-batch-size` per step) | APO gradient phase |
| val | Scores each candidate prompt and decides beam survival | APO selection phase |
| test | Held-out final acceptance; answers "how much better is tuned vs baseline" | `evaluate.py` (manually triggered) |

**Preconditions (check every item before running test):**

1. **APO actually produced a prompt that beats the seed** — check
   `results/<run_id>/report.md` or the end of the log: if the best prompt is
   still v0 (`Best prompt not updated`), running test is pointless; fix the
   data/evaluation setup and rerun APO first.
2. **Verify the `best_prompt.txt` you evaluate came from a full run, not a
   smoke run** — each run (smoke included) writes its own `results/<run_id>/`
   directory, and `results/latest` points at the most recent one, which may be
   a smoke run. Check the beam parameters in that run's `summary.json` (smoke
   is 1/1/1) to confirm the artifacts belong to the full run.
3. **Test must stay unseen** — never probe test scores repeatedly during
   optimization; all tuning and prompt selection uses val only. Every decision
   made against test discounts the credibility of the final number. Run it once
   after an optimization round has converged.

**Steps and outputs:**

```bash
.venv/bin/python evaluate.py --name baseline                                       # baseline prompt
.venv/bin/python evaluate.py --prompt results/latest/best_prompt.txt --name tuned  # tuned prompt
```

The two runs write `results/eval_baseline.json` and `results/eval_tuned.json`
(mean_reward plus per-task details); comparing `mean_reward` is the final
verdict. If the gap is smaller than the evaluation noise observed on val
(about ±0.015 under the pilot configuration), do not claim an improvement —
grow the splits per [doc/dataset-sizing.md](doc/dataset-sizing.md) and
re-verify first.

## Reward

The reward is the part of this project that evolves most, so it lives in a
standalone versioned package: `reward/` with one subfolder per version
(`reward/v1/`, `reward/v2/`, ...), each containing its implementation
(`reward.py`), its tunable parameters (`config.yaml`), and optionally its own
APO meta-prompt overrides (`*.poml`). Every version implements the same
`RewardFunction` interface (`reward/base.py`), so the agent, APO training, and
evaluation are version-agnostic.

**Selecting a version:** pass `--reward-version vN` to `frame_agent.py`,
`apo_train.py`, or `evaluate.py`, or set the `REWARD_VERSION` environment
variable (explicit flag > env var > default `v1`). `apo_train.py` pins the
resolved version in the environment so forked runner processes score with the
same reward, and records it (plus the full reward config) in the run's
`summary.json`.

**v1 (default)** — hybrid reward in `[0, 1]`:

- `0.2` × exact match of `scene_type`
- `0.2` × exact match of `is_courier_action`
- `0.6` × LLM-judge semantic score over `english_detail` / `brief` / `title`

**v2 (upgraded hybrid reward)** — per-field judges, mechanical rule
compliance, and multiplicative gates (redesigned following the SkillOpt-04
analysis article):

- Soft score: `0.45 × judge_detail + 0.20 × judge_brief + 0.10 × judge_title +
  0.25 × rule_compliance`, where `rule_compliance` is the mean of deterministic
  0/1 checks derived from the baseline prompt's hard style rules (exact 5 JSON
  keys, word limits, "person" instead of gendered words, no camera/frame/
  timestamp mentions, Non-Notable trigger consistency).
- Hard gates multiply the soft score: scene error × 0.5, courier false positive
  × 0.3, courier false **negative** × 0.2 (missing a real courier is the most
  expensive mistake; the asymmetry is a customer assumption to confirm).
- Optional judge-noise reduction: set `judge_samples: 3` (and a non-zero
  `judge_temperature`) in `reward/v2/config.yaml` to take the median of
  repeated judge calls.
- `reward/v2/text_gradient_video2frames.poml` describes the v2 objective to
  the APO optimizer (each version owns its text-gradient meta-prompt; see
  "APO Meta-Prompts" below).

Both versions score invalid-JSON output as `0`. Requests rejected by the Azure
OpenAI content safety filter also score `0` — the rejection depends only on
the input frames, so it is identical for every candidate prompt. A probe of
the default 94-task sample found ~3% blocked (not only `ucf_crime`; some
`Charades` videos trigger it too). Sample with `prepare_data.py
--probe-content-filter` to exclude blocked videos up front (the reward-0
fallback still covers anything that slips through), or audit existing splits
with `probe_content_filter.py`.

**Comparing versions:** evaluate the same prompt under both rewards with
distinct `--name` labels, then join the per-task results:

```bash
.venv/bin/python evaluate.py --name baseline_v1 --reward-version v1
.venv/bin/python evaluate.py --name baseline_v2 --reward-version v2
.venv/bin/python compare_rewards.py results/eval_baseline_v1.json results/eval_baseline_v2.json
```

`compare_rewards.py` prints a per-task delta table plus per-family and overall
means, and writes the joined result next to the inputs. Measured results
(baseline calibration, interpretation, and the 2×2 end-to-end comparison plan)
are logged in [doc/reward-comparison.md](doc/reward-comparison.md).

**Adding a new version:** copy an existing folder to `reward/v3/`, adjust
`config.yaml` and `reward.py` (keep the `RewardFunction` interface and the
`get_reward()` factory), and update its `text_gradient_*.poml` to state the new
objective (declared in `apo_meta_prompts`). It is picked up automatically by
`--reward-version v3`.

See [doc/reward-design.md](doc/reward-design.md) for the v1 design rationale
and the assumptions that should be confirmed with the customer before a
large-scale run; the v2 design follows the SkillOpt-04 article
([SkillOpt 系列 04](https://github.com/huqianghui/mindforge/blob/main/Notes/AI/SkillOpt/SkillOpt%E7%B3%BB%E5%88%9704%EF%BC%9AAPO%C3%97SkillOpt%E8%81%94%E5%90%88%E5%B1%95%E6%9C%9B%E2%80%94%E2%80%94%E5%85%88%E6%8E%A2%E7%B4%A2%E5%90%8E%E7%B2%BE%E4%BF%AE%E7%9A%84%E4%B8%A4%E6%AE%B5%E5%BC%8F%E7%AE%A1%E9%81%93%E4%B8%8E%E9%80%89%E5%9E%8B%E7%AE%97%E8%B4%A6%E6%96%B9%E6%B3%95.md)).

## APO Meta-Prompts

APO itself is driven by two meta-prompts: a *text gradient* template that
critiques the current prompt from rollout traces, and an *apply edit* template
that rewrites it. Both are declared per reward version in the
`apo_meta_prompts` section of `reward/<version>/config.yaml` (`null` = use the
shared default in `prompts/`, a filename = use the customized file in the
version folder), and they split by reward coupling:

- **text gradient** states the optimization objective (the reward formula), so
  it is **owned by the reward version** — every version must declare one
  (`reward/v1/text_gradient_video2frames.poml` describes the 0.2/0.2/0.6
  objective, `reward/v2/...` the per-field/gated one); `apo_train.py` refuses
  to run a version without it.
- **apply edit** only encodes reward-independent invariants (5-field JSON
  contract, no frame/`<video>` placeholders), so the shared
  `prompts/apply_edit_video2frames.poml` is the default for all versions.

Pass `--default-poml` to fall back to the framework's built-in templates. See
[doc/apo-poml-customization.md](doc/apo-poml-customization.md) for what the two
files do and the exact changes vs the defaults.

## Execution Strategy & Platform Notes

`apo_train.py` picks the execution strategy automatically (`execution_strategy()`):

- **Linux with Python ≤ 3.13** (multiprocessing start method `fork`, the primary
  target platform): the default client/server strategy with parallel runner
  processes — `--n-runners 4` by default.
- **macOS / Windows** (start method `spawn`, or `forkserver` on Linux Python 3.14+):
  falls back to serial shared-memory mode
  (`strategy={"type": "shm", "n_runners": 1, "main_thread": "algorithm"}`) with a
  warning — fine for smoke tests and small runs.

The fallback exists because the current agent-lightning runtime has two
platform-related limitations:

1. **The default client/server strategy fails on macOS and Windows.**
   `ClientServerExecutionStrategy._spawn_runners` (`agentlightning/execution/client_server.py`)
   starts runner processes via `multiprocessing.get_context()` — the *platform default*
   start method — and passes a locally defined closure (`_runner_sync`) as the process
   entry point. On Linux (Python ≤ 3.13) the default is `fork`, which never pickles the
   entry point, so everything works. On macOS and Windows the default is `spawn`, which
   must pickle it and fails with
   `AttributeError: Can't pickle local object 'ClientServerExecutionStrategy._spawn_runners.<locals>._runner_sync'`.
   Windows has no `fork`, so there is no workaround there.

2. **Shared-memory mode cannot run parallel runners.**
   The shm strategy runs runners as threads inside one process, but the tracer registers
   a *process-global* active tracer (`set_active_tracer` in `agentlightning/tracer/base.py`
   raises `An active tracer is already set`). With 2+ runner threads, every overlapping
   rollout fails, so `n_runners` must stay at 1.

**Practical consequence:** for large-scale runs use **Linux with Python ≤ 3.13**,
where `apo_train.py` parallelizes automatically (tune with `--n-runners`). On
macOS/Windows the same command still runs correctly, just serially.

See [doc/performance-tuning.md](doc/performance-tuning.md) for what the beam
hyperparameters (`--beam-rounds` / `--beam-width` / `--branch-factor`) mean and
how to tune them, the concurrency model, formulas to estimate run time and
Azure OpenAI quota needs, and how to pick `--n-runners` / batch sizes (e.g.
`--n-runners 12 --gradient-batch-size 8` on a large VM).

> **Caveat:** Python 3.14 changes the default start method on Linux to `forkserver`
> (which also pickles the entry point), so those environments auto-fall back to the
> serial mode too. Pin Python ≤ 3.13 for parallel runs until this is fixed upstream
> (microsoft/agent-lightning: use a module-level function as the process entry point,
> and a thread-local/contextvar for the active tracer).

## Dashboard (Optional)

On Linux runs you may see this in the log:

```
ERROR    Dashboard directory not found at .../agentlightning/dashboard
```

**This error is harmless** — the dashboard is an optional web UI for browsing
the store (rollouts, spans, traces), and training works fine without it. It
appears because this project installs agent-lightning from source and the
frontend has not been built. To enable the UI, build it once
(`cd <agent-lightning>/dashboard && npm install && npm run build`) and restart.
See [doc/dashboard.md](doc/dashboard.md) for details, including why the
macOS/Windows shm fallback has no dashboard at all.

## Smoke Test

Offline (no network, no credentials):

```bash
.venv/bin/pytest tests/ -v
```

Online (requires blob access + Azure OpenAI):

```bash
.venv/bin/python prepare_data.py --train-size 2 --val-size 2 --test-size 2
.venv/bin/python frame_agent.py --limit 1
.venv/bin/python apo_train.py --smoke
```

## Included Files

| File | Role |
| --- | --- |
| `original_data/qwen_0318_swift_task.json` | Customer dataset (pandas `to_json` dump). **Never commit.** |
| `blob_utils.py` | Azure Blob helpers: env loading, video→frame-prefix mapping, frame listing, SAS URLs. |
| `prepare_data.py` | Converts the pandas dump into `data/{train,val,test}.jsonl` and `data/baseline_prompt.txt`. Stratifies jointly by (family, `is_courier_action`) with a val courier-positive floor (`--val-courier-min`); `--freeze-test` regrows train/val without touching the test split; `--probe-content-filter` skips videos blocked by the content safety filter. |
| `probe_content_filter.py` | Probes tasks against the Azure content safety filter; caches results per video in `data/content_filter_cache.json`, reports the blocked ratio per split, and optionally removes blocked tasks. |
| `frame_agent.py` | `@rollout` frame-analysis agent, frame placeholder builder, debug CLI; scoring is delegated to the versioned `reward/` package. |
| `reward/` | Versioned reward package: `base.py` (shared `RewardFunction` interface, JSON parsing, judge helpers), `v1/` (hybrid 0.2/0.2/0.6 reward), `v2/` (upgraded hybrid reward: per-field judges + rule compliance + gates). Each version has `reward.py` + `config.yaml` + its own text-gradient POML. Select via `--reward-version` / `REWARD_VERSION`. |
| `compare_rewards.py` | Joins two `evaluate.py` result files task-by-task (e.g. the same prompt under reward v1 vs v2) and prints per-task deltas plus per-family/overall means. |
| `apo_train.py` | APO training entry point; each run writes `log/apo_<run_id>.log` and `results/<run_id>/` (`best_prompt.txt`, `summary.json` with a data fingerprint, run report), and repoints `results/latest`. Meta-prompts come from the reward version's `apo_meta_prompts` config (`--default-poml` reverts to the framework templates). |
| `prompts/apply_edit_video2frames.poml` | Shared reward-agnostic APO apply-edit meta-prompt (5-field JSON contract, frame-placeholder ban). The reward-specific text-gradient meta-prompt lives in each `reward/<version>/`. |
| `generate_report.py` | Parses an APO run log (`--log log/apo_<run_id>.log`) into `report.md` / `report.json` (candidate prompts, rewards, gradient critiques per round), `tree.md` (compact version tree: derivation, scores, beam survival, winner), and — when the best prompt beats the seed — `diffs.md` (per-step derivation diffs plus overall seed → best) under `--output-dir`. |
| `evaluate.py` | Evaluates a prompt file on a dataset split; writes `results/eval_<name>.json` (records the reward version; `--reward-version` selects it). |
| `doc/dataset-sizing.md` / `doc/dataset-sizing.zh.md` | Guide for sizing the splits (noise/SE math), staged scaling, and beam-hyperparameter tuning playbook (English/Chinese). |
| `doc/reward-design.md` / `doc/reward-design.zh.md` | Reward definition, design rationale, and the open questions to confirm with the customer (English/Chinese). |
| `doc/reward-comparison.md` / `doc/reward-comparison.zh.md` | Measured v1-vs-v2 comparison log: baseline scale calibration on the test split, interpretation, and the 2×2 end-to-end comparison playbook (English/Chinese). |
| `doc/apo-poml-customization.md` / `doc/apo-poml-customization.zh.md` | What the APO meta-prompts do, why they are customized, and the exact changes vs the framework defaults (English/Chinese). |
| `doc/dashboard.md` / `doc/dashboard.zh.md` | What the Agent-Lightning dashboard is, why the "Dashboard directory not found" error is harmless, and how to build/access the UI (English/Chinese). |
| `doc/performance-tuning.md` / `doc/performance-tuning.zh.md` | Beam hyperparameters (rounds/width/branch-factor) with a tuning decision table, concurrency model, run-time/quota formulas, and how to choose `--n-runners` and batch sizes (English/Chinese). |
| `README.md` / `README.zh.md` | This document (English/Chinese). |
| `tests/` | Offline unit tests (fixtures only, no customer data, no network). |
| `conftest.py` | Makes project modules importable from `tests/`. |
| `requirements.txt` | Installs agent-lightning 0.3.1 from source (`-e ..[apo]`) plus project deps. |
| `pyrightconfig.json` | Points pyright at the project virtualenv. |
| `.gitignore` | Keeps customer data, generated datasets, logs, results, and env files out of git (folders kept via `.gitkeep`). |
