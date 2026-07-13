# Video2Frames Prompt Tuning (APO)

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

> **Important:** `original_data/qwen_0318_swift_task.json` is customer-provided data.
> The `original_data/` directory is listed in `.gitignore` and must never be committed
> or pushed to GitHub. The generated `data/` and `results/` directories are ignored as
> well.

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
#    --probe-content-filter checks every candidate against the Azure content
#    safety filter during sampling (~3% of videos are rejected regardless of
#    the prompt) and backfills blocked ones, so the splits reach their target
#    sizes with tasks that are guaranteed to pass.
.venv/bin/python prepare_data.py --train-size 40 --val-size 24 --test-size 30 --seed 42 --probe-content-filter

# 2. (Optional) Audit existing splits against the content safety filter.
.venv/bin/python probe_content_filter.py          # report only (data/content_filter_probe.json)
.venv/bin/python probe_content_filter.py --apply  # report + drop blocked tasks (no backfill)
.venv/bin/python probe_content_filter.py --from-report  # re-apply an existing report

# 3. Debug a single rollout with the baseline prompt.
.venv/bin/python frame_agent.py --limit 1

# 4. Smoke-test the APO loop end to end (minimal beam, cheap).
.venv/bin/python apo_train.py --smoke

# 5. Full APO run. Best prompt lands in results/best_prompt.txt, and a full
#    optimization report (per-round candidate prompts, rewards, gradient
#    critiques, validation scores) in results/report.md + results/report.json.
.venv/bin/python apo_train.py

# 5b. (Optional) Re-generate the report from an existing log/apo.log, e.g. for
#     an earlier run in the same log file.
.venv/bin/python generate_report.py --run -1

# 6. Compare baseline vs tuned prompt on the held-out test split.
.venv/bin/python evaluate.py --name baseline
.venv/bin/python evaluate.py --prompt results/best_prompt.txt --name tuned
```

## Reward

Each rollout is scored with a hybrid reward in `[0, 1]`:

- `0.2` × exact match of `scene_type`
- `0.2` × exact match of `is_courier_action`
- `0.6` × LLM-judge semantic score over `english_detail` / `brief` / `title`
- Output that is not a valid JSON object scores `0`.
- Requests rejected by the Azure OpenAI content safety filter score `0` — the
  rejection depends only on the input frames, so it is identical for every
  candidate prompt. A probe of the default 94-task sample found ~3% blocked
  (not only `ucf_crime`; some `Charades` videos trigger it too). Sample with
  `prepare_data.py --probe-content-filter` to exclude blocked videos up front
  (the reward-0 fallback still covers anything that slips through), or audit
  existing splits with `probe_content_filter.py`.

## Execution Strategy & Platform Notes

`apo_train.py` runs the Trainer with `strategy={"type": "shm", "n_runners": 1, "main_thread": "algorithm"}`
(single-threaded shared-memory mode). This is deliberate — the current agent-lightning
runtime has two platform-related limitations:

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

**Practical consequence:** on macOS/Windows this project runs correctly but serially
(fine for smoke tests and small runs). For large-scale runs, use **Linux with
Python ≤ 3.13**, where you can switch to the default client/server strategy with
parallel runner processes:

```python
trainer = Trainer(
    algorithm=algo,
    n_runners=4,  # default client/server strategy, parallel runner processes
    initial_resources={"prompt_template": prompt_template_baseline()},
    adapter=TraceToMessages(),
)
```

> **Caveat:** Python 3.14 changes the default start method on Linux to `forkserver`
> (which also pickles the entry point), so the same failure will appear there. Pin
> Python ≤ 3.13 for parallel runs until this is fixed upstream
> (microsoft/agent-lightning: use a module-level function as the process entry point,
> and a thread-local/contextvar for the active tracer).

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
| `prepare_data.py` | Converts the pandas dump into `data/{train,val,test}.jsonl` and `data/baseline_prompt.txt`; `--probe-content-filter` skips videos blocked by the content safety filter. |
| `probe_content_filter.py` | Probes tasks against the Azure content safety filter; reports the blocked ratio per split and optionally removes blocked tasks. |
| `frame_agent.py` | `@rollout` frame-analysis agent, frame placeholder builder, hybrid reward, debug CLI. |
| `apo_train.py` | APO training entry point; writes `results/best_prompt.txt`, `results/summary.json`, and the run report. |
| `generate_report.py` | Parses `log/apo.log` into `results/report.md` / `report.json` (candidate prompts, rewards, gradient critiques per round). |
| `evaluate.py` | Evaluates a prompt file on a dataset split; writes `results/eval_<name>.json`. |
| `tests/` | Offline unit tests (fixtures only, no customer data, no network). |
| `conftest.py` | Makes project modules importable from `tests/`. |
| `requirements.txt` | Installs agent-lightning 0.3.1 from source (`-e ..[apo]`) plus project deps. |
| `pyrightconfig.json` | Points pyright at the project virtualenv. |
| `.gitignore` | Keeps customer data, generated datasets, results, and env files out of git. |
