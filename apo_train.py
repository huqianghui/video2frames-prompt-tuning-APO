"""Tune the fixed frame-analysis instruction prompt with Agent-Lightning APO.

Runs beam-search prompt optimization over the fixed instruction extracted from
the customer dataset. Only the instruction text is tuned; the per-video
`<frame n | Xs>` placeholders and images are appended by the agent at runtime.

Usage:
    python apo_train.py [--beam-rounds 2] [--beam-width 2] [--branch-factor 2]
                        [--gradient-batch-size 4] [--val-batch-size 24] [--smoke]
                        [--default-poml] [--reward-version v2]

`--smoke` shrinks everything to the minimum (1x1x1 beam, tiny batches) to
verify the end-to-end loop cheaply.

By default APO runs with project-specific meta-prompts (see
`doc/apo-poml-customization.md`): the text-gradient prompt states the
optimization objective and is owned by the reward version (declared in the
`apo_meta_prompts` section of `reward/<version>/config.yaml`), while the
reward-agnostic apply-edit prompt lives in `prompts/` (versions may override
it the same way). `--default-poml` falls back to the framework's built-in
templates.

Each run gets a timestamped run ID: logs go to `log/apo_<run_id>.log`, outputs
(best prompt, score summary, report) to `results/<run_id>/`, and the summary
records a fingerprint (row count + hash) of the `data/` splits used, so runs
never overwrite each other. `results/latest` always points at the newest run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

from agentlightning import Trainer, setup_logging
from agentlightning.adapter import TraceToMessages
from agentlightning.algorithm.apo import APO
from agentlightning.instrumentation.agentops import enable_agentops_service
from agentlightning.tracer.agentops import ENABLE_AGENTOPS_SERVICE_ENV
from blob_utils import PROJECT_ROOT, load_env
from frame_agent import FrameTask, frame_analyzer, load_tasks, prompt_template_baseline
from generate_report import generate_report
from reward import REWARD_VERSION_ENV, load_reward, poml_override, resolve_version

logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "log"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
APPLY_EDIT_POML = PROMPTS_DIR / "apply_edit_video2frames.poml"


def setup_apo_logger(file_path: Path) -> None:
    """Dump a copy of all the logs produced by the APO algorithm to a file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] (Process-%(process)d %(name)s)   %(message)s")
    file_handler.setFormatter(formatter)
    logging.getLogger("agentlightning.algorithm.apo").addHandler(file_handler)


def move_agentops_log(run_id: str) -> None:
    """Move the agentops SDK log into `log/` (the SDK hardcodes `agentops.log` in the cwd)."""
    agentops_log = Path("agentops.log")
    if not agentops_log.exists():
        return
    for handler in logging.getLogger("agentops").handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
    target = LOG_DIR / f"agentops_{run_id}.log"
    agentops_log.replace(target)
    logger.info("Moved agentops.log to %s", target)


def data_fingerprint() -> Dict[str, Any]:
    """Identify the dataset version used by this run (per-split row count + content hash)."""
    fingerprint: Dict[str, Any] = {}
    for split in ("train", "val", "test"):
        path = PROJECT_ROOT / "data" / f"{split}.jsonl"
        if not path.exists():
            fingerprint[split] = None
            continue
        content = path.read_bytes()
        fingerprint[split] = {
            "rows": sum(1 for line in content.splitlines() if line.strip()),
            "sha256": hashlib.sha256(content).hexdigest()[:12],
        }
    return fingerprint


def update_latest_symlink(run_results_dir: Path) -> None:
    """Point `results/latest` at the most recent run directory."""
    latest = RESULTS_DIR / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_results_dir.name)
    except OSError as exc:  # e.g. filesystems without symlink support
        logger.warning("Could not update %s symlink: %s", latest, exc)


def execution_strategy(n_runners: int) -> Dict[str, Any]:
    """Pick the fastest execution strategy the platform supports.

    The default client/server strategy starts runner processes with the platform's
    default multiprocessing start method and only works when that is `fork` (Linux,
    Python <= 3.13): `spawn` and `forkserver` cannot pickle the runner closure. On
    other platforms fall back to single-runner shared-memory mode (the tracer is
    process-global, so shm cannot run parallel runners either).
    """
    start_method = multiprocessing.get_start_method()
    if start_method == "fork":
        logger.info("Start method is 'fork': using client/server strategy with %d parallel runners.", n_runners)
        return {"n_runners": n_runners}
    logger.warning(
        "Start method %r cannot pickle the runner entry point; falling back to serial shared-memory mode.",
        start_method,
    )
    return {"strategy": {"type": "shm", "n_runners": 1, "main_thread": "algorithm"}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run APO on the frame-analysis prompt.")
    parser.add_argument("--beam-rounds", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--branch-factor", type=int, default=2)
    parser.add_argument("--gradient-batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=24)
    parser.add_argument("--n-runners", type=int, default=4, help="Parallel runners (Linux/fork platforms only).")
    parser.add_argument("--smoke", action="store_true", help="Minimal run to verify the end-to-end loop.")
    parser.add_argument(
        "--default-poml",
        action="store_true",
        help="Use the framework's built-in APO meta-prompts instead of the project-specific ones in prompts/.",
    )
    parser.add_argument(
        "--reward-version",
        default=None,
        help="Reward version from reward/ (default: REWARD_VERSION env var or v1).",
    )
    parser.add_argument(
        "--enable-agentops-service",
        action="store_true",
        help="Upload traces to the AgentOps SaaS (app.agentops.ai). Disabled by default: tracing still runs "
        "locally and training is unaffected.",
    )
    args = parser.parse_args()

    if args.smoke:
        args.beam_rounds = 1
        args.beam_width = 1
        args.branch_factor = 1
        args.gradient_batch_size = 2
        args.val_batch_size = 2
        logger.info("Smoke mode: beam 1x1x1, batches of 2.")

    load_dotenv()
    load_env()
    setup_logging()
    # Resolve the reward version once and pin it in the environment so forked
    # runner processes score with the same reward.
    reward_version = resolve_version(args.reward_version)
    os.environ[REWARD_VERSION_ENV] = reward_version
    reward_fn = load_reward(reward_version)
    logger.info("Reward version: %s", reward_version)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    apo_log_path = LOG_DIR / f"apo_{run_id}.log"
    setup_apo_logger(apo_log_path)
    logger.info("Run %s: APO log at %s", run_id, apo_log_path)
    # AgentOps SaaS upload is opt-in; local span collection (required for APO) always runs.
    # The env var propagates the choice to forked runner processes.
    os.environ[ENABLE_AGENTOPS_SERVICE_ENV] = "true" if args.enable_agentops_service else "false"
    enable_agentops_service(args.enable_agentops_service)

    gradient_model = os.environ.get("APO_GRADIENT_MODEL", "gpt-4.1")
    apply_edit_model = os.environ.get("APO_APPLY_EDIT_MODEL", "gpt-4.1-mini")

    poml_kwargs: Dict[str, Any] = {}
    if args.default_poml:
        logger.info("Using the framework's built-in APO meta-prompts (--default-poml).")
    else:
        # The text-gradient prompt states the optimization objective, so every
        # reward version must declare its own in config.yaml; apply_edit is
        # reward-agnostic and falls back to the shared prompts/ default.
        gradient_poml = poml_override(reward_version, "text_gradient")
        if gradient_poml is None:
            raise SystemExit(
                f"reward/{reward_version}/config.yaml must declare apo_meta_prompts.text_gradient "
                "(the text-gradient POML states the objective and is reward-version-specific), "
                "or pass --default-poml to use the framework's built-in templates."
            )
        apply_edit_poml = poml_override(reward_version, "apply_edit") or APPLY_EDIT_POML
        poml_kwargs = {
            "gradient_prompt_files": [gradient_poml],
            "apply_edit_prompt_files": [apply_edit_poml],
        }
        logger.info("Using APO meta-prompts: %s, %s", gradient_poml, apply_edit_poml)

    algo = APO[FrameTask](
        AsyncAzureOpenAI(),
        gradient_model=gradient_model,
        apply_edit_model=apply_edit_model,
        gradient_batch_size=args.gradient_batch_size,
        val_batch_size=args.val_batch_size,
        beam_width=args.beam_width,
        branch_factor=args.branch_factor,
        beam_rounds=args.beam_rounds,
        **poml_kwargs,
    )
    trainer = Trainer(
        algorithm=algo,
        initial_resources={"prompt_template": prompt_template_baseline()},
        adapter=TraceToMessages(),
        **execution_strategy(args.n_runners),
    )

    # The seed-prompt baseline is always scored on the full val split; shrink the
    # datasets in smoke mode so the run stays cheap.
    train_dataset = load_tasks("train", limit=4 if args.smoke else None)
    val_dataset = load_tasks("val", limit=2 if args.smoke else None)
    logger.info(
        "Starting APO: %d train / %d val tasks, gradient=%s, apply_edit=%s",
        len(train_dataset),
        len(val_dataset),
        gradient_model,
        apply_edit_model,
    )
    trainer.fit(agent=frame_analyzer, train_dataset=train_dataset, val_dataset=val_dataset)

    best = algo.get_best_prompt()
    best_score = algo._history_best_score  # pyright: ignore[reportPrivateUsage]
    run_results_dir = RESULTS_DIR / run_id
    run_results_dir.mkdir(parents=True, exist_ok=True)
    (run_results_dir / "best_prompt.txt").write_text(best.template, encoding="utf-8")
    (run_results_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "best_score": best_score,
                "reward_version": reward_version,
                "reward_config": reward_fn.config_dump(),
                "beam_rounds": args.beam_rounds,
                "beam_width": args.beam_width,
                "branch_factor": args.branch_factor,
                "gradient_model": gradient_model,
                "apply_edit_model": apply_edit_model,
                "custom_poml": not args.default_poml,
                "apo_log": str(apo_log_path.relative_to(PROJECT_ROOT)),
                "data": data_fingerprint(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Best score: %s. Best prompt written to %s", best_score, run_results_dir / "best_prompt.txt")

    report_path = generate_report(log_path=apo_log_path, output_dir=run_results_dir)
    update_latest_symlink(run_results_dir)
    move_agentops_log(run_id)

    print(best.template)
    print("best score:", best_score)
    if report_path is not None:
        print("run report:", report_path)


if __name__ == "__main__":
    main()
