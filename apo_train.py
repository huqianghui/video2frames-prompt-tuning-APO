"""Tune the fixed frame-analysis instruction prompt with Agent-Lightning APO.

Runs beam-search prompt optimization over the fixed instruction extracted from
the customer dataset. Only the instruction text is tuned; the per-video
`<frame n | Xs>` placeholders and images are appended by the agent at runtime.

Usage:
    python apo_train.py [--beam-rounds 2] [--beam-width 2] [--branch-factor 2]
                        [--gradient-batch-size 4] [--val-batch-size 24] [--smoke]

`--smoke` shrinks everything to the minimum (1x1x1 beam, tiny batches) to
verify the end-to-end loop cheaply.

The best prompt and a score summary are written to `results/`.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
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

logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "log"


def setup_apo_logger(file_path: Path = LOG_DIR / "apo.log") -> None:
    """Dump a copy of all the logs produced by the APO algorithm to a file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] (Process-%(process)d %(name)s)   %(message)s")
    file_handler.setFormatter(formatter)
    logging.getLogger("agentlightning.algorithm.apo").addHandler(file_handler)


def move_agentops_log() -> None:
    """Move the agentops SDK log into `log/` (the SDK hardcodes `agentops.log` in the cwd)."""
    agentops_log = Path("agentops.log")
    if not agentops_log.exists():
        return
    for handler in logging.getLogger("agentops").handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
    agentops_log.replace(LOG_DIR / "agentops.log")
    logger.info("Moved agentops.log to %s", LOG_DIR / "agentops.log")


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
    setup_apo_logger()
    os.environ[ENABLE_AGENTOPS_SERVICE_ENV] = "true"
    enable_agentops_service(True)

    gradient_model = os.environ.get("APO_GRADIENT_MODEL", "gpt-4.1")
    apply_edit_model = os.environ.get("APO_APPLY_EDIT_MODEL", "gpt-4.1-mini")

    algo = APO[FrameTask](
        AsyncAzureOpenAI(),
        gradient_model=gradient_model,
        apply_edit_model=apply_edit_model,
        gradient_batch_size=args.gradient_batch_size,
        val_batch_size=args.val_batch_size,
        beam_width=args.beam_width,
        branch_factor=args.branch_factor,
        beam_rounds=args.beam_rounds,
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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "best_prompt.txt").write_text(best.template, encoding="utf-8")
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(
            {
                "best_score": best_score,
                "beam_rounds": args.beam_rounds,
                "beam_width": args.beam_width,
                "branch_factor": args.branch_factor,
                "gradient_model": gradient_model,
                "apply_edit_model": apply_edit_model,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Best score: %s. Best prompt written to %s", best_score, RESULTS_DIR / "best_prompt.txt")

    report_path = generate_report(output_dir=RESULTS_DIR)
    move_agentops_log()

    print(best.template)
    print("best score:", best_score)
    if report_path is not None:
        print("run report:", report_path)


if __name__ == "__main__":
    main()
