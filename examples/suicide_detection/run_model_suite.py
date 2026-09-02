"""
Convenience wrapper to execute the sequential model suite for suicide detection.
"""

import sys
from pathlib import Path
import argparse

from EvalRing.logging_utils import configure_logging
from EvalRing.utils.suite_runner import run_suite

script_dir = Path(__file__).resolve().parent

def run_local_suite(
    n_samples: int = 10,
    max_workers: int = 50,
    seed: int = 42,
    agent_mode: str = "single-class",
    base_class: str = "Indicator",
    models_file: str = "model_list.json",
    cache: str = "latest",
    cache_mode: str = "both",
    continue_runs: bool = True,
    ignore_errors: bool = False,
    data_path: str = None,
    no_confirm: bool = False
) -> int:
    eval_script = script_dir / "evaluate_rsd15k_main.py"
    model_list_path = script_dir / models_file
    
    resolved_cache = str(script_dir / "_EvalRing") if cache == "latest" else cache

    return run_suite(
        eval_script=str(eval_script),
        model_list_path=str(model_list_path),
        n_samples=n_samples,
        max_workers=max_workers,
        seed=seed,
        agent_mode=agent_mode,
        base_class=base_class,
        cache=resolved_cache,
        cache_mode=cache_mode,
        continue_runs=continue_runs,
        ignore_errors=ignore_errors,
        data_path=data_path,
        no_confirm=no_confirm
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the suicide detection model suite.")
    parser.add_argument("--n-samples", type=int, default=10000, help="Number of samples to evaluate")
    parser.add_argument("--max-workers", type=int, default=50, help="Parallel workers per model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--agent-mode", type=str, default="single-class", help="Agent execution mode")
    parser.add_argument("--base-class", type=str, default="Indicator", help="Base class for binary modes")
    parser.add_argument("--models-file", type=str, default="model_list.json", help="Model list filename in this directory")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to the dataset CSV. Defaults to data/rsd_15k.csv next to this script.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--cache-mode", type=str, default="both", choices=["runs_only", "cache_file", "both", "none"], help="Global cache operation mode")
    parser.add_argument("--ignore-errors", action="store_true", help="Ignore cached errors, forcing them to re-run.")
    parser.add_argument("-nc", "--no-confirm", action="store_true", help="Bypass interactive run confirmation prompt")
    args = parser.parse_args()

    code = run_local_suite(
        n_samples=args.n_samples,
        max_workers=args.max_workers,
        seed=args.seed,
        agent_mode=args.agent_mode,
        base_class=args.base_class,
        models_file=args.models_file,
        cache=None if args.no_cache else "latest",
        cache_mode="none" if args.no_cache else args.cache_mode,
        continue_runs=not args.no_cache,
        ignore_errors=args.ignore_errors,
        data_path=args.data_path,
        no_confirm=args.no_confirm
    )
    sys.exit(code)
