"""Run the HLE evaluation across a suite of models (mixed providers).

Reads ``model_list.json`` (candidate models + judge + defaults) and invokes
``evaluate_hle_main.py`` once per model, passing each model's LiteLLM routing
(provider prefix + api-key/base env names) on the command line. Models run
concurrently; each writes its own timestamped run directory under a shared
suite directory, and a combined ``suite_report.json`` + markdown summary is
written at the end.

Example (10-entry text-only smoke test, parallel, basic mode, medium thinking)::

    python examples/hle/run_hle_suite.py \\
        --n-samples 10 --max-workers 5
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

current_file = Path(__file__).resolve()
EVAL_SCRIPT = current_file.parent / "evaluate_hle_main.py"


def _load_config(model_list_path: Path) -> Dict[str, Any]:
    with open(model_list_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _model_cmd(model: Dict[str, Any], judge: Dict[str, Any], defaults: Dict[str, Any], args, suite_dir: Path) -> List[str]:
    reasoning = model.get("reasoning_effort") or defaults.get("reasoning_effort", "medium")
    agent_mode = model.get("agent_mode") or defaults.get("agent_mode", "basic")
    # Per-model worker cap (CLI override > model entry > defaults). Radium models
    # use a low value because the gateway drops many concurrent long streams.
    workers = args.max_workers if args.max_workers else model.get("max_workers", defaults.get("max_workers", 5))
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--n-samples", str(args.n_samples),
        "--max-workers", str(workers),
        "--max-retries", str(args.max_retries),
        "--seed", str(args.seed),
        "--agent-mode", agent_mode,
        "--reasoning-effort", reasoning,
        "--max-completion-tokens", str(args.max_completion_tokens),
        "--request-timeout-s", str(args.request_timeout_s),
        "--litellm-model", model["litellm_model"],
        "--api-key-env", model.get("api_key_env", ""),
        "--api-base-env", model.get("api_base_env", ""),
        "--model-label", model["name"],
        "--judge-litellm-model", judge["litellm_model"],
        "--judge-api-key-env", judge.get("api_key_env", ""),
        "--judge-api-base-env", judge.get("api_base_env", ""),
        "--out-dir", str(suite_dir),
        "-nc",
    ]
    if args.data_path:
        cmd.extend(["--data-path", str(args.data_path)])
    if args.include_images:
        cmd.append("--include-images")
    return cmd


def _run_one(model: Dict[str, Any], cmd: List[str], suite_dir: Path) -> Dict[str, Any]:
    name = model["name"]
    log_dir = suite_dir / ".log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{name}.log"
    # Force the child to emit UTF-8 so its log is decodable regardless of the
    # host console codepage (Windows defaults to cp1252 for redirected stdout).
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    with open(log_path, "w", encoding="utf-8") as f_log:
        proc = subprocess.run(cmd, cwd=str(current_file.parent), stdout=f_log, stderr=subprocess.STDOUT, env=env)

    if proc.returncode != 0:
        print(f"  [{name}] FAILED (rc={proc.returncode}); see {log_path}")
        return {"model": name, "error": proc.returncode, "log": str(log_path)}

    # Locate this model's run dir + Meta.json (read tolerantly — logs may still
    # contain stray non-UTF-8 bytes from nested tooling).
    run_dir: Optional[Path] = None
    with open(log_path, "r", encoding="utf-8", errors="replace") as f_log:
        for line in f_log:
            if line.startswith("Output:"):
                run_dir = Path(line.split("Output:", 1)[1].strip())
                break
    if run_dir is None or not (run_dir / "Meta.json").exists():
        candidates = sorted(
            [p for p in suite_dir.iterdir() if p.is_dir() and p.name.startswith("run_") and name in p.name],
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        run_dir = candidates[0] if candidates else None

    if run_dir is None or not (run_dir / "Meta.json").exists():
        print(f"  [{name}] completed but no Meta.json found.")
        return {"model": name, "error": "no_meta", "log": str(log_path)}

    with open(run_dir / "Meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    m = meta.get("aggregate_metrics", {})
    print(f"  [{name}] done — accuracy={m.get('accuracy')} errors={m.get('n_errors')} "
          f"avg_reasoning_tokens={m.get('avg_reasoning_tokens')}")
    return {
        "model": name,
        "run_dir": str(run_dir),
        "accuracy": m.get("accuracy"),
        "n_graded": m.get("n_graded"),
        "n_errors": m.get("n_errors"),
        "calibration_error": m.get("calibration_error"),
        "avg_reasoning_tokens": m.get("avg_reasoning_tokens"),
        "reasoning_token_fraction": m.get("reasoning_token_fraction"),
    }


def _row_from_run_dir(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Build a suite-summary row from a completed run directory's Meta.json."""
    meta_path = run_dir / "Meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    label = meta.get("model_config", {}).get("model_label") or run_dir.name
    m = meta.get("aggregate_metrics", {})
    return {
        "model": label,
        "run_dir": str(run_dir),
        "accuracy": m.get("accuracy"),
        "n_graded": m.get("n_graded"),
        "n_errors": m.get("n_errors"),
        "calibration_error": m.get("calibration_error"),
        "avg_reasoning_tokens": m.get("avg_reasoning_tokens"),
        "reasoning_token_fraction": m.get("reasoning_token_fraction"),
    }


def _write_suite_report(
    suite_dir: Path, rows: List[Dict[str, Any]], stamp: str,
    n_samples: Any, max_workers: Any, judge_model: Any,
) -> None:
    report = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": n_samples,
        "max_workers": max_workers,
        "judge": judge_model,
        "models": rows,
    }
    with open(suite_dir / f"suite_report_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(suite_dir / f"suite_report_{stamp}.md", "w", encoding="utf-8") as f:
        f.write("# HLE Model Suite Report\n\n")
        f.write(f"**timestamp:** {report['timestamp']}\n\n")
        f.write(f"**n_samples:** {n_samples} | **workers/model:** {max_workers} | "
                f"**judge:** {judge_model}\n\n")
        f.write("| Model | Accuracy | Graded | Errors | Calibration Err | Avg Reasoning Tokens | Reasoning Frac |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            if r.get("error") is not None and "accuracy" not in r:
                f.write(f"| {r['model']} | ERROR ({r.get('error')}) | - | - | - | - | - |\n")
            else:
                f.write(
                    f"| {r['model']} | {r.get('accuracy')} | {r.get('n_graded')} | "
                    f"{r.get('n_errors')} | {r.get('calibration_error')} | "
                    f"{r.get('avg_reasoning_tokens')} | {r.get('reasoning_token_fraction')} |\n"
                )
    print(f"Suite report written: {suite_dir / f'suite_report_{stamp}.md'}")

    # Auto-generate visuals (accuracy / errors / reasoning tokens / calibration).
    try:
        from EvalRing.utils.visualizations import generate_generative_suite_visuals
        generate_generative_suite_visuals(suite_dir / f"suite_report_{stamp}.json")
    except Exception as e:  # noqa: BLE001 - visuals are best-effort
        print(f"(visuals skipped: {e})")


def _report_only(suite_dir: Path, models_file: Path) -> int:
    """Rebuild the suite report from existing run dirs (e.g. after an interruption)."""
    if not suite_dir.exists():
        print(f"Suite dir not found: {suite_dir}")
        return 1
    cfg = _load_config(models_file)
    order = {m["name"]: i for i, m in enumerate(cfg.get("models", []))}
    rows: List[Dict[str, Any]] = []
    for d in sorted(suite_dir.iterdir()):
        if d.is_dir() and d.name.startswith("run_") and "suite" not in d.name:
            row = _row_from_run_dir(d)
            if row:
                rows.append(row)
    rows.sort(key=lambda r: order.get(r.get("model"), 999))
    stamp = suite_dir.name.replace("run_suite_", "").replace("_hle", "")
    judge = cfg.get("judge", {}).get("litellm_model")
    n = rows[0].get("n_graded") if rows else None
    _write_suite_report(suite_dir, rows, stamp, n, "-", judge)
    for r in rows:
        print(f"  {r['model']:<18} acc={r.get('accuracy')} errors={r.get('n_errors')}")
    return 0


def _retry_failed_suite(suite_dir: Path, args) -> int:
    """Retry failed cases across every model's run dir in a suite, then rebuild
    the report + visuals. Each model reruns with its own routing/concurrency."""
    if not suite_dir.exists():
        print(f"Suite dir not found: {suite_dir}")
        return 1
    cfg = _load_config(Path(args.models_file))
    models = cfg.get("models", [])
    judge = cfg.get("judge", {})
    defaults = cfg.get("defaults", {})
    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        models = [m for m in models if m["name"] in wanted]

    jobs = []
    for m in models:
        cands = sorted(
            [p for p in suite_dir.iterdir()
             if p.is_dir() and p.name.startswith("run_") and "suite" not in p.name and m["name"] in p.name],
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if not cands:
            print(f"  [{m['name']}] no run dir found in suite — skipping")
            continue
        jobs.append((m, cands[0]))

    if not jobs:
        print("No run dirs to retry.")
        return 1

    def _retry_one(model, run_dir):
        workers = args.max_workers if args.max_workers else model.get("max_workers", defaults.get("max_workers", 5))
        cmd = [
            sys.executable, str(EVAL_SCRIPT),
            "--retry-failed", "--run-dir", str(run_dir),
            "--max-workers", str(workers),
            "--max-retries", str(args.max_retries),
            "--request-timeout-s", str(args.request_timeout_s),
            "--litellm-model", model["litellm_model"],
            "--api-key-env", model.get("api_key_env", ""),
            "--api-base-env", model.get("api_base_env", ""),
            "--model-label", model["name"],
            "--judge-litellm-model", judge["litellm_model"],
            "--judge-api-key-env", judge.get("api_key_env", ""),
            "--judge-api-base-env", judge.get("api_base_env", ""),
            "-nc",
        ]
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        log_path = suite_dir / ".log" / f"retry_{model['name']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f_log:
            proc = subprocess.run(cmd, cwd=str(current_file.parent), stdout=f_log, stderr=subprocess.STDOUT, env=env)
        print(f"  [{model['name']}] retry rc={proc.returncode} (log: {log_path.name})")
        return model["name"], proc.returncode

    print("=" * 80)
    print(f"RETRY failed cases across {len(jobs)} model(s) | timeout="
          f"{'none' if args.request_timeout_s <= 0 else str(args.request_timeout_s)+'s'}")
    print("=" * 80)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futs = [ex.submit(_retry_one, m, d) for m, d in jobs]
        for f in concurrent.futures.as_completed(futs):
            f.result()

    # Rebuild the suite report + visuals from the patched run dirs.
    _report_only(suite_dir, Path(args.models_file))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HLE evaluation across a model suite.")
    parser.add_argument("--models-file", type=str, default=str(current_file.parent / "model_list.json"))
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=0,
                        help="Override workers for ALL models. 0 (default) = use each model's "
                             "own max_workers from the model list.")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated model names to run (subset of the model list).")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-completion-tokens", type=int, default=0,
                        help="0 (default) = no output cap; reasoning models finish naturally.")
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    parser.add_argument("--data-path", type=str, default=None,
                        help="Dataset CSV to evaluate (defaults to the HLE data inside evaluate_hle_main).")
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--report-only", type=str, default=None,
                        help="Path to an existing suite dir; rebuild its report from run dirs and exit.")
    parser.add_argument("--retry-failed", type=str, default=None, metavar="SUITE_DIR",
                        help="Path to an existing suite dir; re-run failed cases in each model's run "
                             "dir (in place), then rebuild the report. Use --request-timeout-s 0 for no limit.")
    args = parser.parse_args()

    if args.report_only:
        return _report_only(Path(args.report_only), Path(args.models_file))
    if args.retry_failed:
        return _retry_failed_suite(Path(args.retry_failed), args)

    cfg = _load_config(Path(args.models_file))
    models = cfg.get("models", [])
    judge = cfg.get("judge", {})
    defaults = cfg.get("defaults", {})
    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        models = [m for m in models if m["name"] in wanted]
    if not models:
        print("No models to run (check --only / model list).")
        return 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(args.out_dir) if args.out_dir else (current_file.parent / "_EvalRing")
    suite_dir = base / f"run_suite_{stamp}_hle"
    suite_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"HLE model suite — {len(models)} models | n_samples={args.n_samples} | "
          f"workers/model={args.max_workers} | judge={judge.get('litellm_model')}")
    print(f"suite_dir={suite_dir}")
    print("=" * 80)

    rows: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as ex:
        futures = {
            ex.submit(_run_one, m, _model_cmd(m, judge, defaults, args, suite_dir), suite_dir): m["name"]
            for m in models
        }
        for fut in concurrent.futures.as_completed(futures):
            rows.append(fut.result())

    # Deterministic ordering by model-list order.
    order = {m["name"]: i for i, m in enumerate(models)}
    rows.sort(key=lambda r: order.get(r.get("model"), 999))

    _write_suite_report(
        suite_dir, rows, stamp,
        n_samples=args.n_samples, max_workers=args.max_workers,
        judge_model=judge.get("litellm_model"),
    )

    print("\n" + "=" * 80)
    print(f"Suite complete. Report: {suite_dir}")
    print("=" * 80)
    for r in rows:
        print(f"  {r['model']:<18} acc={r.get('accuracy')} errors={r.get('n_errors')} "
              f"{'(ERROR)' if r.get('error') is not None and 'accuracy' not in r else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
