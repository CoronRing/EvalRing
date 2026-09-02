"""
Generic utility to run a suite of models alongside an EvalRing evaluation script.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from ..config import CREDENTIAL_ENV_VARS, has_any_credentials
from ..logging_utils import get_logger
from .visualizations import VISUALS_AVAILABLE, generate_suite_visuals

logger = get_logger(__name__)


def _load_model_names(model_list_path: Path) -> list[str]:
    """Load model list JSON file and parse requested models preserving their ordering."""
    if not model_list_path.exists():
        logger.warning("Model list not found: %s", model_list_path)
        return []

    with open(model_list_path, encoding="utf-8") as f:
        data = json.load(f)

    models = data.get("models", [])
    model_names: list[str] = []
    for m in models:
        if isinstance(m, str):
            model_name = m.strip()
            if model_name:
                model_names.append(model_name)
            continue

        if isinstance(m, dict):
            if m.get("available") is False:
                continue
            model_name = (m.get("openrouter_id") or "").strip()
            if model_name:
                model_names.append(model_name)
            continue

    # De-dup while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for name in model_names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)

    return deduped


def _confirm_or_abort() -> None:
    """Prompt the operator to continue, exiting the process on abort.

    Only called from an interactive terminal; automated runs skip it so a suite
    can never hang waiting on stdin.
    """
    while True:
        val = input("Type 'y' to continue with the suite run, or 'q' to quit: ").strip().lower()
        if val == "y":
            logger.info("Proceeding...")
            return
        if val == "q":
            logger.info("Suite aborted.")
            sys.exit(0)
        logger.warning("Invalid input. Type 'y' to confirm or 'q' to abort.")


def _model_env(model_name: str) -> dict[str, str]:
    """Build the child-process environment that pins one model for a run.

    The vendor-neutral ``EVALRING_MODEL`` is authoritative; the provider-specific
    variables are also set so that evaluation scripts written against a single
    provider keep working unchanged.

    Args:
        model_name: Model identifier to pin for the child process.

    Returns:
        A copy of the current environment with the model variables applied.
    """
    env = os.environ.copy()
    for var in (
        "EVALRING_MODEL",
        "OPENAI_MODEL",
        "OPENROUTER_MODEL",
        "OPEN_ROUTER_MODEL",
        "RADIUM_MODEL",
    ):
        env[var] = model_name
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _read_meta(meta_path: Path) -> dict:
    """Read Meta.json output safely."""
    if not meta_path.exists():
        return {}
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _sum_error_histogram(error_histogram: object) -> int:
    """Sum arbitrary error dictionaries robustly."""
    if not isinstance(error_histogram, dict):
        return 0
    total = 0
    for v in error_histogram.values():
        if isinstance(v, bool):
            total += int(v)
            continue
        if isinstance(v, (int, float)):
            total += int(v)
            continue
        if isinstance(v, str):
            s = v.strip()
            if not s:
                continue
            try:
                total += int(float(s))
            except ValueError:
                continue
    return total


def _parse_boolish_int(value: str | None) -> int | None:
    """Safely parse boolean or integer-like truthy strings."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return 1
    if s in {"0", "false", "f", "no", "n"}:
        return 0
    return None


def _count_incorrect_and_errors(all_cases_csv: Path) -> tuple[int, int]:
    """Parse out incorrect cases directly from the target CSV output."""
    if not all_cases_csv.exists():
        return 0, 0
    incorrect_count = 0
    error_count = 0
    with open(all_cases_csv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            correct = _parse_boolish_int(row.get("correct"))
            if correct == 0:
                incorrect_count += 1
            prediction = (row.get("prediction") or "").strip().lower()
            err = (row.get("error") or "").strip()
            if prediction == "error" or err:
                error_count += 1
    return incorrect_count, error_count


def _strip_per_sample_results(md_text: str) -> str:
    """Strips the per-sample results portion to compress markdown reports."""
    lines = md_text.splitlines(keepends=True)
    out: list[str] = []
    skipping = False
    for line in lines:
        if not skipping and line.strip() == "## Per-Sample Results":
            skipping = True
            continue
        if skipping:
            if line.startswith("## ") and line.strip() != "## Per-Sample Results":
                skipping = False
                out.append(line)
            continue
        out.append(line)
    return "".join(out)


def run_suite(
    eval_script: str | Path,
    model_list_path: str | Path,
    n_samples: int = 2,
    max_workers: int = 32,
    seed: int = 0,
    agent_mode: str = "single-class",
    base_class: str | None = None,
    host_model: str | None = None,
    role_models_json: str | None = None,
    max_host_iterations: int = 10,
    continue_runs: bool = False,
    cache: str | None = None,
    cache_mode: str = "both",
    ignore_errors: bool = False,
    out_dir: str | None = None,
    data_path: str | None = None,
    no_confirm: bool = False,
) -> int:
    """Run one evaluation script once per model in a model list.

    Each model runs in its own subprocess with the model pinned through the
    environment (see :func:`_model_env`), so a crash in one model never takes
    down the suite. Per-model artifacts land in ``suite_dir`` alongside a
    combined JSON and Markdown report.

    Args:
        eval_script: Path to the evaluation script to invoke per model. It must
            accept the flags this function forwards (``--n-samples``,
            ``--max-workers``, ``--seed``, ``--agent-mode``, ``--out-dir``,
            ``-nc``, and optionally ``--base-class``, ``--data-path``,
            ``--cache``, ``--cache-mode``, ``--host-model``,
            ``--role-models-json``, ``--continue``, ``--ignore-errors``,
            ``--info-only``).
        model_list_path: Path to the model list JSON produced by
            :func:`EvalRing.utils.generate_model_list.generate_model_list`.
        n_samples: Number of dataset samples to evaluate per model.
        max_workers: Per-model request concurrency.
        seed: Sampling seed forwarded to the evaluation script.
        agent_mode: Agent mode string forwarded to the evaluation script.
        base_class: Optional base class for one-vs-rest modes. Forwarded only
            when set.
        host_model: Optional orchestrator model for multi-role runs.
        role_models_json: Optional JSON mapping of role names to models.
        max_host_iterations: Iteration cap for multi-role orchestration.
        continue_runs: Resume from previous partial runs instead of starting over.
        cache: Optional path to a cache database or run directory.
        cache_mode: One of ``"runs_only"``, ``"cache_file"``, ``"both"``, ``"none"``.
        ignore_errors: Re-run samples whose cached result was an error.
        out_dir: Parent directory for the suite output. Defaults to
            ``_EvalRing/`` beside the evaluation script.
        data_path: Optional dataset path forwarded to the evaluation script.
        no_confirm: Skip the interactive pre-run cache summary confirmation.

    Returns:
        ``0`` when every model completed, ``1`` when the suite could not start.
        Individual model failures are reported in the suite report rather than
        through this return value.
    """
    eval_script = Path(eval_script).resolve()
    model_list_path = Path(model_list_path).resolve()

    script_dir = eval_script.parent
    project_root = script_dir.parent.parent.parent

    model_names = _load_model_names(model_list_path)
    if not model_names:
        logger.error("No models loaded from %s", model_list_path)
        return 1

    load_dotenv(project_root / ".env")

    if not eval_script.exists():
        logger.error("Evaluate script not found: %s", eval_script)
        return 1

    if not has_any_credentials():
        logger.error(
            "No API key found in the environment. Set one of: %s",
            ", ".join(CREDENTIAL_ENV_VARS),
        )
        return 1

    summary_rows = []
    suite_started_at = datetime.now()
    suite_stamp = suite_started_at.strftime("%Y%m%d_%H%M%S")

    if out_dir:
        suite_dir = Path(out_dir).resolve() / f"run_suite_{suite_stamp}_{agent_mode}"
    else:
        suite_dir = script_dir / "_EvalRing" / f"run_suite_{suite_stamp}_{agent_mode}"

    suite_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Model suite start (%s)", suite_started_at.isoformat())
    logger.info("eval_script: %s", eval_script)
    logger.info(
        "n_samples=%s, max_workers=%s, seed=%s, agent_mode=%s, base_class=%s, "
        "host_model=%s, max_host_iterations=%s, continue_runs=%s",
        n_samples,
        max_workers,
        seed,
        agent_mode,
        base_class,
        host_model,
        max_host_iterations,
        continue_runs,
    )
    logger.info("models_loaded=%d from %s", len(model_names), model_list_path)
    logger.info("suite_dir=%s", suite_dir)

    if not no_confirm:
        logger.info("Pre-computing cache hits for the suite...")
        total_cached = 0
        total_uncached = 0

        import concurrent.futures
        import json

        def _get_cache_info(idx_model):
            idx, model_name = idx_model
            env = _model_env(model_name)

            cmd = [
                sys.executable,
                str(eval_script),
                "--n-samples",
                str(n_samples),
                "--max-workers",
                str(max_workers),
                "--seed",
                str(seed),
                "--agent-mode",
                str(agent_mode),
                "--max-host-iterations",
                str(max_host_iterations),
                "--out-dir",
                str(suite_dir),
                "--info-only",
            ]
            if base_class:
                cmd.extend(["--base-class", str(base_class)])
            if data_path:
                cmd.extend(["--data-path", str(data_path)])
            if cache:
                cmd.extend(["--cache", str(cache)])
            if cache_mode:
                cmd.extend(["--cache-mode", cache_mode])
            if host_model:
                cmd.extend(["--host-model", str(host_model)])
            if role_models_json:
                cmd.extend(["--role-models-json", str(role_models_json)])

            try:
                # We use bufsize=0 and binary mode (no text=True) so we can stream \r unbuffered tightly.
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(script_dir),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                )

                final_line = ""

                import threading

                stderr_output = []

                def drain_stderr(pipe):
                    buffer = bytearray()
                    while True:
                        byte = pipe.read(1)
                        if not byte:
                            break
                        buffer.extend(byte)
                        if byte == b"\n" or byte == b"\r":
                            l_clean = buffer.decode("utf-8", errors="replace").strip()
                            buffer.clear()
                            if l_clean:
                                with cache_lock:
                                    cache_progress[model_name] = l_clean
                                update_progress_file()
                                stderr_output.append(l_clean + "\n")
                    if buffer:
                        l_clean = buffer.decode("utf-8", errors="replace").strip()
                        if l_clean:
                            with cache_lock:
                                cache_progress[model_name] = l_clean
                            update_progress_file()
                            stderr_output.append(l_clean + "\n")

                stderr_thread = threading.Thread(target=drain_stderr, args=(proc.stderr,))
                stderr_thread.start()

                # Read stdout in binary and decode line by line natively.
                # Popen was created with stdout=PIPE, so the stream is present.
                assert proc.stdout is not None
                for line_b in proc.stdout:
                    line = line_b.decode("utf-8", errors="replace").strip()
                    if line:
                        final_line = line
                        if not line.startswith("{"):
                            with cache_lock:
                                cache_progress[model_name] = line
                            update_progress_file()

                proc.wait()
                stderr_thread.join()
                if proc.returncode != 0:
                    last_error = ""
                    for line in reversed(stderr_output):
                        if (
                            line.strip()
                            and "tqdm" not in line
                            and "overlapping" not in line.lower()
                        ):
                            if "error" in line.lower() or "exception" in line.lower():
                                last_error = line.strip()
                                break
                            if not last_error:
                                last_error = line.strip()
                    if not last_error:
                        last_error = "Unknown subprocess error."
                    raise subprocess.CalledProcessError(proc.returncode, cmd, stderr=last_error)

                import re

                match = re.search(r"(\{.*\})", final_line)
                if match:
                    cache_info = json.loads(match.group(1))
                else:
                    cache_info = json.loads(final_line)

                return (
                    model_name,
                    cache_info.get("cached", 0),
                    cache_info.get("uncached", 0),
                    idx,
                    None,
                )
            except Exception as e:
                err_msg = str(e)
                if isinstance(e, subprocess.CalledProcessError) and e.stderr:
                    err_msg += f"\nStderr: {e.stderr}"
                return model_name, 0, 0, idx, err_msg

        import threading

        from tqdm import tqdm

        cache_progress = dict.fromkeys(model_names, "Starting...")
        cache_lock = threading.Lock()
        progress_file = suite_dir / "progress.txt"

        def update_progress_file():
            with cache_lock:
                lines = ["Cache Scan Status:", "=" * 80]
                for m_name, status in cache_progress.items():
                    # Format model name to 30 chars
                    m_short = (m_name[:27] + "..") if len(m_name) > 30 else m_name
                    lines.append(f"{m_short:<30} : {status}")
                with open(progress_file, "w", encoding="utf-8") as pf:
                    pf.write("\n".join(lines) + "\n")

        update_progress_file()

        # Process in parallel up to max_workers or len(model_names)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(max_workers, len(model_names))
        ) as executor:
            future_to_model = {
                executor.submit(_get_cache_info, (i, m)): m for i, m in enumerate(model_names, 1)
            }

            completed = 0
            with tqdm(total=len(model_names), desc="Checking cache info") as pbar:
                for future in concurrent.futures.as_completed(future_to_model):
                    m_name = future_to_model[future]
                    completed += 1
                    try:
                        m, c, u, idx, err = future.result()
                        with cache_lock:
                            if err:
                                cache_progress[m_name] = f"Error: {err}"
                            else:
                                cache_progress[m_name] = f"[Cached: {c:<4} | Fresh: {u:<4}] Done."
                                total_cached += c
                                total_uncached += u
                    except Exception as e:
                        with cache_lock:
                            cache_progress[m_name] = f"Error: {e}"

                    update_progress_file()
                    pbar.set_postfix_str(f"{completed}/{len(model_names)} completed")
                    pbar.update(1)

        logger.info(
            "Suite cache summary: %d cached, %d uncached API calls",
            total_cached,
            total_uncached,
        )

        if not sys.stdin.isatty():
            logger.info("Non-interactive session: continuing without confirmation.")
        else:
            _confirm_or_abort()

    def _run_single_model(model_name: str, idx: int) -> dict:
        env = _model_env(model_name)

        cmd = [
            sys.executable,
            str(eval_script),
            "--n-samples",
            str(n_samples),
            "--max-workers",
            str(max_workers),
            "--seed",
            str(seed),
            "--agent-mode",
            str(agent_mode),
            "--max-host-iterations",
            str(max_host_iterations),
            "--out-dir",
            str(suite_dir),
        ]
        if base_class:
            cmd.extend(["--base-class", str(base_class)])
        if data_path:
            cmd.extend(["--data-path", str(data_path)])
        # IMPORTANT: Suite runner must not block on input! Force -nc.
        cmd.append("-nc")
        if continue_runs:
            cmd.append("--continue")
        if cache:
            cmd.extend(["--cache", str(cache)])
        if cache_mode:
            cmd.extend(["--cache-mode", cache_mode])
        if ignore_errors:
            cmd.append("--ignore-errors")
        if host_model:
            cmd.extend(["--host-model", str(host_model)])
        if role_models_json:
            cmd.extend(["--role-models-json", str(role_models_json)])

        clean_name = model_name.replace("/", "_")
        log_dir = suite_dir / ".log"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"run_{clean_name}.log"

        with open(log_path, "w", encoding="utf-8") as f_log:
            proc = subprocess.run(
                cmd, cwd=str(script_dir), env=env, stdout=f_log, stderr=subprocess.STDOUT
            )

        if proc.returncode != 0:
            logger.error(
                "Run failed for model '%s' (return code %s). See %s",
                model_name,
                proc.returncode,
                log_path,
            )
            return {"error": proc.returncode, "model": model_name}

        # Scan log for output dir
        after_dir = None
        with open(log_path, encoding="utf-8") as f_log:
            for line in f_log:
                if line.startswith("Output:   "):
                    after_dir = Path(line.replace("Output:   ", "").strip())
                    break

        if not after_dir or not after_dir.exists():
            model_suffix = model_name.split("/")[-1].lower()
            possible_dirs = []
            for p in suite_dir.iterdir():
                if p.is_dir() and p.name.startswith("run_"):
                    if model_suffix in p.name.lower():
                        possible_dirs.append(p)
            if possible_dirs:
                after_dir = sorted(possible_dirs, key=lambda d: d.stat().st_mtime, reverse=True)[0]

        if not after_dir or not after_dir.exists():
            logger.error("Could not locate new run directory for model '%s'.", model_name)
            return {"error": 2, "model": model_name}

        meta = _read_meta(after_dir / "Meta.json")
        execution_failures = meta.get("execution_failures", [])
        error_histogram = meta.get("error", {})
        incorrect_count, execution_error_count = _count_incorrect_and_errors(
            after_dir / "all_cases.csv"
        )
        error_total = _sum_error_histogram(error_histogram)
        metrics = meta.get("aggregate_metrics", {})
        accuracy = metrics.get("accuracy")
        f1 = metrics.get("f1_score")

        return {
            "model": model_name,
            "run_dir": str(after_dir),
            "accuracy": accuracy,
            "f1_score": f1,
            "execution_failures": len(execution_failures),
            "incorrect": incorrect_count,
            "execution_errors": execution_error_count,
            "error_total": error_total,
            "agent_mode": agent_mode,
            "base_class": base_class,
            "host_model": host_model,
            "role_models_json": role_models_json,
            "max_host_iterations": max_host_iterations,
        }

    futures = []
    has_errors = False
    stop_monitor = False

    def _monitor_progress():
        import shutil

        progress_file = suite_dir / "suite_progress.txt"

        while not stop_monitor:
            total_lines = 0
            total_expected = len(model_names) * n_samples
            per_model = []

            for m in model_names:
                m_suffix = m.split("/")[-1].lower()
                clean_name = m.replace("/", "_")
                log_path = suite_dir / ".log" / f"run_{clean_name}.log"

                cache_hits = 0
                is_done = False
                if log_path.exists():
                    try:
                        with open(log_path, encoding="utf-8") as lf:
                            for log_line in lf:
                                if (
                                    "overlapping with current run" in log_line
                                    and "remaining." in log_line
                                ):
                                    try:
                                        parts = log_line.split("Loaded ")
                                        if len(parts) > 1:
                                            cache_hits = int(parts[1].split(" ")[0])
                                    except Exception:
                                        pass
                                if (
                                    "Evaluation complete" in log_line
                                    or "completely bypassed" in log_line
                                ):
                                    is_done = True
                    except Exception:
                        pass

                lines = 0
                latest_dir = None
                for p in suite_dir.iterdir():
                    if p.is_dir() and p.name.startswith("run_") and m_suffix in p.name.lower():
                        latest_dir = p

                if latest_dir:
                    partial_path = latest_dir / "all_cases_partial.jsonl"
                    if partial_path.exists():
                        try:
                            with open(partial_path, encoding="utf-8") as pf:
                                lines = sum(1 for line in pf if line.strip())
                        except Exception:
                            pass

                if is_done:
                    current_prog = n_samples
                else:
                    current_prog = cache_hits + lines

                current_prog = min(current_prog, n_samples)

                import textwrap

                short_m = textwrap.shorten(m_suffix, width=20, placeholder="..")
                per_model.append(
                    f"{short_m.ljust(20)} : {current_prog} / {n_samples}  [Cache: {cache_hits}, Fresh: {lines}]"
                )
                total_lines += current_prog

            # Write detailed view to progress file
            try:
                with open(progress_file, "w", encoding="utf-8") as pf:
                    pf.write(f"Suite Status: {total_lines} / {total_expected} completed\n")
                    pf.write("=" * 40 + "\n")
                    pf.write("\n".join(per_model) + "\n")
            except Exception:
                pass

            # Terminal view: simple non-wrapping single line
            max_w = min(120, shutil.get_terminal_size().columns - 2)
            pct = (total_lines / max(1, total_expected)) * 100
            out_str = f"Suite Progress: {total_lines}/{total_expected} ({pct:.1f}%) -> Details in {progress_file.name}"

            sys.stdout.write("\r" + out_str.ljust(max_w))
            sys.stdout.flush()
            time.sleep(2)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(model_names)) as executor:
        for idx, model_name in enumerate(model_names, start=1):
            futures.append(executor.submit(_run_single_model, model_name, idx))

        monitor_thread = threading.Thread(target=_monitor_progress, daemon=True)
        monitor_thread.start()

        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                if "error" in res:
                    has_errors = True
                else:
                    summary_rows.append(res)
            except Exception as e:
                logger.error("Executor failed with future: %s", e)
                has_errors = True

        stop_monitor = True
        monitor_thread.join(timeout=3)
        sys.stdout.write("\n")

    if has_errors:
        logger.warning("One or more models encountered errors during execution.")

    suite_report = {
        "timestamp": suite_started_at.isoformat(),
        "n_samples": n_samples,
        "max_workers": max_workers,
        "agent_mode": agent_mode,
        "base_class": base_class,
        "host_model": host_model,
        "role_models_json": role_models_json,
        "max_host_iterations": max_host_iterations,
        "models": summary_rows,
    }

    report_path = suite_dir / f"suite_report_{suite_stamp}.json"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(suite_report, f, indent=2)
    except Exception as e:
        logger.error("Failed to record JSON suite report: %s", e)

    combined_md_path = suite_dir / f"run_suite_{suite_stamp}.md"
    try:
        with open(combined_md_path, "w", encoding="utf-8") as out:
            out.write("# Model Suite Report\n\n")
            out.write(f"**timestamp:** {suite_report['timestamp']}\n\n")
            out.write(
                f"**n_samples:** {n_samples} | **max_workers:** {max_workers} | **seed:** {seed} | "
                f"**agent_mode:** {agent_mode} | **base_class:** {base_class}\n\n"
            )
            for i, row in enumerate(summary_rows, start=1):
                row_model = row.get("model")
                run_dir = Path(str(row.get("run_dir")))
                result_path = run_dir / "result.md"

                out.write("---\n\n")
                out.write(f"## {i}. {row_model}\n\n")
                out.write(f"**run_dir:** {run_dir}\n\n")
                out.write(
                    f"**accuracy:** {row.get('accuracy')} | **f1_score:** {row.get('f1_score')} | "
                    f"**incorrect:** {row.get('incorrect')} | **execution_errors:** {row.get('execution_errors')}\n\n"
                )

                if not result_path.exists():
                    out.write(f"(missing {result_path})\n\n")
                    continue

                text = result_path.read_text(encoding="utf-8", errors="replace")
                text = _strip_per_sample_results(text)
                out.write(text)
                if not text.endswith("\n"):
                    out.write("\n")
    except Exception as e:
        logger.error("Failed to generate composite MD document: %s", e)

    logger.info(
        "Suite completed successfully" if not has_errors else "Suite completed with some failures"
    )
    logger.info("Report: %s", report_path)
    logger.info("Combined markdown: %s", combined_md_path)

    # -------------------------------------------------------------
    # Auto-trigger visuals
    # -------------------------------------------------------------
    if VISUALS_AVAILABLE:
        try:
            logger.info("Generating suite visuals...")
            generate_suite_visuals(report_path, model_list_path)
        except Exception as e:
            logger.error("Visualization generation failed: %s", e)
    else:
        logger.info(
            "Visualizations skipped (matplotlib not installed). "
            "Install with: pip install evalring[viz]"
        )

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EvalRing suite.")
    parser.add_argument(
        "--eval-script", type=str, required=True, help="Path to evaluation main script"
    )
    parser.add_argument("--models-file", type=str, required=True, help="Path to model list JSON")
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agent-mode", type=str, default="single-class")
    parser.add_argument("--base-class", type=str, default=None)
    parser.add_argument("--host-model", type=str, default=None)
    parser.add_argument("--role-models-json", type=str, default=None)
    parser.add_argument("--max-host-iterations", type=int, default=10)
    parser.add_argument("--continue-runs", action="store_true")
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument(
        "--cache-mode", type=str, default="both", choices=["runs_only", "cache_file", "both"]
    )
    parser.add_argument(
        "--ignore-errors", action="store_true", help="Ignore cached errors, forcing them to re-run."
    )
    parser.add_argument("--out-dir", type=str, default=None)

    args = parser.parse_args()

    code = run_suite(
        eval_script=args.eval_script,
        model_list_path=args.models_file,
        n_samples=args.n_samples,
        max_workers=args.max_workers,
        seed=args.seed,
        agent_mode=args.agent_mode,
        base_class=args.base_class,
        host_model=args.host_model,
        role_models_json=args.role_models_json,
        max_host_iterations=args.max_host_iterations,
        continue_runs=args.continue_runs,
        cache=args.cache,
        cache_mode=args.cache_mode,
        ignore_errors=args.ignore_errors,
        out_dir=args.out_dir,
    )
    sys.exit(code)
