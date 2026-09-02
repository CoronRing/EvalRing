"""Standard single-model runner for Humanity's Last Exam (HLE).

Generates answers with one model, grades them with an LLM judge, and writes a
timestamped run directory (result.md / all_cases.csv / all_cases.txt /
all_cases_partial.jsonl / Meta.json) mirroring the suicide-detection example.

Provider routing is explicit so a suite can mix providers in one campaign:

    --litellm-model openai/gpt-5.5      --api-key-env OPENAI_API_KEY
    --litellm-model gemini/gemini-3.5-flash --api-key-env GEMINI_API_KEY
    --litellm-model openai/hal-1.0      --api-key-env RADIUM_API_KEY --api-base-env RADIUM_BASE_URL

Example (10-entry text-only smoke test, parallel, basic mode, medium thinking)::

    python examples/hle/evaluate_hle_main.py \\
        --n-samples 10 --max-workers 5 --agent-mode basic --reasoning-effort medium \\
        --litellm-model openai/hal-1.0 --api-key-env RADIUM_API_KEY --api-base-env RADIUM_BASE_URL
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

current_file = Path(__file__).resolve()
# Sibling example modules (hle_agent, hle_judge, hle_evaluator) are not an installed package.
sys.path.append(str(current_file.parent))

try:
    import pandas as pd
    from EvalRing.dataset import CSVDataset, DataFrameDataset
    from EvalRing.evaluator.base import EvaluationResult
    from hle_agent import HLEAgent
    from hle_judge import HLEJudge
    from hle_evaluator import HLEEvaluator
except ImportError as e:  # pragma: no cover
    print(f"Error importing HLE modules: {e}\nsys.path={sys.path}")
    sys.exit(1)


def _setup_general_logger(eval_dir: Path, mode: str = "w") -> logging.Logger:
    """Attach a file handler writing evaluator/run events to ``run_general.log``.

    Captures the whole ``EvalRing`` logger tree (run + evaluator) with timestamps
    so a run can be debugged/tracked independently of the captured stdout log.
    ``mode="a"`` appends (used by retry so the original run log is preserved).
    """
    log_path = eval_dir / "run_general.log"
    logger = logging.getLogger("EvalRing")
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers if run() is called more than once in a process.
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler) and getattr(h, "_evalring_general", False):
            logger.removeHandler(h)
    handler = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    handler._evalring_general = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    return logging.getLogger("EvalRing.hle.run")


def _env_value(env_name: Optional[str]) -> Optional[str]:
    if not env_name:
        return None
    val = os.environ.get(env_name)
    return val.strip() if isinstance(val, str) and val.strip() else None


def _build_agent(args) -> HLEAgent:
    api_key = _env_value(args.api_key_env)
    api_base = _env_value(args.api_base_env)
    if not api_key:
        print(
            f"WARNING: api-key-env '{args.api_key_env}' is empty; LiteLLM will rely on ambient env keys.",
        )
    agent = HLEAgent(
        name=f"hle-{args.model_label}",
        model_name=args.litellm_model,
        api_key=api_key,
        api_base=api_base,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        reasoning_effort=args.reasoning_effort,
        agent_mode=args.agent_mode,
    )
    agent.initialize()
    return agent


def _build_judge(args) -> HLEJudge:
    judge = HLEJudge(
        model_name=args.judge_litellm_model,
        api_key=_env_value(args.judge_api_key_env),
        api_base=_env_value(args.judge_api_base_env),
        request_timeout_s=args.request_timeout_s,
    )
    judge.initialize()
    return judge


def _load_dataset(data_path: Path, n_samples: int, text_only: bool) -> CSVDataset:
    dataset = CSVDataset(name="hle")
    dataset.load_data(source=data_path, text_field="question", label_field="answer", id_field="ID")
    print(f"Total HLE entries loaded: {len(dataset._samples)}")

    if text_only:
        before = len(dataset._samples)
        dataset._samples = [
            s for s in dataset._samples
            if str(s.metadata.get("has_image", 0)).strip() in {"0", "0.0", "False", "false", ""}
        ]
        print(f"Text-only filter: {before} -> {len(dataset._samples)} entries (dropped image questions).")

    if n_samples >= 0:
        dataset._samples = dataset._samples[:n_samples]
    dataset.assert_unique_ids(expected_count=len(dataset._samples), context="hle.after_sampling")
    print(f"Samples selected: {len(dataset._samples)}")
    return dataset


def _write_reports(*, eval_dir: Path, result, agent: HLEAgent, args, all_cases: List[Dict[str, Any]]):
    metrics = result.metrics.metrics
    per_cat = result.metrics.metadata.get("per_category_accuracy", {})
    incorrect = [c for c in all_cases if int(c.get("correct", 0)) < 1]

    # result.md
    with open(eval_dir / "result.md", "w", encoding="utf-8") as f:
        f.write("# Evaluation Report — Humanity's Last Exam (HLE)\n\n")
        f.write(f"**Timestamp:** {datetime.datetime.now().isoformat()}\n")
        f.write(f"**Agent:** {result.agent_name}\n")
        f.write(f"**Model:** {agent.model_name}\n")
        f.write(f"**Judge Model:** {args.judge_litellm_model}\n")
        f.write(f"**Agent Mode:** {args.agent_mode}\n")
        f.write(f"**Reasoning Effort:** {args.reasoning_effort}\n")
        f.write(f"**Task:** {result.task_name}\n")
        f.write(f"**Dataset:** hle (n={len(all_cases)})\n")
        f.write(f"**Duration:** {result.duration:.2f} s\n")
        f.write(f"**Workers:** {args.max_workers} | **Retries:** {args.max_retries} | **Seed:** {args.seed}\n\n")

        f.write("## Aggregate Metrics\n\n")
        for key in [
            "accuracy", "n_total", "n_graded", "n_correct", "n_errors", "calibration_error",
            "avg_ttft", "avg_tps", "avg_generation_time", "avg_total_time",
            "total_prompt_tokens", "total_completion_tokens", "total_tokens", "avg_completion_tokens",
            "total_reasoning_tokens", "avg_reasoning_tokens", "total_answer_tokens",
            "avg_answer_tokens", "reasoning_token_fraction",
        ]:
            if key in metrics:
                val = metrics[key]
                f.write(f"- **{key}**: {val:.6f}\n" if isinstance(val, float) else f"- **{key}**: {val}\n")
        f.write("\n")

        if per_cat:
            f.write("## Per-Category Accuracy\n\n")
            f.write("| Category | Accuracy | Correct | N |\n|---|---|---|---|\n")
            for cat, d in per_cat.items():
                f.write(f"| {cat} | {d['accuracy']:.4f} | {d['correct']} | {d['n']} |\n")
            f.write("\n")

        f.write("## Per-Sample Results\n\n")
        f.write("| Sample | Type | Category | Correct | Conf | Prediction | Gold |\n|---|---|---|---|---|---|---|\n")
        for c in all_cases:
            pred = str(c.get("prediction", ""))[:60].replace("|", "\\|").replace("\n", " ")
            gold = str(c.get("ground_truth", ""))[:60].replace("|", "\\|").replace("\n", " ")
            conf = c.get("model_confidence")
            conf_s = f"{conf:.0f}" if isinstance(conf, (int, float)) else "-"
            status = "PASS" if int(c.get("correct", 0)) else ("ERR" if c.get("error") else "FAIL")
            f.write(
                f"| {c['sample_id']} | {c.get('answer_type','')} | {c.get('category','')} | "
                f"{status} | {conf_s} | {pred} | {gold} |\n"
            )
        f.write("\n")

    # all_cases.txt
    with open(eval_dir / "all_cases.txt", "w", encoding="utf-8") as f:
        f.write(f"All Cases — Agent: {result.agent_name} | Model: {agent.model_name}\n")
        f.write("=" * 80 + "\n\n")
        for c in all_cases:
            status = "PASS" if int(c.get("correct", 0)) else ("ERROR" if c.get("error") else "FAIL")
            f.write(f"Sample ID : {c['sample_id']} [{status}]\n")
            f.write(f"Type      : {c.get('answer_type','')} | Category: {c.get('category','')}\n")
            if c.get("started_at"):
                f.write(
                    f"Timing    : started={c.get('started_at')} finished={c.get('finished_at')} "
                    f"wall={c.get('wall_time_s')}s attempts={c.get('attempts')}\n"
                )
            f.write(f"Gold      : {c.get('ground_truth','')}\n")
            f.write(f"Pred      : {c.get('prediction','')}\n")
            if c.get("judge_time_s") is not None:
                f.write(
                    f"Judge     : model={c.get('judge_model')} correct={c.get('judge_correct')} "
                    f"time={c.get('judge_time_s')}s tokens={c.get('judge_total_tokens')}\n"
                )
            if isinstance(c.get("model_confidence"), (int, float)):
                f.write(f"Confidence: {c['model_confidence']}\n")
            if c.get("total_time") is not None:
                f.write(
                    f"Latency   : ThinkTTFT={c.get('ttft_reasoning')}s | AnswerTTFT={c.get('ttft')}s | "
                    f"Gen={c.get('generation_time')}s | Total={c.get('total_time')}s | "
                    f"TPS={c.get('tps')} (tok/total-s)\n"
                )
            if c.get("total_tokens") is not None:
                est = " (est)" if c.get("reasoning_tokens_estimated") else ""
                f.write(
                    f"Tokens    : Prompt={c.get('prompt_tokens')} | "
                    f"Completion={c.get('completion_tokens')} "
                    f"(Reasoning={c.get('reasoning_tokens')}{est}, Answer={c.get('answer_tokens')}) | "
                    f"Total={c.get('total_tokens')}\n"
                )
            if c.get("finish_reason"):
                f.write(f"Finish    : {c.get('finish_reason')}\n")
            if c.get("error"):
                f.write(f"Error     : {c['error']}\n")
            f.write(f"Question:\n{c.get('input_text','')}\n")
            if c.get("reasoning_content"):
                f.write(f"Reasoning:\n{c['reasoning_content']}\n")
            if c.get("raw_output"):
                f.write(f"Response:\n{c['raw_output']}\n")
            f.write("-" * 40 + "\n\n")

    # all_cases.csv
    csv_fields = [
        "sample_id", "original_id", "started_at", "finished_at", "wall_time_s", "attempts",
        "model", "reasoning_effort", "judge_model",
        "ground_truth", "prediction", "extracted_answer", "correct", "answer_type", "category",
        "model_confidence", "judge_confidence", "judge_correct", "finish_reason",
        "ttft_reasoning", "ttft", "tps", "agent_time_s", "total_time", "generation_time",
        "prompt_tokens", "completion_tokens", "reasoning_tokens", "answer_tokens", "total_tokens",
        "judge_time_s", "judge_prompt_tokens", "judge_completion_tokens", "judge_total_tokens",
        "error", "question",
    ]
    with open(eval_dir / "all_cases.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for c in all_cases:
            row = {k: c.get(k, "") for k in csv_fields}
            row["question"] = c.get("input_text", "")
            writer.writerow(row)

    # incorrect_cases.txt
    if incorrect:
        with open(eval_dir / "incorrect_cases.txt", "w", encoding="utf-8") as f:
            for c in incorrect:
                f.write(f"Sample ID : {c['sample_id']}\n")
                f.write(f"Gold      : {c.get('ground_truth','')}\n")
                f.write(f"Pred      : {c.get('prediction','')}\n")
                if c.get("error"):
                    f.write(f"Error     : {c['error']}\n")
                f.write(f"Question:\n{c.get('input_text','')}\n")
                f.write("-" * 40 + "\n\n")

    # Meta.json
    meta_payload = {
        **(result.metadata if hasattr(result, "metadata") else {}),
        "run_config": {
            "n_samples": args.n_samples,
            "max_workers": args.max_workers,
            "max_retries": args.max_retries,
            "request_timeout_s": args.request_timeout_s,
            "seed": args.seed,
            "started_at": getattr(args, "run_started_at", None),
            "finished_at": datetime.datetime.now().isoformat(),
            "timestamp": datetime.datetime.now().isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "model_config": {
            "model_name": agent.model_name,
            "model_label": args.model_label,
            "agent_mode": args.agent_mode,
            "reasoning_effort": args.reasoning_effort,
            "temperature": agent.temperature,
            "max_completion_tokens": agent.max_completion_tokens,
            "api_base": agent.base_url,
            "judge_model": args.judge_litellm_model,
        },
        "dataset_config": {
            "source": str(args.data_path),
            "text_field": "question",
            "label_field": "answer",
            "id_field": "ID",
            "text_only": not args.include_images,
        },
        "aggregate_metrics": metrics,
        "per_category_accuracy": per_cat,
    }
    with open(eval_dir / "Meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=4, default=str)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model:     {agent.model_name}  (label: {args.model_label})")
    print(f"Judge:     {args.judge_litellm_model}")
    print(f"Samples:   {len(all_cases)}  |  Errors: {metrics.get('n_errors', 0)}")
    print(f"Accuracy:  {metrics.get('accuracy', 0):.4f}  |  CalErr: {metrics.get('calibration_error', 0):.4f}")
    print(f"Duration:  {result.duration:.2f}s")
    print(f"Output:    {eval_dir}")
    print("=" * 60)


def run(args) -> None:
    args.run_started_at = datetime.datetime.now().isoformat()
    # OpenAIAgent reads this in __init__; set it before building the agent so
    # long-reasoning (silent) requests are not killed by the default timeout.
    os.environ["OPENAI_REQUEST_TIMEOUT_S"] = str(args.request_timeout_s)

    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Dataset not found: {data_path}. Run ingest_hle.py first.")
        sys.exit(1)

    text_only = not args.include_images
    dataset = _load_dataset(data_path, args.n_samples, text_only)

    if args.info_only:
        # Minimal cache-scan compatibility hook for suite runners.
        print(json.dumps({"cached": 0, "uncached": len(dataset._samples)}))
        return

    if len(dataset._samples) == 0:
        print("No samples selected; nothing to evaluate.")
        return

    agent = _build_agent(args)
    judge = _build_judge(args)
    print(f"Agent model: {agent.model_name} (base={agent.base_url}) | Judge: {judge.model_name}")

    # Output directory.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"{args.model_label}_{args.agent_mode}"
    base_dir = Path(args.out_dir) if args.out_dir else (current_file.parent / "_EvalRing")
    eval_dir = base_dir / f"run_{timestamp}_{suffix}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    run_log = _setup_general_logger(eval_dir)
    run_log.info(
        "RUN START: model=%s (base=%s) judge=%s data=%s n_samples=%d workers=%d "
        "retries=%d timeout=%ss reasoning_effort=%s out=%s",
        agent.model_name, agent.base_url, judge.model_name, data_path.name,
        len(dataset._samples), args.max_workers, args.max_retries, args.request_timeout_s,
        args.reasoning_effort, eval_dir,
    )

    partial_path = eval_dir / "all_cases_partial.jsonl"
    write_lock = threading.Lock()
    f_out = open(partial_path, "a", encoding="utf-8")

    def write_partial(sample_metric: Dict[str, Any]) -> None:
        with write_lock:
            f_out.write(json.dumps(sample_metric, ensure_ascii=False, default=str) + "\n")
            f_out.flush()

    def log_error_event(event: Dict[str, Any]) -> None:
        run_log.warning(
            "retry sample=%s type=%s rate_limit=%s attempt=%s: %s",
            event.get("sample_id"), event.get("error_type"), event.get("is_rate_limit"),
            event.get("attempt"), str(event.get("error_message"))[:200],
        )

    evaluator = HLEEvaluator(judge=judge)
    print(f"\nStarting HLE evaluation — workers={args.max_workers}, retries={args.max_retries} ...")
    result = evaluator.evaluate(
        agent=agent,
        dataset=dataset,
        task_name="humanitys_last_exam",
        version="v1.0",
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        seed=args.seed,
        show_progress=True,
        exit_on_first_error=False,
        partial_cb=write_partial,
        error_cb=log_error_event,
    )
    f_out.close()

    all_cases = list(result.metrics.per_sample_metrics)
    print(f"\nSaving reports to: {eval_dir}")
    run_log.info("Saving reports to %s", eval_dir)
    _write_reports(eval_dir=eval_dir, result=result, agent=agent, args=args, all_cases=all_cases)
    run_log.info("RUN DONE: reports written to %s", eval_dir)


def run_retry_failed(args) -> None:
    """Re-run only the failed (error) cases of an existing run, in place.

    Reads ``<run-dir>/all_cases_partial.jsonl``, re-evaluates rows whose
    ``error`` is set (or ``prediction == "Error"``), merges the fresh results
    back over the originals, and regenerates all artifacts. Use a low
    ``--max-workers`` and/or ``--request-timeout-s 0`` (no limit) so previously
    timed-out reasoning has room to finish.
    """
    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir is None or not run_dir.exists():
        print(f"--retry-failed requires an existing --run-dir. Got: {run_dir}")
        sys.exit(1)
    partial = run_dir / "all_cases_partial.jsonl"
    if not partial.exists():
        print(f"No all_cases_partial.jsonl found in {run_dir}")
        sys.exit(1)

    args.run_started_at = datetime.datetime.now().isoformat()
    os.environ["OPENAI_REQUEST_TIMEOUT_S"] = str(args.request_timeout_s)

    all_cases = [json.loads(l) for l in open(partial, encoding="utf-8") if l.strip()]
    failed = [c for c in all_cases if c.get("error") or str(c.get("prediction", "")).strip() == "Error"]

    run_log = _setup_general_logger(run_dir, mode="a")
    run_log.info(
        "RETRY START: run_dir=%s failed=%d/%d timeout=%s workers=%d model=%s",
        run_dir, len(failed), len(all_cases), args.request_timeout_s, args.max_workers, args.litellm_model,
    )
    if not failed:
        print("No failed cases to retry — nothing to do.")
        run_log.info("RETRY: no failed cases.")
        return
    print(f"Retrying {len(failed)} failed case(s) from {run_dir.name} "
          f"(workers={args.max_workers}, timeout={'none' if args.request_timeout_s <= 0 else str(args.request_timeout_s)+'s'}) ...")

    df = pd.DataFrame([{
        "ID": c["sample_id"],
        "question": c.get("input_text", ""),
        "answer": c.get("ground_truth", ""),
        "answer_type": c.get("answer_type", ""),
        "category": c.get("category", ""),
        "original_id": c.get("original_id", ""),
        "has_image": 0,
    } for c in failed])
    dataset = DataFrameDataset(name="hle_retry")
    dataset.load_data(source=df, text_field="question", label_field="answer", id_field="ID")

    agent = _build_agent(args)
    judge = _build_judge(args)

    def log_error_event(event: Dict[str, Any]) -> None:
        run_log.warning(
            "retry sample=%s type=%s rate_limit=%s attempt=%s: %s",
            event.get("sample_id"), event.get("error_type"), event.get("is_rate_limit"),
            event.get("attempt"), str(event.get("error_message"))[:200],
        )

    evaluator = HLEEvaluator(judge=judge)
    result_subset = evaluator.evaluate(
        agent=agent, dataset=dataset, task_name="humanitys_last_exam", version="v1.0",
        max_workers=args.max_workers, max_retries=args.max_retries, seed=args.seed,
        show_progress=True, exit_on_first_error=False, error_cb=log_error_event,
    )

    # Merge fresh results over the originals (keyed by sample_id).
    new_by_id = {str(m["sample_id"]): m for m in result_subset.metrics.per_sample_metrics}
    resolved = 0
    for i, c in enumerate(all_cases):
        sid = str(c["sample_id"])
        if sid in new_by_id:
            all_cases[i] = new_by_id[sid]
            if not new_by_id[sid].get("error"):
                resolved += 1

    with open(partial, "w", encoding="utf-8") as f:
        for c in all_cases:
            f.write(json.dumps(c, ensure_ascii=False, default=str) + "\n")

    merged_metrics = HLEEvaluator._aggregate(all_cases)
    merged_metrics.per_sample_metrics = all_cases
    result = EvaluationResult(
        agent_name=agent.name, dataset_name="hle", metrics=merged_metrics,
        duration=result_subset.duration, timestamp=datetime.datetime.now(),
        task_name="humanitys_last_exam", version="v1.0", metadata={},
    )
    _write_reports(eval_dir=run_dir, result=result, agent=agent, args=args, all_cases=all_cases)
    still = len(failed) - resolved
    run_log.info("RETRY DONE: resolved=%d still_failed=%d", resolved, still)
    print(f"Retry complete: resolved {resolved}/{len(failed)}; {still} still failing. Updated {run_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a model on Humanity's Last Exam (HLE).")
    p.add_argument("--data-path", type=str, default=str(current_file.parent / "data" / "hle.csv"))
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--max-workers", type=int, default=5, help="Parallel worker threads.")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--agent-mode", type=str, default="basic", choices=["basic"])
    p.add_argument("--reasoning-effort", type=str, default="medium",
                   help="Thinking level forwarded to LiteLLM (e.g. low/medium/high; dropped if unsupported).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-completion-tokens", type=int, default=0,
                   help="Output budget. 0 (default) = NO cap: reasoning models finish naturally "
                        "instead of being truncated to an empty answer. Set a positive value to cap.")
    p.add_argument("--request-timeout-s", type=float, default=600.0,
                   help="Per-request timeout. Kept high because OpenAI reasoning models stay silent "
                        "during reasoning and would otherwise hit the default timeout.")
    p.add_argument("--include-images", action="store_true",
                   help="Include image questions (default: text-only).")
    # Candidate model routing.
    p.add_argument("--litellm-model", type=str, default="openai/hal-1.0")
    p.add_argument("--api-key-env", type=str, default="RADIUM_API_KEY")
    p.add_argument("--api-base-env", type=str, default="RADIUM_BASE_URL")
    p.add_argument("--model-label", type=str, default=None, help="Display/dir label (defaults to model tail).")
    # Judge routing. Defaults to OpenAI gpt-5.5 — routing every model's judge
    # calls through one Radium reasoning model saturates that gateway and stalls
    # the suite, whereas OpenAI handles the concurrency.
    p.add_argument("--judge-litellm-model", type=str, default="openai/gpt-5.5")
    p.add_argument("--judge-api-key-env", type=str, default="OPENAI_API_KEY")
    p.add_argument("--judge-api-base-env", type=str, default="")
    # Retry / IO.
    p.add_argument("--retry-failed", action="store_true",
                   help="Re-run only the failed (error) cases of an existing run in place.")
    p.add_argument("--run-dir", type=str, default=None,
                   help="Existing run directory to retry (required with --retry-failed).")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--info-only", action="store_true", help="Print sample count JSON and exit (suite hook).")
    p.add_argument("-nc", "--no-confirm", action="store_true", help="Accepted for suite compatibility.")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if not args.model_label:
        args.model_label = args.litellm_model.split("/")[-1]
    if args.retry_failed:
        run_retry_failed(args)
    else:
        run(args)
    # LiteLLM leaves non-daemon background threads alive, which otherwise makes
    # the interpreter hang at exit (and blocks any parent suite runner waiting on
    # this process). All artifacts are flushed above, so force a clean exit.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
