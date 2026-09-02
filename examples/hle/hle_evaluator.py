"""Evaluator for Humanity's Last Exam: generate answers, then LLM-judge them.

Unlike :class:`EvalRing.evaluator.ClassificationEvaluator` (exact-match over a
fixed label set), HLE requires a two-stage per-sample pipeline:

1. the candidate agent generates a free-form answer + self-reported confidence;
2. an :class:`hle_judge.HLEJudge` decides whether that answer matches the gold
   answer (semantic equivalence).

Parallelism, bounded retries with exponential backoff, rate-limit handling,
partial-result streaming and error callbacks follow the same contract as the
classification evaluator so the surrounding runner/report code stays uniform.

Aggregate metrics: accuracy, RMS calibration error (official HLE style),
per-category accuracy, and token / latency roll-ups.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import logging
import math
import os
import statistics
import time

logger = logging.getLogger("EvalRing.hle.evaluator")
from typing import Any, Callable, Dict, List, Optional, Tuple

from EvalRing.evaluator.base import BaseEvaluator, EvaluationMetrics, EvaluationResult
from EvalRing.agent import classify_error
from EvalRing.agent.base import BaseAgent, AgentResponse
from EvalRing.dataset.base import BaseDataset, DataSample
from EvalRing.utils.timeout import run_with_timeout

from hle_agent import HLEAgent
from hle_judge import HLEJudge


class HLEEvaluator(BaseEvaluator):
    """Generate-then-judge evaluator for HLE."""

    def __init__(self, judge: HLEJudge, max_transient_retries: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.judge = judge
        # Connection/5xx/overload wobble gets its own patient retry budget so a
        # loaded gateway (e.g. Radium under many concurrent streams) can recover.
        self.max_transient_retries = max_transient_retries

    # ── per-sample pipeline ────────────────────────────────
    def _process_sample(
        self,
        sample: DataSample,
        agent: BaseAgent,
        max_retries: int,
        error_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[DataSample, Optional[Dict[str, Any]], Optional[str]]:
        last_error = None
        last_error_type = None
        attempt = 0
        rate_limit_attempt = 0
        transient_attempt = 0
        attempts_total = 0
        started_at = datetime.datetime.now().isoformat()
        started_wall = time.time()
        # Partial output (thinking/answer) from the most recent attempt, kept so a
        # timed-out / errored sample still saves whatever was generated.
        partial_meta: Dict[str, Any] = {}

        # Hard client-side deadlines so a stuck stream/connection can never block
        # a worker forever (the model SDK does not reliably enforce its own
        # timeout). A small margin lets the SDK's own timeout fire first when it
        # works; the hard guard is the guarantee when it does not. A non-positive
        # request timeout means "no limit" — the hard guard is disabled too.
        _agent_to = getattr(agent, "request_timeout_s", 120)
        _judge_to = getattr(self.judge, "request_timeout_s", 120)
        agent_deadline = (_agent_to + 30) if (_agent_to and _agent_to > 0) else 0
        judge_deadline = (_judge_to + 30) if (_judge_to and _judge_to > 0) else 0

        while True:
            attempts_total += 1
            try:
                response: AgentResponse = run_with_timeout(
                    agent.predict, agent_deadline, sample.input_text
                )
                # Keep whatever partial output this attempt produced (used if a
                # later step — or this one — ends up failing).
                partial_meta = response.metadata or {}
                if getattr(response, "error", None) or response.output == "Error":
                    raise RuntimeError(getattr(response, "error", None) or "Unknown prediction error")

                raw_output = response.output if isinstance(response.output, str) else str(response.output)
                meta = response.metadata or {}
                extracted_answer = meta.get("extracted_answer")
                model_confidence = meta.get("model_confidence")
                if extracted_answer is None:
                    extracted_answer, model_confidence = HLEAgent.extract_answer_confidence(raw_output)

                gold = "" if sample.target_output is None else str(sample.target_output)
                verdict = run_with_timeout(
                    self.judge.grade, judge_deadline, sample.input_text, gold, raw_output
                )
                if verdict.error:
                    raise RuntimeError(f"Judge failed: {verdict.error}")

                finished_wall = time.time()
                metric: Dict[str, Any] = {
                    "sample_id": sample.id,
                    "original_id": sample.metadata.get("original_id", ""),
                    "input_text": sample.input_text,
                    "prediction": extracted_answer or (verdict.extracted_answer or ""),
                    "extracted_answer": extracted_answer,
                    "ground_truth": gold,
                    "accuracy": 1.0 if verdict.correct else 0.0,
                    "correct": 1 if verdict.correct else 0,
                    "model_confidence": model_confidence,
                    # Lifecycle / provenance.
                    "started_at": started_at,
                    "finished_at": datetime.datetime.now().isoformat(),
                    "wall_time_s": finished_wall - started_wall,
                    "attempts": attempts_total,
                    "answer_type": sample.metadata.get("answer_type", ""),
                    "category": sample.metadata.get("category", ""),
                    "raw_output": raw_output,
                    "error": None,
                    # Candidate model provenance.
                    "agent_name": getattr(agent, "name", None),
                    "agent_mode": getattr(agent, "agent_mode", None),
                    "reasoning_effort": getattr(agent, "reasoning_effort", None),
                    "temperature": getattr(agent, "temperature", None),
                    "base_url": meta.get("base_url"),
                    "agent_time_s": meta.get("total_time"),
                    # Judge provenance / telemetry.
                    "judge_model": self.judge.model_name,
                    "judge_confidence": verdict.confidence,
                    "judge_correct": verdict.correct,
                    "judge_extracted_answer": verdict.extracted_answer,
                    "judge_raw": verdict.raw,
                    "judge_time_s": verdict.judge_time,
                    "judge_prompt_tokens": verdict.prompt_tokens,
                    "judge_completion_tokens": verdict.completion_tokens,
                    "judge_total_tokens": verdict.total_tokens,
                }
                # Carry telemetry (ttft/tps/tokens/…) emitted by the agent.
                for key in (
                    "ttft", "ttft_reasoning", "tps", "total_time", "generation_time",
                    "prompt_tokens", "completion_tokens", "total_tokens", "model",
                    "finish_reason", "reasoning_chars", "reasoning_content",
                    "reasoning_tokens", "reasoning_tokens_estimated", "answer_tokens", "timed_out",
                ):
                    if key in meta:
                        metric[key] = meta[key]
                return sample, metric, None

            except Exception as e:  # noqa: BLE001
                last_error = str(e)
                last_error_type = e.__class__.__name__
                # Shared retry policy: rate-limit vs transient (retry) vs
                # terminal (don't retry — a retry would only waste tokens).
                cls = classify_error(last_error)
                is_rate_limit = cls.is_rate_limit
                is_transient = cls.is_transient
                is_terminal = cls.is_terminal
                if error_cb:
                    try:
                        error_cb({
                            "timestamp": datetime.datetime.now().isoformat(),
                            "sample_id": str(sample.id),
                            "error_type": last_error_type,
                            "error_message": last_error,
                            "is_rate_limit": is_rate_limit,
                            "attempt": attempt + 1,
                            "rate_limit_attempt": rate_limit_attempt + 1 if is_rate_limit else 0,
                            "max_retries": max_retries,
                        })
                    except Exception:
                        pass

                if is_terminal:
                    # Not worth retrying — surface the complete error as-is.
                    return sample, self._error_metric(
                        sample, agent, last_error, started_at, started_wall, attempts_total, partial_meta
                    ), last_error
                if is_rate_limit:
                    if rate_limit_attempt < 30:
                        time.sleep(10)
                        rate_limit_attempt += 1
                        continue
                    break
                if is_transient:
                    # Longer, capped backoff and its own generous attempt budget.
                    if transient_attempt < self.max_transient_retries:
                        time.sleep(min(5 * (transient_attempt + 1), 30))
                        transient_attempt += 1
                        continue
                    break
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    attempt += 1
                else:
                    break

        err = (
            f"Failed after {max_retries} attempts / {self.max_transient_retries} transient retries "
            f"/ 30 rate limits. Last error type: {last_error_type or 'UnknownError'}. "
            f"Last error: {last_error}"
        )
        return sample, self._error_metric(
            sample, agent, err, started_at, started_wall, attempts_total, partial_meta
        ), err

    def _error_metric(self, sample, agent, error, started_at, started_wall, attempts,
                       partial_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Comprehensive record for a failed sample (same lifecycle fields as success).

        Preserves any *partial* thinking/answer the agent streamed before the
        failure (``partial_meta``) so a timeout/disconnect still saves what it got.
        """
        partial_meta = partial_meta or {}
        metric = {
            "sample_id": sample.id,
            "original_id": sample.metadata.get("original_id", ""),
            "input_text": sample.input_text,
            "prediction": "Error",
            "extracted_answer": partial_meta.get("extracted_answer"),
            "ground_truth": "" if sample.target_output is None else str(sample.target_output),
            "accuracy": 0.0,
            "correct": 0,
            "model_confidence": partial_meta.get("model_confidence"),
            "started_at": started_at,
            "finished_at": datetime.datetime.now().isoformat(),
            "wall_time_s": time.time() - started_wall,
            "attempts": attempts,
            "answer_type": sample.metadata.get("answer_type", ""),
            "category": sample.metadata.get("category", ""),
            "raw_output": partial_meta.get("raw_output", ""),
            "error": error,
            "agent_name": getattr(agent, "name", None),
            "agent_mode": getattr(agent, "agent_mode", None),
            "reasoning_effort": getattr(agent, "reasoning_effort", None),
            "temperature": getattr(agent, "temperature", None),
            "model": partial_meta.get("model") or getattr(agent, "model_name", None),
            "base_url": partial_meta.get("base_url") or getattr(agent, "base_url", None),
            "judge_model": self.judge.model_name,
        }
        # Carry any partial telemetry / thinking captured before the failure.
        for key in (
            "ttft", "ttft_reasoning", "tps", "total_time", "generation_time",
            "prompt_tokens", "completion_tokens", "total_tokens",
            "finish_reason", "reasoning_chars", "reasoning_content",
            "reasoning_tokens", "reasoning_tokens_estimated", "answer_tokens", "timed_out",
        ):
            if key in partial_meta:
                metric[key] = partial_meta[key]
        return metric

    # ── orchestration ──────────────────────────────────────
    def evaluate(
        self,
        agent: BaseAgent,
        dataset: BaseDataset,
        task_name: str,
        version: str = "1.0",
        max_workers: int = 5,
        max_retries: int = 3,
        seed: int = 42,
        show_progress: bool = False,
        exit_on_first_error: bool = False,
        partial_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
        error_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs,
    ) -> EvaluationResult:
        self.validate_inputs(agent, dataset)
        if not agent._is_initialized:
            agent.initialize()

        start_time = time.time()
        samples = list(dataset)
        requested_workers = max(1, int(max_workers))
        planned = min(requested_workers, len(samples)) if samples else 0
        print(
            f"[HLE Concurrency] samples={len(samples)}, max_workers={requested_workers}, "
            f"planned_parallelism={planned}, cpu_count={os.cpu_count() or 1}"
        )
        logger.info(
            "Evaluation START: task=%s agent=%s judge=%s samples=%d workers=%d retries=%d",
            task_name, getattr(agent, "model_name", "?"), self.judge.model_name,
            len(samples), requested_workers, max_retries,
        )

        per_sample_by_id: Dict[str, Dict[str, Any]] = {}
        execution_failures: List[Dict[str, Any]] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=requested_workers) as executor:
            futures = {
                executor.submit(self._process_sample, s, agent, max_retries, error_cb): s
                for s in samples
            }
            iterator = concurrent.futures.as_completed(futures)
            if show_progress:
                try:
                    import sys
                    from tqdm import tqdm
                    iterator = tqdm(iterator, total=len(futures), file=sys.stdout, desc="HLE eval")
                except ImportError:
                    pass

            completed = 0
            for future in iterator:
                sample, metric, error = future.result()
                completed += 1
                if error:
                    print(f"[Sample Failure] sample_id={sample.id}, error={error}")
                    logger.warning("[%d/%d] sample=%s ERROR: %s", completed, len(samples), sample.id, error)
                    if completed == 1 and exit_on_first_error:
                        import sys
                        print(f"\n[Fatal] First request failed: {error}. Exiting.", file=sys.stderr)
                        for f in futures:
                            f.cancel()
                        sys.exit(1)
                    execution_failures.append({"sample_id": sample.id, "error": error})
                    # _process_sample returns a comprehensive error metric (same
                    # lifecycle fields as a success row); fall back defensively.
                    err_metric = metric or {
                        "sample_id": sample.id,
                        "input_text": sample.input_text,
                        "accuracy": 0.0,
                        "correct": 0,
                        "prediction": "Error",
                        "ground_truth": "" if sample.target_output is None else str(sample.target_output),
                        "answer_type": sample.metadata.get("answer_type", ""),
                        "category": sample.metadata.get("category", ""),
                        "error": error,
                    }
                    per_sample_by_id[str(sample.id)] = err_metric
                    if partial_cb:
                        partial_cb(err_metric)
                else:
                    assert metric is not None
                    per_sample_by_id[str(sample.id)] = metric
                    logger.info(
                        "[%d/%d] sample=%s correct=%s conf=%s total_time=%.1fs reasoning_tokens=%s",
                        completed, len(samples), sample.id, metric.get("correct"),
                        metric.get("model_confidence"),
                        float(metric.get("total_time") or 0.0), metric.get("reasoning_tokens"),
                    )
                    if partial_cb:
                        partial_cb(metric)

        per_sample_metrics = [per_sample_by_id[str(s.id)] for s in samples]
        metrics = self._aggregate(per_sample_metrics)
        metrics.per_sample_metrics = per_sample_metrics

        duration = time.time() - start_time
        logger.info(
            "Evaluation COMPLETE: accuracy=%.4f graded=%d correct=%d errors=%d "
            "avg_reasoning_tokens=%.1f duration=%.1fs",
            metrics.metrics.get("accuracy", 0.0), int(metrics.metrics.get("n_graded", 0)),
            int(metrics.metrics.get("n_correct", 0)), int(metrics.metrics.get("n_errors", 0)),
            metrics.metrics.get("avg_reasoning_tokens", 0.0), duration,
        )
        eval_metadata = {
            "task_name": task_name,
            "version": version,
            "max_workers": max_workers,
            "max_retries": max_retries,
            "agent_info": agent.get_info(),
            "dataset_info": dataset.get_statistics(),
            "execution_failures": execution_failures,
            "judge_model": self.judge.model_name,
            "kwargs": kwargs,
        }
        return EvaluationResult(
            agent_name=agent.name,
            dataset_name=dataset.name,
            metrics=metrics,
            duration=duration,
            timestamp=datetime.datetime.now(),
            task_name=task_name,
            version=version,
            metadata={**self._metadata, **eval_metadata},
        )

    # ── metric aggregation ─────────────────────────────────
    @staticmethod
    def _aggregate(per_sample: List[Dict[str, Any]]) -> EvaluationMetrics:
        metrics = EvaluationMetrics()
        graded = [m for m in per_sample if not m.get("error")]
        total = len(per_sample)
        n_graded = len(graded)
        correct = sum(1 for m in graded if int(m.get("correct", 0)) == 1)
        accuracy = correct / n_graded if n_graded else 0.0

        metrics.add_metric("accuracy", accuracy)
        metrics.add_metric("n_total", total)
        metrics.add_metric("n_graded", n_graded)
        metrics.add_metric("n_correct", correct)
        metrics.add_metric("n_errors", total - n_graded)
        metrics.add_metric("calibration_error", _rms_calibration_error(graded))

        # Per-category accuracy.
        cat_totals: Dict[str, int] = {}
        cat_correct: Dict[str, int] = {}
        for m in graded:
            cat = str(m.get("category", "") or "unknown")
            cat_totals[cat] = cat_totals.get(cat, 0) + 1
            cat_correct[cat] = cat_correct.get(cat, 0) + int(m.get("correct", 0))
        metrics.metadata["per_category_accuracy"] = {
            cat: {
                "accuracy": cat_correct[cat] / cat_totals[cat] if cat_totals[cat] else 0.0,
                "n": cat_totals[cat],
                "correct": cat_correct[cat],
            }
            for cat in sorted(cat_totals)
        }

        def _mean(key: str) -> float:
            vals = [m[key] for m in graded if isinstance(m.get(key), (int, float))]
            return statistics.mean(vals) if vals else 0.0

        def _sum(key: str) -> float:
            vals = [m[key] for m in graded if isinstance(m.get(key), (int, float))]
            return sum(vals) if vals else 0

        metrics.add_metric("avg_ttft", _mean("ttft"))
        metrics.add_metric("avg_tps", _mean("tps"))
        metrics.add_metric("avg_generation_time", _mean("generation_time"))
        metrics.add_metric("avg_total_time", _mean("total_time"))
        metrics.add_metric("total_prompt_tokens", _sum("prompt_tokens"))
        metrics.add_metric("total_completion_tokens", _sum("completion_tokens"))
        metrics.add_metric("total_tokens", _sum("total_tokens"))
        metrics.add_metric("avg_completion_tokens", _mean("completion_tokens"))
        # Reasoning ("thinking") token accounting.
        metrics.add_metric("total_reasoning_tokens", _sum("reasoning_tokens"))
        metrics.add_metric("avg_reasoning_tokens", _mean("reasoning_tokens"))
        metrics.add_metric("total_answer_tokens", _sum("answer_tokens"))
        metrics.add_metric("avg_answer_tokens", _mean("answer_tokens"))
        # Share of generated tokens spent on hidden reasoning.
        tct = _sum("completion_tokens")
        rt = _sum("reasoning_tokens")
        metrics.add_metric("reasoning_token_fraction", (rt / tct) if tct else 0.0)
        return metrics


def _rms_calibration_error(graded: List[Dict[str, Any]], n_bins: int = 10) -> float:
    """RMS calibration error over confidence bins (HLE-style).

    Uses the model's self-reported confidence (0-100). Samples without a
    confidence are ignored. Returns 0.0 when no confidences are available.
    """
    pts = [
        (float(m["model_confidence"]) / 100.0, int(m.get("correct", 0)))
        for m in graded
        if isinstance(m.get("model_confidence"), (int, float))
    ]
    if not pts:
        return 0.0

    bins: List[List[Tuple[float, int]]] = [[] for _ in range(n_bins)]
    for conf, corr in pts:
        idx = min(n_bins - 1, max(0, int(conf * n_bins)))
        bins[idx].append((conf, corr))

    total = len(pts)
    acc = 0.0
    for b in bins:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        avg_acc = sum(k for _, k in b) / len(b)
        acc += (len(b) / total) * (avg_conf - avg_acc) ** 2
    return math.sqrt(acc)
