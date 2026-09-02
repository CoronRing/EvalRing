"""
Concrete implementations of evaluators.
"""

import concurrent.futures
import datetime
import json
import os
import time
from pathlib import Path
from typing import Any

from ..agent import resolve_classification_prediction
from ..agent.base import AgentResponse, BaseAgent
from ..dataset.base import BaseDataset, DataSample
from ..logging_utils import get_logger
from .base import BaseEvaluator, EvaluationMetrics, EvaluationResult

logger = get_logger(__name__)


class ClassificationEvaluator(BaseEvaluator):
    """Evaluator for classification tasks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _process_sample(
        self,
        sample: DataSample,
        agent: BaseAgent,
        max_retries: int,
        error_cb=None,
    ) -> tuple[DataSample, AgentResponse | None, str | None]:
        """Process a single sample with retry logic."""
        last_error = None
        attempt = 0
        rate_limit_attempt = 0
        last_error_type = None
        while True:
            try:
                response = agent.predict(sample.input_text)
                if getattr(response, "error", None) or response.output == "Error":
                    raise RuntimeError(
                        getattr(response, "error", None) or "Unknown prediction error"
                    )

                resolved = resolve_classification_prediction(response.output)
                if resolved.label is not None:
                    response.output = resolved.label
                if resolved.confidence is not None and response.confidence is None:
                    response.confidence = resolved.confidence
                if resolved.class_scores is not None:
                    response.metadata = response.metadata or {}
                    response.metadata.setdefault("class_scores", resolved.class_scores)
                response.input_id = sample.id
                return sample, response, None
            except Exception as e:
                last_error = str(e)
                last_error_type = e.__class__.__name__
                error_lower = last_error.lower()
                is_rate_limit = "429" in error_lower or "rate limit" in error_lower

                if error_cb:
                    try:
                        error_cb(
                            {
                                "timestamp": datetime.datetime.now().isoformat(),
                                "sample_id": str(sample.id),
                                "error_type": last_error_type,
                                "error_message": last_error,
                                "is_rate_limit": is_rate_limit,
                                "attempt": attempt + 1,
                                "rate_limit_attempt": rate_limit_attempt + 1
                                if is_rate_limit
                                else 0,
                                "max_retries": max_retries,
                            }
                        )
                    except Exception:
                        pass

                if is_rate_limit:
                    if rate_limit_attempt < 30:
                        time.sleep(10)
                        rate_limit_attempt += 1
                        continue
                    else:
                        break

                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    attempt += 1
                else:
                    break
        return (
            sample,
            None,
            (
                f"Failed after {max_retries} attempts (or 30 rate limits). "
                f"Last error type: {last_error_type or 'UnknownError'}. "
                f"Last error: {last_error}"
            ),
        )

    def evaluate(
        self,
        agent: BaseAgent,
        dataset: BaseDataset,
        task_name: str,
        version: str = "1.0",
        max_workers: int = 5,
        max_retries: int = 3,
        show_progress: bool = False,
        seed: int = 42,
        exit_on_first_error: bool = False,
        partial_cb=None,
        error_cb=None,
        **kwargs,
    ) -> EvaluationResult:
        """
        Evaluate agent on classification task with parallel execution and retries.

        Calculates accuracy, precision, recall, and F1 score.
        """
        self.validate_inputs(agent, dataset)

        if not agent._is_initialized:
            agent.initialize()

        start_time = time.time()

        # Collect predictions (deterministic order: same as dataset iteration order)
        samples = list(dataset)
        requested_workers = max(1, int(max_workers))
        cpu_count = os.cpu_count() or 1
        planned_parallelism = min(requested_workers, len(samples)) if samples else 0

        logger.info(
            "Concurrency plan: samples=%d, requested_max_workers=%d, "
            "planned_parallelism=%d, local_cpu_count=%d",
            len(samples),
            requested_workers,
            planned_parallelism,
            cpu_count,
        )
        per_sample_by_id = {}
        execution_failures = []
        retry_backoff_failures = 0
        rate_limit_failures = 0
        rate_limit_sleep_events = 0

        # Parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=requested_workers) as executor:
            effective_executor_workers = getattr(executor, "_max_workers", requested_workers)
            logger.debug(
                "Executor: max_workers=%s, submitted_tasks=%d",
                effective_executor_workers,
                len(samples),
            )
            futures = {
                executor.submit(self._process_sample, sample, agent, max_retries, error_cb): sample
                for sample in samples
            }

            iterator = concurrent.futures.as_completed(futures)
            if show_progress:
                try:
                    import sys

                    from tqdm import tqdm

                    iterator = tqdm(
                        iterator, total=len(futures), file=sys.stdout, desc="Evaluating"
                    )
                except ImportError:
                    pass

            completed_count = 0
            for future in iterator:
                sample, response, error = future.result()
                completed_count += 1

                if error:
                    logger.warning("Sample failure: sample_id=%s, error=%s", sample.id, error)
                    err_lower = str(error).lower()
                    if (
                        "rate limit" in err_lower
                        or "30 rate limits" in err_lower
                        or "429" in err_lower
                    ):
                        rate_limit_failures += 1
                    else:
                        retry_backoff_failures += 1
                    if "30 rate limits" in err_lower:
                        rate_limit_sleep_events += 30
                    if completed_count == 1 and exit_on_first_error:
                        import sys

                        logger.critical(
                            "First request failed: %s. Exiting to avoid further failures.", error
                        )
                        for f in futures:
                            f.cancel()
                        sys.exit(1)

                    execution_failures.append(
                        {"sample_id": sample.id, "input_text": sample.input_text, "error": error}
                    )
                    err_metric = {
                        "sample_id": sample.id,
                        "input_text": sample.input_text,
                        "accuracy": 0.0,
                        "prediction": "Error",
                        "ground_truth": sample.target_output,
                        "error": error,
                    }
                    per_sample_by_id[str(sample.id)] = err_metric
                    if partial_cb:
                        partial_cb(err_metric)
                else:
                    assert response is not None

                    is_correct = response.output == sample.target_output
                    sample_metric = {
                        "sample_id": sample.id,
                        "input_text": sample.input_text,
                        "accuracy": 1.0 if is_correct else 0.0,
                        "prediction": response.output,
                        "ground_truth": sample.target_output,
                    }
                    if response.confidence is not None:
                        sample_metric["prediction_confidence"] = response.confidence
                    if response.metadata:
                        sample_metric.update(response.metadata)
                    per_sample_by_id[str(sample.id)] = sample_metric
                    if partial_cb:
                        partial_cb(sample_metric)

        if rate_limit_failures or retry_backoff_failures:
            logger.info(
                "Concurrency diagnostics: non_rate_limit_failures=%d, "
                "rate_limit_failures=%d, estimated_rate_limit_sleep_events=%d",
                retry_backoff_failures,
                rate_limit_failures,
                rate_limit_sleep_events,
            )

        # Emit ordered per-sample metrics in the same order as the dataset.
        per_sample_metrics = [per_sample_by_id[str(s.id)] for s in samples]
        predictions = [m["prediction"] for m in per_sample_metrics]
        ground_truth = [m["ground_truth"] for m in per_sample_metrics]

        # Calculate metrics
        metrics = self._calculate_classification_metrics(ground_truth, predictions)
        metrics.per_sample_metrics = per_sample_metrics

        # Calculate aggregate streaming metrics
        import statistics

        def safe_mean(values):
            return statistics.mean(values) if values else 0.0

        def safe_stdev(values):
            return statistics.stdev(values) if len(values) > 1 else 0.0

        ttfts = [m["ttft"] for m in per_sample_metrics if "ttft" in m]
        tps_list = [m["tps"] for m in per_sample_metrics if "tps" in m]
        gen_times = [m["generation_time"] for m in per_sample_metrics if "generation_time" in m]
        total_times = [m["total_time"] for m in per_sample_metrics if "total_time" in m]

        if ttfts:
            metrics.add_metric("avg_ttft", safe_mean(ttfts))
            metrics.add_metric("std_ttft", safe_stdev(ttfts))
        if tps_list:
            metrics.add_metric("avg_tps", safe_mean(tps_list))
            metrics.add_metric("std_tps", safe_stdev(tps_list))
        if gen_times:
            metrics.add_metric("avg_generation_time", safe_mean(gen_times))
        if total_times:
            metrics.add_metric("avg_total_time", safe_mean(total_times))

        # Token usage aggregation
        prompt_toks = [m["prompt_tokens"] for m in per_sample_metrics if "prompt_tokens" in m]
        completion_toks = [
            m["completion_tokens"] for m in per_sample_metrics if "completion_tokens" in m
        ]
        total_toks = [m["total_tokens"] for m in per_sample_metrics if "total_tokens" in m]

        if prompt_toks:
            metrics.add_metric("total_prompt_tokens", sum(prompt_toks))
            metrics.add_metric("total_completion_tokens", sum(completion_toks))
            metrics.add_metric("total_tokens", sum(total_toks))
            metrics.add_metric("avg_prompt_tokens", safe_mean(prompt_toks))
            metrics.add_metric("avg_completion_tokens", safe_mean(completion_toks))

        duration = time.time() - start_time

        # Record metadata for reproducibility
        eval_metadata = {
            "task_name": task_name,
            "version": version,
            "max_workers": max_workers,
            "max_retries": max_retries,
            "agent_info": agent.get_info(),
            "dataset_info": dataset.get_statistics(),
            "execution_failures": execution_failures,
            "kwargs": kwargs,
        }
        full_metadata = {**self._metadata, **eval_metadata}

        from datetime import datetime

        result = EvaluationResult(
            agent_name=agent.name,
            dataset_name=dataset.name,
            metrics=metrics,
            duration=duration,
            timestamp=datetime.now(),
            task_name=task_name,
            version=version,
            metadata=full_metadata,
        )

        return result

    def retry_failed_cases(
        self, meta_file_path: str | Path, agent: BaseAgent, dataset: BaseDataset, **kwargs
    ) -> EvaluationResult | None:
        """
        Read Meta.json, extract execution failures, and re-run evaluation on those specific cases.
        """
        meta_path = Path(meta_file_path)
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_path}")

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        failed_ids = [str(f["sample_id"]) for f in meta.get("execution_failures", [])]
        if not failed_ids:
            logger.info("No execution failures found in Meta.json to retry.")
            return None

        # Create a new dataset with only the failed cases
        retry_dataset = dataset.__class__(name=f"{dataset.name}_retry")
        for sample in dataset:
            if str(sample.id) in failed_ids:
                retry_dataset.add_sample(sample)

        logger.info("Retrying %d failed cases...", len(retry_dataset))

        return self.evaluate(
            agent=agent,
            dataset=retry_dataset,
            task_name=meta.get("task_name", "retry_task"),
            version=meta.get("version", "1.0"),
            max_workers=meta.get("max_workers", 1),
            max_retries=meta.get("max_retries", 3),
            **kwargs,
        )

    def _calculate_classification_metrics(
        self, y_true: list[Any], y_pred: list[Any]
    ) -> EvaluationMetrics:
        """Calculate classification metrics."""
        metrics = EvaluationMetrics()

        # Accuracy
        correct = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
        total = len(y_true)
        accuracy = correct / total if total > 0 else 0.0
        metrics.add_metric("accuracy", accuracy)

        # For multi-class, calculate macro-averaged precision, recall, f1
        classes = set(y_true) | set(y_pred)

        if len(classes) > 0:
            precisions = []
            recalls = []
            f1s = []

            for cls in classes:
                tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == cls and p == cls)
                fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != cls and p == cls)
                fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == cls and p != cls)

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)

            metrics.add_metric("precision", sum(precisions) / len(precisions))
            metrics.add_metric("recall", sum(recalls) / len(recalls))
            metrics.add_metric("f1_score", sum(f1s) / len(f1s))

        return metrics
