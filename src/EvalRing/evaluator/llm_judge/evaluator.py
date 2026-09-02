"""
LLMJudgeEvaluator: Orchestrates multi-metric LLM-as-a-Judge evaluation.

This evaluator takes agent responses, packages them as test cases, and runs
them through one or more JudgeMetrics. It supports parallel execution,
retry logic, and comprehensive reporting — following the same contract as
BaseEvaluator.
"""

import concurrent.futures
import json
import logging
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ...agent.base import AgentResponse, BaseAgent
from ...dataset.base import BaseDataset, DataSample
from ..base import BaseEvaluator, EvaluationMetrics, EvaluationResult
from .judge import OpenAIJudge
from .metric import JudgeMetric
from .rubric import Rubric, ScoringCriteria
from .schema import JudgeVerdict

logger = logging.getLogger(__name__)


class LLMJudgeEvaluator(BaseEvaluator):
    """
    Evaluator that uses LLM-as-a-Judge to score agent responses.

    This is generic: it works for classification, generation, summarization,
    or any task. The evaluation criteria are defined by JudgeMetric instances,
    each with their own criteria/rubric/steps.

    Workflow:
    1. Run the agent on each dataset sample to get predictions
    2. Package each (input, prediction, ground_truth) as a test case
    3. Run each test case through all JudgeMetrics
    4. Aggregate scores across samples and metrics

    Args:
        metrics: List of JudgeMetric instances defining what to evaluate.
        agent_parameters: Which fields the agent provides for judging.
                          Default: ["input", "output", "ground_truth"].
        max_workers: Parallelism for agent prediction + judging.
        max_retries: Retry count for transient failures.
        **kwargs: Passed to BaseEvaluator.
    """

    def __init__(
        self,
        metrics: list[JudgeMetric],
        agent_parameters: list[str] | None = None,
        max_workers: int = 5,
        max_retries: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not metrics:
            raise ValueError("At least one JudgeMetric is required")

        self.metrics = metrics
        self.agent_parameters = agent_parameters or ["input", "output", "ground_truth"]
        self.max_workers = max_workers
        self.max_retries = max_retries

    # ── factory methods ──────────────────────────────────────

    @classmethod
    def from_rubric(
        cls,
        rubric: str | Rubric | dict[str, str | Rubric],
        criteria: str | dict[str, str] | None = None,
        judge_model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        judge_temperature: float = 0.0,
        judge_max_tokens: int = 512,
        weights: dict[str, float] | None = None,
        threshold: float = 0.5,
        strict_mode: bool = False,
        max_workers: int = 5,
        max_retries: int = 3,
        **kwargs,
    ) -> "LLMJudgeEvaluator":
        """
        Create an LLMJudgeEvaluator in one call — just pass a rubric.

        This factory handles creating the judge, criteria, and metrics
        internally so you don't need to assemble them yourself.

        Examples::

            # Single metric with a string rubric
            evaluator = LLMJudgeEvaluator.from_rubric(
                rubric="Score 0 = wrong, 10 = perfect match.",
                criteria="Is the classification correct?",
                judge_model="gpt-5.2",
            )

            # Single metric with a Rubric object
            evaluator = LLMJudgeEvaluator.from_rubric(
                rubric=my_rubric,
                criteria="Is the classification correct?",
                judge_model="gpt-5.2",
            )

            # Multiple metrics — pass dicts keyed by metric name
            evaluator = LLMJudgeEvaluator.from_rubric(
                rubric={
                    "correctness": my_correctness_rubric,
                    "reasoning": "Score 0-10 for reasoning quality.",
                },
                criteria={
                    "correctness": "Is the answer correct?",
                    "reasoning": "Does the answer show good reasoning?",
                },
                weights={"correctness": 2.0, "reasoning": 1.0},
                judge_model="gpt-5.2",
            )

        Args:
            rubric: A string, Rubric, or dict mapping metric names to rubrics.
            criteria: Criteria description(s) matching rubric keys.
                      If None, defaults to rubric text or "Evaluate quality".
            judge_model: Judge model identifier. Defaults to
                      ``$EVALRING_MODEL``, then ``"gpt-4o"``.
            api_key: API key. Resolved from the environment when omitted; see
                      :func:`EvalRing.config.resolve_credentials`.
            base_url: OpenAI-compatible endpoint for the judge. Resolved
                      alongside the API key when omitted.
            judge_temperature: Temperature for the judge model.
            judge_max_tokens: Max completion tokens for the judge.
            weights: Dict of metric-name → weight.  Default 1.0 each.
            threshold: Score threshold for pass/fail.  Default 0.5.
            strict_mode: If True, scoring is binary (0 or 1).
            max_workers: Parallelism for evaluation.
            max_retries: Retry count for transient failures.
            **kwargs: Forwarded to the LLMJudgeEvaluator constructor.

        Returns:
            A fully configured LLMJudgeEvaluator ready to call ``evaluate()``.
        """
        # Normalize rubric(s) into a dict: {name: rubric_value}
        if isinstance(rubric, dict):
            rubric_map = rubric
        else:
            # Single rubric → single metric named "quality"
            rubric_map = {"quality": rubric}

        # Normalize criteria into a dict
        if criteria is None:
            criteria_map = {}
        elif isinstance(criteria, str):
            # Single criteria string → match the single rubric key
            first_key = next(iter(rubric_map))
            criteria_map = {first_key: criteria}
        else:
            criteria_map = criteria

        weights = weights or {}

        # Create the shared judge backend
        judge = OpenAIJudge(
            model_name=judge_model,
            api_key=api_key,
            base_url=base_url,
            temperature=judge_temperature,
            max_completion_tokens=judge_max_tokens,
        )

        # Build one JudgeMetric per rubric entry
        metrics = []
        for name, rubric_val in rubric_map.items():
            sc = ScoringCriteria(
                name=name,
                criteria=criteria_map.get(name, "Evaluate the quality of the output."),
                rubric=rubric_val,
                weight=weights.get(name, 1.0),
            )
            metric = JudgeMetric(
                criteria=sc,
                judge=judge,
                parameters=["input", "output", "ground_truth"],
                strict_mode=strict_mode,
                threshold=threshold,
            )
            metrics.append(metric)

        return cls(
            metrics=metrics,
            max_workers=max_workers,
            max_retries=max_retries,
            **kwargs,
        )

    def _run_agent_on_sample(
        self, sample: DataSample, agent: BaseAgent
    ) -> tuple[DataSample, AgentResponse | None, str | None]:
        """Run the agent on a single sample with retry logic."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = agent.predict(sample.input_text)
                response.input_id = sample.id
                return sample, response, None
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
        return sample, None, f"Agent failed after {self.max_retries} attempts: {last_error}"

    def _judge_sample(
        self,
        test_case: dict[str, str],
        sample_id: str,
    ) -> tuple[str, dict[str, JudgeVerdict], str | None]:
        """
        Run all judge metrics on a single test case with retries.

        Returns:
            (sample_id, {metric_name: verdict}, error_or_none)
        """
        verdicts = {}
        for metric in self.metrics:
            last_error = None
            for attempt in range(self.max_retries):
                try:
                    verdict = metric.score(test_case)
                    verdicts[metric.name] = verdict
                    last_error = None
                    break
                except Exception as e:
                    last_error = str(e)
                    if attempt < self.max_retries - 1:
                        time.sleep(2**attempt)

            if last_error:
                logger.error(
                    f"Judge metric '{metric.name}' failed for sample {sample_id}: {last_error}"
                )
                verdicts[metric.name] = JudgeVerdict(
                    score=0, reason=f"Judging error: {last_error}", metadata={"error": last_error}
                )

        return sample_id, verdicts, None

    def _build_test_case(
        self,
        sample: DataSample,
        response: AgentResponse,
    ) -> dict[str, str]:
        """
        Build a test case dict from a sample + agent response.

        Maps standard fields:
          - "input" → sample.input_text
          - "output" → response.output
          - "ground_truth" → sample.target_output

        Also includes agent metadata for rich judging context.
        """
        test_case = {
            "input": sample.input_text or "",
            "output": str(response.output) if response.output else "",
            "ground_truth": str(sample.target_output) if sample.target_output else "",
        }

        # Include any extra metadata from the sample
        if sample.metadata:
            for key, value in sample.metadata.items():
                if key not in test_case:
                    test_case[key] = str(value)

        # Include agent response metadata (e.g., raw_output, confidence)
        if response.metadata:
            for key, value in response.metadata.items():
                prefixed_key = f"agent_{key}"
                if prefixed_key not in test_case:
                    test_case[prefixed_key] = str(value)

        return test_case

    def evaluate(
        self,
        agent: BaseAgent,
        dataset: BaseDataset,
        task_name: str,
        version: str = "1.0",
        max_workers: int | None = None,
        max_retries: int | None = None,
        **kwargs,
    ) -> EvaluationResult:
        """
        Run the full LLM-as-a-Judge evaluation pipeline.

        Phase 0: Initialize agent, pre-generate evaluation steps for each metric
        Phase 1: Run agent on all samples (parallel)
        Phase 2: Judge all responses with all metrics (parallel)
        Phase 3: Aggregate scores and build result
        """
        self.validate_inputs(agent, dataset)
        workers = max_workers or self.max_workers
        retries = max_retries or self.max_retries

        if not agent._is_initialized:
            agent.initialize()

        start_time = time.time()

        # Phase 0: Pre-generate evaluation steps for all metrics (sequential, cached)
        logger.info("Phase 0: Generating evaluation steps for all metrics...")
        for metric in self.metrics:
            _ = metric.evaluation_steps  # triggers generation if needed
            logger.info(f"  [{metric.name}] Steps: {metric.evaluation_steps}")

        # Phase 1: Run agent on all samples
        logger.info(f"Phase 1: Running agent on {len(dataset)} samples (workers={workers})...")
        agent_results: list[tuple[DataSample, AgentResponse | None, str | None]] = []
        execution_failures = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._run_agent_on_sample, sample, agent): sample
                for sample in dataset
            }
            for future in concurrent.futures.as_completed(futures):
                sample, response, error = future.result()
                if error:
                    execution_failures.append(
                        {"sample_id": sample.id, "phase": "agent", "error": error}
                    )
                agent_results.append((sample, response, error))

        logger.info(
            f"  Agent completed: {len(agent_results)} total, {len(execution_failures)} failures"
        )

        # Phase 2: Judge all successful responses
        logger.info(f"Phase 2: Judging responses with {len(self.metrics)} metrics...")

        # Build test cases for successful predictions only
        judgeable = []
        for sample, response, error in agent_results:
            if response and not error:
                test_case = self._build_test_case(sample, response)
                judgeable.append((sample, response, test_case))

        # Run judge metrics in parallel
        all_verdicts: list[tuple[DataSample, AgentResponse, dict[str, JudgeVerdict]]] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            judge_futures: dict[
                concurrent.futures.Future[tuple[str, dict[str, JudgeVerdict], str | None]],
                tuple[DataSample, AgentResponse],
            ] = {}
            for sample, response, test_case in judgeable:
                judge_future = executor.submit(self._judge_sample, test_case, sample.id)
                judge_futures[judge_future] = (sample, response)

            for judge_future in concurrent.futures.as_completed(judge_futures):
                judged_sample, judged_response = judge_futures[judge_future]
                sample_id, verdicts, judge_error = judge_future.result()
                if judge_error:
                    execution_failures.append(
                        {"sample_id": sample_id, "phase": "judge", "error": judge_error}
                    )
                all_verdicts.append((judged_sample, judged_response, verdicts))

        logger.info(f"  Judging completed: {len(all_verdicts)} samples judged")

        # Phase 3: Aggregate scores
        duration = time.time() - start_time
        eval_metrics = self._aggregate(all_verdicts, execution_failures, duration)

        result = EvaluationResult(
            agent_name=agent.name,
            dataset_name=dataset.name,
            metrics=eval_metrics,
            duration=duration,
            timestamp=datetime.now(),
            task_name=task_name,
            version=version,
            metadata={
                "max_workers": workers,
                "max_retries": retries,
                "num_metrics": len(self.metrics),
                "metric_names": [m.name for m in self.metrics],
                "metric_configs": [m.get_info() for m in self.metrics],
                "execution_failures": execution_failures,
                "total_samples": len(dataset),
                "judged_samples": len(all_verdicts),
            },
        )

        return result

    def _aggregate(
        self,
        all_verdicts: list[tuple[DataSample, AgentResponse, dict[str, JudgeVerdict]]],
        execution_failures: list[dict],
        duration: float,
    ) -> EvaluationMetrics:
        """
        Aggregate per-sample verdicts into summary metrics.

        Produces:
        - Per-metric: avg score, std, pass rate
        - Weighted composite score across all metrics
        - Per-sample detail records
        """
        metrics = EvaluationMetrics()

        if not all_verdicts:
            metrics.add_metric("composite_score", 0.0)
            return metrics

        # Collect per-metric scores
        metric_scores: dict[str, list[float]] = {m.name: [] for m in self.metrics}
        metric_pass_counts: dict[str, int] = {m.name: 0 for m in self.metrics}

        per_sample_records: list[dict[str, Any]] = []

        for sample, response, verdicts in all_verdicts:
            sample_record = {
                "sample_id": sample.id,
                "input_text": sample.input_text[:200],  # truncate for storage
                "agent_output": str(response.output) if response.output else "",
                "ground_truth": str(sample.target_output) if sample.target_output else "",
                "agent_processing_time": response.processing_time,
            }

            sample_weighted_sum = 0.0
            sample_weight_total = 0.0

            for metric in self.metrics:
                verdict = verdicts.get(metric.name)
                if verdict:
                    max_score = verdict.metadata.get("max_score", 10)
                    normalized = verdict.score / max_score if max_score > 0 else 0

                    metric_scores[metric.name].append(normalized)

                    if metric.is_successful(verdict):
                        metric_pass_counts[metric.name] += 1

                    sample_record[f"{metric.name}_score"] = verdict.score
                    sample_record[f"{metric.name}_normalized"] = round(normalized, 4)
                    sample_record[f"{metric.name}_reason"] = verdict.reason
                    sample_record[f"{metric.name}_pass"] = metric.is_successful(verdict)

                    sample_weighted_sum += normalized * metric.weight
                    sample_weight_total += metric.weight

            # Composite score for this sample
            if sample_weight_total > 0:
                sample_record["composite_score"] = round(
                    sample_weighted_sum / sample_weight_total, 4
                )
            else:
                sample_record["composite_score"] = 0.0

            per_sample_records.append(sample_record)

        # Aggregate per-metric stats
        for metric in self.metrics:
            scores = metric_scores[metric.name]
            if scores:
                avg = statistics.mean(scores)
                std = statistics.stdev(scores) if len(scores) > 1 else 0.0
                pass_rate = metric_pass_counts[metric.name] / len(scores)

                metrics.add_metric(f"{metric.name}_avg", round(avg, 4))
                metrics.add_metric(f"{metric.name}_std", round(std, 4))
                metrics.add_metric(f"{metric.name}_pass_rate", round(pass_rate, 4))
                metrics.add_metric(f"{metric.name}_count", len(scores))

        # Composite score across all metrics (weighted average of averages)
        total_weight = sum(m.weight for m in self.metrics)
        if total_weight > 0:
            composite = (
                sum(
                    statistics.mean(metric_scores[m.name]) * m.weight
                    for m in self.metrics
                    if metric_scores[m.name]
                )
                / total_weight
            )
            metrics.add_metric("composite_score", round(composite, 4))

        metrics.add_metric("total_samples_judged", len(all_verdicts))
        metrics.add_metric("execution_failures", len(execution_failures))
        metrics.add_metric("duration_seconds", round(duration, 4))

        metrics.per_sample_metrics = per_sample_records
        metrics.metadata = {
            "metric_names": [m.name for m in self.metrics],
            "execution_failures": execution_failures,
        }

        return metrics

    # ── report saving ────────────────────────────────────────

    def save_reports(
        self,
        result: EvaluationResult,
        output_dir: str | Path,
        run_prefix: str = "judge_run",
    ) -> Path:
        """
        Save evaluation results to a timestamped directory.

        Creates ``<output_dir>/<run_prefix>_<timestamp>/`` containing:

        - **result.md** — human-readable Markdown report with aggregate
          metrics and per-sample verdicts.
        - **Meta.json** — machine-readable metadata (run config, metrics,
          metric configs).

        Args:
            result: The ``EvaluationResult`` returned by :meth:`evaluate`.
            output_dir: Parent directory (e.g., ``"_EvalRing"``).
            run_prefix: Prefix for the timestamped sub-folder.

        Returns:
            Path to the created run directory.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(output_dir) / f"{run_prefix}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        metrics_dict = result.metrics.to_dict()["metrics"]

        # ── result.md ─────────────────────────────────────
        with open(run_dir / "result.md", "w", encoding="utf-8") as f:
            f.write("# LLM-as-a-Judge Evaluation Results\n\n")
            f.write(f"**Agent:** {result.agent_name}\n")
            f.write(f"**Task:** {result.task_name}\n")
            f.write(f"**Duration:** {result.duration:.2f} seconds\n")
            f.write(f"**Samples:** {len(result.metrics.per_sample_metrics)}\n\n")

            f.write("## Aggregate Metrics\n\n")
            for key, value in metrics_dict.items():
                f.write(f"- **{key}**: {value}\n")

            f.write("\n## Per-Sample Verdicts\n\n")
            for record in result.metrics.per_sample_metrics:
                sid = record.get("sample_id", "?")
                output = record.get("agent_output", "")
                gt = record.get("ground_truth", "")
                composite = record.get("composite_score", 0)

                f.write(f"### Sample {sid}\n")
                f.write(f"- **Prediction:** {output}\n")
                f.write(f"- **Ground Truth:** {gt}\n")
                f.write(f"- **Composite Score:** {composite}\n\n")

                for m in self.metrics:
                    name = m.name
                    score = record.get(f"{name}_score", "")
                    normalized = record.get(f"{name}_normalized", "")
                    reason = record.get(f"{name}_reason", "")
                    passed = record.get(f"{name}_pass", "")
                    f.write(f"**{name}** (score={score}, norm={normalized}, pass={passed})\n")
                    f.write(f"> {reason}\n\n")
                f.write("---\n\n")

        # ── Meta.json ─────────────────────────────────────
        meta = {
            "run_id": f"{run_prefix}_{timestamp}",
            "timestamp": datetime.now().isoformat(),
            "agent": result.agent_name,
            "task": result.task_name,
            "num_samples": len(result.metrics.per_sample_metrics),
            "duration": result.duration,
            "metrics_summary": metrics_dict,
            "metric_configs": [m.get_info() for m in self.metrics],
        }
        with open(run_dir / "Meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

        logger.info(f"Reports saved to: {run_dir}")
        return run_dir
