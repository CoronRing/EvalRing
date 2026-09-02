"""
JudgeMetric: The core evaluation unit in the LLM-as-a-Judge pipeline.

A JudgeMetric defines WHAT to measure. It follows the GEval two-phase pattern:
  1. (Optional) Auto-generate evaluation steps from a criteria string
  2. Score a test case using those steps + an optional rubric

Multiple JudgeMetrics can be composed to produce a multi-dimensional
evaluation (e.g., correctness + reasoning quality + safety).
"""

import logging
from typing import Any

from .judge import LLMJudge
from .rubric import Rubric, ScoringCriteria
from .schema import JudgeVerdict, parse_eval_steps, parse_verdict
from .template import JudgeTemplate

logger = logging.getLogger(__name__)


class JudgeMetric:
    """
    A single LLM-judged evaluation metric.

    Follows the criteria → steps → score pipeline:
    - User provides a ScoringCriteria (name, description, optional steps/rubric)
    - If no steps provided, the judge LLM generates them (cached)
    - For each test case, the steps + rubric are sent to the judge for scoring

    Args:
        criteria: A ScoringCriteria instance defining what to measure.
        judge: The LLMJudge backend to use for evaluation.
        parameters: Which test case fields the judge sees
                    (e.g., ["input", "output", "ground_truth"]).
        template: JudgeTemplate class (or subclass) for prompt rendering.
        strict_mode: If True, scoring is binary (0 or 1).
        threshold: Score >= threshold is considered a "pass". Default 0.5 (on 0-1 scale).
    """

    def __init__(
        self,
        criteria: ScoringCriteria,
        judge: LLMJudge,
        parameters: list[str] | None = None,
        template: type[JudgeTemplate] = JudgeTemplate,
        strict_mode: bool = False,
        threshold: float = 0.5,
    ):
        self.criteria = criteria
        self.judge = judge
        self.parameters = parameters or ["input", "output", "ground_truth"]
        self.template = template
        self.strict_mode = strict_mode
        self.threshold = threshold

        # Cached evaluation steps (generated once, reused for all samples)
        self._evaluation_steps: list[str] | None = None

    @property
    def name(self) -> str:
        return self.criteria.name

    @property
    def weight(self) -> float:
        return self.criteria.weight

    @property
    def evaluation_steps(self) -> list[str]:
        """Get evaluation steps, generating them if needed."""
        if self._evaluation_steps is None:
            if self.criteria.evaluation_steps:
                self._evaluation_steps = self.criteria.evaluation_steps
            else:
                self._generate_steps()
        if self._evaluation_steps is None:
            raise RuntimeError(
                f"[{self.name}] Judge returned no evaluation steps for the criteria."
            )
        return self._evaluation_steps

    def _generate_steps(self) -> None:
        """
        Phase 1: Ask the judge LLM to generate evaluation steps from the criteria.
        These steps are cached and reused for all subsequent scorings.
        """
        logger.info(f"[{self.name}] Generating evaluation steps from criteria...")

        prompt = self.template.generate_evaluation_steps(
            criteria=self.criteria.criteria, parameters=self.parameters
        )

        raw_output = self.judge.generate(prompt)
        eval_steps = parse_eval_steps(raw_output)
        self._evaluation_steps = eval_steps.steps

        logger.info(f"[{self.name}] Generated {len(self._evaluation_steps)} evaluation steps")

    def score(self, test_case: dict[str, str]) -> JudgeVerdict:
        """
        Phase 2: Score a single test case using the evaluation steps.

        Args:
            test_case: Dict mapping parameter names to values.
                       e.g., {"input": "...", "output": "...", "ground_truth": "..."}

        Returns:
            JudgeVerdict with score, reason, and metadata.
        """
        steps = self.evaluation_steps  # triggers generation if needed

        # Build the scoring prompt
        if self.strict_mode:
            prompt = self.template.generate_strict_score(
                evaluation_steps=steps,
                test_case=test_case,
                parameters=self.parameters,
            )
            max_score = 1
        else:
            score_range = self.criteria.rubric_score_range
            rubric_text = self.criteria.rubric_text

            prompt = self.template.generate_score(
                evaluation_steps=steps,
                test_case=test_case,
                parameters=self.parameters,
                rubric_text=rubric_text,
                score_range=score_range,
            )
            max_score = score_range[1]

        # Call the judge
        raw_output = self.judge.generate(prompt)
        verdict = parse_verdict(raw_output, max_score=max_score)

        # Add metric metadata
        verdict.metadata["metric_name"] = self.name
        verdict.metadata["strict_mode"] = self.strict_mode
        verdict.metadata["max_score"] = max_score

        return verdict

    def score_simple(
        self,
        input_text: str,
        output: str,
        ground_truth: str = "",
    ) -> JudgeVerdict:
        """
        Score a single case without building a dict manually.

        Convenience wrapper around :meth:`score` that constructs the
        test-case dict for you.

        Args:
            input_text: The original input.
            output: The agent's output / prediction.
            ground_truth: The expected correct answer (optional).

        Returns:
            JudgeVerdict with score, reason, and metadata.
        """
        test_case = {
            "input": input_text,
            "output": output,
            "ground_truth": ground_truth,
        }
        return self.score(test_case)

    def is_successful(self, verdict: JudgeVerdict) -> bool:
        """Check if a verdict passes the threshold."""
        max_score = verdict.metadata.get("max_score", 10)
        normalized = verdict.score / max_score if max_score > 0 else 0
        return normalized >= self.threshold

    def get_info(self) -> dict[str, Any]:
        """Return metric configuration for logging/reproducibility."""
        return {
            "name": self.name,
            "criteria": self.criteria.criteria,
            "evaluation_steps": self._evaluation_steps,
            "rubric": (
                self.criteria.rubric.to_dict()
                if isinstance(self.criteria.rubric, Rubric)
                else self.criteria.rubric
            ),
            "parameters": self.parameters,
            "strict_mode": self.strict_mode,
            "threshold": self.threshold,
            "weight": self.weight,
            "judge_model": self.judge.get_model_name(),
        }
