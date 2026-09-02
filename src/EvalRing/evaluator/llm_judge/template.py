"""
Prompt templates for the LLM-as-a-Judge pipeline.

Follows the GEval two-phase approach:
  Phase 1: criteria → auto-generate evaluation steps (CoT)
  Phase 2: steps + test case → score + reason

All templates are static methods on JudgeTemplate, making them
easily overridable via subclassing for custom domains.
"""


class JudgeTemplate:
    """
    Prompt templates for LLM judge evaluation.

    Override any static method in a subclass to customize prompts
    for your specific domain.
    """

    @staticmethod
    def generate_evaluation_steps(criteria: str, parameters: list[str]) -> str:
        """
        Phase 1 prompt: Ask the LLM to generate chain-of-thought
        evaluation steps from a high-level criteria description.

        Args:
            criteria: The evaluation criterion in natural language.
            parameters: The parameter names available (e.g., ["input", "output", "ground_truth"]).

        Returns:
            Prompt string.
        """
        params_str = ", ".join(parameters)
        return f"""You are an evaluation expert. Given an evaluation criteria, generate 3-5 concise, 
actionable evaluation steps that a human evaluator would follow to score a response.

The evaluation will have access to these parameters: {params_str}

Evaluation Criteria:
{criteria}

Return your answer as JSON with this exact format:
{{"steps": ["Step 1: ...", "Step 2: ...", "Step 3: ..."]}}

Generate only the steps. Be specific and measurable. Each step should guide the evaluator 
toward a concrete assessment. Do not include scoring instructions — those are handled separately."""

    @staticmethod
    def generate_score(
        evaluation_steps: list[str],
        test_case: dict[str, str],
        parameters: list[str],
        rubric_text: str | None = None,
        score_range: tuple = (0, 10),
    ) -> str:
        """
        Phase 2 prompt: Given evaluation steps and a test case,
        produce a score and reasoning.

        Args:
            evaluation_steps: The CoT steps to follow.
            test_case: Dict mapping parameter names to their values.
            parameters: List of parameter names.
            rubric_text: Optional rendered rubric string for scoring anchors.
            score_range: (min, max) score range.

        Returns:
            Prompt string.
        """
        # Build steps section
        steps_text = "\n".join(f"  {i + 1}. {step}" for i, step in enumerate(evaluation_steps))

        # Build test case section
        case_lines = []
        for param in parameters:
            value = test_case.get(param, "N/A")
            case_lines.append(f"[{param}]:\n{value}")
        case_text = "\n\n".join(case_lines)

        # Build rubric section
        rubric_section = ""
        if rubric_text:
            rubric_section = f"""
## Scoring Rubric
{rubric_text}
"""

        return f"""You are a rigorous and fair evaluator. Assess the following test case by 
carefully following the evaluation steps below.

## Evaluation Steps
{steps_text}

## Test Case
{case_text}
{rubric_section}
## Instructions
1. Follow each evaluation step in order.
2. Consider both strengths and weaknesses.
3. Assign a score between {score_range[0]} and {score_range[1]} (inclusive).
4. Provide a clear, concise reason that references specific aspects of the response.

Return your assessment as JSON with this exact format:
{{"score": <integer {score_range[0]}-{score_range[1]}>, "reason": "<your reasoning>"}}

Return ONLY the JSON object. No other text."""

    @staticmethod
    def generate_strict_score(
        evaluation_steps: list[str],
        test_case: dict[str, str],
        parameters: list[str],
    ) -> str:
        """
        Binary scoring prompt (pass/fail). Score is 0 or 1.

        Used when strict_mode=True on a JudgeMetric.
        """
        steps_text = "\n".join(f"  {i + 1}. {step}" for i, step in enumerate(evaluation_steps))

        case_lines = []
        for param in parameters:
            value = test_case.get(param, "N/A")
            case_lines.append(f"[{param}]:\n{value}")
        case_text = "\n\n".join(case_lines)

        return f"""You are a rigorous evaluator performing a pass/fail assessment.

## Evaluation Steps
{steps_text}

## Test Case
{case_text}

## Instructions
Determine if the response PASSES or FAILS based on the evaluation steps.
- Score 1 if the response meets ALL criteria satisfactorily.
- Score 0 if it fails on ANY criterion.

Return your assessment as JSON:
{{"score": <0 or 1>, "reason": "<brief explanation>"}}

Return ONLY the JSON object."""
