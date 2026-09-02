"""
Evaluator module for EvalRing.
"""

from .base import BaseEvaluator, EvaluationMetrics, EvaluationResult
from .implementations import ClassificationEvaluator
from .llm_judge import (
    EvalSteps,
    JudgeMetric,
    JudgeTemplate,
    JudgeVerdict,
    LLMJudge,
    LLMJudgeEvaluator,
    OpenAIJudge,
    Rubric,
    RubricLevel,
    ScoringCriteria,
)

__all__ = [
    "BaseEvaluator",
    "EvaluationResult",
    "EvaluationMetrics",
    "ClassificationEvaluator",
    # LLM-as-a-Judge
    "Rubric",
    "RubricLevel",
    "ScoringCriteria",
    "JudgeVerdict",
    "EvalSteps",
    "JudgeTemplate",
    "JudgeMetric",
    "LLMJudge",
    "OpenAIJudge",
    "LLMJudgeEvaluator",
]
