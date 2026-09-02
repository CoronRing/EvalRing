"""
LLM-as-a-Judge evaluation module for EvalRing.

Inspired by deepeval's GEval architecture, this module provides a generic
framework for using LLMs as evaluators/judges of agent responses.

Key components:
- JudgeMetric: Define what to measure (criteria → CoT steps → score)
- Rubric: Define scoring anchors and grade bands
- LLMJudge: The LLM backend that performs the judging
- LLMJudgeEvaluator: Orchestrates evaluation over datasets
"""

from .evaluator import LLMJudgeEvaluator
from .judge import LLMJudge, OpenAIJudge
from .metric import JudgeMetric
from .rubric import Rubric, RubricLevel, ScoringCriteria
from .schema import EvalSteps, JudgeVerdict
from .template import JudgeTemplate

__all__ = [
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
