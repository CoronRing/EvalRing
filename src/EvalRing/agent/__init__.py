"""
Agent module for EvalRing.
"""

from .base import AgentResponse, BaseAgent
from .classification import (
    ClassificationPrediction,
    aggregate_base_vs_rest_probabilities,
    normalize_probability_distribution,
    parse_json_object,
    resolve_classification_prediction,
)
from .errors import ErrorClass, classify_error, format_exception
from .implementations import MockAgent, OpenAIAgent, RuleBasedAgent
from .multi_role import MultiRoleHostOrchestrator, RoleConfig

__all__ = [
    "BaseAgent",
    "AgentResponse",
    "ClassificationPrediction",
    "aggregate_base_vs_rest_probabilities",
    "normalize_probability_distribution",
    "parse_json_object",
    "resolve_classification_prediction",
    "MultiRoleHostOrchestrator",
    "RoleConfig",
    "MockAgent",
    "RuleBasedAgent",
    "OpenAIAgent",
    "ErrorClass",
    "classify_error",
    "format_exception",
]
