"""
EvalRing: a unified evaluation framework for agents across LLMs and agent versions.

The public API is re-exported here. Three abstractions carry the framework:

- a :class:`~EvalRing.dataset.base.BaseDataset` supplies
  :class:`~EvalRing.dataset.base.DataSample` records,
- a :class:`~EvalRing.agent.base.BaseAgent` turns each sample into an
  :class:`~EvalRing.agent.base.AgentResponse`,
- a :class:`~EvalRing.evaluator.base.BaseEvaluator` scores the responses and
  returns an :class:`~EvalRing.evaluator.base.EvaluationResult`.

Provider credentials are resolved centrally by
:func:`EvalRing.config.resolve_credentials`; see ``docs/CONFIGURATION.md``.
"""

__version__ = "0.2.1"

from .agent import (
    AgentResponse,
    BaseAgent,
    ClassificationPrediction,
    ErrorClass,
    MockAgent,
    MultiRoleHostOrchestrator,
    OpenAIAgent,
    RoleConfig,
    RuleBasedAgent,
    classify_error,
    resolve_classification_prediction,
)
from .config import (
    MissingCredentialsError,
    ProviderCredentials,
    resolve_credentials,
    resolve_model_name,
)
from .dataset import BaseDataset, CSVDataset, DataFrameDataset, DataSample, JSONDataset
from .evaluator import (
    BaseEvaluator,
    ClassificationEvaluator,
    EvalSteps,
    EvaluationMetrics,
    EvaluationResult,
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
from .logging_utils import configure_logging, get_logger
from .utils import GlobalCache, generate_model_list, generate_suite_visuals, run_suite

__all__ = [
    "__version__",
    # Configuration
    "MissingCredentialsError",
    "ProviderCredentials",
    "resolve_credentials",
    "resolve_model_name",
    "configure_logging",
    "get_logger",
    # Datasets
    "BaseDataset",
    "DataSample",
    "JSONDataset",
    "CSVDataset",
    "DataFrameDataset",
    # Agents
    "BaseAgent",
    "AgentResponse",
    "ClassificationPrediction",
    "MockAgent",
    "RuleBasedAgent",
    "OpenAIAgent",
    "MultiRoleHostOrchestrator",
    "RoleConfig",
    "ErrorClass",
    "classify_error",
    "resolve_classification_prediction",
    # Evaluators
    "BaseEvaluator",
    "EvaluationResult",
    "EvaluationMetrics",
    "ClassificationEvaluator",
    # LLM-as-a-judge
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
    # Suite tooling
    "GlobalCache",
    "generate_model_list",
    "generate_suite_visuals",
    "run_suite",
]
