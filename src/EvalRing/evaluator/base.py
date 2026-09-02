"""
Base classes for evaluators in the EvalRing framework.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..agent.base import BaseAgent
from ..dataset.base import BaseDataset
from ..utils.global_cache import GlobalCache


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    metrics: dict[str, float] = field(default_factory=dict)
    per_sample_metrics: list[dict[str, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_metric(self, name: str, value: float) -> None:
        """Add a metric."""
        self.metrics[name] = value

    def get_metric(self, name: str, default: float | None = None) -> float | None:
        """Get a metric by name."""
        return self.metrics.get(name, default)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metrics": self.metrics,
            "per_sample_metrics": self.per_sample_metrics,
            "metadata": self.metadata,
        }


@dataclass
class EvaluationResult:
    """Container for evaluation results."""

    agent_name: str
    dataset_name: str
    metrics: EvaluationMetrics
    duration: float
    timestamp: datetime
    task_name: str
    version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_name": self.agent_name,
            "dataset_name": self.dataset_name,
            "metrics": self.metrics.to_dict(),
            "duration": self.duration,
            "timestamp": self.timestamp.isoformat(),
            "task_name": self.task_name,
            "version": self.version,
            "metadata": self.metadata,
        }

    def save(self, filepath: str | Path) -> None:
        """Save evaluation results to file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class BaseEvaluator(ABC):
    """
    Abstract base class for evaluators in EvalRing.

    This class orchestrates the evaluation process, managing agents,
    datasets, and metric calculation.
    """

    def __init__(self, name: str = "evaluator", output_dir: str | Path | None = None, **kwargs):
        self.name = name
        self.output_dir = Path(output_dir) if output_dir else Path("./results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._metadata = kwargs

        # Initialize global cache
        cache_mode = kwargs.get("cache_mode", "both")
        self.global_cache = GlobalCache(mode=cache_mode)

    @abstractmethod
    def evaluate(
        self, agent: BaseAgent, dataset: BaseDataset, task_name: str, version: str = "1.0", **kwargs
    ) -> EvaluationResult:
        """
        Run evaluation on the given agent and dataset.

        Args:
            agent: The agent to evaluate
            dataset: The dataset to evaluate on
            task_name: Name of the evaluation task
            version: Version of the evaluation
            **kwargs: Additional evaluation parameters

        Returns:
            EvaluationResult containing metrics and metadata
        """
        pass

    def validate_inputs(self, agent: BaseAgent, dataset: BaseDataset) -> bool:
        """
        Validate agent and dataset before evaluation.

        Args:
            agent: Agent to validate
            dataset: Dataset to validate

        Returns:
            True if inputs are valid, False otherwise
        """
        if not isinstance(agent, BaseAgent):
            raise ValueError("Agent must inherit from BaseAgent")

        if not isinstance(dataset, BaseDataset):
            raise ValueError("Dataset must inherit from BaseDataset")

        if len(dataset) == 0:
            raise ValueError("Dataset cannot be empty")

        # Validate data integrity and enforce unique per-sample IDs.
        if hasattr(dataset, "validate_data") and not dataset.validate_data():
            raise ValueError("Dataset validation failed")
        if hasattr(dataset, "assert_unique_ids"):
            dataset.assert_unique_ids(context="evaluator.validate_inputs")

        return True

    def save_results(self, result: EvaluationResult, filename: str | None = None) -> Path:
        """
        Save evaluation results to file.

        Args:
            result: Evaluation result to save
            filename: Optional filename (auto-generated if not provided)

        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_{result.agent_name}_{result.task_name}_{timestamp}.json"

        filepath = self.output_dir / filename
        result.save(filepath)
        return filepath

    def get_info(self) -> dict[str, Any]:
        """Get evaluator information."""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "output_dir": str(self.output_dir),
            "metadata": self._metadata,
        }
