"""
Base classes for agents in the EvalRing framework.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentResponse:
    """Represents an agent's response to an input."""

    input_id: str
    input_text: str
    output: Any
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    processing_time: float | None = None
    error: str | None = None

    def is_successful(self) -> bool:
        """Check if the response was successful."""
        return self.error is None


class BaseAgent(ABC):
    """
    Abstract base class for all agents in EvalRing.

    This class provides a standardized interface for different types of agents
    (LLM-based, rule-based, ensemble, etc.) to ensure compatibility with the
    evaluation framework.
    """

    def __init__(self, name: str, version: str = "1.0", description: str | None = None, **kwargs):
        self.name = name
        self.version = version
        self.description = description or f"Agent: {name}"
        self._metadata = kwargs
        self._is_initialized = False

    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """
        Initialize the agent with required resources.

        This method should be called before making predictions.
        """
        pass

    @abstractmethod
    def predict(self, input_text: str, **kwargs) -> AgentResponse:
        """
        Make a prediction for a single input.

        Args:
            input_text: The input text to process
            **kwargs: Additional parameters for prediction

        Returns:
            AgentResponse containing the prediction and metadata
        """
        pass

    def predict_batch(self, inputs: list[str], **kwargs) -> list[AgentResponse]:
        """
        Make predictions for multiple inputs.

        Default implementation processes inputs sequentially.
        Override this method for batch processing optimization.

        Args:
            inputs: List of input texts to process
            **kwargs: Additional parameters for prediction

        Returns:
            List of AgentResponse objects
        """
        responses = []
        for i, input_text in enumerate(inputs):
            response = self.predict(input_text, **kwargs)
            response.input_id = str(i)
            responses.append(response)
        return responses

    def validate_input(self, input_text: str) -> bool:
        """
        Validate input before processing.

        Args:
            input_text: Input text to validate

        Returns:
            True if input is valid, False otherwise
        """
        return isinstance(input_text, str) and len(input_text.strip()) > 0

    def get_info(self) -> dict[str, Any]:
        """Get agent information."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "type": self.__class__.__name__,
            "metadata": self._metadata,
            "is_initialized": self._is_initialized,
        }

    def save_config(self, filepath: str | Path) -> None:
        """Save agent configuration to file."""
        config = self.get_info()
        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}')"

    def __repr__(self) -> str:
        return self.__str__()
