"""
LLM Judge backends.

An LLMJudge wraps an LLM client and exposes a simple `generate(prompt) -> str`
interface. The evaluator/metric layer calls the judge to get raw outputs,
which are then parsed via the schema module.
"""

from abc import ABC, abstractmethod
from typing import Any

from ...config import resolve_credentials, resolve_model_name


class LLMJudge(ABC):
    """
    Abstract base class for an LLM judge backend.

    Subclass this to integrate any LLM provider (OpenAI, Anthropic,
    local models, etc.) as a judge.
    """

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self._config = kwargs

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a text response from the judge LLM.

        Args:
            prompt: The fully-rendered judge prompt.
            **kwargs: Provider-specific overrides.

        Returns:
            Raw string output from the LLM.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier string."""
        pass

    def get_info(self) -> dict[str, Any]:
        """Return metadata about this judge for logging."""
        return {
            "model_name": self.model_name,
            "type": self.__class__.__name__,
            "config": {k: v for k, v in self._config.items() if k != "api_key"},
        }


class OpenAIJudge(LLMJudge):
    """
    LLM judge backed by any OpenAI-compatible Chat Completions endpoint.

    Args:
        model_name: Judge model identifier. Defaults to ``$EVALRING_MODEL``,
            then ``"gpt-4o"``.
        api_key: API key. Resolved from the environment when omitted; see
            :func:`EvalRing.config.resolve_credentials`.
        base_url: OpenAI-compatible endpoint. Resolved alongside the API key
            when omitted; ``None`` means the OpenAI default.
        temperature: Sampling temperature. Default 0.0 for deterministic judging.
        max_completion_tokens: Max tokens in judge response. Default 512.
        **kwargs: Additional parameters forwarded to the API call.
    """

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_completion_tokens: int = 512,
        **kwargs,
    ):
        resolved_model = resolve_model_name(model_name, default="gpt-4o") or "gpt-4o"
        super().__init__(resolved_model, **kwargs)
        credentials = resolve_credentials(api_key, base_url)
        self.credentials = credentials
        self.api_key = credentials.api_key
        self.base_url = credentials.base_url
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self._client: Any = None

    def _ensure_client(self):
        """Lazily initialize the OpenAI-compatible client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "OpenAIJudge requires the 'openai' package. "
                    "Install it with: pip install evalring[llm]"
                ) from exc
            client_kwargs: dict[str, Any] = {"api_key": self.credentials.require_key()}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._client = OpenAI(**client_kwargs)

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Call the OpenAI Chat Completions API.

        Args:
            prompt: The judge prompt (sent as a user message).
            **kwargs: Overrides for temperature, max_completion_tokens, etc.

        Returns:
            The model's response text.
        """
        self._ensure_client()

        temperature = kwargs.pop("temperature", self.temperature)
        max_tokens = kwargs.pop("max_completion_tokens", self.max_completion_tokens)

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert evaluation judge. Always respond with valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            **kwargs,
        )

        return response.choices[0].message.content.strip()

    def get_model_name(self) -> str:
        return self.model_name
