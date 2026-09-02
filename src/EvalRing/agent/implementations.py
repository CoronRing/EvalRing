"""
Concrete implementations of agents.
"""

import os
import random
import time
from typing import Any

from ..config import resolve_credentials, resolve_model_name
from ..logging_utils import get_logger
from .base import AgentResponse, BaseAgent
from .errors import format_exception

logger = get_logger(__name__)


class MockAgent(BaseAgent):
    """
    A mock agent for testing and demonstration purposes.

    Returns random predictions from a predefined list or a fixed response.
    """

    def __init__(
        self,
        name: str = "mock_agent",
        fixed_response: str | None = None,
        possible_outputs: list[str] | None = None,
        delay: float = 0.1,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.fixed_response = fixed_response
        self.possible_outputs = possible_outputs or ["positive", "negative", "neutral"]
        self.delay = delay

    def initialize(self, **kwargs) -> None:
        """Initialize the mock agent."""
        self._is_initialized = True

    def predict(self, input_text: str, **kwargs) -> AgentResponse:
        """
        Make a mock prediction.

        Args:
            input_text: Input text
            **kwargs: Additional parameters

        Returns:
            AgentResponse with mock prediction
        """
        if not self._is_initialized:
            raise RuntimeError("Agent must be initialized before prediction")

        start_time = time.time()

        # Simulate processing time
        if self.delay > 0:
            time.sleep(self.delay)

        # Generate response
        if self.fixed_response is not None:
            output = self.fixed_response
        else:
            output = random.choice(self.possible_outputs)

        processing_time = time.time() - start_time

        return AgentResponse(
            input_id="",  # Will be set by evaluator or batch processor
            input_text=input_text,
            output=output,
            confidence=random.uniform(0.5, 1.0),
            processing_time=processing_time,
            metadata={"mock": True},
        )


class RuleBasedAgent(BaseAgent):
    """
    A simple rule-based agent that uses keyword matching.
    """

    def __init__(
        self,
        name: str = "rule_based_agent",
        rules: dict | None = None,
        default_output: str = "unknown",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.rules = rules or {}
        self.default_output = default_output

    def initialize(self, **kwargs) -> None:
        """Initialize the rule-based agent."""
        # Convert all keywords to lowercase for case-insensitive matching
        self._processed_rules = {}
        for output, keywords in self.rules.items():
            self._processed_rules[output] = [k.lower() for k in keywords]

        self._is_initialized = True

    def predict(self, input_text: str, **kwargs) -> AgentResponse:
        """
        Make a prediction based on keyword rules.

        Args:
            input_text: Input text
            **kwargs: Additional parameters

        Returns:
            AgentResponse with rule-based prediction
        """
        if not self._is_initialized:
            raise RuntimeError("Agent must be initialized before prediction")

        start_time = time.time()

        text_lower = input_text.lower()
        output = self.default_output
        confidence = 0.0
        matched_keyword = None

        # Simple keyword matching
        for rule_output, keywords in self._processed_rules.items():
            for keyword in keywords:
                if keyword in text_lower:
                    output = rule_output
                    confidence = 1.0
                    matched_keyword = keyword
                    break
            if matched_keyword:
                break

        processing_time = time.time() - start_time

        return AgentResponse(
            input_id="",
            input_text=input_text,
            output=output,
            confidence=confidence,
            processing_time=processing_time,
            metadata={"matched_keyword": matched_keyword} if matched_keyword else {},
        )


class OpenAIAgent(BaseAgent):
    """
    Generic OpenAI-backed agent with streaming, TTFT, TPS, and token tracking.

    Subclass this and set ``self.system_prompt`` to get a working agent.
    Override ``_parse_output`` if you need custom output cleaning.

    Args:
        name: Agent name for reporting.
        model_name: Model identifier (e.g., ``"gpt-4o"``, ``"anthropic/claude-sonnet-4"``).
            Defaults to ``$EVALRING_MODEL`` when unset.
        api_key: API key. When omitted it is resolved from the environment by
            :func:`EvalRing.config.resolve_credentials`.
        base_url: OpenAI-compatible endpoint. When omitted it is resolved
            alongside the API key; ``None`` means the OpenAI default.
        system_prompt: System message sent to the model.
        temperature: Sampling temperature.  Default 0.0.
        max_completion_tokens: Maximum tokens the model may generate.
        **kwargs: Forwarded to BaseAgent.
    """

    def __init__(
        self,
        name: str = "openai-agent",
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.0,
        max_completion_tokens: int = 256,
        reasoning_effort: str | None = None,
        error_on_empty: bool = True,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.model_name = resolve_model_name(model_name, default="gpt-4o")
        credentials = resolve_credentials(api_key, base_url)
        self.credentials = credentials
        self.api_key = credentials.api_key
        self.base_url = credentials.base_url
        self.transport = os.environ.get("EVALRING_LLM_TRANSPORT", "litellm").strip().lower()
        # If the underlying HTTP request hangs (provider/network), streaming can block forever.
        # Use a bounded timeout so the evaluator can retry/record the failure and continue.
        # Env var is intentionally generic so it works with LiteLLM-backed providers.
        self.request_timeout_s = float(os.environ.get("OPENAI_REQUEST_TIMEOUT_S", "120"))
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        # "Thinking" level forwarded to providers that support it (dropped by
        # ``drop_params`` for those that do not). None => not sent.
        self.reasoning_effort = (reasoning_effort or "").strip() or None
        # When True, a completion that returns no visible content is reported as
        # an error (see ``predict``) rather than a silent empty answer.
        self.error_on_empty = error_on_empty
        # Both are provider SDK handles resolved lazily in initialize(); the
        # concrete types depend on which optional extra is installed.
        self.client: Any = None
        self._litellm: Any = None

    # ── lifecycle ──────────────────────────────────────────

    def initialize(self, **kwargs) -> None:
        """Initialize the OpenAI client."""
        self.credentials.require_key()
        logger.debug(
            "Initializing %s: model=%s provider=%s (key from %s) base_url=%s",
            self.name,
            self.model_name,
            self.credentials.provider,
            self.credentials.source,
            self.base_url or "<provider default>",
        )
        if self.transport == "litellm":
            try:
                import litellm

                self._litellm = litellm
                # Silently drop params a given provider/model does not accept
                # (e.g. reasoning_effort on a non-reasoning model, temperature!=1
                # on a reasoning model) so a heterogeneous suite never hard-fails.
                self._litellm.drop_params = True
            except ImportError:
                self._litellm = None

        if self._litellm is None:
            try:
                import httpx
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    f"Required packages missing: {exc}. "
                    "Install them with: pip install evalring[llm]"
                ) from exc

            client_kwargs: dict[str, Any] = {
                "api_key": self.api_key,
                "timeout": self.request_timeout_s,
                "http_client": httpx.Client(
                    limits=httpx.Limits(max_connections=2000, max_keepalive_connections=1000),
                    timeout=httpx.Timeout(self.request_timeout_s),
                ),
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self.client = OpenAI(**client_kwargs)
        self._is_initialized = True

    @property
    def _timeout(self) -> float | None:
        """Request timeout to pass to the SDK; ``None`` (unlimited) when <=0."""
        return (
            self.request_timeout_s
            if (self.request_timeout_s and self.request_timeout_s > 0)
            else None
        )

    def _completion_extra_params(self) -> dict[str, Any]:
        """Optional completion params applied uniformly across transports.

        ``max_completion_tokens`` is omitted when non-positive so reasoning
        models can run to natural completion instead of truncating; a positive
        value still caps output.
        """
        extra: dict[str, Any] = {}
        if self.reasoning_effort:
            extra["reasoning_effort"] = self.reasoning_effort
        if self.max_completion_tokens and self.max_completion_tokens > 0:
            extra["max_completion_tokens"] = self.max_completion_tokens
        return extra

    def _create_completion_stream(self, messages: list[dict[str, str]]):
        extra = self._completion_extra_params()
        if self._litellm is not None:
            return self._litellm.completion(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                stream=True,
                stream_options={"include_usage": True},
                api_key=self.api_key,
                api_base=self.base_url,
                timeout=self._timeout,
                **extra,
            )

        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            stream=True,
            stream_options={"include_usage": True},
            timeout=self._timeout,
            **extra,
        )

    # ── prediction ────────────────────────────────────────

    def predict(self, input_text: str, **kwargs) -> AgentResponse:
        """Call the chat-completions API with streaming.

        On success, ``metadata`` includes: ``raw_output``, ``model``,
        ``base_url``, ``prompt_tokens``, ``completion_tokens``, ``total_tokens``,
        ``reasoning_tokens`` (hidden "thinking" tokens; estimated from streamed
        reasoning text when the provider omits the count), ``answer_tokens``,
        ``reasoning_chars``, ``reasoning_content`` (the full reasoning text when
        the provider streams it; empty when the provider hides it, e.g. OpenAI
        reasoning models), ``ttft``, ``generation_time``, ``tps``,
        ``total_time`` and ``finish_reason``.

        Failure handling:
        - Exceptions are captured with their *complete* message via
          :func:`~EvalRing.agent.errors.format_exception`.
        - When ``error_on_empty`` is set (default) and the model returns no
          visible content — e.g. a reasoning model that exhausts its budget or a
          stream cut at the request timeout — the response is reported as an
          error (``output="Error"``) rather than a silent empty answer.
        """
        if not self._is_initialized:
            raise RuntimeError("Agent not initialized.  Call initialize() first.")

        if not self.validate_input(input_text):
            return AgentResponse(
                input_id="",
                input_text=input_text,
                output=None,
                error="Invalid input",
            )

        start_time = time.time()
        # Accumulators live outside the try so a mid-stream error or timeout can
        # still return whatever partial thinking/answer was produced.
        first_reasoning_time: float | None = None
        first_content_time: float | None = None
        raw_output = ""
        reasoning_output = ""
        reasoning_chars = 0
        reasoning_tokens = 0
        finish_reason = None
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        timed_out = False
        stream_error: str | None = None
        # Soft, in-loop deadline: if the model streams longer than the request
        # timeout we stop and KEEP the partial output. (The evaluator's hard guard
        # is a separate backstop for a totally-stalled stream that yields nothing.)
        soft_deadline = (
            self.request_timeout_s
            if (self.request_timeout_s and self.request_timeout_s > 0)
            else None
        )

        try:
            stream = self._create_completion_stream(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self._build_user_message(input_text)},
                ]
            )
            for chunk in stream:
                if soft_deadline and (time.time() - start_time) > soft_deadline:
                    timed_out = True
                    break
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    delta = getattr(choice, "delta", None)
                    content = getattr(delta, "content", None) if delta else None
                    if content:
                        if first_content_time is None:
                            first_content_time = time.time()
                        raw_output += content
                    # Hidden reasoning ("thinking") streams as ``reasoning_content``.
                    reasoning = getattr(delta, "reasoning_content", None) if delta else None
                    if reasoning:
                        if first_reasoning_time is None:
                            first_reasoning_time = time.time()
                        reasoning_output += reasoning
                        reasoning_chars += len(reasoning)
                    fr = getattr(choice, "finish_reason", None)
                    if fr:
                        finish_reason = fr
                if getattr(chunk, "usage", None) is not None:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens
                    total_tokens = chunk.usage.total_tokens
                    reasoning_tokens = (
                        self._extract_reasoning_tokens(chunk.usage) or reasoning_tokens
                    )
        except Exception as e:  # noqa: BLE001 - captured with partial output below
            stream_error = format_exception(e)

        end_time = time.time()
        total_time = end_time - start_time
        # Two distinct latencies: time to first *thinking* token vs first *answer*
        # token. For a reasoning model the answer TTFT lands after all thinking.
        ttft_reasoning = (
            (first_reasoning_time - start_time) if first_reasoning_time is not None else None
        )
        ttft = (first_content_time - start_time) if first_content_time is not None else None
        generation_time = (end_time - first_content_time) if first_content_time is not None else 0.0
        # Throughput over TOTAL wall time (thinking + answer), not just the answer
        # window — reasoning dominates wall time for these models.
        tps = (completion_tokens / total_time) if (total_time > 0 and completion_tokens) else 0.0

        # Prefer the provider-reported reasoning-token count; otherwise estimate
        # from streamed reasoning text (~4 chars/token).
        reasoning_tokens_estimated = False
        if not reasoning_tokens and reasoning_chars:
            reasoning_tokens = max(1, round(reasoning_chars / 4))
            reasoning_tokens_estimated = True
        answer_tokens = max(0, completion_tokens - reasoning_tokens) if completion_tokens else 0

        metadata = {
            "raw_output": raw_output,
            "model": self.model_name,
            "base_url": self.base_url,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "reasoning_tokens": reasoning_tokens,
            "reasoning_tokens_estimated": reasoning_tokens_estimated,
            "answer_tokens": answer_tokens,
            "reasoning_chars": reasoning_chars,
            "reasoning_content": reasoning_output,
            "ttft": ttft,  # time to first ANSWER token (None if none)
            "ttft_reasoning": ttft_reasoning,  # time to first THINKING token (None if none)
            "generation_time": generation_time,
            "tps": tps,  # completion tokens / total wall time
            "total_time": total_time,
            "finish_reason": finish_reason,
            "timed_out": timed_out,
        }

        # ── error paths (all keep the partial output captured above) ──
        if stream_error is not None:
            return AgentResponse(
                input_id="",
                input_text=input_text,
                output="Error",
                error=stream_error,
                processing_time=total_time,
                metadata=metadata,
            )
        if timed_out:
            err = (
                f"Agent stream exceeded the {soft_deadline:.0f}s request timeout and was stopped; "
                f"partial output kept (content_chars={len(raw_output)}, reasoning_chars={reasoning_chars}, "
                f"finish_reason={finish_reason})."
            )
            return AgentResponse(
                input_id="",
                input_text=input_text,
                output="Error",
                error=err,
                processing_time=total_time,
                metadata=metadata,
            )
        if self.error_on_empty and not raw_output.strip():
            return AgentResponse(
                input_id="",
                input_text=input_text,
                output="Error",
                error=self._empty_response_error(
                    finish_reason, completion_tokens, reasoning_tokens, reasoning_chars, total_time
                ),
                processing_time=total_time,
                metadata=metadata,
            )

        output = self._parse_output(raw_output)
        self._augment_success_metadata(raw_output, metadata)
        return AgentResponse(
            input_id="",
            input_text=input_text,
            output=output,
            confidence=1.0,
            processing_time=total_time,
            metadata=metadata,
        )

    # ── shared helpers ────────────────────────────────────

    def _empty_response_error(
        self,
        finish_reason: str | None,
        completion_tokens: int,
        reasoning_tokens: int,
        reasoning_chars: int,
        total_time: float,
    ) -> str:
        """Build a diagnostic message for a no-visible-content response."""
        if self.request_timeout_s and total_time >= 0.9 * self.request_timeout_s:
            reason = (
                f"stream ended after {total_time:.0f}s (at/near the {self.request_timeout_s:.0f}s "
                f"request timeout — client or provider) while still reasoning, before emitting any "
                f"answer; raise the request timeout (OPENAI_REQUEST_TIMEOUT_S) if the provider allows "
                f"longer streams"
            )
        elif finish_reason == "length":
            reason = "exhausted its output token budget on hidden reasoning without emitting a final answer"
        elif finish_reason == "content_filter":
            reason = "response was blocked by a content filter"
        else:
            reason = "returned no visible content"
        return (
            f"Empty response: model {reason} "
            f"(finish_reason={finish_reason}, completion_tokens={completion_tokens}, "
            f"reasoning_tokens={reasoning_tokens}, reasoning_chars={reasoning_chars}, "
            f"elapsed={total_time:.0f}s)."
        )

    @staticmethod
    def _extract_reasoning_tokens(usage: Any) -> int:
        """Pull reasoning/thinking token count from a usage object.

        LiteLLM normalises most providers to
        ``usage.completion_tokens_details.reasoning_tokens`` (OpenAI o-series /
        gpt-5, Gemini "thinking", Anthropic, and OpenAI-compatible gateways that
        forward it). Falls back to alternate field names, handling both
        pydantic-style objects and plain dicts. Returns 0 when unavailable.
        """

        def _get(obj: Any, key: str) -> Any:
            if obj is None:
                return None
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        details = _get(usage, "completion_tokens_details")
        for key in ("reasoning_tokens", "thoughts_token_count", "thinking_tokens"):
            val = _get(details, key) or _get(usage, key)
            if val:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    pass
        return 0

    # ── hooks for subclasses ──────────────────────────────

    def _build_user_message(self, input_text: str) -> str:
        """
        Build the user-role message from ``input_text``.

        Override this to add task-specific formatting, e.g.::

            def _build_user_message(self, input_text):
                return f"Text to classify:\\n\\n{input_text}"

        Default returns the input as-is.
        """
        return input_text

    def _parse_output(self, raw_output: str) -> str:
        """
        Post-process the model's raw text into the final output.

        Override for label cleaning, JSON extraction, etc.
        Default strips whitespace.
        """
        return raw_output.strip()

    def _augment_success_metadata(self, raw_output: str, metadata: dict[str, Any]) -> None:
        """Hook to add task-specific derived fields to a successful response's
        metadata (e.g. an extracted final answer). Default is a no-op."""
        return None
