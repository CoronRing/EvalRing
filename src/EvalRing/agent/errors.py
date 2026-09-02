"""Reusable error formatting and classification for model-backed agents.

Two concerns that every LLM task hits, factored out so tasks share one policy:

- :func:`format_exception` turns a raised exception (often from LiteLLM / the
  OpenAI SDK) into a *complete* diagnostic string — exception type, full
  message, and useful provider attributes — instead of just the class name.
- :func:`classify_error` labels an error message as rate-limited, transient
  (worth patient retries), and/or terminal (retrying only wastes tokens) so
  evaluators can share one retry policy.
"""

from __future__ import annotations

from dataclasses import dataclass


def format_exception(e: BaseException) -> str:
    """Build a complete, diagnosable error string from an exception.

    Captures the exception type and its full message (falling back to ``repr``
    when the message is empty), plus common LiteLLM/OpenAI attributes
    (``status_code``, ``llm_provider``, ``model``) when present.
    """
    detail = str(e).strip()
    if not detail:
        detail = repr(e)
    parts = [f"{type(e).__name__}: {detail}"]
    for attr in ("status_code", "llm_provider", "model"):
        val = getattr(e, attr, None)
        if val not in (None, ""):
            parts.append(f"{attr}={val}")
    return " | ".join(parts)


# Substrings (matched against a lower-cased message) for each class.
_RATE_LIMIT_MARKERS = ("429", "rate limit", "rate_limit", "too many requests")
_TRANSIENT_MARKERS = (
    "connection error",
    "internalservererror",
    "internal server error",
    "overloaded",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    " 500",
    " 502",
    " 503",
    " 504",
    "502 ",
    "503 ",
    "504 ",
)
_TERMINAL_MARKERS = (
    "empty response",
    "content filter",
    "content_filter",
    "invalid request",
    "invalid_request",
    "badrequest",
    "bad request",
    "context length",
    "maximum context",
    "unsupported",
)


@dataclass(frozen=True)
class ErrorClass:
    """Retry-relevant classification of an error message."""

    is_rate_limit: bool
    is_transient: bool
    is_terminal: bool


def classify_error(message: str) -> ErrorClass:
    """Classify an error message for retry decisions.

    Precedence: an **empty response** (or other terminal condition) is terminal
    even if its message also mentions "timed out" — retrying such a case only
    burns another full timeout window of tokens. Rate limits take precedence
    over generic transient handling so callers can apply a dedicated backoff.
    """
    el = (message or "").lower()
    is_rate_limit = any(m in el for m in _RATE_LIMIT_MARKERS)
    is_transient = any(m in el for m in _TRANSIENT_MARKERS)
    # Always-terminal conditions win over transient-looking text: these all
    # mention "timeout"/"timed out" but retrying only wastes another full
    # deadline — an empty response, a hard client timeout, and a soft
    # (in-stream) timeout where the model was simply too slow for the limit.
    forced_terminal = (
        "empty response" in el
        or "hard client timeout" in el
        or "request timeout and was stopped" in el
    )
    is_terminal = forced_terminal or (
        any(m in el for m in _TERMINAL_MARKERS) and not is_rate_limit and not is_transient
    )
    if is_terminal:
        # A terminal error should never be treated as retryable.
        is_transient = False
        is_rate_limit = False
    return ErrorClass(
        is_rate_limit=is_rate_limit, is_transient=is_transient, is_terminal=is_terminal
    )
