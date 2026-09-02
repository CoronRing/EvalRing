"""
Provider configuration and credential resolution.

EvalRing talks to any OpenAI-compatible chat-completions endpoint. Rather than
hard-coding a vendor, every component resolves its credentials through
:func:`resolve_credentials`, which reads a documented, ordered list of
environment variables.

See ``docs/CONFIGURATION.md`` for the full table and worked examples.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

#: Base URL applied when a provider is selected but no explicit URL is given.
#: ``None`` means "let the OpenAI SDK use its own default".
_PROVIDER_DEFAULT_BASE_URL: Final[dict[str, str | None]] = {
    "evalring": None,
    "openai": None,
    "openrouter": "https://openrouter.ai/api/v1",
    "radium": "https://api.radium.cloud/v1",
}

#: Ordered credential sources: (provider, api-key var, base-url var).
#: The first variable that is set wins. ``EVALRING_*`` is the vendor-neutral
#: form and should be preferred by new users; the remaining entries exist so
#: that existing provider-specific environments keep working unchanged.
_CREDENTIAL_SOURCES: Final[tuple[tuple[str, str, str], ...]] = (
    ("evalring", "EVALRING_API_KEY", "EVALRING_BASE_URL"),
    ("openai", "OPENAI_API_KEY", "OPENAI_BASE_URL"),
    ("openrouter", "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL"),
    ("openrouter", "OPEN_ROUTER_KEY", "OPENROUTER_BASE_URL"),
    ("radium", "RADIUM_API_KEY", "RADIUM_BASE_URL"),
)

#: Ordered model-name variables, checked when no model is passed explicitly.
_MODEL_VARS: Final[tuple[str, ...]] = (
    "EVALRING_MODEL",
    "OPENAI_MODEL",
    "OPENROUTER_MODEL",
    "OPEN_ROUTER_MODEL",
    "RADIUM_MODEL",
)

#: Every variable EvalRing will read for an API key, in precedence order.
CREDENTIAL_ENV_VARS: Final[tuple[str, ...]] = tuple(
    key_var for _, key_var, _ in _CREDENTIAL_SOURCES
)


@dataclass(frozen=True)
class ProviderCredentials:
    """Resolved endpoint configuration for an OpenAI-compatible provider.

    Attributes:
        api_key: The API key, or ``None`` when nothing was configured.
        base_url: The endpoint to call, or ``None`` to use the SDK default
            (``https://api.openai.com/v1``).
        provider: Short identifier of the source that supplied the key, one of
            ``"evalring"``, ``"openai"``, ``"openrouter"``, ``"radium"``, or
            ``"explicit"`` when the key was passed in code.
        source: Human-readable description of where the key came from, safe to
            print in diagnostics (never contains the key itself).
    """

    api_key: str | None
    base_url: str | None
    provider: str
    source: str

    def require_key(self) -> str:
        """Return the API key, raising a descriptive error when absent.

        Raises:
            MissingCredentialsError: If no API key was resolved.
        """
        if not self.api_key:
            raise MissingCredentialsError()
        return self.api_key


class MissingCredentialsError(ValueError):
    """Raised when no API key could be resolved from arguments or environment."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or (
                "No API key found. Pass api_key=... explicitly, or set one of: "
                + ", ".join(CREDENTIAL_ENV_VARS)
                + ". For a non-OpenAI endpoint also set EVALRING_BASE_URL. "
                "See docs/CONFIGURATION.md."
            )
        )


def _clean(value: str | None) -> str | None:
    """Normalize an environment value, treating blank strings as unset."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def resolve_credentials(
    api_key: str | None = None,
    base_url: str | None = None,
    *,
    env: dict[str, str] | None = None,
) -> ProviderCredentials:
    """Resolve the API key and base URL for an OpenAI-compatible provider.

    Explicit arguments always win. Otherwise the variables listed in
    ``_CREDENTIAL_SOURCES`` are consulted in order; the first one that is set
    determines both the key and, unless overridden, the base URL.

    Args:
        api_key: Explicit API key. When given, no key variable is consulted.
        base_url: Explicit endpoint. When given, no URL variable is consulted.
        env: Environment mapping to read from. Defaults to ``os.environ``;
            supplying one makes the function trivially testable.

    Returns:
        A :class:`ProviderCredentials` describing the selected endpoint. The
        ``api_key`` field may be ``None``; call
        :meth:`ProviderCredentials.require_key` at the point of use so the
        error message is raised where it is actionable.

    Example:
        >>> creds = resolve_credentials(env={"EVALRING_API_KEY": "sk-test"})
        >>> creds.provider
        'evalring'
    """
    environ = os.environ if env is None else env

    explicit_key = _clean(api_key)
    explicit_url = _clean(base_url)

    if explicit_key:
        return ProviderCredentials(
            api_key=explicit_key,
            base_url=explicit_url or _clean(environ.get("EVALRING_BASE_URL")),
            provider="explicit",
            source="api_key argument",
        )

    for provider, key_var, url_var in _CREDENTIAL_SOURCES:
        key = _clean(environ.get(key_var))
        if not key:
            continue
        resolved_url = (
            explicit_url
            or _clean(environ.get("EVALRING_BASE_URL"))
            or _clean(environ.get(url_var))
            or _PROVIDER_DEFAULT_BASE_URL.get(provider)
        )
        return ProviderCredentials(
            api_key=key,
            base_url=resolved_url,
            provider=provider,
            source=f"${key_var}",
        )

    return ProviderCredentials(
        api_key=None,
        base_url=explicit_url or _clean(environ.get("EVALRING_BASE_URL")),
        provider="none",
        source="unset",
    )


def resolve_model_name(
    model_name: str | None = None,
    *,
    default: str | None = None,
    env: dict[str, str] | None = None,
) -> str | None:
    """Resolve the model identifier to evaluate.

    Args:
        model_name: Explicit model identifier; returned unchanged when given.
        default: Value returned when neither the argument nor any environment
            variable is set.
        env: Environment mapping to read from. Defaults to ``os.environ``.

    Returns:
        The resolved model identifier, or ``default`` when nothing is set.
    """
    environ = os.environ if env is None else env

    explicit = _clean(model_name)
    if explicit:
        return explicit

    for var in _MODEL_VARS:
        value = _clean(environ.get(var))
        if value:
            return value

    return default


def has_any_credentials(env: dict[str, str] | None = None) -> bool:
    """Report whether any recognized API-key variable is set.

    Args:
        env: Environment mapping to read from. Defaults to ``os.environ``.

    Returns:
        ``True`` if at least one credential variable holds a non-blank value.
    """
    environ = os.environ if env is None else env
    return any(_clean(environ.get(var)) for var in CREDENTIAL_ENV_VARS)
