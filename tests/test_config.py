"""Credential and model resolution.

These tests pin the precedence order that ``docs/CONFIGURATION.md`` documents.
Changing the order here is a breaking change for every user whose environment
sets more than one variable.
"""

from __future__ import annotations

import pytest

from EvalRing.config import (
    CREDENTIAL_ENV_VARS,
    MissingCredentialsError,
    has_any_credentials,
    resolve_credentials,
    resolve_model_name,
)


def test_explicit_key_beats_every_environment_variable() -> None:
    creds = resolve_credentials(
        api_key="explicit-key",
        env={"EVALRING_API_KEY": "env-key", "OPENAI_API_KEY": "other"},
    )
    assert creds.api_key == "explicit-key"
    assert creds.provider == "explicit"
    assert "argument" in creds.source


def test_evalring_variable_takes_precedence_over_provider_variables() -> None:
    creds = resolve_credentials(
        env={
            "EVALRING_API_KEY": "neutral",
            "OPENAI_API_KEY": "openai",
            "RADIUM_API_KEY": "radium",
        }
    )
    assert creds.api_key == "neutral"
    assert creds.provider == "evalring"
    assert creds.source == "$EVALRING_API_KEY"


def test_openai_precedes_openrouter_and_legacy_vendor_keys() -> None:
    creds = resolve_credentials(
        env={"OPENAI_API_KEY": "openai", "OPEN_ROUTER_KEY": "or", "RADIUM_API_KEY": "radium"}
    )
    assert creds.provider == "openai"
    # No base URL is invented for OpenAI; the SDK default applies.
    assert creds.base_url is None


def test_openrouter_supplies_its_default_base_url() -> None:
    creds = resolve_credentials(env={"OPENROUTER_API_KEY": "or-key"})
    assert creds.provider == "openrouter"
    assert creds.base_url == "https://openrouter.ai/api/v1"


def test_legacy_open_router_key_still_resolves() -> None:
    creds = resolve_credentials(env={"OPEN_ROUTER_KEY": "legacy"})
    assert creds.api_key == "legacy"
    assert creds.provider == "openrouter"


def test_explicit_base_url_overrides_the_provider_default() -> None:
    creds = resolve_credentials(
        base_url="https://gateway.internal/v1", env={"OPENROUTER_API_KEY": "or-key"}
    )
    assert creds.base_url == "https://gateway.internal/v1"


def test_evalring_base_url_overrides_provider_specific_url() -> None:
    creds = resolve_credentials(
        env={
            "OPENAI_API_KEY": "openai",
            "OPENAI_BASE_URL": "https://ignored.example/v1",
            "EVALRING_BASE_URL": "https://wins.example/v1",
        }
    )
    assert creds.base_url == "https://wins.example/v1"


def test_blank_values_are_treated_as_unset() -> None:
    creds = resolve_credentials(env={"EVALRING_API_KEY": "   ", "OPENAI_API_KEY": "real"})
    assert creds.api_key == "real"
    assert creds.provider == "openai"


def test_no_credentials_yields_an_empty_result_not_an_exception() -> None:
    creds = resolve_credentials(env={})
    assert creds.api_key is None
    assert creds.provider == "none"


def test_require_key_raises_with_an_actionable_message() -> None:
    creds = resolve_credentials(env={})
    with pytest.raises(MissingCredentialsError) as excinfo:
        creds.require_key()
    # The error must name the variables the user can actually set.
    assert "EVALRING_API_KEY" in str(excinfo.value)


def test_no_vendor_endpoint_is_used_without_that_vendors_key() -> None:
    """A private endpoint must never be the default for an unrelated key."""
    creds = resolve_credentials(env={"OPENAI_API_KEY": "openai"})
    assert creds.base_url is None
    assert "radium" not in (creds.base_url or "")


class TestModelResolution:
    def test_explicit_model_wins(self) -> None:
        assert resolve_model_name("gpt-4o", env={"EVALRING_MODEL": "other"}) == "gpt-4o"

    def test_neutral_variable_precedes_provider_variables(self) -> None:
        env = {"EVALRING_MODEL": "neutral", "OPENAI_MODEL": "openai"}
        assert resolve_model_name(env=env) == "neutral"

    def test_falls_back_to_provider_variable(self) -> None:
        assert resolve_model_name(env={"RADIUM_MODEL": "hal-1.0"}) == "hal-1.0"

    def test_default_used_when_nothing_is_set(self) -> None:
        assert resolve_model_name(env={}, default="fallback") == "fallback"


class TestCredentialDetection:
    def test_detects_any_recognized_variable(self) -> None:
        assert has_any_credentials({"OPEN_ROUTER_KEY": "k"}) is True

    def test_reports_false_for_empty_environment(self) -> None:
        assert has_any_credentials({}) is False

    def test_unrelated_variables_do_not_count(self) -> None:
        assert has_any_credentials({"SOME_OTHER_TOKEN": "k"}) is False

    def test_public_variable_list_is_ordered_neutral_first(self) -> None:
        assert CREDENTIAL_ENV_VARS[0] == "EVALRING_API_KEY"
