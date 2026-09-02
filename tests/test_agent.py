"""Agent contract, classification parsing, and error classification."""

from __future__ import annotations

import pytest

from EvalRing.agent import (
    AgentResponse,
    MockAgent,
    OpenAIAgent,
    RuleBasedAgent,
    aggregate_base_vs_rest_probabilities,
    classify_error,
    format_exception,
    normalize_probability_distribution,
    parse_json_object,
    resolve_classification_prediction,
)
from EvalRing.config import MissingCredentialsError


class TestMockAgent:
    def test_predict_requires_initialization(self) -> None:
        agent = MockAgent(delay=0)
        with pytest.raises(RuntimeError, match="initialized"):
            agent.predict("hello")

    def test_fixed_response_is_deterministic(self) -> None:
        agent = MockAgent(fixed_response="positive", delay=0)
        agent.initialize()
        assert all(agent.predict(f"input {i}").output == "positive" for i in range(5))

    def test_response_carries_input_text_and_success_flag(self) -> None:
        agent = MockAgent(fixed_response="positive", delay=0)
        agent.initialize()
        response = agent.predict("some input")
        assert response.input_text == "some input"
        assert response.is_successful() is True

    def test_batch_assigns_positional_ids(self) -> None:
        agent = MockAgent(fixed_response="x", delay=0)
        agent.initialize()
        responses = agent.predict_batch(["a", "b", "c"])
        assert [r.input_id for r in responses] == ["0", "1", "2"]

    def test_blank_input_fails_validation(self) -> None:
        agent = MockAgent(delay=0)
        assert agent.validate_input("   ") is False
        assert agent.validate_input("real") is True

    def test_get_info_reports_type_and_state(self) -> None:
        agent = MockAgent(name="m", version="2.0", delay=0)
        agent.initialize()
        info = agent.get_info()
        assert info["name"] == "m"
        assert info["version"] == "2.0"
        assert info["type"] == "MockAgent"
        assert info["is_initialized"] is True


class TestRuleBasedAgent:
    def test_matches_keywords_case_insensitively(self) -> None:
        agent = RuleBasedAgent(rules={"positive": ["Great"], "negative": ["awful"]})
        agent.initialize()
        assert agent.predict("this is GREAT").output == "positive"

    def test_returns_default_when_nothing_matches(self) -> None:
        agent = RuleBasedAgent(rules={"positive": ["great"]}, default_output="unknown")
        agent.initialize()
        response = agent.predict("nothing relevant here")
        assert response.output == "unknown"
        assert response.confidence == 0.0


class TestOpenAIAgentConfiguration:
    """Construction must stay offline; only initialize() needs credentials."""

    def test_construction_without_credentials_does_not_raise(self) -> None:
        agent = OpenAIAgent(name="a")
        assert agent.api_key is None

    def test_initialize_without_credentials_raises_actionable_error(self) -> None:
        agent = OpenAIAgent(name="a")
        with pytest.raises(MissingCredentialsError, match="EVALRING_API_KEY"):
            agent.initialize()

    def test_model_comes_from_the_neutral_environment_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EVALRING_MODEL", "some-model-v2")
        assert OpenAIAgent(name="a").model_name == "some-model-v2"

    def test_explicit_arguments_win_over_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EVALRING_API_KEY", "env-key")
        agent = OpenAIAgent(name="a", api_key="explicit", base_url="https://x.example/v1")
        assert agent.api_key == "explicit"
        assert agent.base_url == "https://x.example/v1"

    def test_openai_key_alone_implies_no_custom_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert OpenAIAgent(name="a").base_url is None


class TestClassificationParsing:
    def test_plain_label_passes_through(self) -> None:
        assert resolve_classification_prediction("positive").label == "positive"

    def test_highest_scoring_class_wins(self) -> None:
        prediction = resolve_classification_prediction(
            {"positive": 0.2, "negative": 0.7, "neutral": 0.1}
        )
        assert prediction.label == "negative"
        assert prediction.confidence == pytest.approx(0.7)

    def test_ties_resolve_by_valid_label_order(self) -> None:
        prediction = resolve_classification_prediction(
            {"b": 0.5, "a": 0.5}, valid_labels=["b", "a"]
        )
        assert prediction.label == "b"

    def test_scores_outside_valid_labels_are_dropped(self) -> None:
        prediction = resolve_classification_prediction(
            {"positive": 0.9, "bogus": 0.95}, valid_labels=["positive", "negative"]
        )
        assert prediction.label == "positive"

    def test_none_output_yields_no_label(self) -> None:
        assert resolve_classification_prediction(None).label is None

    def test_aliases_are_applied(self) -> None:
        prediction = resolve_classification_prediction("POS", label_aliases={"pos": "positive"})
        assert prediction.label == "positive"


class TestJSONExtraction:
    def test_parses_exact_json(self) -> None:
        assert parse_json_object('{"a": 1}') == {"a": 1}

    def test_parses_json_embedded_in_prose(self) -> None:
        assert parse_json_object('Sure! {"a": 1} hope that helps') == {"a": 1}

    def test_returns_none_for_unparseable_text(self) -> None:
        assert parse_json_object("no json here") is None

    def test_returns_none_for_blank_text(self) -> None:
        assert parse_json_object("") is None


class TestProbabilityHelpers:
    def test_distribution_sums_to_one(self) -> None:
        result = normalize_probability_distribution({"a": 2.0, "b": 2.0})
        assert sum(result.values()) == pytest.approx(1.0)
        assert result["a"] == pytest.approx(0.5)

    def test_negative_scores_are_clamped(self) -> None:
        result = normalize_probability_distribution({"a": -1.0, "b": 1.0})
        assert result["a"] == 0.0
        assert result["b"] == pytest.approx(1.0)

    def test_all_zero_scores_do_not_divide_by_zero(self) -> None:
        assert normalize_probability_distribution({"a": 0.0}) == {"a": 0.0}

    def test_base_vs_rest_produces_a_valid_distribution(self) -> None:
        result = aggregate_base_vs_rest_probabilities(
            base_label="base", target_vs_base_probs={"x": 0.75, "y": 0.5}
        )
        assert sum(result.values()) == pytest.approx(1.0)
        # p(x|x vs base) > p(y|y vs base), so x must outrank y.
        assert result["x"] > result["y"]

    def test_empty_base_label_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="base_label"):
            aggregate_base_vs_rest_probabilities(base_label="", target_vs_base_probs={})


class TestErrorClassification:
    def test_rate_limit_is_detected(self) -> None:
        result = classify_error("Error code: 429 - rate limit exceeded")
        assert result.is_rate_limit is True
        assert result.is_terminal is False

    def test_transient_server_error_is_retryable(self) -> None:
        result = classify_error("APIConnectionError: connection error")
        assert result.is_transient is True
        assert result.is_terminal is False

    def test_empty_response_is_terminal_even_when_it_mentions_timeout(self) -> None:
        result = classify_error("Empty response after request timed out")
        assert result.is_terminal is True
        assert result.is_transient is False
        assert result.is_rate_limit is False

    def test_context_length_is_terminal(self) -> None:
        assert classify_error("maximum context length exceeded").is_terminal is True

    def test_format_exception_includes_type_and_message(self) -> None:
        formatted = format_exception(ValueError("something broke"))
        assert "ValueError" in formatted
        assert "something broke" in formatted

    def test_format_exception_handles_a_blank_message(self) -> None:
        assert "ValueError" in format_exception(ValueError())

    def test_format_exception_surfaces_provider_attributes(self) -> None:
        class ProviderError(Exception):
            status_code = 503
            llm_provider = "example"

        assert "status_code=503" in format_exception(ProviderError("upstream"))


def test_agent_response_reports_failure_when_error_is_set() -> None:
    response = AgentResponse(input_id="1", input_text="t", output=None, error="boom")
    assert response.is_successful() is False
