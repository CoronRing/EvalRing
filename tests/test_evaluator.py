"""End-to-end classification evaluation and the metric arithmetic behind it."""

from __future__ import annotations

from pathlib import Path

import pytest

from EvalRing.agent.base import AgentResponse, BaseAgent
from EvalRing.dataset import DataSample, JSONDataset
from EvalRing.evaluator import ClassificationEvaluator, EvaluationMetrics, EvaluationResult


class ScriptedAgent(BaseAgent):
    """Agent that replays a fixed input-to-output mapping.

    Makes evaluator assertions exact: no randomness, no network, and a
    controllable failure mode for the retry path.

    Args:
        mapping: Input text mapped to the label the agent should return.
        failing_inputs: Inputs for which ``predict`` raises instead.
    """

    def __init__(self, mapping: dict, failing_inputs: tuple = (), **kwargs) -> None:
        super().__init__(name=kwargs.pop("name", "scripted"), **kwargs)
        self.mapping = mapping
        self.failing_inputs = set(failing_inputs)
        self.call_count = 0

    def initialize(self, **kwargs) -> None:
        self._is_initialized = True

    def predict(self, input_text: str, **kwargs) -> AgentResponse:
        self.call_count += 1
        if input_text in self.failing_inputs:
            raise RuntimeError("scripted failure")
        return AgentResponse(
            input_id="",
            input_text=input_text,
            output=self.mapping[input_text],
            confidence=0.9,
        )


def _dataset(pairs: list[tuple]) -> JSONDataset:
    dataset = JSONDataset(name="scripted")
    for index, (text, label) in enumerate(pairs):
        dataset.add_sample(DataSample(id=str(index), input_text=text, target_output=label))
    return dataset


class TestClassificationEvaluation:
    def test_perfect_agent_scores_one(self, tmp_path: Path) -> None:
        pairs = [("a", "positive"), ("b", "negative")]
        agent = ScriptedAgent(mapping=dict(pairs))
        evaluator = ClassificationEvaluator(output_dir=tmp_path)

        result = evaluator.evaluate(agent, _dataset(pairs), task_name="t", max_workers=2)

        assert result.metrics.get_metric("accuracy") == pytest.approx(1.0)
        assert result.metrics.get_metric("f1_score") == pytest.approx(1.0)

    def test_accuracy_reflects_partial_correctness(self, tmp_path: Path) -> None:
        pairs = [("a", "positive"), ("b", "negative"), ("c", "neutral"), ("d", "positive")]
        mapping = {"a": "positive", "b": "positive", "c": "neutral", "d": "positive"}
        evaluator = ClassificationEvaluator(output_dir=tmp_path)

        result = evaluator.evaluate(
            ScriptedAgent(mapping=mapping), _dataset(pairs), task_name="t", max_workers=2
        )

        assert result.metrics.get_metric("accuracy") == pytest.approx(0.75)

    def test_per_sample_metrics_follow_dataset_order(self, tmp_path: Path) -> None:
        pairs = [(letter, "x") for letter in "abcdefgh"]
        agent = ScriptedAgent(mapping={letter: "x" for letter, _ in pairs})
        evaluator = ClassificationEvaluator(output_dir=tmp_path)

        result = evaluator.evaluate(agent, _dataset(pairs), task_name="t", max_workers=4)

        ids = [m["sample_id"] for m in result.metrics.per_sample_metrics]
        assert ids == [str(i) for i in range(len(pairs))]

    def test_failing_samples_are_recorded_not_raised(self, tmp_path: Path) -> None:
        pairs = [("a", "positive"), ("b", "negative")]
        agent = ScriptedAgent(mapping=dict(pairs), failing_inputs=("b",))
        evaluator = ClassificationEvaluator(output_dir=tmp_path)

        result = evaluator.evaluate(
            agent, _dataset(pairs), task_name="t", max_workers=1, max_retries=1
        )

        failures = result.metadata["execution_failures"]
        assert len(failures) == 1
        assert failures[0]["sample_id"] == "1"
        assert result.metrics.get_metric("accuracy") == pytest.approx(0.5)

    def test_failed_samples_are_retried_up_to_the_limit(self, tmp_path: Path) -> None:
        pairs = [("a", "positive")]
        agent = ScriptedAgent(mapping=dict(pairs), failing_inputs=("a",))
        evaluator = ClassificationEvaluator(output_dir=tmp_path)

        evaluator.evaluate(agent, _dataset(pairs), task_name="t", max_workers=1, max_retries=3)

        assert agent.call_count == 3

    def test_empty_dataset_is_rejected(self, tmp_path: Path) -> None:
        evaluator = ClassificationEvaluator(output_dir=tmp_path)
        with pytest.raises(ValueError, match="empty"):
            evaluator.evaluate(ScriptedAgent(mapping={}), _dataset([]), task_name="t")

    def test_duplicate_sample_ids_are_rejected_before_any_call(self, tmp_path: Path) -> None:
        dataset = JSONDataset(name="dupes")
        for _ in range(2):
            dataset.add_sample(DataSample(id="same", input_text="a", target_output="x"))
        agent = ScriptedAgent(mapping={"a": "x"})
        evaluator = ClassificationEvaluator(output_dir=tmp_path)

        with pytest.raises(ValueError, match="Duplicate sample IDs"):
            evaluator.evaluate(agent, dataset, task_name="t")
        assert agent.call_count == 0

    def test_agent_is_initialized_automatically(self, tmp_path: Path) -> None:
        pairs = [("a", "x")]
        agent = ScriptedAgent(mapping=dict(pairs))
        assert agent._is_initialized is False

        ClassificationEvaluator(output_dir=tmp_path).evaluate(agent, _dataset(pairs), task_name="t")
        assert agent._is_initialized is True

    def test_partial_callback_fires_once_per_sample(self, tmp_path: Path) -> None:
        pairs = [("a", "x"), ("b", "x")]
        seen: list[dict] = []
        evaluator = ClassificationEvaluator(output_dir=tmp_path)

        evaluator.evaluate(
            ScriptedAgent(mapping={"a": "x", "b": "x"}),
            _dataset(pairs),
            task_name="t",
            partial_cb=seen.append,
        )
        assert len(seen) == 2


class TestMetricArithmetic:
    """Macro-averaged precision/recall/F1 computed without scikit-learn."""

    def _metrics(self, y_true: list[str], y_pred: list[str]) -> EvaluationMetrics:
        return ClassificationEvaluator()._calculate_classification_metrics(y_true, y_pred)

    def test_perfect_prediction(self) -> None:
        metrics = self._metrics(["a", "b"], ["a", "b"])
        assert metrics.get_metric("accuracy") == pytest.approx(1.0)
        assert metrics.get_metric("precision") == pytest.approx(1.0)

    def test_all_wrong_prediction(self) -> None:
        metrics = self._metrics(["a", "a"], ["b", "b"])
        assert metrics.get_metric("accuracy") == pytest.approx(0.0)
        assert metrics.get_metric("f1_score") == pytest.approx(0.0)

    def test_macro_average_weights_classes_equally(self) -> None:
        # Class "a": perfect. Class "b": never predicted.
        metrics = self._metrics(["a", "a", "a", "b"], ["a", "a", "a", "a"])
        assert metrics.get_metric("accuracy") == pytest.approx(0.75)
        # Macro recall = mean(recall_a=1.0, recall_b=0.0) = 0.5
        assert metrics.get_metric("recall") == pytest.approx(0.5)

    def test_empty_inputs_do_not_divide_by_zero(self) -> None:
        assert self._metrics([], []).get_metric("accuracy") == pytest.approx(0.0)


class TestResultSerialization:
    def test_result_round_trips_through_json(self, tmp_path: Path) -> None:
        pairs = [("a", "x")]
        evaluator = ClassificationEvaluator(output_dir=tmp_path)
        result = evaluator.evaluate(
            ScriptedAgent(mapping=dict(pairs)), _dataset(pairs), task_name="task"
        )

        path = evaluator.save_results(result)
        assert path.exists()

        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["task_name"] == "task"
        assert payload["metrics"]["metrics"]["accuracy"] == pytest.approx(1.0)

    def test_metrics_to_dict_includes_per_sample_rows(self) -> None:
        metrics = EvaluationMetrics()
        metrics.add_metric("accuracy", 0.5)
        metrics.per_sample_metrics = [{"sample_id": "1"}]
        payload = metrics.to_dict()
        assert payload["metrics"]["accuracy"] == 0.5
        assert payload["per_sample_metrics"] == [{"sample_id": "1"}]

    def test_missing_metric_returns_the_default(self) -> None:
        assert EvaluationMetrics().get_metric("absent", default=-1.0) == -1.0


def test_evaluator_rejects_a_non_agent(tmp_path: Path) -> None:
    evaluator = ClassificationEvaluator(output_dir=tmp_path)
    with pytest.raises(ValueError, match="BaseAgent"):
        evaluator.validate_inputs("not an agent", _dataset([("a", "x")]))  # type: ignore[arg-type]


def test_evaluation_result_is_constructible_standalone() -> None:
    from datetime import datetime

    result = EvaluationResult(
        agent_name="a",
        dataset_name="d",
        metrics=EvaluationMetrics(),
        duration=1.0,
        timestamp=datetime.now(),
        task_name="t",
    )
    assert result.to_dict()["agent_name"] == "a"
