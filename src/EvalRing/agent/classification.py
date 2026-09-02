"""
Reusable utilities for classification-style outputs.

These helpers let agents return either:
- a plain class label string, or
- a structured mapping of class -> confidence score.

The evaluator can then resolve the top class consistently.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class ClassificationPrediction:
    """Normalized prediction object used by evaluators."""

    label: str | None
    confidence: float | None = None
    class_scores: dict[str, float] | None = None


def parse_json_object(raw_text: str) -> dict[str, Any] | None:
    """
    Parse a JSON object from raw model text.

    Supports either:
    - exact JSON object text, or
    - text that contains one object block.
    """
    text = (raw_text or "").strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def canonicalize_label(raw_label: Any, label_aliases: Mapping[str, str] | None = None) -> str:
    """Map a raw label to its canonical form (case-insensitive aliases)."""
    text = str(raw_label).strip()
    if not text:
        return ""

    if not label_aliases:
        return text

    alias_map = {k.lower(): v for k, v in label_aliases.items()}
    return alias_map.get(text.lower(), text)


def normalize_class_scores(
    output: Mapping[Any, Any],
    *,
    label_aliases: Mapping[str, str] | None = None,
) -> dict[str, float]:
    """Convert a raw mapping into a canonical class-score dict."""
    merged: dict[str, float] = {}
    for raw_label, raw_score in output.items():
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue

        label = canonicalize_label(raw_label, label_aliases=label_aliases)
        if not label:
            continue

        merged[label] = merged.get(label, 0.0) + score
    return merged


def normalize_probability_distribution(scores: Mapping[str, float]) -> dict[str, float]:
    """Normalize non-negative class scores into a probability distribution."""
    normalized: dict[str, float] = {}
    for label, value in scores.items():
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        normalized[label] = max(0.0, v)

    total = sum(normalized.values())
    if total <= 0:
        return dict.fromkeys(normalized, 0.0)

    return {label: value / total for label, value in normalized.items()}


def aggregate_base_vs_rest_probabilities(
    *,
    base_label: str,
    target_vs_base_probs: Mapping[str, float],
    all_labels: list[str] | None = None,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    """
    Convert pairwise binary probabilities into a full multi-class distribution.

    Expected input per target class is p(target | target vs base).
    We convert each pairwise probability into odds r_t = p_t / (1-p_t), and use:
        P(base) = 1 / (1 + sum_t r_t)
        P(target_t) = r_t * P(base)

    This makes a reusable bridge from base-vs-rest binary runs to multi-class output.
    """
    if not base_label:
        raise ValueError("base_label must be a non-empty string")

    ratios: dict[str, float] = {}
    for label, probability in target_vs_base_probs.items():
        if label == base_label:
            continue

        try:
            p_target = float(probability)
        except (TypeError, ValueError):
            continue

        p_target = min(max(p_target, epsilon), 1.0 - epsilon)
        ratios[label] = p_target / (1.0 - p_target)

    base_prob = 1.0 / (1.0 + sum(ratios.values()))
    scores: dict[str, float] = {base_label: base_prob}
    for label, ratio in ratios.items():
        scores[label] = ratio * base_prob

    if all_labels:
        for label in all_labels:
            scores.setdefault(label, 0.0)

    return normalize_probability_distribution(scores)


def resolve_classification_prediction(
    output: Any,
    *,
    valid_labels: list[str] | None = None,
    label_aliases: Mapping[str, str] | None = None,
) -> ClassificationPrediction:
    """
    Resolve top-class prediction from string or class-score mapping output.

    If output is a mapping, the highest-scoring class is selected.
    Ties are resolved by valid_labels order (if provided), then alphabetically.
    """
    if isinstance(output, Mapping):
        scores = normalize_class_scores(output, label_aliases=label_aliases)

        if valid_labels:
            valid_set = set(valid_labels)
            scores = {k: v for k, v in scores.items() if k in valid_set}

        if not scores:
            return ClassificationPrediction(label=None, confidence=None, class_scores={})

        order = {label: idx for idx, label in enumerate(valid_labels or [])}
        best_label, best_score = min(
            scores.items(),
            key=lambda kv: (
                -kv[1],
                order.get(kv[0], 10**9),
                kv[0].lower(),
            ),
        )
        return ClassificationPrediction(
            label=best_label, confidence=float(best_score), class_scores=scores
        )

    if output is None:
        return ClassificationPrediction(label=None, confidence=None, class_scores=None)

    label = canonicalize_label(str(output), label_aliases=label_aliases)
    return ClassificationPrediction(label=label, confidence=None, class_scores=None)
