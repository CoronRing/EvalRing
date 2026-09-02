"""
Response schemas for structured LLM judge output.

These define the expected JSON shapes returned by the judge LLM.
We use simple dataclass parsing (no pydantic dependency) with
a robust JSON extraction fallback.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class JudgeVerdict:
    """
    The judge's verdict for a single evaluation.

    Attributes:
        score: Numeric score (within the rubric's range, e.g., 0-10)
        reason: The judge's chain-of-thought reasoning for the score.
        metadata: Any additional data returned by the judge.
    """

    score: float
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_score(self) -> float:
        """Score normalized to 0-1 range (assuming 0-10 raw scale)."""
        return self.score / 10.0

    def to_dict(self) -> dict[str, Any]:
        return {"score": self.score, "reason": self.reason, "metadata": self.metadata}


@dataclass
class EvalSteps:
    """
    Auto-generated evaluation steps from a criteria string.

    Attributes:
        steps: Ordered list of evaluation step descriptions.
    """

    steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"steps": self.steps}


def extract_json(text: str) -> dict[str, Any]:
    """
    Robustly extract the first JSON object from LLM output.

    Handles:
    - Clean JSON responses
    - JSON wrapped in markdown code blocks (```json ... ```)
    - JSON mixed with other text
    - Trailing commas (common LLM error)

    Args:
        text: Raw LLM output string.

    Returns:
        Parsed dictionary.

    Raises:
        ValueError: If no valid JSON object can be extracted.
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first { ... } block
    brace_depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                candidate = text[start : i + 1]
                # Fix trailing commas before } or ]
                candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    start = None
                    continue

    raise ValueError(f"Could not extract valid JSON from LLM output: {text[:200]}...")


def parse_verdict(raw_output: str, max_score: int = 10) -> JudgeVerdict:
    """
    Parse a JudgeVerdict from raw LLM judge output.

    Expected JSON format:
        {"score": <int>, "reason": "<string>"}

    Args:
        raw_output: Raw string from the judge LLM.
        max_score: Maximum valid score (for clamping).

    Returns:
        Parsed JudgeVerdict instance.
    """
    data = extract_json(raw_output)

    score = float(data.get("score", 0))
    score = max(0, min(score, max_score))  # clamp

    reason = str(data.get("reason", "No reason provided."))

    # Capture any extra fields as metadata
    metadata = {k: v for k, v in data.items() if k not in ("score", "reason")}

    return JudgeVerdict(score=score, reason=reason, metadata=metadata)


def parse_eval_steps(raw_output: str) -> EvalSteps:
    """
    Parse auto-generated evaluation steps from LLM output.

    Expected JSON format:
        {"steps": ["step 1", "step 2", ...]}
    """
    data = extract_json(raw_output)
    steps = data.get("steps", [])
    if isinstance(steps, list):
        steps = [str(s) for s in steps]
    else:
        steps = [str(steps)]
    return EvalSteps(steps=steps)
