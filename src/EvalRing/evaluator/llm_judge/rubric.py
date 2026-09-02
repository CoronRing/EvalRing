"""
Rubric definitions for LLM-as-a-Judge scoring.

A Rubric anchors the judge's scoring by defining discrete grade bands,
each with a score range and a description of what that score means.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class RubricLevel:
    """
    A single level/band within a rubric.

    Examples:
        RubricLevel(score=5, label="Excellent", description="Perfectly matches ground truth...")
        RubricLevel(score=1, label="Poor", description="Completely incorrect or irrelevant...")
    """

    score: int
    label: str
    description: str

    def to_prompt_string(self) -> str:
        return f"Score {self.score} ({self.label}): {self.description}"


@dataclass
class Rubric:
    """
    A complete scoring rubric composed of ordered levels.

    The rubric is injected into the judge prompt to provide scoring anchors,
    improving calibration and consistency across evaluations.

    Args:
        name: Descriptive name for this rubric (e.g., "classification_accuracy")
        levels: Ordered list of RubricLevel defining the scoring scale
        score_range: (min, max) score bounds. Auto-derived from levels if not set.
        description: Optional high-level description of what this rubric measures.
    """

    name: str
    levels: list[RubricLevel]
    score_range: tuple[int, int] | None = None
    description: str = ""

    def __post_init__(self):
        if not self.levels:
            raise ValueError("Rubric must have at least one level")
        # Sort levels by score
        self.levels = sorted(self.levels, key=lambda level: level.score)
        # Auto-derive score range
        if self.score_range is None:
            self.score_range = (self.levels[0].score, self.levels[-1].score)

    @property
    def _range(self) -> tuple[int, int]:
        """Score range, always populated once ``__post_init__`` has run."""
        assert self.score_range is not None
        return self.score_range

    def to_prompt_string(self) -> str:
        """Render the rubric as a string suitable for injection into a judge prompt."""
        lines = []
        if self.description:
            lines.append(f"Rubric: {self.description}")
        lines.append(f"Score range: {self._range[0]} to {self._range[1]}")
        lines.append("")
        for level in self.levels:
            lines.append(level.to_prompt_string())
        return "\n".join(lines)

    @property
    def max_score(self) -> int:
        return self._range[1]

    @property
    def min_score(self) -> int:
        return self._range[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "score_range": list(self._range),
            "levels": [
                {"score": level.score, "label": level.label, "description": level.description}
                for level in self.levels
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Rubric":
        levels = [RubricLevel(**level) for level in data["levels"]]
        return cls(
            name=data["name"],
            levels=levels,
            score_range=tuple(data["score_range"]) if "score_range" in data else None,
            description=data.get("description", ""),
        )


@dataclass
class ScoringCriteria:
    """
    A named evaluation criterion with an optional rubric.

    This is the user-facing config object: you provide a criteria description
    (and optionally explicit evaluation_steps), and the framework either uses
    your steps or asks the judge LLM to generate them.

    Args:
        name: Short name for the criterion (e.g., "correctness", "relevance")
        criteria: Natural language description of what this criterion measures.
        evaluation_steps: Optional pre-defined steps. If None, the judge LLM
                          will auto-generate them from the criteria string.
        rubric: Optional Rubric to anchor scoring. If None, a default 0-10
                scale is used.
        weight: Weight for this criterion when combining multiple metrics.
                Defaults to 1.0.
    """

    name: str
    criteria: str
    evaluation_steps: list[str] | None = None
    rubric: Rubric | str | None = None
    weight: float = 1.0

    @property
    def rubric_text(self) -> str | None:
        """Get the rubric as a prompt-ready string, regardless of input type."""
        if self.rubric is None:
            return None
        if isinstance(self.rubric, str):
            return self.rubric
        return self.rubric.to_prompt_string()

    @property
    def rubric_score_range(self) -> tuple[int, int]:
        """Get the score range from the rubric, defaulting to (0, 10)."""
        if isinstance(self.rubric, Rubric):
            return (self.rubric.min_score, self.rubric.max_score)
        return (0, 10)

    def to_dict(self) -> dict[str, Any]:
        rubric_val: dict[str, Any] | str | None = None
        if isinstance(self.rubric, Rubric):
            rubric_val = self.rubric.to_dict()
        elif isinstance(self.rubric, str):
            rubric_val = self.rubric
        return {
            "name": self.name,
            "criteria": self.criteria,
            "evaluation_steps": self.evaluation_steps,
            "rubric": rubric_val,
            "weight": self.weight,
        }
