"""LLM-as-a-judge grader for Humanity's Last Exam responses.

HLE answers are free-form (or a multiple-choice letter) and cannot be graded by
exact string match — a judge model decides whether the candidate's final answer
is semantically equivalent to the gold answer. This mirrors the official HLE
grading contract: the judge extracts the candidate's final answer, decides
``correct`` yes/no, and reports its own confidence.

The judge uses LiteLLM directly (non-streaming) and supports the same
per-provider routing (model prefix + explicit ``api_base``/``api_key``) as
:class:`hle_agent.HLEAgent`, so the judge can run on any configured provider.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Optional

from EvalRing.agent import format_exception

JUDGE_PROMPT_TEMPLATE = """Judge whether the following [response] to a [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must focus only on whether the final answer in [response] matches the [correct_answer]. Ignore differences in formatting, wording, or the amount of explanation.

[correct_answer]: {correct_answer}

Return ONLY a JSON object with exactly these keys and no extra text:
{{"extracted_final_answer": "<the exact final answer extracted from [response], or 'None' if not found>", "correct": "yes" or "no", "confidence": <integer 0-100 = how confident the [response] itself sounded>}}"""


@dataclass
class JudgeVerdict:
    """Result of grading one response."""
    correct: bool
    extracted_answer: str
    confidence: Optional[float]
    raw: str
    error: Optional[str] = None
    # Telemetry for comprehensive per-sample records.
    judge_time: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class HLEJudge:
    """Grades HLE responses via an LLM, returning a structured verdict."""

    def __init__(
        self,
        model_name: str = "openai/gpt-5.5",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: Optional[float] = None,  # None => use the provider default
        max_completion_tokens: int = 0,  # <=0 => no cap
        request_timeout_s: float = 600.0,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.request_timeout_s = request_timeout_s
        self._litellm = None

    def initialize(self) -> None:
        import litellm

        litellm.drop_params = True
        self._litellm = litellm

    def grade(self, question: str, correct_answer: str, response: str) -> JudgeVerdict:
        if self._litellm is None:
            self.initialize()

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            response=response if response else "(no response)",
            correct_answer=correct_answer,
        )
        extra = {}
        if self.max_completion_tokens and self.max_completion_tokens > 0:
            extra["max_completion_tokens"] = self.max_completion_tokens
        # Only send temperature when explicitly set — several reasoning models
        # (e.g. gpt-5.5) reject any value other than their default of 1.
        if self.temperature is not None:
            extra["temperature"] = self.temperature
        t0 = time.time()
        try:
            completion = self._litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                api_key=self.api_key,
                api_base=self.api_base,
                timeout=(self.request_timeout_s if self.request_timeout_s and self.request_timeout_s > 0 else None),
                **extra,
            )
            raw = (completion.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001 - surfaced to caller as an errored verdict
            return JudgeVerdict(correct=False, extracted_answer="", confidence=None, raw="",
                                error=format_exception(e), judge_time=time.time() - t0)

        judge_time = time.time() - t0
        usage = getattr(completion, "usage", None)
        verdict = self._parse_verdict(raw)
        verdict.judge_time = judge_time
        if usage is not None:
            verdict.prompt_tokens = getattr(usage, "prompt_tokens", None)
            verdict.completion_tokens = getattr(usage, "completion_tokens", None)
            verdict.total_tokens = getattr(usage, "total_tokens", None)
        return verdict

    @staticmethod
    def _parse_verdict(raw: str) -> JudgeVerdict:
        parsed = None
        # Prefer a JSON object anywhere in the text.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None

        if isinstance(parsed, dict):
            correct_field = str(parsed.get("correct", "")).strip().lower()
            correct = correct_field in {"yes", "true", "1", "correct"}
            extracted = str(parsed.get("extracted_final_answer", "")).strip()
            confidence: Optional[float] = None
            try:
                confidence = float(parsed.get("confidence"))
            except (TypeError, ValueError):
                confidence = None
            return JudgeVerdict(correct=correct, extracted_answer=extracted, confidence=confidence, raw=raw)

        # Fallback: loose textual scan when the judge ignored the JSON contract.
        correct = bool(re.search(r"\bcorrect\b\W+(yes|true)\b", raw, re.IGNORECASE)) or bool(
            re.search(r"\byes\b", raw, re.IGNORECASE)
        )
        return JudgeVerdict(correct=correct, extracted_answer="", confidence=None, raw=raw)
