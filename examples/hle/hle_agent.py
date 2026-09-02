"""Agent for answering Humanity's Last Exam (HLE) questions.

``HLEAgent`` is intentionally thin: all of the reusable machinery — streaming
telemetry, ``finish_reason`` capture, reasoning ("thinking") token accounting,
empty/timeout responses surfaced as errors, complete error formatting, and
``reasoning_effort`` support — now lives in the core
:class:`EvalRing.agent.OpenAIAgent`. This class only adds:

- the HLE answer contract (explanation + succinct final answer + confidence);
- extraction of the final answer / confidence from the response, exposed in the
  response metadata for the judge and reports.

Per-model provider routing (OpenAI / Gemini / Radium gateway) is achieved purely
by the ``model_name`` (LiteLLM provider prefix), explicit ``api_base`` and
``api_key`` — so a suite can mix providers in one run.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from EvalRing.agent import OpenAIAgent

MODE_BASIC = "basic"
VALID_MODES = {MODE_BASIC}

# Official-HLE-style answer contract. Works for both exactMatch and
# multipleChoice questions (the choices are embedded in the question text).
SYSTEM_PROMPT_BASIC = """You are an expert answering an extremely difficult exam question.

Read the question carefully and reason step by step. Your response MUST end with the following three lines, in this exact format:

Explanation: {a concise explanation of your reasoning}
Exact Answer: {your final answer, as succinct as possible — for multiple choice give only the letter of the correct option}
Confidence: {an integer from 0 to 100 representing how confident you are that your answer is correct}%"""

_EXACT_ANSWER_RE = re.compile(r"Exact Answer\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)
_CONFIDENCE_RE = re.compile(r"Confidence\s*:\s*(\d{1,3})", re.IGNORECASE)


class HLEAgent(OpenAIAgent):
    """LiteLLM-backed agent that answers HLE questions in the standard format."""

    def __init__(
        self,
        name: str = "hle-agent",
        model_name: str = "openai/gpt-5.4-mini",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_completion_tokens: int = 0,  # <=0 => no cap (reasoning models finish naturally)
        reasoning_effort: Optional[str] = "medium",
        agent_mode: str = MODE_BASIC,
        **kwargs,
    ):
        if agent_mode not in VALID_MODES:
            raise ValueError(f"Unsupported agent_mode='{agent_mode}'. Valid modes: {sorted(VALID_MODES)}")
        self.agent_mode = agent_mode

        super().__init__(
            name=name,
            model_name=model_name,
            api_key=api_key,
            base_url=api_base,
            system_prompt=SYSTEM_PROMPT_BASIC,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
            **kwargs,
        )

        # OpenAIAgent applies an env-based fallback chain for api_key/base_url.
        # For a mixed-provider suite that is wrong: force the explicit values so
        # OpenAI/Gemini calls are NOT redirected to a Radium base_url that may be
        # present in the environment.
        if api_key is not None:
            self.api_key = api_key
        self.base_url = api_base  # None => LiteLLM uses the provider's native endpoint.

    def _build_user_message(self, input_text: str) -> str:
        # HLE questions already embed multiple-choice options where relevant.
        return input_text

    def _augment_success_metadata(self, raw_output: str, metadata: Dict[str, Any]) -> None:
        answer, confidence = self.extract_answer_confidence(raw_output)
        metadata["extracted_answer"] = answer
        metadata["model_confidence"] = confidence

    @staticmethod
    def extract_answer_confidence(raw_output: str) -> Tuple[str, Optional[float]]:
        """Pull the ``Exact Answer`` and ``Confidence`` fields from a response.

        Falls back to the last non-empty line for the answer when the model does
        not follow the format, and to ``None`` confidence when absent.
        """
        answer = ""
        m = _EXACT_ANSWER_RE.search(raw_output or "")
        if m:
            answer = m.group(1).strip().strip("*").strip()
        else:
            for line in reversed((raw_output or "").splitlines()):
                if line.strip():
                    answer = line.strip()
                    break

        confidence: Optional[float] = None
        c = _CONFIDENCE_RE.search(raw_output or "")
        if c:
            try:
                confidence = max(0.0, min(100.0, float(c.group(1))))
            except ValueError:
                confidence = None
        return answer, confidence
