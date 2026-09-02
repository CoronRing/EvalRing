"""Reusable multi-role host orchestration for classification tasks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .classification import (
    normalize_probability_distribution,
    parse_json_object,
    resolve_classification_prediction,
)


@dataclass
class RoleConfig:
    """Configuration for one specialist role in the multi-role system."""

    name: str
    persona: str
    model_name: str
    temperature: float = 0.0
    max_completion_tokens: int = 400


class MultiRoleHostOrchestrator:
    """Orchestrates role-by-role reviews plus bounded host-led questioning."""

    def __init__(
        self,
        *,
        client: Any,
        labels: list[str],
        task_name: str,
        task_instructions: str,
        host_model_name: str,
        role_configs: list[RoleConfig],
        host_temperature: float = 0.0,
        host_max_completion_tokens: int = 500,
        max_iterations: int = 10,
    ):
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if not role_configs:
            raise ValueError("role_configs cannot be empty")

        self.client = client
        self.labels = labels
        self.task_name = task_name
        self.task_instructions = task_instructions
        self.host_model_name = host_model_name
        self.role_configs = role_configs
        self.role_map = {role.name: role for role in role_configs}
        self.host_temperature = host_temperature
        self.host_max_completion_tokens = host_max_completion_tokens
        self.max_iterations = max_iterations

    def _call_json(
        self,
        *,
        model_name: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_completion_tokens: int,
    ) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            stream=False,
        )
        raw_output = response.choices[0].message.content or ""
        parsed = parse_json_object(raw_output) or {}

        usage = getattr(response, "usage", None)
        token_usage = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
        return {"raw_output": raw_output, "parsed": parsed, "token_usage": token_usage}

    def _one_hot(self, label: str) -> dict[str, float]:
        return {candidate: (1.0 if candidate == label else 0.0) for candidate in self.labels}

    def _normalize_role_judgment(self, parsed: Mapping[str, Any]) -> dict[str, Any]:
        class_scores = parsed.get("class_scores")
        conclusion = str(parsed.get("conclusion", "")).strip()

        if isinstance(class_scores, Mapping):
            prediction = resolve_classification_prediction(class_scores, valid_labels=self.labels)
            label = prediction.label or (self.labels[0] if self.labels else "")
            scores = normalize_probability_distribution(
                {k: float(v) for k, v in prediction.class_scores.items()}
                if prediction.class_scores
                else self._one_hot(label)
            )
        else:
            prediction = resolve_classification_prediction(conclusion, valid_labels=self.labels)
            label = prediction.label or (self.labels[0] if self.labels else "")
            scores = self._one_hot(label)

        return {
            "analysis": str(parsed.get("analysis", "")).strip(),
            "conclusion": label,
            "class_scores": scores,
            "confidence": parsed.get("confidence", None),
        }

    def _aggregate_scores(self, role_reviews: list[dict[str, Any]]) -> dict[str, float]:
        sums = dict.fromkeys(self.labels, 0.0)
        if not role_reviews:
            return sums

        for review in role_reviews:
            scores = review.get("class_scores", {})
            for label in self.labels:
                sums[label] += float(scores.get(label, 0.0))

        return normalize_probability_distribution(sums)

    def run_case(self, input_text: str) -> dict[str, Any]:
        role_reviews: list[dict[str, Any]] = []
        host_dialogue: list[dict[str, Any]] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0

        labels_text = ", ".join(self.labels)

        # 1) Independent role evaluations.
        for role in self.role_configs:
            call = self._call_json(
                model_name=role.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are role: {role.name}. Persona: {role.persona}\n"
                            f"Task: {self.task_name}\n"
                            f"Allowed labels: {labels_text}\n"
                            f"Instructions: {self.task_instructions}\n"
                            "Return ONLY JSON with keys: analysis, conclusion, class_scores, confidence."
                        ),
                    },
                    {"role": "user", "content": f"Case:\n\n{input_text}"},
                ],
                temperature=role.temperature,
                max_completion_tokens=role.max_completion_tokens,
            )
            normalized = self._normalize_role_judgment(call["parsed"])
            role_reviews.append(
                {
                    "role": role.name,
                    "model": role.model_name,
                    "raw_output": call["raw_output"],
                    **normalized,
                }
            )

            usage = call["token_usage"]
            total_prompt_tokens += usage["prompt_tokens"]
            total_completion_tokens += usage["completion_tokens"]
            total_tokens += usage["total_tokens"]

        # 2) Host iterative questioning (bounded).
        for iteration in range(1, self.max_iterations + 1):
            host_call = self._call_json(
                model_name=self.host_model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are the host coordinator for a multi-role classifier.\n"
                            f"Allowed labels: {labels_text}.\n"
                            "Decide either:\n"
                            "1) ask one role one follow-up question, or\n"
                            "2) finalize with a conclusion.\n"
                            "Return ONLY JSON with either:\n"
                            '- {"action":"ask","target_role":"...","question":"..."}\n'
                            "or\n"
                            '- {"action":"finalize","conclusion":"...","class_scores":{...},"rationale":"..."}.'
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Case:\n{input_text}\n\n"
                            f"Role reviews JSON:\n{role_reviews}\n\n"
                            f"Dialogue so far JSON:\n{host_dialogue}\n\n"
                            f"Iteration: {iteration}/{self.max_iterations}"
                        ),
                    },
                ],
                temperature=self.host_temperature,
                max_completion_tokens=self.host_max_completion_tokens,
            )

            usage = host_call["token_usage"]
            total_prompt_tokens += usage["prompt_tokens"]
            total_completion_tokens += usage["completion_tokens"]
            total_tokens += usage["total_tokens"]

            action = str(host_call["parsed"].get("action", "")).strip().lower()
            if action == "finalize":
                final_scores_raw = host_call["parsed"].get("class_scores")
                if isinstance(final_scores_raw, Mapping):
                    normalized_scores = normalize_probability_distribution(
                        {label: float(final_scores_raw.get(label, 0.0)) for label in self.labels}
                    )
                else:
                    label = (
                        resolve_classification_prediction(
                            host_call["parsed"].get("conclusion", ""),
                            valid_labels=self.labels,
                        ).label
                        or self.labels[0]
                    )
                    normalized_scores = self._one_hot(label)

                final_prediction = resolve_classification_prediction(
                    normalized_scores, valid_labels=self.labels
                )
                return {
                    "final_label": final_prediction.label,
                    "class_scores": normalized_scores,
                    "rationale": str(host_call["parsed"].get("rationale", "")).strip(),
                    "iterations_used": iteration,
                    "role_reviews": role_reviews,
                    "host_dialogue": host_dialogue,
                    "token_usage": {
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": total_completion_tokens,
                        "total_tokens": total_tokens,
                    },
                }

            if action != "ask":
                break

            target_role = str(host_call["parsed"].get("target_role", "")).strip()
            question = str(host_call["parsed"].get("question", "")).strip()
            if not question or target_role not in self.role_map:
                break

            target_cfg = self.role_map[target_role]
            role_reply = self._call_json(
                model_name=target_cfg.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are role: {target_cfg.name}. Persona: {target_cfg.persona}\n"
                            f"Allowed labels: {labels_text}\n"
                            "Answer host questions. Return ONLY JSON with keys: answer, updated_conclusion, class_scores."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Case:\n{input_text}\n\n"
                            f"Host question:\n{question}\n\n"
                            f"Your original review context:\n{role_reviews}"
                        ),
                    },
                ],
                temperature=target_cfg.temperature,
                max_completion_tokens=target_cfg.max_completion_tokens,
            )

            usage = role_reply["token_usage"]
            total_prompt_tokens += usage["prompt_tokens"]
            total_completion_tokens += usage["completion_tokens"]
            total_tokens += usage["total_tokens"]

            host_dialogue.append(
                {
                    "iteration": iteration,
                    "target_role": target_role,
                    "question": question,
                    "answer": role_reply["parsed"],
                    "raw_answer": role_reply["raw_output"],
                }
            )

        # 3) Fallback finalize if host never finalized.
        fallback_scores = self._aggregate_scores(role_reviews)
        fallback_prediction = resolve_classification_prediction(
            fallback_scores, valid_labels=self.labels
        )
        return {
            "final_label": fallback_prediction.label,
            "class_scores": fallback_scores,
            "rationale": "Fallback finalize from averaged role distributions.",
            "iterations_used": self.max_iterations,
            "role_reviews": role_reviews,
            "host_dialogue": host_dialogue,
            "token_usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
            },
        }
