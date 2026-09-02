from typing import Any, Dict, Optional
import os
import json
import time
from EvalRing.agent import OpenAIAgent
from EvalRing.agent.classification import (
    aggregate_base_vs_rest_probabilities,
    parse_json_object,
)
from EvalRing.agent.multi_role import MultiRoleHostOrchestrator, RoleConfig
from EvalRing.agent.base import AgentResponse


class OpenAISuicideDetectionAgent(OpenAIAgent):
    """
    Suicide Risk Classification agent using OpenAI.
    Classifies text into: Ideation, Behavior, Indicator, Attempt.

    Subclasses OpenAIAgent — inherits streaming, TTFT/TPS tracking,
    and token counting.  Only the system prompt and output parsing
    are domain-specific.
    """

    SYSTEM_PROMPT_SINGLE_CLASS = """Classify the post into exactly ONE class: Indicator, Ideation, Behavior, or Attempt.

Use these class definitions:

**Indicator**
- The post does NOT indicate the author's own suicidal risk.
- Includes: discussion of another person's suicidality, general discussion/news, or explicit denial of own suicidal intent.
- Also includes concern for someone else at risk.

**Ideation**
- The author expresses suicidal thoughts, wishes, or desire to die/end life.
- Includes passive or active suicidal thinking, including hypothetical or unrealistic methods.
- No concrete self-harm/suicide action has been taken in the post.

**Behavior**
- The author describes preparatory or self-harm behavior beyond verbal thoughts.
- Includes planning/preparation actions (for example, gathering means, writing notes, preparing for death).
- Includes non-suicidal self-harm when self-inflicted harm is described, even without explicit intent to die.

**Attempt**
- The post mentions a suicide attempt (current or past).
- A past attempt should still be labeled Attempt, even if the current post does not express ongoing suicidal thoughts.

Decision rules:
- If the post contains mixed signals, choose the class supported by the clearest explicit evidence.

Output format:
- Return ONLY one word: Indicator, Ideation, Behavior, or Attempt."""

    SYSTEM_PROMPT_MULTI_CLASS_CHANCE = """You are a suicide-risk classification assistant.

Classify the post using these 4 classes:
- Indicator
- Ideation
- Behavior
- Attempt

Use the same definitions and decision rules as the single-label setting:
- Indicator: no self-risk signal from the author.
- Ideation: suicidal thoughts/wishes without concrete self-harm action.
- Behavior: preparatory/self-harm behavior beyond verbal thoughts.
- Attempt: explicit mention of suicide attempt (current or past).

Output format requirements:
1) Return ONLY a JSON object.
2) Include ALL 4 keys: "ideation", "behavior", "indicator", "attempt".
3) Values must be numbers in [0, 1].
4) The values should sum to 1.0 (or very close).
5) No extra text, no markdown, no explanation.

Example:
{"ideation": 0.80, "behavior": 0.10, "indicator": 0.05, "attempt": 0.05}"""



    VALID_LABELS = ["Ideation", "Behavior", "Indicator", "Attempt"]
    MODE_SINGLE_CLASS = "single-class"
    MODE_MULTI_CLASS_CHANCE = "multi-class-chance"
    MODE_BASE_VS_REST_BINARY = "base-vs-rest-binary"
    MODE_MULTI_AGENT_HOST = "multi-agent-host"
    MODE_PER_CLASS_SCORE = "per-class-score"
    VALID_MODES = {
        MODE_SINGLE_CLASS,
        MODE_MULTI_CLASS_CHANCE,
        MODE_BASE_VS_REST_BINARY,
        MODE_MULTI_AGENT_HOST,
        MODE_PER_CLASS_SCORE,
    }
    LABEL_ALIASES = {
        "ideation": "Ideation",
        "behavior": "Behavior",
        "indicator": "Indicator",
        "attempt": "Attempt",
    }
    DEFAULT_ROLE_PERSONAS = {
        "internet_advisor": "Focus on online context clues, posting behavior patterns, and potential environmental triggers.",
        "mental_health_practitioner": "Use clinical risk reasoning and prioritize safety-relevant evidence in the text.",
        "language_specialist": "Analyze wording, intensity, ambiguity, and discourse cues in the user's language.",
    }

    def __init__(
        self,
        name: str = "gpt-5-mini-suicide-detector",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        agent_mode: str = MODE_SINGLE_CLASS,
        base_class: str = "Indicator",
        host_model_name: Optional[str] = None,
        role_model_map: Optional[Dict[str, str]] = None,
        max_host_iterations: int = 10,
        **kwargs,
    ):
        if agent_mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported agent_mode='{agent_mode}'. Valid modes: {sorted(self.VALID_MODES)}")

        if model_name is None:
            if os.environ.get("RADIUM_API_KEY"):
                model_name = os.environ.get("RADIUM_MODEL", "hal-1.0")
            elif os.environ.get("OPENAI_API_KEY"):
                model_name = os.environ.get("OPENAI_MODEL", "gpt-5.2")
            elif os.environ.get("OPEN_ROUTER_KEY"):
                model_name = os.environ.get("OPEN_ROUTER_MODEL", "openai/gpt-4o-mini")
            else:
                model_name = os.environ.get("OPENAI_MODEL", "gpt-5.2")

        self.agent_mode = agent_mode
        self.base_class = self.LABEL_ALIASES.get(base_class.lower(), base_class)
        if self.base_class not in self.VALID_LABELS:
            raise ValueError(f"Unsupported base_class='{base_class}'. Valid classes: {self.VALID_LABELS}")
        self.max_host_iterations = max(1, min(int(max_host_iterations), 10))

        env_role_model_map = None
        role_map_json = os.environ.get("SUICIDE_ROLE_MODELS_JSON", "").strip()
        if role_map_json:
            try:
                parsed = json.loads(role_map_json)
                if isinstance(parsed, dict):
                    env_role_model_map = {str(k): str(v) for k, v in parsed.items()}
            except Exception:
                env_role_model_map = None

        self.role_model_map = role_model_map or env_role_model_map or {}
        self.host_model_name = host_model_name or os.environ.get("SUICIDE_HOST_MODEL") or model_name

        if agent_mode == self.MODE_MULTI_CLASS_CHANCE:
            system_prompt = self.SYSTEM_PROMPT_MULTI_CLASS_CHANCE
        else:
            system_prompt = self.SYSTEM_PROMPT_SINGLE_CLASS

        super().__init__(
            name=name,
            model_name=model_name,
            api_key=api_key,
            system_prompt=system_prompt,
            temperature=temperature,
            max_completion_tokens=500,  # reasoning models (e.g. gpt-5-mini) consume tokens internally before outputting; 500 covers reasoning budget + label output
            **kwargs,
        )

    def _build_user_message(self, input_text: str) -> str:
        return f"Text to classify:\n\n{input_text}"

    def _parse_single_label(self, raw_output: str) -> str:
        clean = raw_output.strip('\'". \n').lower()
        for label in self.VALID_LABELS:
            if label.lower() in clean:
                return label
        raise ValueError(f"Could not parse a valid label from output: {raw_output}")

    def _parse_multi_class_chance(self, raw_output: str) -> Dict[str, float]:
        parsed = parse_json_object(raw_output)
        if not parsed:
            # Throw an error if JSON is malformed instead of falling back
            raise ValueError(f"Could not parse valid JSON probabilities from output: {raw_output}")

        scores: Dict[str, float] = {}
        for key, value in parsed.items():
            canonical = self.LABEL_ALIASES.get(str(key).strip().lower())
            if not canonical:
                continue
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            scores[canonical] = max(0.0, score)

        # Ensure all expected labels exist and normalize to a probability distribution.
        for label in self.VALID_LABELS:
            scores.setdefault(label, 0.0)

        total = sum(scores.values())
        if total <= 0:
            raise ValueError(f"Extracted probabilities sum to zero for output: {raw_output}")

        return {label: scores[label] / total for label in self.VALID_LABELS}

    def _parse_output(self, raw_output: str) -> Any:
        """Parse output based on the selected run mode."""
        if self.agent_mode == self.MODE_MULTI_CLASS_CHANCE:
            return self._parse_multi_class_chance(raw_output)
        return self._parse_single_label(raw_output)

    def _per_class_prompt(self, target_class: str) -> str:
        target_key = target_class.lower()
        return (
            "You are a suicide-risk classification assistant.\n\n"
            f"You are being asked to evaluate whether the post matches the class: {target_class}\n"
            "Use the same definitions as the single-label setting:\n"
            "- Indicator: no self-risk signal from the author.\n"
            "- Ideation: suicidal thoughts/wishes without concrete self-harm action.\n"
            "- Behavior: preparatory/self-harm behavior beyond verbal thoughts.\n"
            "- Attempt: explicit mention of suicide attempt (current or past).\n\n"
            "Output format requirements:\n"
            "1) Return ONLY a JSON object.\n"
            f'2) Include exactly one key: "{target_key}".\n'
            f"3) The value must be a number from 1 to 10 based on how well the post matches {target_class} (e.g. 10 means a confirmed and complete match, 6 means reasonable, 1 means no match).\n"
            "4) No extra text, no markdown, no explanation.\n\n"
            "Example:\n"
            f'{{"{target_key}": 8}}'
        )

    def _build_multi_role_configs(self) -> list[RoleConfig]:
        role_configs: list[RoleConfig] = []
        for role_name, persona in self.DEFAULT_ROLE_PERSONAS.items():
            role_configs.append(
                RoleConfig(
                    name=role_name,
                    persona=persona,
                    model_name=self.role_model_map.get(role_name, self.model_name),
                    temperature=self.temperature,
                    max_completion_tokens=self.max_completion_tokens,
                )
            )
        return role_configs

    def _binary_prompt_tournament(self, class_a: str, class_b: str) -> str:
        return (
            "You are a suicide-risk classifier.\n"
            f"Compare the two classes: '{class_a}' vs '{class_b}' for the given post. Which class is fundamentally more accurate?\n"
            "Output exactly ONE word, which is the exact name of the winning class.\n"
            f"Do not output anything else. Your output MUST be either '{class_a}' or '{class_b}'."
        )


    def predict(self, input_text: str, **kwargs) -> AgentResponse:
        if self.agent_mode == self.MODE_MULTI_AGENT_HOST:
            if not self._is_initialized:
                raise RuntimeError("Agent not initialized.  Call initialize() first.")

            if not self.validate_input(input_text):
                return AgentResponse(
                    input_id="",
                    input_text=input_text,
                    output=None,
                    error="Invalid input",
                )

            start_time = time.time()
            try:
                orchestrator = MultiRoleHostOrchestrator(
                    client=self.client,
                    labels=self.VALID_LABELS,
                    task_name="suicide_risk_classification",
                    task_instructions=(
                        "Classify into exactly one of: Indicator, Ideation, Behavior, Attempt. "
                        "Use evidence from the text only. Attempt indicates current or past suicide attempt."
                    ),
                    host_model_name=self.host_model_name,
                    role_configs=self._build_multi_role_configs(),
                    host_temperature=self.temperature,
                    host_max_completion_tokens=self.max_completion_tokens,
                    max_iterations=self.max_host_iterations,
                )
                run = orchestrator.run_case(input_text)

                total_time = time.time() - start_time
                token_usage = run.get("token_usage", {})
                class_scores = run.get(
                    "class_scores",
                    {label: (1.0 if label == "Indicator" else 0.0) for label in self.VALID_LABELS},
                )
                return AgentResponse(
                    input_id="",
                    input_text=input_text,
                    output=class_scores,
                    confidence=max(class_scores.values()) if class_scores else None,
                    processing_time=total_time,
                    metadata={
                        "raw_output": {
                            "role_reviews": run.get("role_reviews", []),
                            "host_dialogue": run.get("host_dialogue", []),
                        },
                        "model": self.model_name,
                        "host_model": self.host_model_name,
                        "role_model_map": self.role_model_map,
                        "base_url": self.base_url,
                        "prompt_tokens": token_usage.get("prompt_tokens", 0),
                        "completion_tokens": token_usage.get("completion_tokens", 0),
                        "total_tokens": token_usage.get("total_tokens", 0),
                        "ttft": 0.0,
                        "generation_time": total_time,
                        "tps": (token_usage.get("completion_tokens", 0) / total_time) if total_time > 0 else 0.0,
                        "total_time": total_time,
                        "host_iterations_used": run.get("iterations_used", 0),
                        "host_max_iterations": self.max_host_iterations,
                        "host_rationale": run.get("rationale", ""),
                    },
                )
            except Exception as e:
                processing_time = time.time() - start_time
                return AgentResponse(
                    input_id="",
                    input_text=input_text,
                    output="Error",
                    error=str(e),
                    processing_time=processing_time,
                )

        if self.agent_mode not in (self.MODE_BASE_VS_REST_BINARY, self.MODE_PER_CLASS_SCORE):
            return super().predict(input_text, **kwargs)

        if not self._is_initialized:
            raise RuntimeError("Agent not initialized.  Call initialize() first.")

        if not self.validate_input(input_text):
            return AgentResponse(
                input_id="",
                input_text=input_text,
                output=None,
                error="Invalid input",
            )

        start_time = time.time()
        prompt_tokens_total = 0
        completion_tokens_total = 0
        total_tokens_total = 0
        
        if self.agent_mode == self.MODE_PER_CLASS_SCORE:
            class_scores: Dict[str, float] = {}
            per_class_details = []
            
            try:
                for target_class in self.VALID_LABELS:
                    raw_response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": self._per_class_prompt(target_class)},
                            {"role": "user", "content": self._build_user_message(input_text)},
                        ],
                        temperature=self.temperature,
                        max_completion_tokens=self.max_completion_tokens,
                        stream=False,
                    )
                    raw_output = raw_response.choices[0].message.content or ""
                    
                    parsed = parse_json_object(raw_output)
                    target_key = target_class.lower()
                    score = 1.0
                    if isinstance(parsed, dict):
                        raw_score = parsed.get(target_key, parsed.get(target_class))
                        if raw_score is not None:
                            try:
                                score = float(raw_score)
                            except (ValueError, TypeError):
                                pass
                    
                    class_scores[target_class] = max(1.0, min(10.0, score))
                    
                    usage = getattr(raw_response, "usage", None)
                    if usage is not None:
                        prompt_tokens_total += getattr(usage, "prompt_tokens", 0) or 0
                        completion_tokens_total += getattr(usage, "completion_tokens", 0) or 0
                        total_tokens_total += getattr(usage, "total_tokens", 0) or 0

                    per_class_details.append(
                        {
                            "target_class": target_class,
                            "raw_output": raw_output,
                            "score": score,
                        }
                    )
                
                total_time = time.time() - start_time
                return AgentResponse(
                    input_id="",
                    input_text=input_text,
                    output=class_scores,
                    confidence=max(class_scores.values()) if class_scores else None,
                    processing_time=total_time,
                    metadata={
                        "raw_output": per_class_details,
                        "model": self.model_name,
                        "base_url": self.base_url,
                        "prompt_tokens": prompt_tokens_total,
                        "completion_tokens": completion_tokens_total,
                        "total_tokens": total_tokens_total,
                        "ttft": 0.0,
                        "generation_time": total_time,
                        "tps": (completion_tokens_total / total_time) if total_time > 0 else 0.0,
                        "total_time": total_time,
                    },
                )
            except Exception as e:
                processing_time = time.time() - start_time
                return AgentResponse(
                    input_id="",
                    input_text=input_text,
                    output="Error",
                    error=str(e),
                    processing_time=processing_time,
                )

        pairwise_details = []

        try:
            current_winner = self.base_class
            targets = [label for label in self.VALID_LABELS if label != self.base_class]
            
            for target_class in targets:
                raw_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self._binary_prompt_tournament(current_winner, target_class)},
                        {"role": "user", "content": self._build_user_message(input_text)},
                    ],
                    temperature=self.temperature,
                    max_completion_tokens=self.max_completion_tokens,
                    stream=False,
                )

                raw_output = raw_response.choices[0].message.content or ""
                chosen = target_class if target_class.lower() in raw_output.lower() else current_winner

                usage = getattr(raw_response, "usage", None)
                if usage is not None:
                    prompt_tokens_total += getattr(usage, "prompt_tokens", 0) or 0
                    completion_tokens_total += getattr(usage, "completion_tokens", 0) or 0
                    total_tokens_total += getattr(usage, "total_tokens", 0) or 0

                pairwise_details.append(
                    {
                        "matchup": f"{current_winner} vs {target_class}",
                        "raw_output": raw_output,
                        "winner": chosen,
                    }
                )
                
                current_winner = chosen

            class_scores = {label: 1.0 if label == current_winner else 0.0 for label in self.VALID_LABELS}

            total_time = time.time() - start_time
            return AgentResponse(
                input_id="",
                input_text=input_text,
                output=class_scores,
                confidence=1.0,
                processing_time=total_time,
                metadata={
                    "raw_output": pairwise_details,
                    "model": self.model_name,
                    "base_url": self.base_url,
                    "prompt_tokens": prompt_tokens_total,
                    "completion_tokens": completion_tokens_total,
                    "total_tokens": total_tokens_total,
                    "ttft": 0.0,
                    "generation_time": total_time,
                    "tps": (completion_tokens_total / total_time) if total_time > 0 else 0.0,
                    "total_time": total_time,
                    "base_class": self.base_class,
                },
            )
        except Exception as e:
            processing_time = time.time() - start_time
            return AgentResponse(
                input_id="",
                input_text=input_text,
                output="Error",
                error=str(e),
                processing_time=processing_time,
            )
