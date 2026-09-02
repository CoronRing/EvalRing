"""
Utility to fetch and generate a list of models and their pricing from OpenRouter.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import requests

from ..logging_utils import get_logger

logger = get_logger(__name__)


MANUAL_MODELS = {
    "Clarke 1.0": ("clarke-1.0", 2026),
    "Tycho 1.0": ("tycho-1.0", 2026),
    "HAL 1.0": ("hal-1.0", 2026),
}


def _manual_model_ids() -> set[str]:
    return {model_id for model_id, _ in MANUAL_MODELS.values()}


def get_openrouter_models() -> list[dict]:
    """Fetch model data from OpenRouter API."""
    url = "https://openrouter.ai/api/v1/models"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.json().get("data", [])
    except Exception as e:
        logger.error("Failed to fetch OpenRouter models: %s", e)
        return []


def generate_model_list(output_path: Path, requested_mapping: dict | None = None) -> None:
    """
    Generate model config JSON used by suite_runner.

    Args:
        output_path: Target path for the output JSON.
        requested_mapping: Optional custom mapping configuration.
    """
    if requested_mapping is None:
        requested_mapping = {
            "Gemini-3.2 pro": ("google/gemini-3.1-pro-preview", 2025),
            "Grok-4": ("x-ai/grok-4.1-fast", 2025),
            "Sonnect-4.6": ("anthropic/claude-sonnet-4.6", 2025),
            "GLM5": ("z-ai/glm-5", 2025),
            "GPT-4o": ("openai/gpt-4o", 2024),
            "Gemini-3 Flash": ("google/gemini-3-flash-preview", 2024),
            "Qwen3.5": ("qwen/qwen3.5-plus-02-15", 2024),
            "GPT-3.5 turbo": ("openai/gpt-3.5-turbo", 2023),
            "Llama-3": ("meta-llama/llama-3.3-70b-instruct", 2023),
            "KIMI": ("moonshotai/kimi-k2.5", None),
            "Minimax": ("minimax/minimax-m2.5", None),
            "DeepSeek V3.2": ("deepseek/deepseek-v3.2", 2024),
            "GPT-5 mini": ("openai/gpt-5-mini", 2024),
            **MANUAL_MODELS,
        }

    models_data = get_openrouter_models()
    models_by_id = {m["id"]: m for m in models_data}
    manual_model_ids = _manual_model_ids()

    output_list = []

    for req_name, (target_id, year) in requested_mapping.items():
        m = models_by_id.get(target_id)
        if m:
            prompt_price = float(m.get("pricing", {}).get("prompt", 0))
            completion_price = float(m.get("pricing", {}).get("completion", 0))
            total_cost_1m = (prompt_price + completion_price) * 1_000_000

            output_list.append(
                {
                    "requested": req_name,
                    "year": year,
                    "available": True,
                    "exact_match": target_id == m["id"] and req_name != "Gemini-3.2 pro",
                    "openrouter_id": target_id,
                    "openrouter_name": m.get("name"),
                    "pricing": {
                        "prompt_usd_per_token": prompt_price,
                        "completion_usd_per_token": completion_price,
                        "total_usd_per_1m_tokens": round(total_cost_1m, 2),
                    },
                    "note": "Exact preferred id 'google/gemini-3.2-pro' not found, using closest available."
                    if req_name == "Gemini-3.2 pro"
                    else None,
                }
            )
        elif target_id in manual_model_ids:
            output_list.append(
                {
                    "requested": req_name,
                    "year": year,
                    "available": True,
                    "exact_match": True,
                    "openrouter_id": target_id,
                    "openrouter_name": None,
                    "pricing": None,
                    "note": "LiteLLM/Radium model; pricing is not sourced from OpenRouter.",
                }
            )
        else:
            output_list.append(
                {
                    "requested": req_name,
                    "year": year,
                    "available": False,
                    "exact_match": False,
                    "openrouter_id": target_id,
                    "openrouter_name": None,
                    "pricing": None,
                    "note": "Model ID not found on OpenRouter.",
                }
            )

    # Sort by cost
    def sort_key(x):
        if x["pricing"]:
            return x["pricing"]["total_usd_per_1m_tokens"]
        return float("inf")

    output_list.sort(key=sort_key)

    final_output = {
        "generated_at_utc": datetime.utcnow().isoformat() + "+00:00",
        "source": "https://openrouter.ai/api/v1/models",
        "sort_key": "pricing.total_usd_per_1m_tokens (ascending)",
        "models": output_list,
    }

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2)

        logger.info("Generated model list at %s (%d models)", output_path, len(output_list))
    except Exception as e:
        logger.error("Failed to save generated model list to %s: %s", output_path, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=str, default="model_list.json", help="Output path for the json list"
    )
    args = parser.parse_args()

    generate_model_list(Path(args.out).resolve())
