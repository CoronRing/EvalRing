"""
Convenience wrapper to generate the model list for the suicide detection suite.
"""

import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent

from EvalRing.utils.generate_model_list import generate_model_list

if __name__ == "__main__":
    out_path = script_dir / "model_list.json"
    requested_mapping = {
        "GPT-5 mini": ("openai/gpt-5-mini", 2024),
        "Clarke 1.0": ("clarke-1.0", 2026),
        "Tycho 1.0": ("tycho-1.0", 2026),
        "HAL 1.0": ("hal-1.0", 2026),
    }
    generate_model_list(out_path, requested_mapping)
