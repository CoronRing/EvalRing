"""Ingest the ARC-Challenge dataset (``allenai/ai2_arc``) to a local CSV.

ARC-Challenge is a set of grade-school-level multiple-choice science questions —
far lighter on reasoning than HLE — useful for exercising the pipeline quickly.

The output CSV uses the SAME schema the HLE runner consumes, so it can be fed
straight to ``evaluate_hle_main.py --data-path`` / ``run_hle_suite.py --data-path``:
columns ``ID, original_id, question, answer, answer_type, category, has_image``,
where ``question`` embeds the answer choices and ``answer`` is the gold option
label (e.g. ``A``).

``allenai/ai2_arc`` is public (no token required).

Usage::

    python examples/hle/ingest_arc.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = CURRENT_FILE.parent / "data"

CSV_FIELDS = [
    "ID", "original_id", "question", "answer",
    "answer_type", "category", "has_image",
]


def _format_question(question: str, choices: dict) -> str:
    """Render the stem plus labelled answer choices into one prompt string."""
    labels = list(choices.get("label", []))
    texts = list(choices.get("text", []))
    lines = [str(question).strip(), "", "Answer Choices:"]
    for label, text in zip(labels, texts):
        lines.append(f"{label}. {text}")
    return "\n".join(lines)


def ingest(config: str = "ARC-Challenge", split: str = "test") -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed. Run: uv pip install datasets", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "arc_challenge.csv"

    print(f"Loading allenai/ai2_arc config='{config}' split='{split}'...")
    ds = load_dataset("allenai/ai2_arc", config, split=split)
    total = len(ds)
    print(f"Loaded {total} entries. Writing CSV...")

    with open(csv_path, "w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for i in range(total):
            row = ds[i]
            writer.writerow({
                "ID": i,
                "original_id": row.get("id", ""),
                "question": _format_question(row.get("question", ""), row.get("choices", {})),
                "answer": row.get("answerKey", ""),
                "answer_type": "multipleChoice",
                "category": f"ARC-Challenge/{config}",
                "has_image": 0,
            })

    print("=" * 70)
    print(f"Ingest complete. total_entries={total}")
    print(f"  csv: {csv_path}")
    print("  (answer_type=multipleChoice, has_image=0 for all rows)")
    print("=" * 70)


if __name__ == "__main__":
    ingest()
