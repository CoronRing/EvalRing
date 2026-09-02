"""Ingest GPQA (``Idavidrein/gpqa``) to a local CSV in the shared schema.

GPQA is a graduate-level, "Google-proof" multiple-choice QA benchmark (much
harder than ARC). Each row stores a correct answer plus three incorrect answers
rather than a pre-formatted MC item, so this script assembles the four options,
shuffles them **deterministically per record** (reproducible answer key), and
writes the shared CSV schema consumed by the runner via ``--data-path``:
``ID, original_id, question, answer, answer_type, category, has_image``.

GPQA is **gated** — accept the terms on the dataset page and set ``HF_TOKEN`` in
``.env``.

Usage::

    python examples/hle/ingest_gpqa.py            # gpqa_main (448)
    python examples/hle/ingest_gpqa.py --config gpqa_diamond
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = CURRENT_FILE.parent / "data"
LETTERS = ["A", "B", "C", "D"]
CSV_FIELDS = ["ID", "original_id", "question", "answer", "answer_type", "category", "has_image"]


def _clean(v) -> str:
    return "" if v is None else str(v).strip()


def _format_mc(question: str, correct: str, incorrects: list[str], seed_key: str) -> tuple[str, str]:
    """Assemble a shuffled A–D question. Returns (question_text, correct_label)."""
    options = [(correct, True)] + [(i, False) for i in incorrects]
    # Deterministic per-record shuffle so the answer key is stable across runs.
    random.Random(seed_key).shuffle(options)
    lines = [question.strip(), "", "Answer Choices:"]
    correct_label = "A"
    for letter, (text, is_correct) in zip(LETTERS, options):
        lines.append(f"{letter}. {text}")
        if is_correct:
            correct_label = letter
    return "\n".join(lines), correct_label


def ingest(config: str = "gpqa_main", split: str = "train") -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("ERROR: No HuggingFace token found. Set HF_TOKEN in .env (GPQA is gated).", file=sys.stderr)
        sys.exit(1)
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed. Run: uv pip install datasets", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / f"{config}.csv"

    print(f"Loading Idavidrein/gpqa config='{config}' split='{split}'...")
    ds = load_dataset("Idavidrein/gpqa", config, split=split, token=token)
    total = len(ds)
    print(f"Loaded {total} entries. Writing CSV...")

    with open(csv_path, "w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for i in range(total):
            row = ds[i]
            record_id = _clean(row.get("Record ID")) or str(i)
            incorrects = [_clean(row.get(f"Incorrect Answer {n}")) for n in (1, 2, 3)]
            question, answer = _format_mc(
                _clean(row.get("Question")), _clean(row.get("Correct Answer")), incorrects, record_id
            )
            writer.writerow({
                "ID": i,
                "original_id": record_id,
                "question": question,
                "answer": answer,
                "answer_type": "multipleChoice",
                "category": _clean(row.get("High-level domain")) or "GPQA",
                "has_image": 0,
            })

    print("=" * 70)
    print(f"Ingest complete. total_entries={total}")
    print(f"  csv: {csv_path}")
    print("  (answer=A/B/C/D, deterministic per-record shuffle, has_image=0)")
    print("=" * 70)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ingest GPQA into the shared CSV schema.")
    p.add_argument("--config", default="gpqa_main",
                   choices=["gpqa_main", "gpqa_diamond", "gpqa_extended", "gpqa_experts"])
    p.add_argument("--split", default="train")
    args = p.parse_args()
    ingest(args.config, args.split)
