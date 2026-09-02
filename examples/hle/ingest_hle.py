"""Ingest the full ``cais/hle`` (Humanity's Last Exam) dataset to local files.

Downloads every entry of the HLE ``test`` split from HuggingFace and writes two
artifacts under ``examples/hle/data``:

- ``hle_full.jsonl``: one JSON object per row, full fidelity (including the
  ``image`` data-URI when present). This is the lossless archive.
- ``hle.csv``: text-usable columns only (no image bytes), consumable directly by
  :class:`EvalRing.dataset.CSVDataset` with ``text_field="question"``,
  ``label_field="answer"`` and ``id_field="ID"``.

``cais/hle`` is a gated dataset. Access requires accepting the terms on
https://huggingface.co/datasets/cais/hle and providing an HF token via the
``HF_TOKEN`` (or ``HUGGING_FACE_HUB_TOKEN``) environment variable / ``.env``.

Usage::

    python examples/hle/ingest_hle.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()

CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = CURRENT_FILE.parent / "data"

# Columns preserved in the flat CSV (image bytes deliberately excluded).
CSV_FIELDS = [
    "ID",
    "original_id",
    "question",
    "answer",
    "answer_type",
    "category",
    "raw_subject",
    "author_name",
    "has_image",
]


def _clean(value: Any) -> str:
    """Coerce a field to a clean string, mapping None-ish values to ""."""
    if value is None:
        return ""
    s = str(value)
    if s.strip().lower() in {"none", "nan"}:
        return ""
    return s


def ingest(dataset_name: str = "cais/hle", split: str = "test") -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            "ERROR: No HuggingFace token found. Set HF_TOKEN in .env (cais/hle is gated).",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed. Run: uv pip install datasets", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = DATA_DIR / "hle_full.jsonl"
    csv_path = DATA_DIR / "hle.csv"

    print(f"Loading {dataset_name} split='{split}' (this downloads and caches parquet shards)...")
    ds = load_dataset(dataset_name, split=split, token=token)
    total = len(ds)
    print(f"Loaded {total} entries. Writing archive + CSV...")

    answer_types: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    image_count = 0

    with open(jsonl_path, "w", encoding="utf-8") as jf, open(
        csv_path, "w", encoding="utf-8", newline=""
    ) as cf:
        writer = csv.DictWriter(cf, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for row_idx in range(total):
            row = ds[row_idx]
            image_uri = _clean(row.get("image"))
            has_image = bool(image_uri.strip())
            if has_image:
                image_count += 1

            answer_type = _clean(row.get("answer_type"))
            category = _clean(row.get("category"))
            answer_types[answer_type or "unknown"] += 1
            categories[category or "unknown"] += 1

            # Full-fidelity archive record (image_preview is a non-serialisable PIL
            # object and is intentionally omitted; the raw ``image`` data-URI is kept).
            record: Dict[str, Any] = {
                "ID": row_idx,
                "original_id": _clean(row.get("id")),
                "question": _clean(row.get("question")),
                "answer": _clean(row.get("answer")),
                "answer_type": answer_type,
                "category": category,
                "raw_subject": _clean(row.get("raw_subject")),
                "author_name": _clean(row.get("author_name")),
                "rationale": _clean(row.get("rationale")),
                "has_image": has_image,
                "image": image_uri,
            }
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")

            writer.writerow({k: (record[k] if k != "has_image" else int(has_image)) for k in CSV_FIELDS})

    print("=" * 70)
    print(f"Ingest complete. total_entries={total}")
    print(f"  archive (full):     {jsonl_path}")
    print(f"  flat csv (no imgs): {csv_path}")
    print(f"  entries with image: {image_count} ({image_count / max(1, total) * 100:.1f}%)")
    print(f"  text-only entries:  {total - image_count}")
    print("  answer_type distribution:")
    for k, v in answer_types.most_common():
        print(f"    {k}: {v}")
    print("  top categories:")
    for k, v in categories.most_common(12):
        print(f"    {k}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    ingest()
