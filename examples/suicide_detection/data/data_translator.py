"""Create filtered suicide-detection datasets by user post frequency.

Outputs (in addition to the original dataset):
- *_simple.csv: users that occur exactly once
- *_multi_round.csv: users that occur more than once
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List


def _detect_user_column(fieldnames: List[str], user_column: str | None = None) -> str:
    if user_column:
        if user_column not in fieldnames:
            raise ValueError(f"Requested user column '{user_column}' not found. Available columns: {fieldnames}")
        return user_column

    if "users" in fieldnames:
        return "users"

    for name in fieldnames:
        if name.lstrip("\ufeff") == "users":
            return name

    raise ValueError(
        "Could not detect user column automatically. Pass --user-column explicitly. "
        f"Available columns: {fieldnames}"
    )


def _write_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_dataset_by_user_frequency(
    input_csv: Path,
    *,
    user_column: str | None = None,
    output_simple: Path | None = None,
    output_multi_round: Path | None = None,
    dry_run: bool = False,
) -> dict:
    """Split dataset into single-post and multi-post user subsets."""
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Input CSV has no header: {input_csv}")

        fieldnames = list(reader.fieldnames)
        user_key = _detect_user_column(fieldnames, user_column=user_column)
        rows = list(reader)

    user_counts = Counter(row.get(user_key, "") for row in rows)

    simple_rows = [row for row in rows if user_counts.get(row.get(user_key, ""), 0) == 1]
    multi_rows = [row for row in rows if user_counts.get(row.get(user_key, ""), 0) > 1]

    stem = input_csv.stem
    output_simple = output_simple or input_csv.with_name(f"{stem}_simple.csv")
    output_multi_round = output_multi_round or input_csv.with_name(f"{stem}_multi_round.csv")

    if not dry_run:
        _write_rows(output_simple, fieldnames, simple_rows)
        _write_rows(output_multi_round, fieldnames, multi_rows)

    unique_users = len(user_counts)
    single_users = sum(1 for count in user_counts.values() if count == 1)
    multi_users = sum(1 for count in user_counts.values() if count > 1)

    return {
        "input_csv": str(input_csv),
        "user_column": user_key,
        "total_rows": len(rows),
        "unique_users": unique_users,
        "single_post_users": single_users,
        "multi_post_users": multi_users,
        "simple_rows": len(simple_rows),
        "multi_round_rows": len(multi_rows),
        "output_simple": str(output_simple),
        "output_multi_round": str(output_multi_round),
        "dry_run": dry_run,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split suicide detection CSV into *_simple and *_multi_round datasets by user occurrence count."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "rsd_15k.csv",
        help="Path to source CSV (default: data/rsd_15k.csv)",
    )
    parser.add_argument(
        "--user-column",
        type=str,
        default=None,
        help="User identifier column. If omitted, auto-detects 'users'.",
    )
    parser.add_argument(
        "--output-simple",
        type=Path,
        default=None,
        help="Optional explicit output path for *_simple CSV.",
    )
    parser.add_argument(
        "--output-multi-round",
        type=Path,
        default=None,
        help="Optional explicit output path for *_multi_round CSV.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print stats only; do not write files.",
    )
    args = parser.parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    summary = split_dataset_by_user_frequency(
        args.input_csv,
        user_column=args.user_column,
        output_simple=args.output_simple,
        output_multi_round=args.output_multi_round,
        dry_run=args.dry_run,
    )

    print("Dataset translation summary")
    print("=" * 60)
    for key, value in summary.items():
        print(f"{key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
