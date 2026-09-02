"""Add a deterministic per-row ID column to rsd_15k.csv.

Why this exists:
- The RSD_15K dataset includes a `users` column that is *not unique* (one user can post multiple times).
- EvalRing requires a unique per-sample ID for correct resume/merge/retry behavior.

This script prepends an `ID` column (0-based row index) as the *first* column.

Usage:
  python add_id_column.py --in data/rsd_15k.csv

Notes:
- If an `ID` column already exists as the first column, this is a no-op.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def add_id_column_inplace(csv_path: Path) -> dict:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.reader(fin)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"Empty CSV: {csv_path}")

        if header and header[0] == "ID":
            return {"status": "noop", "reason": "ID column already present", "path": str(csv_path)}

        new_header = ["ID", *header]

        with open(tmp_path, "w", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(new_header)

            row_count = 0
            for i, row in enumerate(reader):
                writer.writerow([i, *row])
                row_count += 1

    tmp_path.replace(csv_path)
    return {"status": "updated", "rows": row_count, "path": str(csv_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepend an ID column to a CSV (in-place).")
    parser.add_argument("--in", dest="input_csv", type=Path, required=True, help="Path to CSV to update in-place")
    args = parser.parse_args()

    result = add_id_column_inplace(args.input_csv)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
