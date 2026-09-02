"""De-duplicate rsd_15k.csv by full text content.

- Keeps the first occurrence of each unique text.
- Preserves original row order and existing ID values (no reindexing).
- Does not assume any particular index exists.

Default output name follows user request: res_15k_no_duplication.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def _normalize_text(text: str) -> str:
    # Normalize only enough to catch trivial line-ending differences.
    return text.replace("\r\n", "\n").strip()


def _text_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def dedupe_csv_by_text(*, in_path: Path, out_path: Path, text_field: str = "text") -> tuple[int, int, int]:
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    total_rows = 0
    kept_rows = 0

    with in_path.open("r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header")
        if text_field not in reader.fieldnames:
            raise ValueError(f"Missing required column '{text_field}'. Found: {reader.fieldnames}")

        with out_path.open("w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
            writer.writeheader()

            for row in reader:
                total_rows += 1
                raw_text = row.get(text_field) or ""
                norm = _normalize_text(str(raw_text))
                fp = _text_fingerprint(norm)

                if fp in seen:
                    continue
                seen.add(fp)

                writer.writerow(row)
                kept_rows += 1

    removed_rows = total_rows - kept_rows
    return total_rows, kept_rows, removed_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=Path(__file__).parent / "data" / "rsd_15k.csv",
        help="Input CSV path (default: data/rsd_15k.csv)",
    )
    ap.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        default=Path(__file__).parent / "data" / "res_15k_no_duplication.csv",
        help="Output CSV path (default: data/res_15k_no_duplication.csv)",
    )
    ap.add_argument(
        "--text-field",
        dest="text_field",
        type=str,
        default="text",
        help="Column name containing text (default: text)",
    )

    args = ap.parse_args()

    total, kept, removed = dedupe_csv_by_text(in_path=args.in_path, out_path=args.out_path, text_field=args.text_field)
    print(f"input_rows={total}")
    print(f"output_rows={kept}")
    print(f"removed_duplicates={removed}")
    print(f"output_path={args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
