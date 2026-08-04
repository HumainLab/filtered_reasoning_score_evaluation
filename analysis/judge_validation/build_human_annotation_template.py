#!/usr/bin/env python3
"""Build an empty human annotation CSV aligned row-for-row with validation_500_samples_full.csv."""

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "validation_500_samples_full.csv"
OUT = HERE / "human_annotations_template.csv"

OUT_FIELDS = [
    "model",
    "dataset",
    "idx",
    "human_faith",
    "human_utili",
    "human_coher",
    "human_factu",
    "annotator_id",
    "notes",
]


def main() -> None:
    rows_out = []
    with SRC.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_out.append(
                {
                    "model": row["model"],
                    "dataset": row["dataset"],
                    "idx": row["idx"],
                    "human_faith": "",
                    "human_utili": "",
                    "human_coher": "",
                    "human_factu": "",
                    "annotator_id": "",
                    "notes": "",
                }
            )
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        w.writerows(rows_out)
    print(f"Wrote {OUT} ({len(rows_out)} rows). Fill integer 1–5 in human_* columns.")


if __name__ == "__main__":
    main()
