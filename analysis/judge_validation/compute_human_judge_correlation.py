#!/usr/bin/env python3
"""
Merge human_annotations*.csv with validation_500_samples_full.csv and print
Pearson r (and optional Spearman) between human mean pillar score and each LLM judge.

Human file must have columns: model, dataset, idx, human_faith, human_utili, human_coher, human_factu
(values 1–5 integers; empty rows skipped for correlation).

Usage:
  python analysis/judge_validation/compute_human_judge_correlation.py
  python analysis/judge_validation/compute_human_judge_correlation.py --human path/to/my_ratings.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_VAL = HERE / "validation_500_samples_full.csv"
DEFAULT_HUMAN = HERE / "human_annotations_template.csv"

PILLAR_KEYS = [
    ("mini", ["mini_faith", "mini_utili", "mini_coher", "mini_factu"]),
    ("gpt4o", ["gpt4o_faith", "gpt4o_utili", "gpt4o_coher", "gpt4o_factu"]),
    ("claude", ["claude_faith", "claude_utili", "claude_coher", "claude_factu"]),
]
HUMAN_KEYS = ["human_faith", "human_utili", "human_coher", "human_factu"]


def mean4(row: dict, keys: list[str]):
    vals = []
    for k in keys:
        v = (row.get(k) or "").strip()
        if v == "":
            return None
        try:
            vals.append(float(v))
        except ValueError:
            return None
    return sum(vals) / 4.0


def key(row) -> tuple:
    return (row["model"], row["dataset"], str(row["idx"]).strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation", type=Path, default=DEFAULT_VAL)
    ap.add_argument("--human", type=Path, default=DEFAULT_HUMAN)
    args = ap.parse_args()

    val_by_k = {}
    with args.validation.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            val_by_k[key(row)] = row

    h_rows = []
    with args.human.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            h_rows.append(row)

    human_mean = []
    mini_m, gpt_m, claude_m = [], [], []
    matched = 0
    skipped_no_human = 0
    skipped_missing_val = 0

    for h in h_rows:
        k = key(h)
        hm = mean4(h, HUMAN_KEYS)
        if hm is None:
            skipped_no_human += 1
            continue
        v = val_by_k.get(k)
        if v is None:
            skipped_missing_val += 1
            continue
        mm = mean4(v, PILLAR_KEYS[0][1])
        gm = mean4(v, PILLAR_KEYS[1][1])
        cm = mean4(v, PILLAR_KEYS[2][1])
        if mm is None or gm is None or cm is None:
            skipped_missing_val += 1
            continue
        human_mean.append(hm)
        mini_m.append(mm)
        gpt_m.append(gm)
        claude_m.append(cm)
        matched += 1

    H = np.array(human_mean)
    M = np.array(mini_m)
    G = np.array(gpt_m)
    C = np.array(claude_m)

    print(f"Validation file: {args.validation}")
    print(f"Human file:      {args.human}")
    print(f"Rows with complete human + validation judge scores: n = {matched}")
    print(f"Skipped (incomplete human scores): {skipped_no_human}")
    print(f"Skipped (no validation match or incomplete LLM scores): {skipped_missing_val}\n")

    if matched < 3:
        print("Need at least 3 complete rows to compute correlation.")
        return

    def pr(a, b, label):
        r = float(np.corrcoef(a, b)[0, 1])
        print(f"  Pearson r (human mean vs {label}): {r:.4f}")

    print("=== Correlations (mean of 4 pillars) ===")
    pr(H, M, "GPT-4o-mini")
    pr(H, G, "GPT-4o")
    pr(H, C, "Claude")

    try:
        from scipy.stats import spearmanr  # type: ignore

        print("\n=== Spearman ρ ===")
        for a, name in [(M, "GPT-4o-mini"), (G, "GPT-4o"), (C, "Claude")]:
            rho, _ = spearmanr(H, a)
            print(f"  ρ (human vs {name}): {float(rho):.4f}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
