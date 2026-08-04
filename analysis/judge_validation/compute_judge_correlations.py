#!/usr/bin/env python3
"""
Compute inter-judge correlations on the stratified 500-sample validation CSV.

Scores: mean of four pillars (1–5) per row, and per-pillar breakdown.
Also reports feasible range for corr(human, GPT-4o) given corr(human, mini) = target
and empirical corr(mini, GPT-4o) (PSD constraint on 3x3 correlation matrix).

Usage:
  python analysis/judge_validation/compute_judge_correlations.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

CSV_PATH = Path(__file__).resolve().parent / "validation_500_samples_full.csv"

PILLAR_SUFFIXES = [
    ("faithfulness", "faith"),
    ("utility", "utili"),
    ("coherence", "coher"),
    ("factuality", "factu"),
]


def load_means() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Return M, G, C (n,) mean pillar scores and raw rows for debugging."""
    m_list, g_list, c_list = [], [], []
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ms, gs, cs = [], [], []
            ok = True
            for _, suf in PILLAR_SUFFIXES:
                try:
                    ms.append(float(row[f"mini_{suf}"]))
                    gs.append(float(row[f"gpt4o_{suf}"]))
                    cs.append(float(row[f"claude_{suf}"]))
                except (KeyError, ValueError):
                    ok = False
                    break
            if not ok:
                continue
            m_list.append(sum(ms) / 4.0)
            g_list.append(sum(gs) / 4.0)
            c_list.append(sum(cs) / 4.0)
    return np.array(m_list), np.array(g_list), np.array(c_list), []


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    from scipy.stats import spearmanr  # type: ignore

    r, _ = spearmanr(a, b)
    return float(r)


def feasible_r_hg(rho_hm: float, rho_mg: float) -> tuple[float, float]:
    """Range of rho_hg such that [[1,rho_hm,rho_hg],[rho_hm,1,rho_mg],[rho_hg,rho_mg,1]] is PSD."""
    a, b = rho_hm, rho_mg
    A, B, Ccoef = -1.0, 2 * a * b, (1 - a * a - b * b)
    disc = B * B - 4 * A * Ccoef
    if disc < 0:
        return float("nan"), float("nan")
    r1 = (-B - math.sqrt(disc)) / (2 * A)
    r2 = (-B + math.sqrt(disc)) / (2 * A)
    return (min(r1, r2), max(r1, r2))


def main() -> None:
    M, G, C, _ = load_means()
    n = len(M)
    print(f"File: {CSV_PATH}")
    print(f"Rows with all three judges (4 pillars each): n = {n}\n")

    # --- Headline numbers (use these in the paper if you report mean score) ---
    r_mg_p = pearson(M, G)
    r_mc_p = pearson(M, C)
    r_gc_p = pearson(G, C)
    print("=== Pearson r (mean of 4 pillars per sample) ===")
    print(f"  GPT-4o-mini  vs  GPT-4o:   {r_mg_p:.4f}")
    print(f"  GPT-4o-mini  vs  Claude:   {r_mc_p:.4f}")
    print(f"  GPT-4o       vs  Claude:   {r_gc_p:.4f}")

    try:
        r_mg_s = spearman(M, G)
        r_mc_s = spearman(M, C)
        r_gc_s = spearman(G, C)
        print("\n=== Spearman ρ (same scores) ===")
        print(f"  GPT-4o-mini  vs  GPT-4o:   {r_mg_s:.4f}")
        print(f"  GPT-4o-mini  vs  Claude:   {r_mc_s:.4f}")
        print(f"  GPT-4o       vs  Claude:   {r_gc_s:.4f}")
    except ImportError:
        print("\n(install scipy for Spearman: pip install scipy)")

    print("\n=== Pearson r per pillar (mini vs GPT-4o) ===")
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for pname, suf in PILLAR_SUFFIXES:
        a, b = [], []
        for row in rows:
            try:
                a.append(float(row[f"mini_{suf}"]))
                b.append(float(row[f"gpt4o_{suf}"]))
            except (KeyError, ValueError):
                pass
        if len(a) >= 3:
            aa, bb = np.array(a), np.array(b)
            print(f"  {pname:12s}  r = {pearson(aa, bb):.4f}  (n={len(a)})")

    # Human validation: if human tracks mini at rho_hm, compatible rho with GPT-4o
    print("\n=== If human ↔ GPT-4o-mini = target, feasible human ↔ GPT-4o (PSD bound) ===")
    print(f"(using empirical corr(mini, GPT-4o) = {r_mg_p:.4f})\n")
    for target in (0.70, 0.75, 0.80):
        lo, hi = feasible_r_hg(target, r_mg_p)
        print(f"  corr(human, mini) = {target:.2f}  →  corr(human, GPT-4o) ∈ [{lo:.3f}, {hi:.3f}]")

    print(
        "\n--- Suggested paper line (fill in your measured human r) ---\n"
        f"On n={n} stratified items, mean-pillar scores correlate across judges with\n"
        f"Pearson r = {r_mg_p:.2f} (mini vs GPT-4o), {r_mc_p:.2f} (mini vs Claude), "
        f"{r_gc_p:.2f} (GPT-4o vs Claude)."
    )


if __name__ == "__main__":
    main()
