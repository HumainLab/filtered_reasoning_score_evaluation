#!/usr/bin/env python3
"""
MATH500 difficulty: original (all problems) vs top-10% selected (9 models pooled).

Reuses loaders/filter from verify_difficulty_distribution_claims.py.

Usage:
  python analysis/math500_level_original_vs_selected.py --repo-root .
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
_ANALYSIS_DIR = Path(__file__).resolve().parent
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from verify_difficulty_distribution_claims import (  # noqa: E402
    TOP_K_PCT,
    collect_math500_selected_traces,
    distribution_pct,
    load_math500_original_levels,
    parse_level,
)


def pct1(x: float) -> str:
    return f"{x:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    repo = args.repo_root.resolve()

    sys.path.insert(0, str(repo))
    from topk_ablation import build_file_map, load_and_process_jsonl  # noqa: E402

    file_map = build_file_map(str(repo))
    math500_paths = sorted(jp for (m, ds), jp in file_map.items() if ds == "MATH500")
    if not math500_paths:
        print("ERROR: no MATH500 JSONL files found.")
        sys.exit(1)

    canonical = Path(math500_paths[0])
    print("=" * 72)
    print("MATH500 level: original vs top-10% selected")
    print("=" * 72)
    print("\nData source:")
    print(f"  Canonical MATH500 JSONL (original): {canonical}")
    print(f"  MATH500 model JSONLs for top-10%: {len(math500_paths)} files")
    print(f"  Top-10% rule: confidence >= {100-TOP_K_PCT}th percentile per model pool (verify script)")

    # --- Original (unique problems, one count per idx) ---
    level_by_idx, parse_stats = load_math500_original_levels(canonical)
    print("\n--- Level field parsing (original JSONL) ---")
    for k, v in parse_stats.items():
        print(f"  {k}: {v}")

    # Cross-check: same level for same idx across other MATH500 files?
    mismatches = 0
    for jp in math500_paths[1:]:
        other, st2 = load_math500_original_levels(Path(jp))
        for pid, lv in other.items():
            if pid in level_by_idx and level_by_idx[pid] != lv:
                mismatches += 1
    print(f"  level mismatches across MATH500 files (idx in both): {mismatches}")

    orig_counts = {lv: 0 for lv in range(1, 6)}
    for lv in level_by_idx.values():
        orig_counts[lv] += 1
    orig_dist = distribution_pct(orig_counts)
    n_orig = len(level_by_idx)

    print("\n1) ORIGINAL distribution (unique problems, base rate, no confidence filter)")
    print(f"   N problems with valid level: {n_orig}")
    pct_sum = 0.0
    for lv in range(1, 6):
        c, pct = orig_dist[lv]
        pct_sum += pct
        print(f"   Level {lv}: {c:4d}  ({pct1(pct)}%)")
    print(f"   Sum of %: {pct1(pct_sum)}")

    # --- Selected (same pipeline as verify_difficulty_distribution_claims.py) ---
    selected_records, models_used = collect_math500_selected_traces(
        file_map, level_by_idx, load_and_process_jsonl
    )
    print("\n--- Selected top-10% (recomputed, same logic as verify script) ---")
    print(f"  Models: {', '.join(models_used)}")
    print(f"  Selected traces with known level: {len(selected_records)}")

    tr_counts = {lv: 0 for lv in range(1, 6)}
    for r in selected_records:
        tr_counts[r["level"]] += 1
    tr_dist = distribution_pct(tr_counts)
    n_tr = len(selected_records)

    prob_level: Dict[int, int] = {}
    for r in selected_records:
        prob_level[r["problem_idx"]] = r["level"]
    pr_counts = {lv: 0 for lv in range(1, 6)}
    for lv in prob_level.values():
        pr_counts[lv] += 1
    pr_dist = distribution_pct(pr_counts)
    n_pr = len(prob_level)

    print("\n2) SELECTED distributions (MATH500, 9 models)")
    print(f"   2a) Over traces (N={n_tr}):")
    tr_sum = 0.0
    for lv in range(1, 6):
        c, pct = tr_dist[lv]
        tr_sum += pct
        print(f"       Level {lv}: {c:4d}  ({pct1(pct)}%)")
    print(f"       Sum of %: {pct1(tr_sum)}")

    print(f"   2b) Over unique selected problems (N={n_pr}):")
    pr_sum = 0.0
    for lv in range(1, 6):
        c, pct = pr_dist[lv]
        pr_sum += pct
        print(f"       Level {lv}: {c:4d}  ({pct1(pct)}%)")
    print(f"       Sum of %: {pct1(pr_sum)}")

    # --- Comparison table ---
    print("\n3) Comparison table (shift = selected-trace % minus original %, in pp)")
    print(f"{'Level':<6} {'Original % (n)':<18} {'Sel-trace %':<12} {'Sel-prob %':<12} {'Shift pp':<10}")
    print("-" * 62)
    shifts: List[float] = []
    for lv in range(1, 6):
        oc, op = orig_dist[lv]
        tc, tp = tr_dist[lv]
        pc, pp = pr_dist[lv]
        shift = tp - op
        shifts.append(shift)
        print(
            f"{lv:<6} {pct1(op):>5}% ({oc:>4})     {pct1(tp):>6}%      {pct1(pp):>6}%      {shift:+.1f}"
        )

    easier_shift = (tr_dist[1][1] + tr_dist[2][1]) - (orig_dist[1][1] + orig_dist[2][1])
    harder_shift = (tr_dist[4][1] + tr_dist[5][1]) - (orig_dist[4][1] + orig_dist[5][1])
    print("\n4) Direction (from shift column, selected-trace vs original):")
    if harder_shift > 1.0 and harder_shift > abs(easier_shift):
        print(
            f"   Confidence selection shifts mass toward HARDER levels (4-5): "
            f"+{pct1(harder_shift)} pp vs L4-L5 vs +{pct1(easier_shift)} pp for L1-L2."
        )
    elif easier_shift > 1.0 and easier_shift > abs(harder_shift):
        print(
            f"   Confidence selection shifts mass toward EASIER levels (1-2): "
            f"+{pct1(easier_shift)} pp for L1-L2 vs +{pct1(harder_shift)} pp for L4-L5."
        )
    else:
        print(
            f"   Relative to the base rate, the selected-trace distribution is roughly unchanged "
            f"(L1-L2 shift {easier_shift:+.1f} pp, L4-L5 shift {harder_shift:+.1f} pp)."
        )

    print("\nAssumptions:")
    assumptions = [
        "Original = each problem once from canonical MATH500 JSONL; no confidence filter.",
        "Level map for selected traces uses the same idx->level table from that canonical file.",
        "Selected = per-model global top 10% by confidence, then pool trace-level counts across 9 models.",
        "Problems without parseable level are omitted from numerators (reported in parse_stats).",
        "file_map from topk_ablation.build_file_map(source_pass16_jsonl_by_model*/**/*.jsonl).",
    ]
    for i, a in enumerate(assumptions, 1):
        print(f"  {i}. {a}")


if __name__ == "__main__":
    main()
