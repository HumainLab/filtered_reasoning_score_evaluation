#!/usr/bin/env python3
"""
Pairwise Spearman correlations among faithfulness, coherence, utility, factuality
on all judged traces (all confidence bins), from checkpoint JSON only — no new judge calls.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from k_sensitivity_analysis import discover_judge_files

DIMS = ["faithfulness", "coherence", "utility", "factuality"]


def load_all_traces_from_checkpoint(path: str) -> Tuple[str, str, List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("metadata", {})
    model = str(meta.get("model", ""))
    benchmark = str(meta.get("dataset", ""))
    rows: List[Dict[str, Any]] = []
    for s in data.get("judged_samples", []):
        js = s.get("judge_scores") or {}
        row = {
            "model": model,
            "benchmark": benchmark,
            "bin_label": str(s.get("bin_label", "")),
            "trace_id": f"{s.get('idx')}::{s.get('trace_idx')}",
            "idx": s.get("idx"),
            "trace_idx": s.get("trace_idx"),
            "judge_ok": s.get("judge_ok", True),
        }
        for d in DIMS:
            row[d] = js.get(d)
        rows.append(row)
    return model, benchmark, rows


def main() -> None:
    p = argparse.ArgumentParser(description="Reasoning dimension correlation analysis")
    p.add_argument(
        "--judging_dir",
        type=str,
        default="outputs/reasoning_confidence_bins_results/judging_checkpoints",
        help="Directory containing judged_*.json",
    )
    p.add_argument("--data_dir", type=str, default=".", help="Project root (prepended to relative paths)")
    p.add_argument(
        "--output_csv",
        type=str,
        default="outputs/diagnostics/dimension_correlations.csv",
        help="Where to save the correlation matrix CSV",
    )
    args = p.parse_args()

    root = os.path.abspath(args.data_dir)
    judging_dir = args.judging_dir
    if not os.path.isabs(judging_dir):
        judging_dir = os.path.join(root, judging_dir)

    judge_map = discover_judge_files(judging_dir)
    all_rows: List[Dict[str, Any]] = []
    for (_m, _b), path in sorted(judge_map.items()):
        _model, _bench, rows = load_all_traces_from_checkpoint(path)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    n_raw = len(df)
    n_pairs = df.groupby(["model", "benchmark"]).ngroups

    # Missing counts before drop
    miss = {d: int(df[d].isna().sum()) for d in DIMS}
    n_miss_any = int(df[DIMS].isna().any(axis=1).sum())

    df_clean = df.dropna(subset=DIMS).copy()
    for d in DIMS:
        df_clean[d] = pd.to_numeric(df_clean[d], errors="coerce")
    df_clean = df_clean.dropna(subset=DIMS)
    n_analyze = len(df_clean)

    # Composites for reweighting (1–5 scale)
    df_clean["reasoning_equal"] = df_clean[DIMS].mean(axis=1)
    df_clean["reasoning_util_heavy"] = (
        df_clean["faithfulness"]
        + df_clean["coherence"]
        + 2 * df_clean["utility"]
        + df_clean["factuality"]
    ) / 5.0
    df_clean["reasoning_util_only"] = df_clean["utility"]

    corr_matrix = pd.DataFrame(index=DIMS, columns=DIMS, dtype=float)
    for d1 in DIMS:
        for d2 in DIMS:
            rho, _p = spearmanr(df_clean[d1], df_clean[d2])
            corr_matrix.loc[d1, d2] = float(rho)

    off_diag = []
    for i, d1 in enumerate(DIMS):
        for j, d2 in enumerate(DIMS):
            if i < j:
                off_diag.append(corr_matrix.loc[d1, d2])
    mean_rho = float(np.mean(off_diag))
    min_rho = float(np.min(off_diag))
    max_rho = float(np.max(off_diag))

    # Per-dimension variance
    variances = {d: float(df_clean[d].var(ddof=0)) for d in DIMS}

    # Model-level means for ranking stability
    model_scores = df_clean.groupby("model", as_index=True)[
        ["reasoning_equal", "reasoning_util_heavy", "reasoning_util_only"]
    ].mean()
    for col in ["reasoning_equal", "reasoning_util_heavy", "reasoning_util_only"]:
        model_scores[f"rank_{col}"] = model_scores[col].rank(ascending=False, method="average")

    n_models = len(model_scores)
    if n_models >= 3:
        rho_heavy, _ = spearmanr(
            model_scores["rank_reasoning_equal"],
            model_scores["rank_reasoning_util_heavy"],
        )
        rho_only, _ = spearmanr(
            model_scores["rank_reasoning_equal"],
            model_scores["rank_reasoning_util_only"],
        )
        rho_heavy = float(rho_heavy) if np.isfinite(rho_heavy) else float("nan")
        rho_only = float(rho_only) if np.isfinite(rho_only) else float("nan")
    else:
        rho_heavy = rho_only = float("nan")

    out_path = args.output_csv
    if not os.path.isabs(out_path):
        out_path = os.path.join(root, out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    corr_matrix.to_csv(out_path)

    # Conclusion wording
    if mean_rho > 0.8:
        corr_label = "highly"
        weight_label = "justified"
    elif mean_rho > 0.5:
        corr_label = "moderately"
        weight_label = "potentially problematic"
    else:
        corr_label = "weakly"
        weight_label = "potentially problematic"

    def fmt_cm() -> str:
        lines = []
        header = "              " + "".join(f"{d[:5]:>8s}" for d in DIMS)
        lines.append(header)
        for d1 in DIMS:
            row = f"  {d1[:5]:5s}    " + "".join(f"{corr_matrix.loc[d1, d2]:8.3f}" for d2 in DIMS)
            lines.append(row)
        return "\n".join(lines)

    print(
        f"""
================================================================
         REASONING DIMENSION CORRELATION ANALYSIS
================================================================

Judged traces analyzed: {n_analyze} (of {n_raw} raw rows; {n_miss_any} rows with any missing dimension dropped)
(Model, Benchmark) pairs: {n_pairs}
Missing values per dimension (before drop): {miss}

PAIRWISE SPEARMAN CORRELATIONS:
{fmt_cm()}

Mean pairwise ρ: {mean_rho:.3f}
Range: [{min_rho:.3f}, {max_rho:.3f}]

PER-DIMENSION VARIANCE:
  faithfulness: var = {variances['faithfulness']:.3f}
  coherence:    var = {variances['coherence']:.3f}
  utility:      var = {variances['utility']:.3f}
  factuality:   var = {variances['factuality']:.3f}

RANKING STABILITY UNDER REWEIGHTING (mean score per model, {n_models} models):
  Equal vs utility-heavy (2x): ρ = {rho_heavy:.3f}
  Equal vs utility-only:       ρ = {rho_only:.3f}

CONCLUSION:
  Dimensions are {corr_label} correlated (mean pairwise ρ = {mean_rho:.3f}).
  Equal weighting is {weight_label}.

Correlation matrix saved to: {out_path}
================================================================
"""
    )


if __name__ == "__main__":
    main()
