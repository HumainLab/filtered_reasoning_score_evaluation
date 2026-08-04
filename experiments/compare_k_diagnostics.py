#!/usr/bin/env python3
"""
Build k=8 vs k=4 diagnostic comparison from existing CSV outputs (no hardcoded metrics).
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Optional

import numpy as np
import pandas as pd


def _fmt(x: Any, prec: int = 6) -> str:
    if x is None:
        return "N/A"
    if isinstance(x, (float, np.floating)) and (np.isnan(x) or not np.isfinite(x)):
        return "N/A"
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{prec}f}"
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=".")
    ap.add_argument("--aggregate_csv", type=str, default="outputs/results/k_sensitivity_aggregate.csv")
    ap.add_argument("--overlap_csv", type=str, default="outputs/results/k_sensitivity_overlap.csv")
    ap.add_argument("--diag8_dir", type=str, default="outputs/diagnostics")
    ap.add_argument("--diag4_dir", type=str, default="outputs/diagnostics_k4")
    ap.add_argument("--out_txt", type=str, default="outputs/diagnostics_k4/comparison_summary.txt")
    ap.add_argument("--out_per_model", type=str, default="outputs/diagnostics_k4/per_model_recall_comparison.csv")
    args = ap.parse_args()

    root = os.path.abspath(args.data_dir)

    def pjoin(rel: str) -> str:
        return rel if os.path.isabs(rel) else os.path.join(root, rel)

    agg = pd.read_csv(pjoin(args.aggregate_csv))
    ov = pd.read_csv(pjoin(args.overlap_csv))

    row8 = agg[agg["k"] == 8].iloc[0] if len(agg[agg["k"] == 8]) else None
    row4 = agg[agg["k"] == 4].iloc[0] if len(agg[agg["k"] == 4]) else None

    gmr8 = row8["grand_mean_recall"] if row8 is not None else np.nan
    gmr4 = row4["grand_mean_recall"] if row4 is not None else np.nan
    rf8 = row8["regime_flip_rate"] if row8 is not None else np.nan
    rf4 = row4["regime_flip_rate"] if row4 is not None else np.nan
    mfs8 = row8["mean_frs_shift"] if row8 is not None else np.nan
    mfs4 = row4["mean_frs_shift"] if row4 is not None else np.nan

    def fmt_shift(v: float) -> str:
        if v is None or (isinstance(v, float) and (np.isnan(v) or not np.isfinite(v))):
            return "N/A"
        return _fmt(v, 4)

    b8 = pd.read_csv(pjoin(os.path.join(args.diag8_dir, "boundary_summary.csv")))
    b4 = pd.read_csv(pjoin(os.path.join(args.diag4_dir, "boundary_summary.csv")))

    mean_med8 = float(b8["median_k16_pct_new"].mean())
    mean_med4 = float(b4["median_k16_pct_new"].mean())
    mean_p958 = float(b8["p95_k16_pct_new"].mean())
    mean_p954 = float(b4["p95_k16_pct_new"].mean())

    d8 = pjoin(os.path.join(args.diag8_dir, "regime_flip_detail.csv"))
    d4 = pjoin(os.path.join(args.diag4_dir, "regime_flip_detail.csv"))
    detail8 = pd.read_csv(d8) if os.path.isfile(d8) else pd.DataFrame()
    detail4 = pd.read_csv(d4) if os.path.isfile(d4) else pd.DataFrame()

    ov8 = ov[ov["k"] == 8]
    ov4 = ov[ov["k"] == 4]
    flips8 = int(ov8["regime_flip"].astype(bool).sum()) if len(ov8) else 0
    flips4 = int(ov4["regime_flip"].astype(bool).sum()) if len(ov4) else 0
    trials8 = len(ov8)
    trials4 = len(ov4)

    max_flip8 = float(detail8["abs_snr_k16"].max()) if len(detail8) and "abs_snr_k16" in detail8.columns else float("nan")
    max_flip4 = float(detail4["abs_snr_k16"].max()) if len(detail4) and "abs_snr_k16" in detail4.columns else float("nan")

    r8path = pjoin(os.path.join(args.diag8_dir, "ranking_stability.csv"))
    r4path = pjoin(os.path.join(args.diag4_dir, "ranking_stability.csv"))
    rk8 = pd.read_csv(r8path) if os.path.isfile(r8path) else pd.DataFrame()
    rk4 = pd.read_csv(r4path) if os.path.isfile(r4path) else pd.DataFrame()

    def rho_stats(df: pd.DataFrame) -> tuple[float, float, int, int]:
        if df.empty or "spearman_rho" not in df.columns:
            return float("nan"), float("nan"), 0, len(df)
        ok = df["spearman_rho"].notna() & np.isfinite(df["spearman_rho"])
        if "num_models_ranked" in df.columns:
            ok = ok & (df["num_models_ranked"] >= 5)
        sub = df[ok]
        n_comp = len(sub)
        n_tot = len(df)
        if n_comp == 0:
            return float("nan"), float("nan"), 0, n_tot
        return float(sub["spearman_rho"].mean()), float(sub["spearman_rho"].std(ddof=0)), n_comp, n_tot

    rm8, sd8, nc8, nt8 = rho_stats(rk8)
    rm4, sd4, nc4, nt4 = rho_stats(rk4)

    def fmt_rho(m: float, s: float) -> str:
        if not np.isfinite(m):
            return "N/A"
        if not np.isfinite(s):
            return _fmt(m)
        return f"{_fmt(m)}±{_fmt(s)}"

    pm8 = pd.read_csv(pjoin(os.path.join(args.diag8_dir, "per_model_recall.csv")))
    pm4 = pd.read_csv(pjoin(os.path.join(args.diag4_dir, "per_model_recall.csv")))

    lo8 = pm8.loc[pm8["mean_recall"].idxmin()]
    hi8 = pm8.loc[pm8["mean_recall"].idxmax()]
    lo4 = pm4.loc[pm4["mean_recall"].idxmin()]
    hi4 = pm4.loc[pm4["mean_recall"].idxmax()]

    merged = pm8[["model", "mean_recall"]].merge(
        pm4[["model", "mean_recall"]],
        on="model",
        suffixes=("_k8", "_k4"),
    )
    merged["recall_drop"] = merged["mean_recall_k8"] - merged["mean_recall_k4"]
    merged = merged.rename(columns={"mean_recall_k8": "recall_k8", "mean_recall_k4": "recall_k4"})
    merged = merged.sort_values("recall_k4", ascending=True)
    merged.to_csv(pjoin(args.out_per_model), index=False)

    # Conclusions
    q1 = mean_med4 < 12.0

    def _flip_snr_ok(m: float, n_detail: int) -> bool:
        if n_detail == 0:
            return True
        return bool(np.isfinite(m) and m < 0.5)

    q2 = _flip_snr_ok(max_flip8, len(detail8)) and _flip_snr_ok(max_flip4, len(detail4))

    q3 = False
    if not rk4.empty and "num_models_ranked" in rk4.columns:
        q3 = bool(
            (
                rk4["spearman_rho"].notna()
                & np.isfinite(rk4["spearman_rho"])
                & (rk4["num_models_ranked"] >= 5)
            ).any()
        )

    q4 = bool(gmr4 > 0.95) if np.isfinite(gmr4) else False

    q5 = "UNKNOWN"
    if q3 and np.isfinite(rm4):
        q5 = "YES" if rm4 > 0.90 else "NO"
    elif not q3:
        q5 = "UNKNOWN"

    txt = f"""================================================================
         k=4 vs k=8 DIAGNOSTIC COMPARISON
         (all numbers read from output files)
================================================================

                                        k=8             k=4
                                        ----            ----
OVERLAP (from k_sensitivity_aggregate.csv):
  Grand mean recall:                    {_fmt(gmr8)}           {_fmt(gmr4)}
  Regime flip rate:                     {_fmt(rf8)}           {_fmt(rf4)}
  Mean FRS shift:                       {fmt_shift(mfs8)}           {fmt_shift(mfs4)}

BOUNDARY (from boundary_summary.csv):
  Median k=16 pctile of new entrants:   {_fmt(mean_med8, 4)}%          {_fmt(mean_med4, 4)}%
  95th pctile of new entrants:          {_fmt(mean_p958, 4)}%          {_fmt(mean_p954, 4)}%

REGIME FLIPS (overlap CSV counts; max |SNR| from regime_flip_detail.csv):
  Total flips / total trials:           {flips8}/{trials8}         {flips4}/{trials4}
  Max |SNR_k16| among flips:            {_fmt(max_flip8)}           {_fmt(max_flip4)}

RANKING STABILITY (from ranking_stability.csv, ρ only if ≥5 models ranked):
  Spearman ρ (mean ± std):              {fmt_rho(rm8, sd8)}     {fmt_rho(rm4, sd4)}
  Resamples with computable ranking:    {nc8}/{nt8}         {nc4}/{nt4}

PER-MODEL (from per_model_recall.csv):
  Least stable model (recall):          {lo8['model']} ({_fmt(float(lo8['mean_recall']))})  {lo4['model']} ({_fmt(float(lo4['mean_recall']))})
  Most stable model (recall):           {hi8['model']} ({_fmt(float(hi8['mean_recall']))})  {hi4['model']} ({_fmt(float(hi4['mean_recall']))})

================================================================
CONCLUSIONS:
  1. New entrants at k=4 are borderline
     (median pctile < 12%)?              {"YES" if mean_med4 < 12 else "NO"}
  2. Regime flips confined to near-zero
     SNR (max |SNR| < 0.5)?              {"YES" if q2 else "NO"}
  3. Ranking stability estimable
     (≥5 models rankable)?               {"YES" if q3 else "NO"}
  4. Filtered set stable at k=4?         {"YES" if q4 else "NO"}
  5. FRS rankings stable at k=4?         {q5}
================================================================
"""
    out_txt = pjoin(args.out_txt)
    os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    print("\n--- per_model_recall_comparison (sorted by recall_k4) ---")
    print(merged.to_string(index=False))
    flagged = merged[merged["recall_k4"] < 0.95]
    if len(flagged):
        print("\nModels with recall_k4 < 0.95:")
        print(flagged["model"].tolist())
    else:
        print("\nNo models with recall_k4 < 0.95")
    print(f"\nWrote {out_txt}")
    print(f"Wrote {pjoin(args.out_per_model)}")


if __name__ == "__main__":
    main()
