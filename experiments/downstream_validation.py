#!/usr/bin/env python3
"""
Downstream validation experiment for Filtered Reasoning Score (FRS).

Split-half design: per benchmark, 50/50 problem split (calibration / test),
multiple re-split seeds for uncertainty quantification.

Dependencies: pandas, numpy, scipy, matplotlib, seaborn, pyarrow (for parquet I/O).

Expected parquet columns (or infer ``model`` / ``benchmark`` from ``<model>__<benchmark>.parquet``):
``problem_id``, ``trace_id`` (or ``trace_idx``), ``confidence``, ``correct``, ``reasoning_score``
(0–100 or 0–1; 0–1 is auto-scaled to 0–100).

Usage:
  python downstream_validation.py --data_dir results/ --output_dir downstream_results/
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, spearmanr

# ── Publication-style defaults ─────────────────────────────────────────────
mpl.rcParams.update(
    {
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "text.color": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "grid.color": "#cccccc",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
    }
)

# Canonical model names -> regime (user taxonomy)
REGIME_BY_MODEL: Dict[str, str] = {
    "DS-R1-7B": "reliable",
    "DS-R1-1.5B": "reliable",
    "Qwen3-4B": "reliable",
    "LLaMA-3.1-8B": "reliable",
    "Phi-4-Reas.": "inverted",
    "Phi-4-Reasoning": "inverted",
    "Qwen2.5-Math": "inverted",
    "Qwen2.5-Math-7B": "inverted",
    "Gemma-7B": "inverted",
    "Phi-4": "flat",
    "Qwen2.5-7B": "flat",
}

# Filename / slug aliases -> canonical model name
MODEL_ALIASES: Dict[str, str] = {
    "ds_r1_7b": "DS-R1-7B",
    "ds_r1_1_5b": "DS-R1-1.5B",
    "deepseek_r1_distill_qwen_7b": "DS-R1-7B",
    "deepseek_r1_distill_qwen_1_5b": "DS-R1-1.5B",
    "qwen3_4b": "Qwen3-4B",
    "llama_3_1_8b": "LLaMA-3.1-8B",
    "llama_3_1_8b_instruct": "LLaMA-3.1-8B",
    "qwen2_5_7b": "Qwen2.5-7B",
    "qwen2_5_7b_instruct": "Qwen2.5-7B",
    "qwen2_5_math_7b": "Qwen2.5-Math",
    "qwen2_5_math": "Qwen2.5-Math",
    "gemma_7b": "Gemma-7B",
    "phi_4": "Phi-4",
    "phi4": "Phi-4",
    "phi_4_reasoning": "Phi-4-Reas.",
    "phi4_reasoning": "Phi-4-Reas.",
}

BENCHMARK_ALIASES: Dict[str, str] = {
    "gsm8k": "gsm8k",
    "math500": "math500",
    "svamp": "svamp",
    "aqua": "aqua",
    "gpqa": "gpqa",
    "commonsenseqa": "commonsenseqa",
    "commonsense_qa": "commonsenseqa",
}


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def top_frac_count(n: int, frac: float) -> int:
    if n <= 0:
        return 0
    return max(1, int(np.ceil(n * frac)))


def parse_model_benchmark_from_path(path: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse ``model__benchmark.parquet`` from basename."""
    base = os.path.splitext(os.path.basename(path))[0]
    if "__" not in base:
        return None, None
    a, b = base.split("__", 1)
    ma = MODEL_ALIASES.get(_slug(a))
    bench = BENCHMARK_ALIASES.get(_slug(b))
    return ma, bench


def normalize_reasoning_series(s: pd.Series) -> pd.Series:
    """Map judge-derived scores to 0–100 if stored in 0–1."""
    v = s.astype(float)
    if v.notna().any():
        mx = float(v.max())
        if mx <= 1.5:
            v = v * 100.0
    return v


def discover_parquet_files(data_dir: str) -> List[str]:
    out: List[str] = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".parquet"):
                out.append(os.path.join(root, f))
    out.sort()
    return out


def load_unified_table(data_dir: str) -> pd.DataFrame:
    paths = discover_parquet_files(data_dir)
    if not paths:
        raise FileNotFoundError(f"No parquet files under {data_dir!r}")

    frames: List[pd.DataFrame] = []
    for p in paths:
        df = pd.read_parquet(p)
        df.columns = [str(c).lower() for c in df.columns]

        file_model, file_bench = parse_model_benchmark_from_path(p)
        if "model" not in df.columns and file_model:
            df["model"] = file_model
        if "benchmark" not in df.columns and file_bench:
            df["benchmark"] = file_bench
        if "model" not in df.columns or "benchmark" not in df.columns:
            raise ValueError(
                f"{p}: need columns 'model' and 'benchmark' or filename model__benchmark.parquet"
            )

        # trace index
        if "trace_id" not in df.columns:
            if "trace_idx" in df.columns:
                df["trace_id"] = df["trace_idx"]
            else:
                df["trace_id"] = df.groupby(["model", "benchmark", "problem_id"]).cumcount()

        for col in ("problem_id", "confidence", "correct", "reasoning_score"):
            if col not in df.columns:
                raise ValueError(f"{p}: missing required column {col!r}")

        df["reasoning_score"] = normalize_reasoning_series(df["reasoning_score"])
        df["correct"] = df["correct"].astype(bool)
        df["confidence"] = df["confidence"].astype(float)
        df["model"] = df["model"].astype(str)
        df["benchmark"] = df["benchmark"].astype(str).str.lower()

        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    # Canonicalize model names when obvious
    out["model"] = out["model"].apply(lambda m: MODEL_ALIASES.get(_slug(m), m))
    return out


def regime_for_model(model: str) -> str:
    return REGIME_BY_MODEL.get(model, "unknown")


def split_problem_ids(
    problem_ids: np.ndarray,
    rng: np.random.Generator,
    frac_cal: float = 0.5,
) -> Tuple[Set[Any], Set[Any]]:
    """Random 50/50 split of unique problem ids."""
    ids = np.unique(problem_ids)
    rng.shuffle(ids)
    n_cal = int(np.floor(len(ids) * frac_cal))
    cal = set(ids[:n_cal])
    test = set(ids[n_cal:])
    if not cal or not test:
        raise ValueError("Split produced empty calibration or test set.")
    return cal, test


def compute_frs_cal(
    df_cal: pd.DataFrame,
    *,
    top_frac: float = 0.10,
) -> float:
    """Mean reasoning score on top ``top_frac`` of pooled calibration traces by confidence."""
    if df_cal.empty:
        return float("nan")
    sub = df_cal.dropna(subset=["reasoning_score", "confidence"])
    if sub.empty:
        return float("nan")
    sub = sub.sort_values(
        ["confidence", "problem_id", "trace_id"],
        ascending=[False, True, True],
    )
    k = top_frac_count(len(sub), top_frac)
    top = sub.head(k)
    return float(top["reasoning_score"].mean())


def compute_acc_cal(df_cal: pd.DataFrame) -> float:
    """Mean accuracy (fraction correct) over all calibration traces."""
    if df_cal.empty:
        return float("nan")
    return float(df_cal["correct"].astype(float).mean())


def select_bon_and_random(
    df_test: pd.DataFrame,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float]:
    """
    Per problem: BoN = argmax confidence (random tie-break); Random = mean over traces.

    Returns BoN_reasoning, BoN_accuracy, Random_reasoning, Random_accuracy.
    """
    if df_test.empty:
        return (float("nan"),) * 4

    bon_rs: List[float] = []
    bon_ok: List[float] = []
    rand_rs: List[float] = []
    rand_ok: List[float] = []

    for pid, g in df_test.groupby("problem_id"):
        g = g.dropna(subset=["confidence", "reasoning_score"])
        if g.empty:
            continue
        max_conf = g["confidence"].max()
        ties = g[g["confidence"] == max_conf]
        if len(ties) > 1:
            j = int(rng.integers(0, len(ties)))
            pick = ties.iloc[[j]]
        else:
            pick = ties
        bon_rs.append(float(pick["reasoning_score"].iloc[0]))
        bon_ok.append(float(pick["correct"].iloc[0]))

        rand_rs.append(float(g["reasoning_score"].mean()))
        rand_ok.append(float(g["correct"].astype(float).mean()))

    if not bon_rs:
        return (float("nan"),) * 4

    return (
        float(np.mean(bon_rs)),
        float(np.mean(bon_ok)),
        float(np.mean(rand_rs)),
        float(np.mean(rand_ok)),
    )


def spearman_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 0,
) -> Tuple[float, float, float, float]:
    """Bootstrap 95% CI for Spearman rho (paired rows, drop NaN). Returns rho, lo, hi, p."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        r, p = spearmanr(x, y)
        return float(r), float("nan"), float("nan"), float(p) if np.isfinite(p) else float("nan")

    rho_obs, p_two = spearmanr(x, y)
    rho_obs = float(rho_obs)
    rng = np.random.default_rng(seed)
    rhos: List[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(x), size=len(x))
        r = float(spearmanr(x[idx], y[idx]).correlation)
        if np.isfinite(r):
            rhos.append(r)
    lo, hi = (
        (float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5)))
        if rhos
        else (float("nan"), float("nan"))
    )
    return rho_obs, lo, hi, float(p_two)


@dataclass
class SplitResult:
    frs_cal: float
    acc_cal: float
    bon_reason: float
    bon_acc: float
    rand_reason: float
    rand_acc: float


def metrics_for_pair(
    df: pd.DataFrame,
    model: str,
    benchmark: str,
    cal_ids: Set[Any],
    test_ids: Set[Any],
    rng: np.random.Generator,
    top_frac: float,
) -> Optional[SplitResult]:
    """Calibration/test IDs are shared across all models for this benchmark."""
    sub = df[(df["model"] == model) & (df["benchmark"] == benchmark)]
    if sub.empty:
        return None
    df_cal = sub[sub["problem_id"].isin(cal_ids)]
    df_test = sub[sub["problem_id"].isin(test_ids)]
    if df_cal.empty or df_test.empty:
        return None

    frs = compute_frs_cal(df_cal, top_frac=top_frac)
    acc = compute_acc_cal(df_cal)
    bon_r, bon_a, rand_r, rand_a = select_bon_and_random(df_test, rng)

    if not np.isfinite(frs) or not np.isfinite(bon_r):
        return None
    return SplitResult(frs, acc, bon_r, bon_a, rand_r, rand_a)


def main() -> None:
    ap = argparse.ArgumentParser(description="FRS downstream validation (split-half).")
    ap.add_argument("--data_dir", type=str, required=True, help="Directory with model__bench.parquet files")
    ap.add_argument("--output_dir", type=str, default="outputs/downstream_results", help="Figures and logs")
    ap.add_argument("--n_splits", type=int, default=10, help="Number of random re-splits")
    ap.add_argument("--base_seed", type=int, default=42, help="First RNG seed for splits")
    ap.add_argument("--top_frac", type=float, default=0.10, help="Top fraction for FRS (default 10%%)")
    ap.add_argument("--bootstrap_iters", type=int, default=1000)
    args = ap.parse_args()
    top_frac = float(args.top_frac)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading parquet files from", args.data_dir, flush=True)
    df = load_unified_table(args.data_dir)
    print(f"Loaded {len(df)} rows.", flush=True)

    models = sorted(df["model"].unique())
    benchmarks = sorted(df["benchmark"].unique())
    pairs = [(m, b) for m in models for b in benchmarks]

    # Accumulate per (model, bench), per split
    records: List[Dict[str, Any]] = []

    for split_idx in range(args.n_splits):
        seed = args.base_seed + split_idx
        rng = np.random.default_rng(seed)
        print(f"--- Re-split {split_idx + 1}/{args.n_splits} (seed={seed}) ---", flush=True)

        # Same problem split for every model on a given benchmark (shared held-out test).
        bench_splits: Dict[str, Tuple[Set[Any], Set[Any]]] = {}
        for bench in benchmarks:
            pids = df[df["benchmark"] == bench]["problem_id"].unique()
            bench_splits[bench] = split_problem_ids(pids, rng)

        for model, bench in pairs:
            cal_ids, test_ids = bench_splits[bench]
            sr = metrics_for_pair(df, model, bench, cal_ids, test_ids, rng, top_frac)
            if sr is None:
                continue
            records.append(
                {
                    "split": split_idx,
                    "seed": seed,
                    "model": model,
                    "benchmark": bench,
                    "regime": regime_for_model(model),
                    "frs_cal": sr.frs_cal,
                    "acc_cal": sr.acc_cal,
                    "bon_reasoning": sr.bon_reason,
                    "bon_accuracy": sr.bon_acc,
                    "random_reasoning": sr.rand_reason,
                    "random_accuracy": sr.rand_acc,
                    "lift": sr.bon_reason - sr.rand_reason,
                }
            )

    long_df = pd.DataFrame(records)
    if long_df.empty:
        print("No valid model-benchmark results; check data.", file=sys.stderr)
        sys.exit(1)

    long_df.to_csv(os.path.join(args.output_dir, "per_split_metrics.csv"), index=False)
    print(f"Wrote {len(long_df)} rows to per_split_metrics.csv", flush=True)

    # Aggregate mean/std per (model, benchmark) across splits
    agg = (
        long_df.groupby(["model", "benchmark", "regime"], as_index=False)
        .agg(
            frs_cal_mean=("frs_cal", "mean"),
            frs_cal_std=("frs_cal", "std"),
            acc_cal_mean=("acc_cal", "mean"),
            bon_reason_mean=("bon_reasoning", "mean"),
            bon_reason_std=("bon_reasoning", "std"),
            random_reason_mean=("random_reasoning", "mean"),
            lift_mean=("lift", "mean"),
            lift_std=("lift", "std"),
            bon_acc_mean=("bon_accuracy", "mean"),
            random_acc_mean=("random_accuracy", "mean"),
        )
    )
    agg.to_csv(os.path.join(args.output_dir, "aggregated_model_benchmark.csv"), index=False)

    # --- Primary: Spearman FRS_cal vs BoN_reasoning (one point per pair per split) ---
    rhos: List[float] = []
    acc_rhos: List[float] = []
    for sp in range(args.n_splits):
        sub = long_df[long_df["split"] == sp]
        if len(sub) < 3:
            continue
        r, _ = spearmanr(sub["frs_cal"], sub["bon_reasoning"])
        if np.isfinite(r):
            rhos.append(float(r))
        r2, _ = spearmanr(sub["acc_cal"], sub["bon_reasoning"])
        if np.isfinite(r2):
            acc_rhos.append(float(r2))

    rho_mean = float(np.mean(rhos)) if rhos else float("nan")
    rho_std = float(np.std(rhos, ddof=1)) if len(rhos) > 1 else 0.0

    # Pooled correlation: split-averaged points (one per model–benchmark pair)
    pivot_frs = agg["frs_cal_mean"].values
    pivot_bon = agg["bon_reason_mean"].values
    pivot_acc = agg["acc_cal_mean"].values
    rho_full, rho_lo, rho_hi, p_fb = spearman_bootstrap_ci(
        pivot_frs,
        pivot_bon,
        n_resamples=args.bootstrap_iters,
        seed=args.base_seed + 999,
    )
    rho_ab, rho_ab_lo, rho_ab_hi, p_ab = spearman_bootstrap_ci(
        pivot_acc,
        pivot_bon,
        n_resamples=args.bootstrap_iters,
        seed=args.base_seed + 1000,
    )

    print("\n=== Spearman correlations (split-averaged, N pairs = {}) ===".format(len(agg)), flush=True)
    print(
        f"FRS_cal vs BoN_reasoning: rho={rho_full:.4f}, p={p_fb:.2e}, 95% CI [{rho_lo:.4f}, {rho_hi:.4f}]",
        flush=True,
    )
    print(
        f"Acc_cal vs BoN_reasoning: rho={rho_ab:.4f}, p={p_ab:.2e}, 95% CI [{rho_ab_lo:.4f}, {rho_ab_hi:.4f}]",
        flush=True,
    )
    print(f"Across-split rho(FRS, BoN): mean={rho_mean:.4f}, std={rho_std:.4f}", flush=True)
    print(
        f"Tertiary — Spearman rho gap (FRS − Acc) vs BoN_reasoning: {rho_full - rho_ab:+.4f}",
        flush=True,
    )

    # --- Mann-Whitney: lift by model (reliable vs inverted) ---
    model_lift = (
        long_df.groupby(["model", "regime"], as_index=False)
        .agg(lift=("lift", "mean"))
    )
    rel = model_lift[model_lift["regime"] == "reliable"]["lift"].values
    inv = model_lift[model_lift["regime"] == "inverted"]["lift"].values
    mw_p = float("nan")
    mw_stat = float("nan")
    if len(rel) >= 1 and len(inv) >= 1:
        try:
            mw_stat, mw_p = mannwhitneyu(rel, inv, alternative="two-sided")
        except ValueError:
            pass

    print("\n=== Mann-Whitney U (selection lift: reliable vs inverted) ===", flush=True)
    print(f"U={mw_stat}, p={mw_p}", flush=True)

    # Per-regime lift CI (bootstrap over splits, mean lift per split pooled)
    print("\n=== Per-regime mean lift (95% bootstrap CI) ===", flush=True)
    regime_ci: Dict[str, Tuple[float, float, float]] = {}
    for reg in ["reliable", "inverted", "flat", "unknown"]:
        lifts = long_df[long_df["regime"] == reg]["lift"].values
        if len(lifts) == 0:
            continue
        m = float(np.mean(lifts))
        rng = np.random.default_rng(12345)
        bs = []
        for _ in range(args.bootstrap_iters):
            sample = rng.choice(lifts, size=len(lifts), replace=True)
            bs.append(float(np.mean(sample)))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        regime_ci[reg] = (m, float(lo), float(hi))
        print(f"  {reg}: mean={m:.4f}, CI=[{lo:.4f}, {hi:.4f}]", flush=True)

    # --- Summary table (stdout) ---
    print("\n" + "=" * 100, flush=True)
    print(
        f"{'Model':<18} | {'Regime':<10} | {'FRS_cal':>8} | {'BoN_Reas':>9} | {'Rand_Reas':>10} | {'Lift':>7} | {'BoN_Acc':>8} | {'Rand_Acc':>9}",
        flush=True,
    )
    print("=" * 100, flush=True)
    tbl = (
        long_df.groupby(["model", "regime"], as_index=False)
        .agg(
            frs_cal=("frs_cal", "mean"),
            bon_r=("bon_reasoning", "mean"),
            rand_r=("random_reasoning", "mean"),
            lift=("lift", "mean"),
            bon_a=("bon_accuracy", "mean"),
            rand_a=("random_accuracy", "mean"),
        )
        .sort_values(["regime", "model"])
    )
    for _, r in tbl.iterrows():
        print(
            f"{r['model']:<18} | {r['regime']:<10} | {r['frs_cal']:8.2f} | {r['bon_r']:9.2f} | {r['rand_r']:10.2f} | {r['lift']:7.3f} | {r['bon_a']:8.3f} | {r['rand_a']:9.3f}",
            flush=True,
        )

    # --- Figure A: scatter ---
    fig_a, ax = plt.subplots(figsize=(6.5, 5.5))
    palette = {"reliable": "#2ca02c", "inverted": "#d62728", "flat": "#7f7f7f", "unknown": "#bbbbbb"}
    for reg in agg["regime"].unique():
        sub = agg[agg["regime"] == reg]
        ax.errorbar(
            sub["frs_cal_mean"],
            sub["bon_reason_mean"],
            xerr=sub["frs_cal_std"].fillna(0),
            yerr=sub["bon_reason_std"].fillna(0),
            fmt="o",
            label=reg,
            color=palette.get(reg, "#333333"),
            markersize=6,
            alpha=0.85,
            capsize=2,
            elinewidth=0.8,
        )

    lo = float(min(agg["frs_cal_mean"].min(), agg["bon_reason_mean"].min()) - 5)
    hi = float(max(agg["frs_cal_mean"].max(), agg["bon_reason_mean"].max()) + 5)
    ax.plot([lo, hi], [lo, hi], ls="--", color="#999999", lw=1, zorder=0)
    ax.set_xlabel(
        rf"FRS$_\mathrm{{cal}}$ (top {top_frac * 100:.0f}% mean reasoning, 0–100)"
    )
    ax.set_ylabel(r"BoN reasoning (confidence-selected, test split)")
    ax.set_title(
        rf"FRS predicts confidence-selected reasoning quality ($\rho_s={rho_full:.3f}$, $p={p_fb:.2e}$)"
        f"\n95% CI [{rho_lo:.3f}, {rho_hi:.3f}] | N={len(agg)} model–benchmark pairs"
    )
    ax.legend(title="Regime", frameon=False, loc="lower right")
    ax.grid(True, alpha=0.4)
    sns.despine(ax=ax)
    plt.tight_layout()
    fig_a.savefig(os.path.join(args.output_dir, "fig_a_frs_vs_bon.pdf"))
    fig_a.savefig(os.path.join(args.output_dir, "fig_a_frs_vs_bon.png"))
    plt.close(fig_a)

    # --- Figure B: bar chart per model (mean lift ± std across re-splits, after averaging benchmarks) ---
    lift_per_split = long_df.groupby(["model", "split"], as_index=False).agg(
        lift_ms=("lift", "mean")
    )
    lift_by_m = lift_per_split.groupby("model").agg(
        lift_mean=("lift_ms", "mean"), lift_std=("lift_ms", "std")
    )
    lift_by_m["lift_std"] = lift_by_m["lift_std"].fillna(0.0)
    model_order = lift_by_m["lift_mean"].sort_values(ascending=False).index.tolist()
    lift_by_m = lift_by_m.reindex(model_order)
    regimes = [regime_for_model(m) for m in model_order]
    colors = [palette.get(r, "#888888") for r in regimes]

    fig_b, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(model_order))
    ax.bar(x, lift_by_m["lift_mean"], yerr=lift_by_m["lift_std"], color=colors, ecolor="#444444", capsize=3, edgecolor="#333333", linewidth=0.5)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(model_order, rotation=35, ha="right")
    ax.set_ylabel(r"Selection lift (BoN reasoning $-$ random mean)")
    ax.set_title("Confidence-based selection: reasoning lift by model (mean ± std over re-splits)")
    sns.despine(ax=ax)
    plt.tight_layout()
    fig_b.savefig(os.path.join(args.output_dir, "fig_b_lift_by_model.pdf"))
    fig_b.savefig(os.path.join(args.output_dir, "fig_b_lift_by_model.png"))
    plt.close(fig_b)

    # Save stats summary
    with open(os.path.join(args.output_dir, "summary_stats.txt"), "w", encoding="utf-8") as f:
        f.write(f"Spearman FRS_cal vs BoN_reasoning: rho={rho_full}, p={p_fb}, CI=[{rho_lo},{rho_hi}]\n")
        f.write(
            f"Spearman Acc_cal vs BoN_reasoning: rho={rho_ab}, p={p_ab}, CI=[{rho_ab_lo},{rho_ab_hi}]\n"
        )
        f.write(f"Mann-Whitney U (reliable vs inverted lift): U={mw_stat}, p={mw_p}\n")
        f.write(f"Tertiary Spearman rho gap (FRS - Acc) vs BoN_reasoning: {rho_full - rho_ab}\n")
        for reg, (m, lo, hi) in regime_ci.items():
            f.write(f"Regime {reg}: mean_lift={m}, CI=[{lo},{hi}]\n")

    print(f"\nSaved figures and stats to {args.output_dir}/", flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
