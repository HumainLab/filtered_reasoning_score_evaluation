#!/usr/bin/env python3
"""
Restricted pairwise analysis: FRS vs accuracy as predictors of held-out BoN reasoning,
among accuracy-similar model pairs (within benchmark).

Reads: downstream_results/aggregated_model_benchmark.csv (from downstream_validation.py)

Usage:
  python restricted_pair_analysis.py \\
    --input_csv downstream_results/aggregated_model_benchmark.csv \\
    --output_dir downstream_results/restricted_pairs/
"""

from __future__ import annotations

import argparse
import os
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

mpl.rcParams.update(
    {
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "savefig.bbox": "tight",
    }
)

REGIME_PALETTE = {
    "reliable": "#2ca02c",
    "inverted": "#d62728",
    "flat": "#7f7f7f",
    "unknown": "#bbbbbb",
}


def acc_pp_diff(a: float, b: float) -> float:
    """Absolute accuracy gap in percentage points (0–100 scale)."""
    return abs(float(a) - float(b)) * 100.0


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """All unordered within-benchmark model pairs with gap features."""
    rows: List[Dict] = []
    for bench, g in df.groupby("benchmark"):
        models = g["model"].tolist()
        idxs = list(range(len(g)))
        for i1, i2 in combinations(idxs, 2):
            r1 = g.iloc[i1]
            r2 = g.iloc[i2]
            frs1, frs2 = float(r1["frs_cal_mean"]), float(r2["frs_cal_mean"])
            a1, a2 = float(r1["acc_cal_mean"]), float(r2["acc_cal_mean"])
            b1, b2 = float(r1["bon_reason_mean"]), float(r2["bon_reason_mean"])
            acc_gap_pp = acc_pp_diff(a1, a2)
            frs_gap = abs(frs1 - frs2)
            bon_gap = abs(b1 - b2)
            acc_gap_raw = abs(a1 - a2)
            rows.append(
                {
                    "benchmark": bench,
                    "model_a": r1["model"],
                    "model_b": r2["model"],
                    "regime_a": r1["regime"],
                    "regime_b": r2["regime"],
                    "acc_gap_pp": acc_gap_pp,
                    "acc_gap_raw": acc_gap_raw,
                    "frs_gap": frs_gap,
                    "bon_gap": bon_gap,
                    "frs_gt_acc_pp": frs_gap > acc_gap_pp,
                    "sign_frs": np.sign(frs1 - frs2),
                    "sign_acc": np.sign(a1 - a2),
                    "sign_bon": np.sign(b1 - b2),
                    "frs1": frs1,
                    "frs2": frs2,
                    "acc1": a1,
                    "acc2": a2,
                    "bon1": b1,
                    "bon2": b2,
                }
            )
    return pd.DataFrame(rows)


def concordance_signed(
    sign_pred: np.ndarray,
    sign_bon: np.ndarray,
) -> Tuple[float, int]:
    """Fraction where signs match, excluding ties in prediction or BoN."""
    ok = (sign_pred != 0) & (sign_bon != 0)
    n = int(ok.sum())
    if n == 0:
        return float("nan"), 0
    return float(np.mean(sign_pred[ok] == sign_bon[ok])), n


def bootstrap_concordance_ci(
    pair_df: pd.DataFrame,
    mask: np.ndarray,
    n_boot: int,
    seed: int,
) -> Tuple[float, float, float, float, float, float]:
    """Bootstrap 95% CI for FRS and Acc concordance on masked pairs."""
    sub = pair_df.loc[mask].copy()
    if len(sub) == 0:
        return (float("nan"),) * 6

    rng = np.random.default_rng(seed)

    def rate(pred: str) -> float:
        sp = sub[pred].values
        sb = sub["sign_bon"].values
        ok = (sp != 0) & (sb != 0)
        if not ok.any():
            return float("nan")
        return float(np.mean(sp[ok] == sb[ok]))

    obs_frs = rate("sign_frs")
    obs_acc = rate("sign_acc")

    frs_bs: List[float] = []
    acc_bs: List[float] = []
    n = len(sub)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = sub.iloc[idx]
        sp, sb = s["sign_frs"].values, s["sign_bon"].values
        ok = (sp != 0) & (sb != 0)
        if ok.any():
            frs_bs.append(float(np.mean(sp[ok] == sb[ok])))
        sp, sb = s["sign_acc"].values, s["sign_bon"].values
        ok = (sp != 0) & (sb != 0)
        if ok.any():
            acc_bs.append(float(np.mean(sp[ok] == sb[ok])))

    def ci(xs: List[float]) -> Tuple[float, float]:
        if not xs:
            return float("nan"), float("nan")
        a = np.array(xs)
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    f_lo, f_hi = ci(frs_bs)
    a_lo, a_hi = ci(acc_bs)
    return obs_frs, f_lo, f_hi, obs_acc, a_lo, a_hi


def spearman_safe(x: np.ndarray, y: np.ndarray, min_n: int = 5) -> Tuple[float, float]:
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < min_n:
        return float("nan"), float("nan")
    r, p = spearmanr(x, y)
    return float(r), float(p)


def analyze_threshold(
    pair_df: pd.DataFrame,
    tau_pp: Optional[float],
) -> Dict:
    """tau_pp None means all pairs (infinity)."""
    if tau_pp is None:
        mask = np.ones(len(pair_df), dtype=bool)
        label = "All"
    else:
        mask = pair_df["acc_gap_pp"].values <= tau_pp + 1e-9
        label = f"≤{tau_pp:g}pp"

    sub = pair_df.loc[mask]
    n_pairs = len(sub)
    frs_gap = sub["frs_gap"].values
    acc_gap_pp = sub["acc_gap_pp"].values
    bon_gap = sub["bon_gap"].values

    rho_frs, p_frs = spearman_safe(frs_gap, bon_gap)
    # Rank correlation invariant to linear scale; use pp gaps for Acc vs BoN gap
    rho_acc, p_acc = spearman_safe(acc_gap_pp, bon_gap)

    frac_frs_gt = (
        float(np.mean(sub["frs_gt_acc_pp"]))
        if n_pairs
        else float("nan")
    )
    mean_amp = (
        float(np.mean(frs_gap) / np.mean(acc_gap_pp))
        if n_pairs and np.mean(acc_gap_pp) > 1e-12
        else float("nan")
    )

    cf, nf = concordance_signed(sub["sign_frs"].values, sub["sign_bon"].values)
    ca, na = concordance_signed(sub["sign_acc"].values, sub["sign_bon"].values)

    return {
        "threshold_label": label,
        "tau_pp": np.inf if tau_pp is None else tau_pp,
        "n_pairs": n_pairs,
        "rho_frs_gap_bon_gap": rho_frs,
        "p_frs": p_frs,
        "rho_acc_gap_bon_gap": rho_acc,
        "p_acc": p_acc,
        "frac_frs_gap_gt_acc_pp": frac_frs_gt,
        "amplification_mean_frs_over_mean_acc_pp": mean_amp,
        "frs_concordance": cf,
        "frs_concordance_n": nf,
        "acc_concordance": ca,
        "acc_concordance_n": na,
    }


def per_benchmark_spearman(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for bench, g in df.groupby("benchmark"):
        if len(g) < 3:
            continue
        x_f = g["frs_cal_mean"].values
        y = g["bon_reason_mean"].values
        x_a = g["acc_cal_mean"].values
        rf, pf = spearmanr(x_f, y)
        ra, pa = spearmanr(x_a, y)
        out.append(
            {
                "benchmark": bench,
                "n_models": len(g),
                "rho_frs_bon": float(rf),
                "p_frs": float(pf),
                "rho_acc_bon": float(ra),
                "p_acc": float(pa),
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_csv",
        type=str,
        default="outputs/downstream_results/aggregated_model_benchmark.csv",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default="outputs/downstream_results/restricted_pairs",
    )
    ap.add_argument("--bootstrap_iters", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    if "bon_reason_mean" not in df.columns:
        for alt in ("bon_reasoning_mean", "BoN_reasoning_mean", "bon_reasoning"):
            if alt in df.columns:
                df = df.rename(columns={alt: "bon_reason_mean"})
                break
    required = {
        "model",
        "benchmark",
        "regime",
        "frs_cal_mean",
        "acc_cal_mean",
        "bon_reason_mean",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    pair_df = build_pairs(df)
    pair_df.to_csv(os.path.join(args.output_dir, "all_pairs.csv"), index=False)

    taus: List[Optional[float]] = [3, 5, 10, 15, 20, None]

    summary_rows: List[Dict] = []
    for tau in taus:
        row = analyze_threshold(pair_df, tau)
        if tau is None:
            m = np.ones(len(pair_df), dtype=bool)
        else:
            m = pair_df["acc_gap_pp"].values <= tau + 1e-9
        _, f_lo, f_hi, _, a_lo, a_hi = bootstrap_concordance_ci(
            pair_df, m, args.bootstrap_iters, args.seed + (0 if tau is None else int(tau))
        )
        row["frs_concordance_boot_ci_low"] = f_lo
        row["frs_concordance_boot_ci_high"] = f_hi
        row["acc_concordance_boot_ci_low"] = a_lo
        row["acc_concordance_boot_ci_high"] = a_hi
        summary_rows.append(row)

    # Part 2: dedicated τ≤5pp row already in summary; duplicate explicit Part-2 table
    mask_5 = pair_df["acc_gap_pp"].values <= 5 + 1e-9
    p2_frs, p2_f_lo, p2_f_hi, p2_a, p2_a_lo, p2_a_hi = bootstrap_concordance_ci(
        pair_df, mask_5, args.bootstrap_iters, args.seed + 1
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.output_dir, "restricted_pair_results.csv"), index=False)

    bench_tbl = per_benchmark_spearman(df)
    bench_tbl.to_csv(os.path.join(args.output_dir, "per_benchmark_spearman.csv"), index=False)

    # ── Print table ─────────────────────────────────────────────────────
    print("\n=== Threshold summary (within-benchmark pairs; acc gap in pp) ===\n")
    print(
        f"{'Threshold':<12} | {'N_pairs':>7} | {'ρ(FRS_gap,BoN_gap)':>18} | {'ρ(Acc_pp,BoN_gap)':>18} | "
        f"{'FRS_conc':>9} | {'Acc_conc':>9} | {'Amplif':>8} | {'P(FRS>Acc_pp)':>13}"
    )
    for _, r in summary_df.iterrows():
        tau_lbl = r["threshold_label"]
        print(
            f"{tau_lbl:<12} | {int(r['n_pairs']):7d} | {r['rho_frs_gap_bon_gap']:18.4f} | "
            f"{r['rho_acc_gap_bon_gap']:18.4f} | {r['frs_concordance']:9.3f} | {r['acc_concordance']:9.3f} | "
            f"{r['amplification_mean_frs_over_mean_acc_pp']:8.3f} | {r['frac_frs_gap_gt_acc_pp']:13.3f}"
        )

    print("\n=== Part 2: τ ≤ 5pp concordance (bootstrap 95% CI) ===")
    print(f"  FRS sign vs BoN sign: {p2_frs:.3f} [{p2_f_lo:.3f}, {p2_f_hi:.3f}]")
    print(f"  Acc sign vs BoN sign: {p2_a:.3f}   [{p2_a_lo:.3f}, {p2_a_hi:.3f}]")

    print("\n=== Part 3: Per-benchmark Spearman (9 models) ===\n")
    print(bench_tbl.to_string(index=False))

    # ── Figure C: amplification + fraction ──────────────────────────────
    fig_c, ax1 = plt.subplots(figsize=(7, 4))
    taus_finite = [t for t in taus if t is not None]
    amps = []
    fracs = []
    ns = []
    for tau in taus_finite:
        sub = pair_df[pair_df["acc_gap_pp"] <= tau]
        ns.append(len(sub))
        if len(sub) and sub["acc_gap_pp"].mean() > 1e-12:
            amps.append(sub["frs_gap"].mean() / sub["acc_gap_pp"].mean())
        else:
            amps.append(float("nan"))
        fracs.append(float(sub["frs_gt_acc_pp"].mean()) if len(sub) else float("nan"))

    ax1.plot(taus_finite, amps, "o-", color="#1f77b4", label="Mean |ΔFRS| / mean |ΔAcc| (pp)")
    ax1.set_xlabel(r"Accuracy-gap threshold $\tau$ (pp)")
    ax1.set_ylabel("Amplification ratio")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(taus_finite, fracs, "s--", color="#ff7f0e", label="P(|ΔFRS| > |ΔAcc| pp)")
    ax2.set_ylabel("Fraction pairs")
    ax2.set_ylim(0, 1.05)
    lines, labels = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + l2, labels + lab2, loc="upper right", frameon=False)
    ax1.set_title("C: FRS amplification vs accuracy gap threshold (downstream)")
    plt.tight_layout()
    fig_c.savefig(os.path.join(args.output_dir, "fig_c_amplification.pdf"))
    fig_c.savefig(os.path.join(args.output_dir, "fig_c_amplification.png"))
    plt.close(fig_c)

    # ── Figure D: concordance bars at τ ∈ {3,5,10,15,20} ───────────────
    fig_d, ax = plt.subplots(figsize=(8, 4))
    plot_taus = [3, 5, 10, 15, 20]
    x = np.arange(len(plot_taus))
    w = 0.35
    frs_m, frs_lo, frs_hi = [], [], []
    acc_m, acc_lo, acc_hi = [], [], []
    for tau in plot_taus:
        m = pair_df["acc_gap_pp"].values <= tau
        o_f, f_lo, f_hi, o_a, a_lo, a_hi = bootstrap_concordance_ci(
            pair_df, m, args.bootstrap_iters, args.seed + tau
        )
        frs_m.append(o_f)
        frs_lo.append(f_lo)
        frs_hi.append(f_hi)
        acc_m.append(o_a)
        acc_lo.append(a_lo)
        acc_hi.append(a_hi)

    ax.bar(x - w / 2, frs_m, w, label="FRS sign concordance", color="#1f77b4", yerr=[np.array(frs_m) - np.array(frs_lo), np.array(frs_hi) - np.array(frs_m)], capsize=3, ecolor="#333")
    ax.bar(x + w / 2, acc_m, w, label="Acc sign concordance", color="#ff7f0e", yerr=[np.array(acc_m) - np.array(acc_lo), np.array(acc_hi) - np.array(acc_m)], capsize=3, ecolor="#333")
    ax.set_xticks(x)
    ax.set_xticklabels([f"≤{t}pp" for t in plot_taus])
    ax.set_ylabel("Concordance rate")
    ax.set_ylim(0, 1.05)
    ax.axvspan(-0.5, 1.5, alpha=0.08, color="green", label="τ≤5pp")
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("D: Directional prediction vs BoN reasoning (within-benchmark pairs)")
    plt.tight_layout()
    fig_d.savefig(os.path.join(args.output_dir, "fig_d_concordance.pdf"))
    fig_d.savefig(os.path.join(args.output_dir, "fig_d_concordance.png"))
    plt.close(fig_d)

    # ── Figure E: small multiples ───────────────────────────────────────
    benchmarks = sorted(df["benchmark"].unique())
    n_b = len(benchmarks)
    ncols = 3
    nrows = int(np.ceil(n_b / ncols))
    fig_e, axes = plt.subplots(nrows, ncols, figsize=(10, 3.2 * nrows), squeeze=False)
    regime_order = sorted(df["regime"].dropna().unique(), key=str)
    legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            markersize=7,
            markerfacecolor=REGIME_PALETTE.get(reg, "#333333"),
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=reg,
        )
        for reg in regime_order
    ]
    for idx, bench in enumerate(benchmarks):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        g = df[df["benchmark"] == bench]
        present = set(g["regime"].dropna().unique())
        for reg in regime_order:
            if reg not in present:
                continue
            sub = g[g["regime"] == reg]
            ax.scatter(
                sub["frs_cal_mean"],
                sub["bon_reason_mean"],
                c=REGIME_PALETTE.get(reg, "#333333"),
                s=45,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.5,
            )
        if len(g) >= 3:
            rf, _ = spearmanr(g["frs_cal_mean"], g["bon_reason_mean"])
            ax.text(
                0.05,
                0.95,
                rf"$\rho_s={rf:.2f}$",
                transform=ax.transAxes,
                va="top",
                fontsize=9,
            )
        ax.set_xlabel(r"FRS$_\mathrm{cal}$")
        ax.set_ylabel("BoN reasoning")
        ax.set_title(bench)
        ax.grid(True, alpha=0.3)
    # hide empty
    for j in range(len(benchmarks), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)
    fig_e.legend(
        handles=legend_handles,
        labels=[h.get_label() for h in legend_handles],
        loc="upper center",
        ncol=min(4, len(legend_handles)),
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )
    plt.tight_layout()
    fig_e.savefig(os.path.join(args.output_dir, "fig_e_benchmark_grid.pdf"))
    fig_e.savefig(os.path.join(args.output_dir, "fig_e_benchmark_grid.png"))
    plt.close(fig_e)

    print(f"\nSaved figures and CSVs to {args.output_dir}/", flush=True)


if __name__ == "__main__":
    main()
