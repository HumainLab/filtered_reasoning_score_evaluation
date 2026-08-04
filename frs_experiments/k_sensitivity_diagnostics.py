#!/usr/bin/env python3
"""
Deep diagnostics for FRS k-sensitivity (why high recall, boundary cases, regime flips).

Reuses loaders and helpers from ``k_sensitivity_analysis``.

Outputs (CSV under ``--output_dir``, figures under ``--figures_dir``):
  variance_decomposition.csv, top10_concentration.csv,
  boundary_analysis.csv, boundary_summary.csv,
  regime_flip_detail.csv, per_model_recall.csv, per_benchmark_recall.csv,
  ranking_stability.csv

Requires ``results/k_sensitivity_summary.csv`` (from ``k_sensitivity_analysis.py``) for Analysis 5.
Runtime ~8–10 min on the full 54 pair × 10 resample setting (large JSONL reads).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from typing import Any, Dict, List, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from k_sensitivity_analysis import (
    REGIME_COLOR,
    build_problems_from_raw,
    discover_judge_files,
    load_frs_at_10_csv,
    load_judge_scores_top_bin,
    load_jsonl_raw,
    pool_traces,
    regime_for_model,
    snr_cohen_style,
    subsample_problems,
    top_frac_count,
    top_frac_set,
)
from topk_ablation import build_file_map

LOG = logging.getLogger("k_sensitivity_diagnostics")


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(ch)


def set_pub_rc() -> None:
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def stable_pair_seed(model: str, benchmark: str, seed: int, res_id: int, kv: int) -> np.random.Generator:
    pair_h = int(hashlib.md5(f"{model}::{benchmark}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed + res_id * 10007 + kv * 131 + (pair_h % 100000))


def variance_decomposition_row(
    problems: List[Dict[str, Any]],
) -> Dict[str, float]:
    """ANOVA-style SS_between / SS_total for confidence."""
    all_conf: List[float] = []
    group_means: List[float] = []
    group_ss_within: List[float] = []
    ns: List[int] = []
    for p in problems:
        c = [t["confidence"] for t in p["traces"]]
        all_conf.extend(c)
        arr = np.asarray(c, dtype=np.float64)
        ns.append(len(arr))
        group_means.append(float(np.mean(arr)))
        group_ss_within.append(float(np.sum((arr - np.mean(arr)) ** 2)))

    x = np.asarray(all_conf, dtype=np.float64)
    mu = float(np.mean(x))
    N = len(x)
    J = len(problems)
    total_ss = float(np.sum((x - mu) ** 2))
    between_ss = sum(nj * (mj - mu) ** 2 for nj, mj in zip(ns, group_means))
    within_ss = sum(group_ss_within)
    total_var = float(np.var(x, ddof=0))
    between_var = between_ss / N if N else 0.0
    within_var_mean = float(np.mean([ss / n for ss, n in zip(group_ss_within, ns) if n > 0])) if ns else 0.0

    between_fraction = (between_ss / total_ss) if total_ss > 0 else float("nan")
    return {
        "between_ss": between_ss,
        "within_ss": within_ss,
        "total_ss": total_ss,
        "between_var": between_var,
        "within_var_mean": within_var_mean,
        "total_var": total_var,
        "between_fraction": between_fraction,
    }


def trace_rank_map_k16(pool: List[Dict[str, Any]]) -> Dict[str, int]:
    """Rank 1 = highest confidence among N traces."""
    sorted_t = sorted(
        pool,
        key=lambda t: (-t["confidence"], t["problem_id"], t["trace_idx"]),
    )
    return {sorted_t[i]["trace_id"]: i + 1 for i in range(len(sorted_t))}


def k16_rank_percentile(rank: int, n: float) -> float:
    """100 * rank / N: top-10% have values ≤ 10 (same scale as 'top fraction * 100')."""
    if n <= 0:
        return float("nan")
    return 100.0 * float(rank) / float(n)


def top10_traces_per_problem(
    pool: List[Dict[str, Any]],
    top_ids: Set[str],
) -> Tuple[Dict[Any, int], int]:
    """Count traces per problem_id within top set."""
    counts: Dict[Any, int] = {}
    for t in pool:
        tid = t["trace_id"]
        if tid not in top_ids:
            continue
        pid = t["problem_id"]
        counts[pid] = counts.get(pid, 0) + 1
    return counts, len(counts)


def concentration_ratio_top20pct(counts: Dict[Any, int]) -> float:
    """Fraction of traces in top set from top 20% of contributing problems by count."""
    if not counts:
        return float("nan")
    vals = sorted(counts.values(), reverse=True)
    n_contrib = len(vals)
    k_top = max(1, int(np.ceil(0.2 * n_contrib)))
    num_traces = sum(vals)
    top_sum = sum(vals[:k_top])
    return top_sum / num_traces if num_traces else float("nan")


def run(
    data_dir: str,
    judging_dir: str,
    frs_csv: str,
    summary_csv: str,
    k_sub: int,
    num_resamples: int,
    seed: int,
    output_dir: str,
    figures_dir: str,
    top_frac: float = 0.10,
    expected_k: int = 16,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    set_pub_rc()

    file_map = build_file_map(data_dir)
    judge_map = discover_judge_files(judging_dir)
    frs_map = load_frs_at_10_csv(frs_csv)

    var_rows: List[Dict[str, Any]] = []
    conc_rows: List[Dict[str, Any]] = []
    boundary_rows: List[Dict[str, Any]] = []
    flip_rows: List[Dict[str, Any]] = []
    traces_per_problem_all: List[int] = []
    new_entrant_pcts: List[float] = []
    pair_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for (model, benchmark), jsonl_path in sorted(file_map.items()):
        raw = load_jsonl_raw(jsonl_path)
        problems, _warns = build_problems_from_raw(raw, expected_k=expected_k)
        if not problems:
            continue

        pool = pool_traces(problems)
        N = len(pool)
        n_prob = len(problems)
        baseline_top = top_frac_set(pool, top_frac)
        n_top = len(baseline_top)

        # --- Analysis 1 ---
        vd = variance_decomposition_row(problems)
        var_rows.append(
            {
                "model": model,
                "benchmark": benchmark,
                "between_ss": vd["between_ss"],
                "within_ss": vd["within_ss"],
                "total_ss": vd["total_ss"],
                "between_var": vd["between_var"],
                "within_var": vd["within_var_mean"],
                "total_var": vd["total_var"],
                "between_fraction": vd["between_fraction"],
            }
        )

        # --- Analysis 2 ---
        counts, n_contrib = top10_traces_per_problem(pool, baseline_top)
        traces_per_problem_all.extend(counts.values())
        conc_ratio = concentration_ratio_top20pct(counts)
        mean_tpc = float(np.mean(list(counts.values()))) if counts else float("nan")
        max_tpc = int(max(counts.values())) if counts else 0
        n_zero = n_prob - n_contrib
        conc_rows.append(
            {
                "model": model,
                "benchmark": benchmark,
                "num_contributing_problems": n_contrib,
                "total_problems": n_prob,
                "fraction_contributing": n_contrib / n_prob if n_prob else float("nan"),
                "mean_traces_per_contributing_problem": mean_tpc,
                "max_traces_per_contributing_problem": max_tpc,
                "concentration_ratio_top20pct": conc_ratio,
                "n_problems_with_0_in_top10": n_zero,
                "n_top10_traces": n_top,
            }
        )

        rank_map = trace_rank_map_k16(pool)
        conf_all = np.array([t["confidence"] for t in pool], dtype=np.float64)
        corr_mask = np.array([t["correct"] for t in pool], dtype=bool)
        snr_k16 = snr_cohen_style(conf_all[corr_mask], conf_all[~corr_mask])

        judge_path = judge_map.get((model, benchmark))
        judge_top: Dict[str, float] = load_judge_scores_top_bin(judge_path) if judge_path else {}

        pair_cache[(model, benchmark)] = {
            "problems": problems,
            "pool": pool,
            "baseline_top": baseline_top,
            "rank_map": rank_map,
            "N": N,
            "snr_k16": snr_k16,
            "judge_path": judge_path,
            "judge_top": judge_top,
        }

        for res_id in range(num_resamples):
            rng = stable_pair_seed(model, benchmark, seed, res_id, k_sub)
            sub_problems = subsample_problems(problems, k_sub, rng)
            if not sub_problems:
                continue
            sub_pool = pool_traces(sub_problems)
            sub_top = top_frac_set(sub_pool, top_frac)

            new_entrants = sub_top - baseline_top
            displaced = baseline_top - sub_top

            for tid in new_entrants:
                r = rank_map.get(tid)
                if r is not None:
                    new_entrant_pcts.append(k16_rank_percentile(r, N))

            percentiles: List[float] = []
            for tid in new_entrants:
                r = rank_map.get(tid)
                if r is not None:
                    percentiles.append(k16_rank_percentile(r, N))

            med = float(np.median(percentiles)) if percentiles else float("nan")
            p95 = float(np.percentile(percentiles, 95)) if len(percentiles) else float("nan")

            boundary_rows.append(
                {
                    "model": model,
                    "benchmark": benchmark,
                    "resample_id": res_id,
                    "num_new_entrants": len(new_entrants),
                    "num_displaced": len(displaced),
                    "median_k16_percentile_of_new_entrants": med,
                    "p95_k16_percentile_of_new_entrants": p95,
                }
            )

            conf_s = np.array([t["confidence"] for t in sub_pool], dtype=np.float64)
            cm = np.array([t["correct"] for t in sub_pool], dtype=bool)
            snr_sub = snr_cohen_style(conf_s[cm], conf_s[~cm])
            flip = bool(
                np.isfinite(snr_k16)
                and np.isfinite(snr_sub)
                and (np.sign(snr_k16) != np.sign(snr_sub))
            )
            flip_rows.append(
                {
                    "model": model,
                    "benchmark": benchmark,
                    "resample_id": res_id,
                    "snr_k16": snr_k16,
                    "snr_k8": snr_sub,
                    "abs_snr_k16": abs(snr_k16) if np.isfinite(snr_k16) else float("nan"),
                    "regime_flip": flip,
                }
            )

    df_var = pd.DataFrame(var_rows)
    df_conc = pd.DataFrame(conc_rows)
    df_boundary = pd.DataFrame(boundary_rows)
    df_flip = pd.DataFrame(flip_rows)

    df_var.to_csv(os.path.join(output_dir, "variance_decomposition.csv"), index=False)
    df_conc.to_csv(os.path.join(output_dir, "top10_concentration.csv"), index=False)
    df_boundary.to_csv(os.path.join(output_dir, "boundary_analysis.csv"), index=False)

    df_boundary_sum = (
        df_boundary.groupby(["model", "benchmark"], as_index=False)
        .agg(
            num_new_entrants=("num_new_entrants", "mean"),
            num_displaced=("num_displaced", "mean"),
            median_k16_pct_new=("median_k16_percentile_of_new_entrants", "mean"),
            p95_k16_pct_new=("p95_k16_percentile_of_new_entrants", "mean"),
        )
    )
    df_boundary_sum.to_csv(os.path.join(output_dir, "boundary_summary.csv"), index=False)

    df_flip_detail = df_flip[df_flip["regime_flip"]].copy()
    df_flip_detail.to_csv(os.path.join(output_dir, "regime_flip_detail.csv"), index=False)

    # --- Figures: histogram traces per problem in top-10% ---
    if traces_per_problem_all:
        fig, ax = plt.subplots(figsize=(8, 4))
        bins = np.arange(0.5, max(traces_per_problem_all) + 1.5, 1.0)
        ax.hist(traces_per_problem_all, bins=bins, color="steelblue", edgecolor="white")
        ax.set_xlabel("Traces per problem (within global top-10% set, k=16)")
        ax.set_ylabel("Count (problem × model-benchmark cells)")
        ax.set_title("Distribution: how many top-10% traces each problem contributes")
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "traces_per_problem_histogram.png"))
        plt.close(fig)

    if new_entrant_pcts:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(new_entrant_pcts, bins=40, color="coral", edgecolor="white")
        ax.axvline(10.0, color="black", ls="--", lw=1, label="Top-10% cutoff (10)")
        ax.set_xlabel("k=16 pool rank percentile (100×rank/N; lower = more confident)")
        ax.set_ylabel("Count (new entrants, all resamples)")
        ax.set_title(f"New entrants (k={k_sub} top-10% not in k=16 top-10%): k=16 rank percentile")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "new_entrant_percentile_distribution.png"))
        plt.close(fig)

    # --- Regime flip figure ---
    if not df_flip.empty:
        sub = df_flip[np.isfinite(df_flip["abs_snr_k16"])].copy()
        sub["flip_str"] = sub["regime_flip"].map({True: "regime flip", False: "no flip"})
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.violinplot(
            data=sub,
            x="flip_str",
            y="abs_snr_k16",
            order=["no flip", "regime flip"],
            ax=ax,
            inner="box",
            palette={"no flip": "#a6cee3", "regime flip": "#fb9a99"},
        )
        ax.set_xlabel("")
        ax.set_ylabel("|SNR| at k=16")
        ax.set_title(f"Distribution of |SNR_k16| vs sign flip of k={k_sub} subsampled SNR")
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "regime_flip_snr_distribution.png"))
        plt.close(fig)

    # --- Analysis 5: per-model recall from summary CSV ---
    df_sum = pd.read_csv(summary_csv)
    df_sum_k = df_sum[df_sum["k"] == k_sub].copy()
    per_model = (
        df_sum_k.groupby("model", as_index=False)
        .agg(mean_recall=("mean_recall", "mean"), std_recall=("mean_recall", "std"))
    )
    per_bench = (
        df_sum_k.groupby("benchmark", as_index=False)
        .agg(mean_recall=("mean_recall", "mean"), std_recall=("mean_recall", "std"))
    )

    model_cv: Dict[str, float] = {}
    for model in per_model["model"].unique():
        confs: List[float] = []
        for (m, _), d in pair_cache.items():
            if m != model:
                continue
            for t in d["pool"]:
                confs.append(t["confidence"])
        arr = np.asarray(confs, dtype=np.float64)
        model_cv[model] = float(np.std(arr, ddof=0) / np.mean(arr)) if len(arr) and np.mean(arr) > 0 else float("nan")

    per_model["confidence_cv"] = per_model["model"].map(model_cv)
    per_model["regime"] = per_model["model"].map(regime_for_model)

    per_model.to_csv(os.path.join(output_dir, "per_model_recall.csv"), index=False)
    per_bench.to_csv(os.path.join(output_dir, "per_benchmark_recall.csv"), index=False)

    # recall vs CV scatter
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, r in per_model.iterrows():
        ax.scatter(
            r["confidence_cv"],
            r["mean_recall"],
            c=REGIME_COLOR.get(regime_for_model(str(r["model"])), "#888888"),
            s=45,
            edgecolors="k",
            linewidths=0.3,
        )
        ax.annotate(str(r["model"]), (r["confidence_cv"], r["mean_recall"]), fontsize=7, alpha=0.9)
    ax.set_xlabel("Coefficient of variation of confidence (std/mean, pooled traces)")
    ax.set_ylabel(f"Mean recall (k={k_sub} vs k=16 top-10%)")
    ax.set_title("Recall vs confidence spread by model")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "recall_vs_confidence_cv.png"))
    plt.close(fig)

    # Bar chart recall by model
    fig, ax = plt.subplots(figsize=(10, 4))
    order = per_model.sort_values("mean_recall", ascending=False)["model"].tolist()
    colors = [REGIME_COLOR.get(regime_for_model(m), "#888888") for m in order]
    y = [per_model.set_index("model").loc[m, "mean_recall"] for m in order]
    err = [per_model.set_index("model").loc[m, "std_recall"] for m in order]
    ax.bar(range(len(order)), y, yerr=err, color=colors, ecolor="#333", capsize=2, edgecolor="k", linewidth=0.3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylabel("Mean recall")
    ax.set_title(f"Mean recall by model (across benchmarks, k={k_sub})")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "recall_by_model.png"))
    plt.close(fig)

    corr_r_cv = (float("nan"), float("nan"))
    if len(per_model) >= 3:
        msk = np.isfinite(per_model["confidence_cv"]) & np.isfinite(per_model["mean_recall"])
        if msk.sum() >= 3:
            try:
                corr_r_cv = pearsonr(
                    per_model.loc[msk, "confidence_cv"],
                    per_model.loc[msk, "mean_recall"],
                )
            except Exception:
                corr_r_cv = (float("nan"), float("nan"))

    # --- Analysis 6: ranking stability ---
    ranking_rows: List[Dict[str, Any]] = []
    models_sorted = sorted({m for m, _ in file_map.keys()})
    benchmarks_sorted = sorted({b for _, b in file_map.keys()})

    for res_id in range(num_resamples):
        vec_frs: List[float] = []
        vec_p: List[float] = []
        for model in models_sorted:
            partials: List[float] = []
            frs_same_b: List[float] = []
            for benchmark in benchmarks_sorted:
                key = (model, benchmark)
                if key not in pair_cache:
                    continue
                d = pair_cache[key]
                problems = d["problems"]
                rng = stable_pair_seed(model, benchmark, seed, res_id, k_sub)
                sub_problems = subsample_problems(problems, k_sub, rng)
                if not sub_problems:
                    continue
                sub_pool = pool_traces(sub_problems)
                sub_top = top_frac_set(sub_pool, top_frac)
                judge_top = d.get("judge_top") or {}
                if not judge_top:
                    continue
                survived = [sc * 100.0 for tid, sc in judge_top.items() if tid in sub_top]
                if len(survived) >= 20 and key in frs_map:
                    partials.append(float(np.mean(survived)))
                    frs_same_b.append(frs_map[key])
            if partials:
                vec_p.append(float(np.mean(partials)))
                vec_frs.append(float(np.mean(frs_same_b)))

        rho = float("nan")
        n_ranked = len(vec_frs)
        if n_ranked >= 5:
            rho, _ = spearmanr(vec_frs, vec_p)

        ranking_rows.append(
            {
                "resample_id": res_id,
                "spearman_rho": rho,
                "num_models_ranked": n_ranked,
            }
        )

    df_rank = pd.DataFrame(ranking_rows)
    df_rank.to_csv(os.path.join(output_dir, "ranking_stability.csv"), index=False)

    # Bump chart: resample with Spearman closest to median ρ (need ≥5 models ranked)
    df_rank_ok = df_rank[(df_rank["num_models_ranked"] >= 5) & np.isfinite(df_rank["spearman_rho"])]
    if not df_rank_ok.empty:
        med = float(df_rank_ok["spearman_rho"].median())
        med_idx = int(df_rank_ok.iloc[(df_rank_ok["spearman_rho"] - med).abs().argsort()]["resample_id"].iloc[0])
        rows_bump: List[Tuple[str, float, float]] = []
        for model in models_sorted:
            parts: List[float] = []
            frs_list: List[float] = []
            for benchmark in benchmarks_sorted:
                key = (model, benchmark)
                if key not in pair_cache:
                    continue
                d = pair_cache[key]
                problems = d["problems"]
                rng = stable_pair_seed(model, benchmark, seed, med_idx, k_sub)
                sub_problems = subsample_problems(problems, k_sub, rng)
                sub_pool = pool_traces(sub_problems)
                sub_top = top_frac_set(sub_pool, top_frac)
                judge_top = d.get("judge_top") or {}
                if not judge_top:
                    continue
                survived = [sc * 100.0 for tid, sc in judge_top.items() if tid in sub_top]
                if len(survived) >= 20 and key in frs_map:
                    parts.append(float(np.mean(survived)))
                    frs_list.append(frs_map[key])
            if parts and frs_list and len(parts) == len(frs_list):
                rows_bump.append(
                    (
                        model,
                        float(np.mean(frs_list)),
                        float(np.mean(parts)),
                    )
                )

        if len(rows_bump) >= 2:
            df_b = pd.DataFrame(rows_bump, columns=["model", "frs_k16_mean", "partial_k8_mean"])
            df_b["rank_k16"] = df_b["frs_k16_mean"].rank(ascending=False, method="min")
            df_b["rank_k8"] = df_b["partial_k8_mean"].rank(ascending=False, method="min")
            fig, ax = plt.subplots(figsize=(8, 5))
            df_b = df_b.sort_values("rank_k16")
            for _, row in df_b.iterrows():
                m = row["model"]
                r0 = row["rank_k16"]
                r1 = row["rank_k8"]
                ax.plot([0, 1], [r0, r1], "o-", color="gray", alpha=0.65, lw=0.9, markersize=4)
                ax.text(-0.06, r0, m, ha="right", va="center", fontsize=7)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["k=16 FRS (mean over qualifying benchmarks)", f"k={k_sub} partial FRS"])
            ax.invert_yaxis()
            ax.set_ylabel("Rank (1 = best)")
            ax.set_title(f"Model ranking: resample {med_idx} (Spearman ρ nearest median = {med:.3f})")
            fig.tight_layout()
            fig.savefig(os.path.join(figures_dir, "ranking_comparison.png"))
            plt.close(fig)

    # --- Console tables ---
    print("\n--- Variance decomposition (first 8 rows) ---")
    print(df_var[["model", "benchmark", "between_fraction"]].head(8).to_string(index=False))
    print("\n--- Top-10% concentration (first 8 rows) ---")
    print(
        df_conc[
            [
                "model",
                "benchmark",
                "num_contributing_problems",
                "total_problems",
                "mean_traces_per_contributing_problem",
                "max_traces_per_contributing_problem",
                "concentration_ratio_top20pct",
            ]
        ]
        .head(8)
        .to_string(index=False)
    )

    # Consolidated summary
    bf_mean = float(df_var["between_fraction"].mean())
    bf_std = float(df_var["between_fraction"].std(ddof=0))
    conc_mean_prob = float(df_conc["num_contributing_problems"].mean())
    conc_mean_total = float(df_conc["total_problems"].mean())
    conc_frac = conc_mean_prob / conc_mean_total if conc_mean_total else float("nan")
    mean_tpc_g = float(df_conc["mean_traces_per_contributing_problem"].mean())
    max_tpc_g = int(df_conc["max_traces_per_contributing_problem"].max())
    mean_cr = float(df_conc["concentration_ratio_top20pct"].mean())

    med_new = float(df_boundary["median_k16_percentile_of_new_entrants"].median())
    p95_new = float(df_boundary["p95_k16_percentile_of_new_entrants"].median())

    n_flip = int(df_flip["regime_flip"].sum())
    n_tot = len(df_flip)
    n_pairs_with_any_flip = (
        df_flip[df_flip["regime_flip"]].groupby(["model", "benchmark"]).ngroups if n_flip else 0
    )
    max_abs_flip = float(df_flip.loc[df_flip["regime_flip"], "abs_snr_k16"].max()) if n_flip else 0.0

    mean_rho = float(np.nanmean(df_rank["spearman_rho"].values)) if len(df_rank) else float("nan")
    std_rho = float(np.nanstd(df_rank["spearman_rho"].values, ddof=0)) if len(df_rank) else float("nan")

    stable_m = per_model.sort_values("mean_recall", ascending=False).iloc[0]
    unstable_m = per_model.sort_values("mean_recall", ascending=True).iloc[0]

    def _interp_bf(x: float) -> str:
        if x > 0.8:
            return "mostly between-problem (problem means explain most variance)"
        if x < 0.4:
            return "not dominated by between-problem means (within-problem variance is large)"
        return "mixed"

    def _interp_conc(mtpc: float, frac_prob: float, cr: float) -> str:
        if frac_prob > 0.55 and mtpc <= 3.0:
            return "spread across many problems (typically 1–3 traces each)"
        if mtpc > 3.2 or cr > 0.46:
            return "skewed: some problems supply many of the top traces"
        return "mixed"

    def _interp_bnd(med: float, p95: float) -> str:
        if med <= 12 and p95 <= 20:
            return "borderline"
        return "substantively different"

    def _interp_rf(mx: float) -> str:
        if mx < 1.5:
            return "borderline (|SNR| < 1.5)"
        if mx > 3.0:
            return "some are substantive"
        return "moderate"

    def _interp_rank(r: float) -> str:
        if not np.isfinite(r):
            return "not estimable (insufficient models with ≥20 scored traces in top-10%)"
        if r > 0.95:
            return "preserved"
        if r > 0.85:
            return "partially preserved"
        return "unstable"

    print(
        f"""
============================================================
                 K-SENSITIVITY DEEP DIAGNOSTICS
============================================================

VARIANCE DECOMPOSITION
  Between-problem fraction of total variance: {bf_mean:.4f} ± {bf_std:.4f}
  → {_interp_bf(bf_mean)}
  → Note: the “easy problems dominate” story requires high between-fraction; values ≪ 0.8
     mean high recall is NOT primarily explained by problem-level confidence alone.

TOP-10% CONCENTRATION  
  Mean problems contributing to top-10%: {conc_mean_prob:.1f} of {conc_mean_total:.1f} ({100*conc_frac:.1f}%)
  Mean traces per contributing problem: {mean_tpc_g:.2f}
  Mean concentration (top-20% problems share of top-10% traces): {mean_cr:.3f}
  Max traces from single problem (any pair): {max_tpc_g}
  → {_interp_conc(mean_tpc_g, conc_frac, mean_cr)}

BOUNDARY ANALYSIS
  Median k=16 percentile of k={k_sub} new entrants (vs k=16 baseline): {med_new:.2f}%
  95th percentile (median across pairs): {p95_new:.2f}%
  → Non-overlapping traces are {_interp_bnd(med_new, p95_new)}

REGIME FLIPS
  Total flips: {n_flip} / {n_tot} resamples ({100*n_flip/n_tot if n_tot else 0:.2f}%)
  (Model×benchmark pairs with ≥1 flip: {n_pairs_with_any_flip})
  Max |SNR_k16| among flipping resamples: {max_abs_flip:.4f}
  → {_interp_rf(max_abs_flip)}; flips only when SNR is near 0 (sign is ill-defined)

PER-MODEL VARIATION
  Most stable model: {stable_m['model']} (recall = {stable_m['mean_recall']:.4f})
  Least stable model: {unstable_m['model']} (recall = {unstable_m['mean_recall']:.4f})
  Recall-CV correlation: r = {corr_r_cv[0]:.4f}, p = {corr_r_cv[1]:.4g}

RANKING STABILITY (k={k_sub} partial FRS vs k=16)
  Mean Spearman ρ vs k=16 FRS: {mean_rho:.4f} ± {std_rho:.4f}
  → Rankings are {_interp_rank(mean_rho)}
============================================================
"""
    )


def main() -> None:
    p = argparse.ArgumentParser(description="FRS k-sensitivity deep diagnostics")
    p.add_argument("--data_dir", type=str, default=".")
    p.add_argument("--judging_dir", type=str, default="reasoning_confidence_bins_results/judging_checkpoints")
    p.add_argument("--frs_csv", type=str, default="reasoning_confidence_bins_results/reasoning_by_confidence_bin.csv")
    p.add_argument("--summary_csv", type=str, default="results/k_sensitivity_summary.csv")
    p.add_argument("--k_values", type=int, nargs="+", default=[8])
    p.add_argument("--num_resamples", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="diagnostics")
    p.add_argument("--figures_dir", type=str, default="figures")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    setup_logging(args.verbose)
    data_dir = os.path.abspath(args.data_dir)
    judging_dir = os.path.join(data_dir, args.judging_dir) if not os.path.isabs(args.judging_dir) else args.judging_dir
    frs_csv = os.path.join(data_dir, args.frs_csv) if not os.path.isabs(args.frs_csv) else args.frs_csv
    out_dir = os.path.join(data_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    fig_dir = os.path.join(data_dir, args.figures_dir) if not os.path.isabs(args.figures_dir) else args.figures_dir

    k_sub = args.k_values[0]
    LOG.info("Using k_sub=%d (first of k_values)", k_sub)

    run(
        data_dir=data_dir,
        judging_dir=judging_dir,
        frs_csv=frs_csv,
        summary_csv=(
        os.path.join(data_dir, args.summary_csv)
        if not os.path.isabs(args.summary_csv)
        else args.summary_csv
    ),
        k_sub=k_sub,
        num_resamples=args.num_resamples,
        seed=args.seed,
        output_dir=out_dir,
        figures_dir=fig_dir,
    )


if __name__ == "__main__":
    main()
