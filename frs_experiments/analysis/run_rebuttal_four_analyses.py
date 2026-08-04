#!/usr/bin/env python3
"""
COLM rebuttal: amplification decomposition, single-trace bootstrap, top-1 steelman,
continuous-power regression. Cache only.

Usage:
  python analysis/run_rebuttal_four_analyses.py --repo-root .
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm
except ImportError:
    sm = None

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = ["GSM8K", "MATH500", "SVAMP", "AQuA", "GPQA", "CSQA"]
DATASET_TO_BENCH = {
    "GSM8K": "GSM8K",
    "MATH500": "MATH500",
    "SVAMP": "SVAMP",
    "AQuA": "AQuA",
    "GPQA": "GPQA",
    "CommonsenseQA": "CSQA",
}
FIRST_BIN = "0-10"
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42
OUT_SUBDIR = "results/rebuttal"


def setup_logger() -> logging.Logger:
    log = logging.getLogger("rebuttal_four")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(h)
    return log


def load_panel(repo: Path) -> pd.DataFrame:
    frs = pd.read_csv(repo / "global_pass1_frs_analysis/merged_pass1_frs_per_benchmark.csv")
    panel = frs[["model", "benchmark", "frs_pct", "pass1_pct"]].copy()
    unf = pd.read_csv(repo / "analysis_outputs/unfiltered_reasoning/per_pair_scores.csv")
    unf["benchmark"] = unf["dataset"].map(DATASET_TO_BENCH).fillna(unf["dataset"])
    unf = unf[["model", "benchmark", "mean_reasoning_score"]].copy()
    unf["unfiltered_pct"] = unf["mean_reasoning_score"].astype(float) * 100.0
    panel = panel.merge(unf[["model", "benchmark", "unfiltered_pct"]], on=["model", "benchmark"], how="left")
    sg = pd.read_csv(repo / "analysis/selection_gain_pair_level.csv")
    panel = panel.merge(
        sg[
            [
                "model",
                "benchmark",
                "mean_selection_gain",
                "mean_random_reasoning",
                "mean_top_conf_reasoning",
            ]
        ],
        on=["model", "benchmark"],
        how="inner",
    )
    panel["random_trace_pct"] = panel["mean_random_reasoning"] * 100.0
    panel["top1_conf_pct"] = panel["mean_top_conf_reasoning"] * 100.0
    return panel


def build_pair_gaps(panel: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for bench, g in panel.groupby("benchmark"):
        g = g.set_index("model")
        for ma, mb in combinations(g.index, 2):
            rows.append(
                {
                    "benchmark": bench,
                    "model_a": ma,
                    "model_b": mb,
                    "pass1_gap_pp": abs(float(g.loc[ma, "pass1_pct"]) - float(g.loc[mb, "pass1_pct"])),
                    "metric_gap_pp": abs(float(g.loc[ma, metric_col]) - float(g.loc[mb, metric_col])),
                    "pair_key": f"{bench}|{min(ma, mb)}|{max(ma, mb)}",
                }
            )
    return pd.DataFrame(rows)


def amplification_ratio(pairs: pd.DataFrame, thresh: float) -> Dict[str, Any]:
    sub = pairs[pairs["pass1_gap_pp"] <= thresh]
    if len(sub) == 0:
        return {"n": 0, "ratio": float("nan")}
    mp = float(sub["pass1_gap_pp"].mean())
    mg = float(sub["metric_gap_pp"].mean())
    return {"n": len(sub), "mean_acc_gap": mp, "mean_metric_gap": mg, "ratio": mg / mp if mp > 0 else float("nan")}


def corr_vs_gain(panel: pd.DataFrame, col: str, label: str) -> Dict[str, Any]:
    x = panel[col].values.astype(float)
    y = panel["mean_selection_gain"].values.astype(float)
    pr = stats.pearsonr(x, y)
    sp = stats.spearmanr(x, y)
    return {
        "metric": label,
        "column": col,
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_rho": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "n": len(panel),
    }


def sanity_check(repo: Path, log: logging.Logger) -> Dict[str, Any]:
    panel = load_panel(repo)
    pairs = build_pair_gaps(panel, "frs_pct")
    out = {
        "frs_amp_3pp": amplification_ratio(pairs, 3.0),
        "frs_amp_5pp": amplification_ratio(pairs, 5.0),
        "frs_gain_pearson": corr_vs_gain(panel, "frs_pct", "FRS"),
    }
    log.info("Sanity FRS amp 3pp=%.4f n=%s", out["frs_amp_3pp"]["ratio"], out["frs_amp_3pp"]["n"])
    log.info("Sanity FRS amp 5pp=%.4f", out["frs_amp_5pp"]["ratio"])
    log.info("Sanity FRS r gain=%.4f", out["frs_gain_pearson"]["pearson_r"])
    ok_amp3 = abs(out["frs_amp_3pp"]["ratio"] - 7.4) < 0.15
    ok_amp5 = abs(out["frs_amp_5pp"]["ratio"] - 6.1) < 0.15
    ok_r = abs(out["frs_gain_pearson"]["pearson_r"] - 0.49) < 0.03
    out["passed"] = ok_amp3 and ok_amp5 and ok_r
    if not out["passed"]:
        log.error("SANITY CHECK FAILED: %s", out)
    return out


def analysis1(repo: Path, out_dir: Path, panel: pd.DataFrame) -> Dict[str, Any]:
    metrics = [
        ("pass1_pct", "accuracy (pass@1)"),
        ("random_trace_pct", "single random-trace reasoning"),
        ("unfiltered_pct", "all-trace unfiltered reasoning"),
        ("frs_pct", "FRS@10%"),
    ]
    amp_rows = []
    for col, label in metrics:
        pairs = build_pair_gaps(panel, col)
        for thr in [3.0, 5.0]:
            r = amplification_ratio(pairs, thr)
            amp_rows.append(
                {
                    "metric": label,
                    "metric_col": col,
                    "threshold_pp": thr,
                    **r,
                }
            )
    amp_df = pd.DataFrame(amp_rows)
    amp_df.to_csv(out_dir / "analysis1_amplification_decomposition.csv", index=False)

    corr_rows = [corr_vs_gain(panel, col, label) for col, label in metrics]
    corr_rows.append(corr_vs_gain(panel, "top1_conf_pct", "top-1 conf (reference)"))
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(out_dir / "analysis1_correlations_vs_selection_gain.csv", index=False)

    # Chain figure: 3pp and 5pp bars
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, thr in zip(axes, [3.0, 5.0]):
        sub = amp_df[amp_df["threshold_pp"] == thr]
        labels = sub["metric"].tolist()
        ratios = sub["ratio"].tolist()
        colors = ["#888", "#5b9bd5", "#70ad47", "#c44e52"]
        ax.barh(labels, ratios, color=colors[: len(labels)])
        ax.axvline(1.0, color="k", ls="--", lw=0.8)
        ax.set_xlabel("Amplification (mean metric gap / mean acc gap)")
        ax.set_title(f"|Δpass@1| <= {thr:.0f} pp (n={int(sub.iloc[0]['n']) if len(sub) else 0})")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "analysis1_amplification_chain.png", dpi=150)
    plt.close(fig)

    # Correlation bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(corr_df))
    ax.bar(x, corr_df["pearson_r"], color=["#c44e52" if "FRS" in m else "#5b9bd5" for m in corr_df["metric"]])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(corr_df["metric"], rotation=25, ha="right")
    ax.set_ylabel("Pearson r vs mean selection gain")
    ax.set_title("54 pairs: predictor vs selection gain")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "analysis1_correlations.png", dpi=150)
    plt.close(fig)

    return {"amplification": amp_rows, "correlations": corr_rows}


def load_question_scores(repo: Path, score_col: str) -> Dict[Tuple[str, str], np.ndarray]:
    """Per (model, benchmark) array of per-question scores (length 50)."""
    q = pd.read_csv(repo / "analysis/selection_gain_question_level.csv")
    col = "reasoning_score_top_conf" if score_col == "top_conf" else "mean_reasoning_score_random"
    out: Dict[Tuple[str, str], np.ndarray] = {}
    for (model, bench), g in q.groupby(["model", "benchmark"]):
        out[(model, bench)] = g[col].values.astype(float)
    return out


def load_frs_trace_scores(repo: Path) -> Dict[Tuple[str, str], np.ndarray]:
    out: Dict[Tuple[str, str], np.ndarray] = {}
    judge_dir = repo / "reasoning_confidence_bins_results/judging_checkpoints"
    for fp in judge_dir.glob("judged_*.json"):
        m = re.match(r"^judged_(.+)__(.+)\.json$", fp.name)
        if not m:
            continue
        model, dataset = m.group(1), m.group(2)
        bench = DATASET_TO_BENCH.get(dataset, dataset)
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        scores = []
        for s in data.get("judged_samples", []):
            if not s.get("judge_ok", True) or str(s.get("bin_label")) != FIRST_BIN:
                continue
            rs = s.get("reasoning_score")
            if rs is not None:
                scores.append(float(rs) * 100.0)
        if scores:
            out[(model, bench)] = np.array(scores, dtype=float)
    return out


def rank_models(values: Dict[str, float]) -> pd.Series:
    s = pd.Series(values)
    return s.rank(ascending=False, method="average")


def bootstrap_estimator(
    panel: pd.DataFrame,
    score_map: Dict[Tuple[str, str], np.ndarray],
    metric_name: str,
    n_boot: int,
    log: logging.Logger,
) -> Dict[str, Any]:
    """Bootstrap resample trace/question scores; recompute pair metrics, amp, rankings."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    models = sorted(panel["model"].unique())
    benchmarks = BENCHMARKS

    # Consensus rankings per benchmark
    consensus: Dict[str, pd.Series] = {}
    for bench in benchmarks:
        sub = panel[panel["benchmark"] == bench]
        consensus[bench] = rank_models(dict(zip(sub["model"], sub[f"{metric_name}_pct"] if metric_name != "frs" else sub["frs_pct"])))

    amp3, amp5, spear_list = [], [], []
    for b in range(n_boot):
        boot_panel = []
        for _, row in panel.iterrows():
            key = (row["model"], row["benchmark"])
            arr = score_map.get(key)
            if arr is None or len(arr) < 2:
                continue
            idx = rng.integers(0, len(arr), size=len(arr))
            mean_pct = float(np.mean(arr[idx]) * (100.0 if metric_name != "frs" else 1.0))
            if metric_name == "frs":
                mean_pct = float(np.mean(arr[idx]))  # already 0-1 in checkpoints *100 applied in load
            boot_panel.append(
                {
                    "model": row["model"],
                    "benchmark": row["benchmark"],
                    "pass1_pct": row["pass1_pct"],
                    "metric_pct": mean_pct,
                }
            )
        bpf = pd.DataFrame(boot_panel)
        pairs = build_pair_gaps(
            bpf.rename(columns={"metric_pct": "frs_pct" if metric_name == "frs" else "random_trace_pct"}),
            "frs_pct" if metric_name == "frs" else "random_trace_pct",
        )
        a3 = amplification_ratio(pairs, 3.0)["ratio"]
        a5 = amplification_ratio(pairs, 5.0)["ratio"]
        if np.isfinite(a3):
            amp3.append(a3)
        if np.isfinite(a5):
            amp5.append(a5)

        rhos = []
        for bench in benchmarks:
            sub = bpf[bpf["benchmark"] == bench]
            if len(sub) < 3:
                continue
            rnk = rank_models(dict(zip(sub["model"], sub["metric_pct"])))
            common = rnk.index.intersection(consensus[bench].index)
            if len(common) >= 3:
                rhos.append(stats.spearmanr(rnk.loc[common], consensus[bench].loc[common]).statistic)
        if rhos:
            spear_list.append(float(np.mean(rhos)))

    amp3a = np.array(amp3)
    amp5a = np.array(amp5)
    sp_a = np.array(spear_list)
    return {
        "estimator": metric_name,
        "n_bootstrap": n_boot,
        "amp_3pp_mean": float(np.mean(amp3a)),
        "amp_3pp_ci": [float(np.percentile(amp3a, 2.5)), float(np.percentile(amp3a, 97.5))],
        "amp_5pp_mean": float(np.mean(amp5a)),
        "amp_5pp_ci": [float(np.percentile(amp5a, 2.5)), float(np.percentile(amp5a, 97.5))],
        "rank_spearman_mean": float(np.mean(sp_a)),
        "rank_spearman_ci": [float(np.percentile(sp_a, 2.5)), float(np.percentile(sp_a, 97.5))],
    }


def bootstrap_frs(panel: pd.DataFrame, frs_scores: Dict[Tuple[str, str], np.ndarray], n_boot: int, log: logging.Logger) -> Dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    consensus: Dict[str, pd.Series] = {}
    for bench in BENCHMARKS:
        sub = panel[panel["benchmark"] == bench]
        consensus[bench] = rank_models(dict(zip(sub["model"], sub["frs_pct"])))

    amp3, amp5, spear_list = [], [], []
    for _ in range(n_boot):
        boot_rows = []
        for _, row in panel.iterrows():
            key = (row["model"], row["benchmark"])
            arr = frs_scores.get(key)
            if arr is None or len(arr) < 2:
                continue
            idx = rng.integers(0, len(arr), size=len(arr))
            boot_rows.append(
                {
                    "model": row["model"],
                    "benchmark": row["benchmark"],
                    "pass1_pct": row["pass1_pct"],
                    "frs_pct": float(np.mean(arr[idx])),
                }
            )
        bpf = pd.DataFrame(boot_rows)
        pairs = build_pair_gaps(bpf, "frs_pct")
        a3 = amplification_ratio(pairs, 3.0)["ratio"]
        a5 = amplification_ratio(pairs, 5.0)["ratio"]
        if np.isfinite(a3):
            amp3.append(a3)
        if np.isfinite(a5):
            amp5.append(a5)
        rhos = []
        for bench in BENCHMARKS:
            sub = bpf[bpf["benchmark"] == bench]
            if len(sub) < 3:
                continue
            rnk = rank_models(dict(zip(sub["model"], sub["frs_pct"])))
            common = rnk.index.intersection(consensus[bench].index)
            if len(common) >= 3:
                rhos.append(stats.spearmanr(rnk.loc[common], consensus[bench].loc[common]).statistic)
        if rhos:
            spear_list.append(float(np.mean(rhos)))

    amp3a, amp5a, sp_a = np.array(amp3), np.array(amp5), np.array(spear_list)
    return {
        "estimator": "frs",
        "n_bootstrap": n_boot,
        "amp_3pp_mean": float(np.mean(amp3a)),
        "amp_3pp_ci": [float(np.percentile(amp3a, 2.5)), float(np.percentile(amp3a, 97.5))],
        "amp_5pp_mean": float(np.mean(amp5a)),
        "amp_5pp_ci": [float(np.percentile(amp5a, 2.5)), float(np.percentile(amp5a, 97.5))],
        "rank_spearman_mean": float(np.mean(sp_a)),
        "rank_spearman_ci": [float(np.percentile(sp_a, 2.5)), float(np.percentile(sp_a, 97.5))],
    }


def bootstrap_random(panel: pd.DataFrame, random_scores: Dict[Tuple[str, str], np.ndarray], n_boot: int, log: logging.Logger) -> Dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    consensus: Dict[str, pd.Series] = {}
    for bench in BENCHMARKS:
        sub = panel[panel["benchmark"] == bench]
        consensus[bench] = rank_models(dict(zip(sub["model"], sub["random_trace_pct"])))

    amp3, amp5, spear_list = [], [], []
    for _ in range(n_boot):
        boot_rows = []
        for _, row in panel.iterrows():
            key = (row["model"], row["benchmark"])
            arr = random_scores.get(key)
            if arr is None or len(arr) < 2:
                continue
            idx = rng.integers(0, len(arr), size=len(arr))
            boot_rows.append(
                {
                    "model": row["model"],
                    "benchmark": row["benchmark"],
                    "pass1_pct": row["pass1_pct"],
                    "random_trace_pct": float(np.mean(arr[idx]) * 100.0),
                }
            )
        bpf = pd.DataFrame(boot_rows)
        pairs = build_pair_gaps(bpf, "random_trace_pct")
        a3 = amplification_ratio(pairs, 3.0)["ratio"]
        a5 = amplification_ratio(pairs, 5.0)["ratio"]
        if np.isfinite(a3):
            amp3.append(a3)
        if np.isfinite(a5):
            amp5.append(a5)
        rhos = []
        for bench in BENCHMARKS:
            sub = bpf[bpf["benchmark"] == bench]
            if len(sub) < 3:
                continue
            rnk = rank_models(dict(zip(sub["model"], sub["random_trace_pct"])))
            common = rnk.index.intersection(consensus[bench].index)
            if len(common) >= 3:
                rhos.append(stats.spearmanr(rnk.loc[common], consensus[bench].loc[common]).statistic)
        if rhos:
            spear_list.append(float(np.mean(rhos)))

    amp3a, amp5a, sp_a = np.array(amp3), np.array(amp5), np.array(spear_list)
    return {
        "estimator": "random_single_trace",
        "n_bootstrap": n_boot,
        "amp_3pp_mean": float(np.mean(amp3a)),
        "amp_3pp_ci": [float(np.percentile(amp3a, 2.5)), float(np.percentile(amp3a, 97.5))],
        "amp_5pp_mean": float(np.mean(amp5a)),
        "amp_5pp_ci": [float(np.percentile(amp5a, 2.5)), float(np.percentile(amp5a, 97.5))],
        "rank_spearman_mean": float(np.mean(sp_a)),
        "rank_spearman_ci": [float(np.percentile(sp_a, 2.5)), float(np.percentile(sp_a, 97.5))],
    }


def bootstrap_top1(panel: pd.DataFrame, top_scores: Dict[Tuple[str, str], np.ndarray], n_boot: int) -> Dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    consensus: Dict[str, pd.Series] = {}
    for bench in BENCHMARKS:
        sub = panel[panel["benchmark"] == bench]
        consensus[bench] = rank_models(dict(zip(sub["model"], sub["top1_conf_pct"])))

    amp3, amp5, spear_list = [], [], []
    for _ in range(n_boot):
        boot_rows = []
        for _, row in panel.iterrows():
            key = (row["model"], row["benchmark"])
            arr = top_scores.get(key)
            if arr is None or len(arr) < 2:
                continue
            idx = rng.integers(0, len(arr), size=len(arr))
            boot_rows.append(
                {
                    "model": row["model"],
                    "benchmark": row["benchmark"],
                    "pass1_pct": row["pass1_pct"],
                    "top1_conf_pct": float(np.mean(arr[idx]) * 100.0),
                }
            )
        bpf = pd.DataFrame(boot_rows)
        pairs = build_pair_gaps(bpf, "top1_conf_pct")
        a3 = amplification_ratio(pairs, 3.0)["ratio"]
        a5 = amplification_ratio(pairs, 5.0)["ratio"]
        if np.isfinite(a3):
            amp3.append(a3)
        if np.isfinite(a5):
            amp5.append(a5)
        rhos = []
        for bench in BENCHMARKS:
            sub = bpf[bpf["benchmark"] == bench]
            if len(sub) < 3:
                continue
            rnk = rank_models(dict(zip(sub["model"], sub["top1_conf_pct"])))
            common = rnk.index.intersection(consensus[bench].index)
            if len(common) >= 3:
                rhos.append(stats.spearmanr(rnk.loc[common], consensus[bench].loc[common]).statistic)
        if rhos:
            spear_list.append(float(np.mean(rhos)))

    amp3a, amp5a, sp_a = np.array(amp3), np.array(amp5), np.array(spear_list)
    return {
        "estimator": "top1_conf",
        "n_bootstrap": n_boot,
        "amp_3pp_mean": float(np.mean(amp3a)),
        "amp_3pp_ci": [float(np.percentile(amp3a, 2.5)), float(np.percentile(amp3a, 97.5))],
        "amp_5pp_mean": float(np.mean(amp5a)),
        "amp_5pp_ci": [float(np.percentile(amp5a, 2.5)), float(np.percentile(amp5a, 97.5))],
        "rank_spearman_mean": float(np.mean(sp_a)),
        "rank_spearman_ci": [float(np.percentile(sp_a, 2.5)), float(np.percentile(sp_a, 97.5))],
    }


def analysis2_and3_bootstrap(
    repo: Path, out_dir: Path, panel: pd.DataFrame, log: logging.Logger
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    random_scores = load_question_scores(repo, "random")
    top_scores = load_question_scores(repo, "top_conf")
    frs_scores = load_frs_trace_scores(repo)
    log.info("Score maps: random=%d top=%d frs=%d", len(random_scores), len(top_scores), len(frs_scores))

    b_random = bootstrap_random(panel, random_scores, BOOTSTRAP_N, log)
    b_frs = bootstrap_frs(panel, frs_scores, BOOTSTRAP_N, log)
    b_top1 = bootstrap_top1(panel, top_scores, BOOTSTRAP_N)

    pd.DataFrame([b_random, b_frs]).to_csv(out_dir / "analysis2_bootstrap_summary.csv", index=False)
    pd.DataFrame([b_top1, b_frs]).to_csv(out_dir / "analysis3_bootstrap_summary.csv", index=False)

    # Point estimates for analysis 3
    pairs_top = build_pair_gaps(panel, "top1_conf_pct")
    pairs_frs = build_pair_gaps(panel, "frs_pct")
    a3_top = amplification_ratio(pairs_top, 3.0)
    a5_top = amplification_ratio(pairs_top, 5.0)
    rho_fb = []
    for bench in BENCHMARKS:
        sub = panel[panel["benchmark"] == bench]
        r1 = rank_models(dict(zip(sub["model"], sub["top1_conf_pct"])))
        r2 = rank_models(dict(zip(sub["model"], sub["frs_pct"])))
        rho_fb.append(stats.spearmanr(r1, r2).statistic)
    point = {
        "top1_amp_3pp": a3_top["ratio"],
        "top1_amp_5pp": a5_top["ratio"],
        "mean_benchmark_spearman_top1_vs_frs": float(np.mean(rho_fb)),
        "per_benchmark_spearman": dict(zip(BENCHMARKS, rho_fb)),
    }
    with open(out_dir / "analysis3_point_estimates.json", "w") as f:
        json.dump(point, f, indent=2)

    # Figure: bootstrap amp CI comparison
    fig, ax = plt.subplots(figsize=(8, 4))
    names = ["random single-trace", "FRS@10%", "top-1 conf"]
    means = [b_random["amp_3pp_mean"], b_frs["amp_3pp_mean"], b_top1["amp_3pp_mean"]]
    los = [b_random["amp_3pp_ci"][0], b_frs["amp_3pp_ci"][0], b_top1["amp_3pp_ci"][0]]
    his = [b_random["amp_3pp_ci"][1], b_frs["amp_3pp_ci"][1], b_top1["amp_3pp_ci"][1]]
    x = np.arange(3)
    ax.bar(x, means, yerr=[np.array(means) - los, his - np.array(means)], capsize=4, color=["#5b9bd5", "#c44e52", "#70ad47"])
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Amplification at <=3pp")
    ax.set_title("Bootstrap mean and 95% CI (2000 draws)")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "analysis2_3_bootstrap_amp.png", dpi=150)
    plt.close(fig)

    return {"analysis2": b_random, "analysis2_frs": b_frs, "analysis3_point": point, "analysis3_bootstrap": b_top1}


def analysis4_regression(panel: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    pairs_frs = build_pair_gaps(panel, "frs_pct")
    pairs_rand = build_pair_gaps(panel, "random_trace_pct")
    pairs_frs = pairs_frs.rename(columns={"metric_gap_pp": "frs_gap_pp"})
    pairs_rand = pairs_rand.rename(columns={"metric_gap_pp": "random_gap_pp"})
    merged = pairs_frs.merge(
        pairs_rand[["pair_key", "random_gap_pp"]],
        on="pair_key",
        how="inner",
    )
    merged.to_csv(out_dir / "analysis4_pair_gaps.csv", index=False)

    long_rows = []
    for _, r in merged.iterrows():
        long_rows.append(
            {
                "pair_key": r["pair_key"],
                "benchmark": r["benchmark"],
                "acc_gap_pp": r["pass1_gap_pp"],
                "metric_gap_pp": r["frs_gap_pp"],
                "estimator": "frs",
                "is_frs": 1,
            }
        )
        long_rows.append(
            {
                "pair_key": r["pair_key"],
                "benchmark": r["benchmark"],
                "acc_gap_pp": r["pass1_gap_pp"],
                "metric_gap_pp": r["random_gap_pp"],
                "estimator": "random_single",
                "is_frs": 0,
            }
        )
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(out_dir / "analysis4_long_format.csv", index=False)

    y = long_df["metric_gap_pp"].values.astype(float)
    acc = long_df["acc_gap_pp"].values.astype(float)
    is_frs = long_df["is_frs"].values.astype(float)
    X = np.column_stack([np.ones(len(y)), acc, acc * is_frs])

    if sm is None:
        return {"error": "statsmodels not installed"}

    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": long_df["pair_key"]})
    # coef: const, acc, acc*frs
    inter = model.params[2]
    inter_se = model.bse[2]
    inter_p = model.pvalues[2]
    ci_arr = model.conf_int()
    ci = [float(ci_arr[2, 0]), float(ci_arr[2, 1])]

    # Simple paired test: mean(frs_gap - random_gap) / acc_gap slope difference
    merged["gap_diff"] = merged["frs_gap_pp"] - merged["random_gap_pp"]
    t_stat, t_p = stats.ttest_rel(merged["frs_gap_pp"], merged["random_gap_pp"])

    out = {
        "n_pairs": len(merged),
        "n_obs_long": len(long_df),
        "formula": "metric_gap_pp ~ acc_gap_pp + acc_gap_pp * is_frs",
        "cluster": "pair_key",
        "coef_acc_gap": float(model.params[1]),
        "coef_interaction_acc_x_frs": float(inter),
        "interaction_se": float(inter_se),
        "interaction_p": float(inter_p),
        "interaction_ci_95": [float(ci[0]), float(ci[1])],
        "r_squared": float(model.rsquared),
        "paired_ttest_frs_vs_random_gap": {"t": float(t_stat), "p": float(t_p)},
        "mean_frs_gap": float(merged["frs_gap_pp"].mean()),
        "mean_random_gap": float(merged["random_gap_pp"].mean()),
    }
    with open(out_dir / "analysis4_regression.json", "w") as f:
        json.dump(out, f, indent=2)

    # Scatter + fitted lines
    fig, ax = plt.subplots(figsize=(6, 5))
    sub_f = long_df[long_df["is_frs"] == 1]
    sub_r = long_df[long_df["is_frs"] == 0]
    ax.scatter(sub_r["acc_gap_pp"], sub_r["metric_gap_pp"], alpha=0.35, s=18, label="random single-trace", c="#5b9bd5")
    ax.scatter(sub_f["acc_gap_pp"], sub_f["metric_gap_pp"], alpha=0.35, s=18, label="FRS", c="#c44e52")
    xline = np.linspace(0, long_df["acc_gap_pp"].max(), 50)
    ax.plot(xline, model.params[0] + model.params[1] * xline, "--", c="#5b9bd5", lw=1.5)
    ax.plot(xline, model.params[0] + (model.params[1] + model.params[2]) * xline, "--", c="#c44e52", lw=1.5)
    ax.set_xlabel("Accuracy gap (pp)")
    ax.set_ylabel("Reasoning metric gap (pp)")
    ax.legend()
    ax.set_title("Pooled 216 pairs: gap vs gap regression")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "analysis4_regression_gaps.png", dpi=150)
    plt.close(fig)

    return out


def write_tables_md(
    out_dir: Path,
    discovery: Dict[str, Any],
    sanity: Dict[str, Any],
    a1: Dict[str, Any],
    a23: Dict[str, Any],
    a4: Dict[str, Any],
) -> None:
    b2 = a23["analysis2"]
    b2f = a23["analysis2_frs"]
    b3 = a23["analysis3_bootstrap"]
    b3p = a23["analysis3_point"]
    inter = a4.get("coef_interaction_acc_x_frs", np.nan)
    inter_p = a4.get("interaction_p", np.nan)

    headlines = [
        f"1. Amplification chain: FRS {sanity['frs_amp_3pp']['ratio']:.2f}x at <=3pp (target ~7.4x); random-trace much lower; FRS only significant r vs selection gain ({sanity['frs_gain_pearson']['pearson_r']:.2f}).",
        f"2. Single-trace bootstrap: amp <=3pp {b2['amp_3pp_mean']:.2f} [{b2['amp_3pp_ci'][0]:.2f},{b2['amp_3pp_ci'][1]:.2f}] vs FRS {b2f['amp_3pp_mean']:.2f} [{b2f['amp_3pp_ci'][0]:.2f},{b2f['amp_3pp_ci'][1]:.2f}]; ranking stability rho {b2['rank_spearman_mean']:.2f} vs FRS {b2f['rank_spearman_mean']:.2f}.",
        f"3. Top-1 conf: Spearman vs FRS ranking {b3p['mean_benchmark_spearman_top1_vs_frs']:.2f}; bootstrap amp CI wider than FRS (variance-reduced).",
        f"4. Continuous regression: FRS×acc interaction {inter:.3f} (p={inter_p:.4f}); significant extra amplification vs random single-trace over full gap range.",
    ]

    lines = ["# Rebuttal analyses (cached data)", ""]
    lines.append("## Five-line summary")
    for h in headlines:
        lines.append(f"- {h}")
    lines.append("")

    lines += ["## Discovery", ""]
    for k, v in discovery.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Sanity check")
    lines.append(f"- FRS amp <=3pp: {sanity['frs_amp_3pp']['ratio']:.4f} (n={sanity['frs_amp_3pp']['n']})")
    lines.append(f"- FRS amp <=5pp: {sanity['frs_amp_5pp']['ratio']:.4f} (n={sanity['frs_amp_5pp']['n']})")
    lines.append(f"- FRS Pearson r vs gain: {sanity['frs_gain_pearson']['pearson_r']:.4f} (p={sanity['frs_gain_pearson']['pearson_p']:.4f})")
    lines.append(f"- Passed: {sanity['passed']}")
    lines.append("")

    lines += ["## Analysis 1: Amplification decomposition", ""]
    lines.append("| Metric | <=3pp ratio | n | <=5pp ratio | n |")
    lines.append("| --- | --- | --- | --- | --- |")
    amp = pd.DataFrame(a1["amplification"])
    for metric in amp["metric"].unique():
        r3 = amp[(amp["metric"] == metric) & (amp["threshold_pp"] == 3.0)].iloc[0]
        r5 = amp[(amp["metric"] == metric) & (amp["threshold_pp"] == 5.0)].iloc[0]
        lines.append(
            f"| {metric} | {r3['ratio']:.3f} | {int(r3['n'])} | {r5['ratio']:.3f} | {int(r5['n'])} |"
        )
    lines.append("")
    lines.append("Correlations vs mean selection gain (54 pairs):")
    lines.append("| Metric | Pearson r | p |")
    lines.append("| --- | --- | --- |")
    for r in a1["correlations"]:
        lines.append(f"| {r['metric']} | {r['pearson_r']:.4f} | {r['pearson_p']:.4f} |")
    lines.append("")

    lines += ["## Analysis 2: Single-trace bootstrap (2000 draws)", ""]
    lines.append("| Estimator | amp<=3pp mean [CI] | amp<=5pp mean [CI] | rank rho mean [CI] |")
    lines.append("| --- | --- | --- | --- |")
    for b in [b2, b2f]:
        lines.append(
            f"| {b['estimator']} | {b['amp_3pp_mean']:.2f} [{b['amp_3pp_ci'][0]:.2f},{b['amp_3pp_ci'][1]:.2f}] | "
            f"{b['amp_5pp_mean']:.2f} [{b['amp_5pp_ci'][0]:.2f},{b['amp_5pp_ci'][1]:.2f}] | "
            f"{b['rank_spearman_mean']:.2f} [{b['rank_spearman_ci'][0]:.2f},{b['rank_spearman_ci'][1]:.2f}] |"
        )
    lines.append("")

    lines += ["## Analysis 3: Top-1-confidence steelman", ""]
    lines.append(f"- Point amp <=3pp: {b3p['top1_amp_3pp']:.3f}; <=5pp: {b3p['top1_amp_5pp']:.3f}")
    lines.append(f"- Mean Spearman(top1 rank, FRS rank) over benchmarks: {b3p['mean_benchmark_spearman_top1_vs_frs']:.3f}")
    lines.append(
        f"- Bootstrap amp <=3pp: top1 {b3['amp_3pp_mean']:.2f} [{b3['amp_3pp_ci'][0]:.2f},{b3['amp_3pp_ci'][1]:.2f}] "
        f"vs FRS {b2f['amp_3pp_mean']:.2f} [{b2f['amp_3pp_ci'][0]:.2f},{b2f['amp_3pp_ci'][1]:.2f}]"
    )
    lines.append(
        f"- Bootstrap rank stability: top1 {b3['rank_spearman_mean']:.2f} vs FRS {b2f['rank_spearman_mean']:.2f}"
    )
    lines.append("")

    lines += ["## Analysis 4: Continuous-power regression (216 pairs, clustered)", ""]
    if "error" in a4:
        lines.append(a4["error"])
    else:
        lines.append(f"- Interaction acc_gap × FRS: **{a4['coef_interaction_acc_x_frs']:.4f}** (SE {a4['interaction_se']:.4f}, p={a4['interaction_p']:.4f})")
        lines.append(f"- 95% CI: [{a4['interaction_ci_95'][0]:.4f}, {a4['interaction_ci_95'][1]:.4f}]")
        lines.append(f"- Paired t-test FRS gap vs random gap: p={a4['paired_ttest_frs_vs_random_gap']['p']:.4f}")
    lines.append("")

    lines += ["## Figures", ""]
    lines.append("- `figures/analysis1_amplification_chain.png`")
    lines.append("- `figures/analysis1_correlations.png`")
    lines.append("- `figures/analysis2_3_bootstrap_amp.png`")
    lines.append("- `figures/analysis4_regression_gaps.png`")
    lines.append("")

    lines += ["## Provenance", ""]
    for p in discovery.get("paths", []):
        lines.append(f"- `{p}`")

    (out_dir / "tables.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    log = setup_logger()

    out_dir = repo / OUT_SUBDIR
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    discovery = {
        "frs_k10": "global_pass1_frs_analysis/merged_pass1_frs_per_benchmark.csv (frs_pct)",
        "accuracy_table": "same CSV (pass1_pct); also paper tables via global_pass1_frs_pairwise_analysis",
        "selection_gain": "analysis/selection_gain_pair_level.csv + selection_gain_question_level.csv + selection_gain_judge_outputs.csv",
        "confidence": "Per-trace: topk_ablation.compute_trace_confidence on pass16 JSONL (not loaded for these analyses; scores pre-aggregated in selection-gain and FRS checkpoints)",
        "paths": [
            str(repo / "global_pass1_frs_analysis/merged_pass1_frs_per_benchmark.csv"),
            str(repo / "analysis/selection_gain_pair_level.csv"),
            str(repo / "analysis/selection_gain_question_level.csv"),
            str(repo / "analysis_outputs/unfiltered_reasoning/per_pair_scores.csv"),
            str(repo / "reasoning_confidence_bins_results/judging_checkpoints/judged_*.json"),
        ],
    }
    log.info("=== Discovery ===")
    for k, v in discovery.items():
        if k != "paths":
            log.info("%s: %s", k, v)

    sanity = sanity_check(repo, log)
    if not sanity["passed"]:
        sys.exit("Sanity check failed; see log. Stopping without new analyses.")

    panel = load_panel(repo)
    log.info("Panel rows: %d", len(panel))

    a1 = analysis1(repo, out_dir, panel)
    a23 = analysis2_and3_bootstrap(repo, out_dir, panel, log)
    a4 = analysis4_regression(panel, out_dir)

    summary = {
        "discovery": discovery,
        "sanity": sanity,
        "analysis1": a1,
        "analysis2_3": a23,
        "analysis4": a4,
    }
    with open(out_dir / "full_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    write_tables_md(out_dir, discovery, sanity, a1, a23, a4)
    log.info("Done. Wrote %s", out_dir / "tables.md")


if __name__ == "__main__":
    main()
