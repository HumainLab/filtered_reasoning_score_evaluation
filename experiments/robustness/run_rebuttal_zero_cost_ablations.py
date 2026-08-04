#!/usr/bin/env python3
"""
Zero-cost rebuttal analyses from cached artifacts only (no generation, no judge API, no GPU).

Outputs under analysis_outputs/rebuttal_zero_cost/

Usage:
  python analysis/run_rebuttal_zero_cost_ablations.py --repo-root .
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR_DEFAULT = REPO_ROOT / "outputs/analysis_outputs" / "rebuttal_zero_cost"

DATASET_TO_BENCHMARK = {
    "GSM8K": "GSM8K",
    "MATH500": "MATH500",
    "SVAMP": "SVAMP",
    "AQuA": "AQuA",
    "GPQA": "GPQA",
    "CommonsenseQA": "CSQA",
}

BENCHMARKS = ["GSM8K", "MATH500", "SVAMP", "AQuA", "GPQA", "CSQA"]

DIMS = ["faithfulness", "utility", "coherence", "factuality"]

RUBRIC_VARIANTS = {
    "full_4d": None,
    "drop_faithfulness": ["utility", "coherence", "factuality"],
    "drop_coherence": ["faithfulness", "utility", "factuality"],
    "drop_utility": ["faithfulness", "coherence", "factuality"],
    "drop_factuality": ["faithfulness", "utility", "coherence"],
    "only_faithfulness": ["faithfulness"],
    "only_coherence": ["coherence"],
    "only_utility": ["utility"],
    "only_factuality": ["factuality"],
}

FIRST_BIN_LABEL = "0-10"
PASS1_TIE_TOL_PP = 2.0

LONG_TO_SHORT = {
    "DeepSeek-R1-Distill-Qwen-1.5B": "DS-R1-1.5B",
    "DeepSeek-R1-Distill-Qwen-7B": "DS-R1-7B",
    "LLaMA-3.1-8B-Instruct": "LLaMA-3.1-8B",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "Qwen2.5-Math-7B": "Qwen2.5-Math",
    "Phi-4": "Phi-4",
    "Phi-4-Reasoning": "Phi-4-Reas.",
    "Qwen3-4B-thinking": "Qwen3-4B",
    "Gemma-7B": "Gemma-7B",
}


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("rebuttal_zero_cost")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def safe_pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    r, p = pearsonr(x[m], y[m])
    return float(r), float(p), n


def safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    r, p = spearmanr(x[m], y[m])
    return float(r), float(p), n


def rs_from_dims(scores: Dict[str, Any], dims: List[str]) -> float:
    vals: List[float] = []
    for d in dims:
        v = scores.get(d)
        if v is None:
            return float("nan")
        vals.append(float(v))
    n = len(vals)
    return (sum(vals) - n) / (4.0 * n)


def discover_judge_files(judging_dir: Path) -> Dict[Tuple[str, str], Path]:
    out: Dict[Tuple[str, str], Path] = {}
    for fp in judging_dir.glob("judged_*.json"):
        m = re.match(r"^judged_(.+)__(.+)\.json$", fp.name)
        if m:
            out[(m.group(1), m.group(2))] = fp
    return out


def bench_from_dataset(ds: str) -> str:
    return DATASET_TO_BENCHMARK.get(ds, ds.replace("CommonsenseQA", "CSQA"))


@dataclass
class ArtifactStatus:
    path: str
    exists: bool
    n_rows: int
    note: str


def load_optional_csv(path: Path, logger: logging.Logger) -> Tuple[Optional[pd.DataFrame], ArtifactStatus]:
    if not path.is_file():
        logger.warning("MISSING: %s", path)
        return None, ArtifactStatus(str(path), False, 0, "missing")
    df = pd.read_csv(path)
    logger.info("Loaded %s (%d rows)", path.name, len(df))
    return df, ArtifactStatus(str(path), True, len(df), "ok")


def build_canonical_panel(repo: Path, logger: logging.Logger) -> Tuple[pd.DataFrame, List[ArtifactStatus]]:
    statuses: List[ArtifactStatus] = []
    path_merged = repo / "outputs/global_pass1_frs_analysis" / "merged_pass1_frs_per_benchmark.csv"
    df_m, st = load_optional_csv(path_merged, logger)
    statuses.append(st)
    if df_m is None:
        raise FileNotFoundError(f"Required: {path_merged}")

    panel = df_m.copy()
    expected = 54
    if len(panel) != expected:
        logger.warning("merged panel has %d rows (expected %d)", len(panel), expected)

    # Unfiltered
    path_unf = repo / "outputs/analysis_outputs" / "unfiltered_reasoning" / "per_pair_scores.csv"
    df_u, st_u = load_optional_csv(path_unf, logger)
    statuses.append(st_u)
    if df_u is not None:
        df_u = df_u.copy()
        df_u["benchmark"] = df_u["dataset"].map(DATASET_TO_BENCHMARK)
        miss = df_u["benchmark"].isna()
        if miss.any():
            logger.error("Unmapped datasets: %s", df_u.loc[miss, "dataset"].unique().tolist())
        panel = panel.merge(
            df_u[["model", "benchmark", "mean_reasoning_score", "n_judged_traces"]].rename(
                columns={
                    "mean_reasoning_score": "unfiltered_reasoning_mean",
                    "n_judged_traces": "unfiltered_n_judged",
                }
            ),
            on=["model", "benchmark"],
            how="left",
            indicator="_unf_merge",
        )
        n_miss = int(panel["unfiltered_reasoning_mean"].isna().sum())
        if n_miss:
            logger.warning("%d pairs missing unfiltered scores", n_miss)
        panel["unfiltered_reasoning_mean_pct"] = panel["unfiltered_reasoning_mean"] * 100.0

    # Top-10% accuracy
    path_topk = repo / "outputs/topk_ablation_results" / "topk_ablation_results.csv"
    df_t, st_t = load_optional_csv(path_topk, logger)
    statuses.append(st_t)
    if df_t is not None:
        df_t10 = df_t[df_t["top_k_pct"] == 10].copy()
        df_t10["benchmark"] = df_t10["dataset"].replace({"CommonsenseQA": "CSQA"})
        panel = panel.merge(
            df_t10[["model", "benchmark", "accuracy"]].rename(
                columns={"accuracy": "high_conf_accuracy_pct"}
            ),
            on=["model", "benchmark"],
            how="left",
        )

    # pass@16
    path_p16 = repo / "analysis" / "pass16_recomputed.csv"
    df_p16, st_p16 = load_optional_csv(path_p16, logger)
    statuses.append(st_p16)
    if df_p16 is not None and "pass16_pct" in df_p16.columns:
        panel = panel.merge(
            df_p16[["model", "benchmark", "pass16_pct"]],
            on=["model", "benchmark"],
            how="left",
            suffixes=("", "_p16dup"),
        )
        if "pass16_pct_p16dup" in panel.columns:
            panel["pass16_pct"] = panel["pass16_pct"].fillna(panel["pass16_pct_p16dup"])
            panel.drop(columns=["pass16_pct_p16dup"], inplace=True)

    # base_reasoning (pass@1-era proxy)
    path_p1r = repo / "outputs/global_pass1_frs_analysis" / "paper_pass1_reasoning_by_benchmark.csv"
    df_p1r, st_p1r = load_optional_csv(path_p1r, logger)
    statuses.append(st_p1r)
    if df_p1r is not None:
        df_p1r = df_p1r.copy()
        df_p1r["model"] = df_p1r["model"].map(LONG_TO_SHORT).fillna(df_p1r["model"])
        p1_cols = ["model", "benchmark", "base_acc"]
        if "base_reasoning" not in panel.columns:
            p1_cols.append("base_reasoning")
        panel = panel.merge(df_p1r[p1_cols], on=["model", "benchmark"], how="left")

    # Selection gain
    path_sg = repo / "analysis" / "selection_gain_pair_level.csv"
    df_sg, st_sg = load_optional_csv(path_sg, logger)
    statuses.append(st_sg)
    if df_sg is not None:
        panel = panel.merge(
            df_sg[["model", "benchmark", "mean_selection_gain"]],
            on=["model", "benchmark"],
            how="left",
        )

    # Cumulative top-10 reasoning (0-1)
    path_cum = repo / "outputs/reasoning_confidence_bins_results" / "reasoning_cumulative_topk.csv"
    df_cum, st_cum = load_optional_csv(path_cum, logger)
    statuses.append(st_cum)
    if df_cum is not None:
        df_c10 = df_cum[df_cum["topk_pct"] == 10].copy()
        df_c10["benchmark"] = df_c10["dataset"].replace({"CommonsenseQA": "CSQA"})
        panel = panel.merge(
            df_c10[["model", "benchmark", "mean_reasoning_score"]].rename(
                columns={"mean_reasoning_score": "cum_top10_reasoning_0_1"}
            ),
            on=["model", "benchmark"],
            how="left",
        )
        panel["cum_top10_reasoning_pct"] = panel["cum_top10_reasoning_0_1"] * 100.0

    panel["pair_key"] = panel["model"] + "||" + panel["benchmark"]
    logger.info("Canonical panel: %d rows, %d cols", len(panel), len(panel.columns))
    return panel, statuses


def pairwise_discrimination(
    panel: pd.DataFrame,
    value_col: str,
    pass1_col: str = "pass1_pct",
    tolerance_pp: float = PASS1_TIE_TOL_PP,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    sub = panel.dropna(subset=[value_col, pass1_col])
    for bench, g in sub.groupby("benchmark"):
        models = g["model"].tolist()
        vals = g[value_col].astype(float).tolist()
        p1s = g[pass1_col].astype(float).tolist()
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                rows.append(
                    {
                        "benchmark": bench,
                        "abs_pass1_gap_pp": abs(p1s[i] - p1s[j]),
                        "abs_metric_gap": abs(vals[i] - vals[j]),
                    }
                )
    if not rows:
        return {
            "median_abs_delta_at_pass1_tie_2pp": float("nan"),
            "mean_abs_delta_at_pass1_tie_2pp": float("nan"),
            "spearman_abs_delta_vs_pass1": float("nan"),
            "spearman_p": float("nan"),
            "n_pairs_total": 0,
            "n_pairs_near_tie_2pp": 0,
        }
    df = pd.DataFrame(rows)
    near = df[df["abs_pass1_gap_pp"] <= tolerance_pp]
    sp_r, sp_p, _ = safe_spearman(
        df["abs_pass1_gap_pp"].values.astype(float),
        df["abs_metric_gap"].values.astype(float),
    )
    return {
        "median_abs_delta_at_pass1_tie_2pp": float(near["abs_metric_gap"].median()) if len(near) else float("nan"),
        "mean_abs_delta_at_pass1_tie_2pp": float(near["abs_metric_gap"].mean()) if len(near) else float("nan"),
        "spearman_abs_delta_vs_pass1": sp_r,
        "spearman_p": sp_p,
        "n_pairs_total": len(df),
        "n_pairs_near_tie_2pp": len(near),
    }


def model_macro_ranks(panel: pd.DataFrame, value_col: str) -> pd.Series:
    macro = panel.groupby("model")[value_col].mean()
    return macro.rank(ascending=False, method="average")


def run_lobo_spearman(panel: pd.DataFrame, metric_col: str) -> Tuple[float, int]:
    """Mean LOBO Spearman: train-benchmark aggregate vs held-out (same metric)."""
    benchmarks = sorted(panel["benchmark"].dropna().unique())
    rs: List[float] = []
    sub = panel.dropna(subset=[metric_col])
    for held in benchmarks:
        train = sub[sub["benchmark"] != held]
        test = sub[sub["benchmark"] == held]
        train_agg = train.groupby("model")[metric_col].mean()
        test_agg = test.groupby("model")[metric_col].mean()
        common = sorted(set(train_agg.index) & set(test_agg.index))
        if len(common) < 3:
            continue
        x = train_agg.loc[common].values.astype(float)
        y = test_agg.loc[common].values.astype(float)
        r, _, _ = safe_spearman(x, y)
        if np.isfinite(r):
            rs.append(r)
    if not rs:
        return float("nan"), 0
    return float(np.mean(rs)), len(rs)


def compute_loro_rubric(repo: Path, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    judge_dir = repo / "outputs/reasoning_confidence_bins_results" / "judging_checkpoints"
    judge_map = discover_judge_files(judge_dir)
    logger.info("Judge checkpoints: %d files", len(judge_map))

    pair_rows: List[Dict[str, Any]] = []
    trace_rows_total = 0
    trace_rows_bin = 0

    for (model, dataset), jpath in sorted(judge_map.items()):
        benchmark = bench_from_dataset(dataset)
        with open(jpath, encoding="utf-8") as f:
            data = json.load(f)
        for s in data.get("judged_samples", []):
            if s.get("judge_ok") is False:
                continue
            js = s.get("judge_scores") or {}
            if not js:
                continue
            trace_rows_total += 1
            if str(s.get("bin_label")) != FIRST_BIN_LABEL:
                continue
            trace_rows_bin += 1
            for variant, dims in RUBRIC_VARIANTS.items():
                use_dims = dims if dims is not None else DIMS
                rs01 = rs_from_dims(js, use_dims)
                if not np.isfinite(rs01):
                    continue
                pair_rows.append(
                    {
                        "model": model,
                        "benchmark": benchmark,
                        "dataset": dataset,
                        "variant": variant,
                        "idx": int(s["idx"]),
                        "trace_idx": int(s["trace_idx"]),
                        "reasoning_score_0_1": rs01,
                        "reasoning_score_pct": rs01 * 100.0,
                        "correct": s.get("correct"),
                        "confidence": s.get("confidence"),
                    }
                )

    df_traces = pd.DataFrame(pair_rows)
    logger.info(
        "LORO rubric: %d judged traces total, %d in bin %s, %d variant-rows",
        trace_rows_total,
        trace_rows_bin,
        FIRST_BIN_LABEL,
        len(df_traces),
    )

    if df_traces.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    pair_agg = (
        df_traces.groupby(["model", "benchmark", "variant"], as_index=False)
        .agg(
            frs_bin010_pct=("reasoning_score_pct", "mean"),
            n_judged_in_bin=("reasoning_score_pct", "count"),
            mean_correct=("correct", "mean"),
        )
    )

    full_ref = pair_agg[pair_agg["variant"] == "full_4d"][
        ["model", "benchmark", "frs_bin010_pct"]
    ].rename(columns={"frs_bin010_pct": "full_frs_ref_pct"})

    macro_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for variant, g in pair_agg.groupby("variant"):
        macro = g.groupby("model")["frs_bin010_pct"].mean()
        rank = macro.rank(ascending=False, method="average")
        macro_df = pd.DataFrame(
            {
                "variant": variant,
                "model": macro.index,
                "macro_frs_bin010_pct": macro.values,
                "macro_rank": rank.values,
            }
        )
        macro_rows.extend(macro_df.to_dict("records"))

        merged_v = g.merge(full_ref, on=["model", "benchmark"], how="left")
        sp_full, sp_p, n_pairs = safe_spearman(
            merged_v["frs_bin010_pct"].values.astype(float),
            merged_v["full_frs_ref_pct"].values.astype(float),
        )

        # vs paper FRS if available
        path_merged = repo / "outputs/global_pass1_frs_analysis" / "merged_pass1_frs_per_benchmark.csv"
        sp_paper = float("nan")
        if path_merged.is_file():
            paper = pd.read_csv(path_merged)
            m2 = g.merge(paper[["model", "benchmark", "frs_pct"]], on=["model", "benchmark"], how="inner")
            if len(m2) >= 3:
                sp_paper, _, _ = safe_spearman(
                    m2["frs_bin010_pct"].values.astype(float),
                    m2["frs_pct"].values.astype(float),
                )

        summary_rows.append(
            {
                "variant": variant,
                "n_model_benchmark_pairs": len(g),
                "n_judged_traces_in_bin": int(g["n_judged_in_bin"].sum()),
                "spearman_vs_full_4d_recomputed": round(sp_full, 4) if np.isfinite(sp_full) else sp_full,
                "spearman_vs_paper_frs_pct": round(sp_paper, 4) if np.isfinite(sp_paper) else sp_paper,
                "mean_macro_frs_pct": round(float(macro.mean()), 2),
            }
        )

    df_macro = pd.DataFrame(macro_rows)
    df_summary = pd.DataFrame(summary_rows)
    return pair_agg, df_macro, df_summary


def trace0_sparse_metrics(repo: Path, logger: logging.Logger) -> pd.DataFrame:
    judge_dir = repo / "outputs/reasoning_confidence_bins_results" / "judging_checkpoints"
    rows: List[Dict[str, Any]] = []
    n_judged_all = 0
    n_t0 = 0
    for jpath in sorted(judge_dir.glob("judged_*.json")):
        m = re.match(r"^judged_(.+)__(.+)\.json$", jpath.name)
        if not m:
            continue
        model, dataset = m.group(1), m.group(2)
        benchmark = bench_from_dataset(dataset)
        with open(jpath, encoding="utf-8") as f:
            data = json.load(f)
        scores_t0: List[float] = []
        n_all = 0
        for s in data.get("judged_samples", []):
            if s.get("judge_ok") is False:
                continue
            rs = s.get("reasoning_score")
            if rs is None:
                continue
            n_all += 1
            n_judged_all += 1
            v = float(rs)
            rs01 = v if v <= 1.5 else v / 100.0
            if int(s.get("trace_idx", -1)) == 0:
                scores_t0.append(rs01)
                n_t0 += 1
        rows.append(
            {
                "model": model,
                "benchmark": benchmark,
                "n_judged_total": n_all,
                "n_judged_trace0": len(scores_t0),
                "trace0_coverage_fraction": len(scores_t0) / n_all if n_all else float("nan"),
                "trace0_rs_mean_pct": 100.0 * float(np.mean(scores_t0)) if scores_t0 else float("nan"),
            }
        )
    df = pd.DataFrame(rows)
    logger.info(
        "Trace-0 RS: %d/%d judged traces are trace_idx==0 (%.1f%%)",
        n_t0,
        n_judged_all,
        100.0 * n_t0 / n_judged_all if n_judged_all else 0,
    )
    return df


def load_trace0_accuracy(repo: Path, logger: logging.Logger) -> Optional[pd.DataFrame]:
    path = (
        repo
        / "outputs/analysis_exports"
        / "reasoning_converges_faster_csv"
        / "per_problem_accuracy_pass1_trace0_from_pass16.csv"
    )
    if not path.is_file():
        logger.warning("Missing trace-0 accuracy export: %s", path)
        return None
    df = pd.read_csv(path)
    agg = (
        df.groupby(["model", "benchmark"], as_index=False)["pass1_trace0"]
        .mean()
        .rename(columns={"pass1_trace0": "trace0_accuracy_pct"})
    )
    agg["trace0_accuracy_pct"] = agg["trace0_accuracy_pct"] * 100.0
    logger.info("Trace-0 accuracy: %d problems -> %d pair rows", len(df), len(agg))
    return agg


def confidence_filtering_summary(repo: Path, panel: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    # Top-K accuracy spread
    path_topk = repo / "outputs/topk_ablation_results" / "topk_ablation_results.csv"
    if path_topk.is_file():
        df_t = pd.read_csv(path_topk)
        for (model, dataset), g in df_t.groupby(["model", "dataset"]):
            acc = g.set_index("top_k_pct")["accuracy"]
            if 10 in acc.index and 100 in acc.index:
                rows.append(
                    {
                        "summary_type": "topk_accuracy_spread",
                        "model": model,
                        "benchmark": bench_from_dataset(dataset),
                        "metric": "accuracy_10pct_minus_100pct_pp",
                        "value": round(float(acc[10] - acc[100]), 2),
                    }
                )
        spreads = [r["value"] for r in rows if r["summary_type"] == "topk_accuracy_spread"]
        if spreads:
            rows.append(
                {
                    "summary_type": "topk_accuracy_spread",
                    "model": "_MACRO_",
                    "benchmark": "_ALL_",
                    "metric": "mean_spread_pp",
                    "value": round(float(np.mean(spreads)), 2),
                }
            )

    # Correctness-conditioned gap
    path_cc = repo / "outputs/correctness_conditioned_results" / "correctness_conditioned.csv"
    if path_cc.is_file():
        df_cc = pd.read_csv(path_cc)
        df_cc["benchmark"] = df_cc["dataset"].replace({"CommonsenseQA": "CSQA"})
        for _, r in df_cc.iterrows():
            rows.append(
                {
                    "summary_type": "median_split_accuracy_gap",
                    "model": r["model"],
                    "benchmark": r["benchmark"],
                    "metric": "high_minus_low_conf_accuracy_pp",
                    "value": round(float(r["gap_pp"]), 2),
                }
            )
        rows.append(
            {
                "summary_type": "median_split_accuracy_gap",
                "model": "_MACRO_",
                "benchmark": "_ALL_",
                "metric": "mean_gap_pp",
                "value": round(float(df_cc["gap_pp"].mean()), 2),
            }
        )

    # Top-10 concentration
    path_conc = repo / "outputs/diagnostics" / "top10_concentration.csv"
    if path_conc.is_file():
        df_c = pd.read_csv(path_conc)
        rows.append(
            {
                "summary_type": "top10_concentration",
                "model": "_MACRO_",
                "benchmark": "_ALL_",
                "metric": "mean_fraction_problems_contributing_to_top10_pool",
                "value": round(float(df_c["fraction_contributing"].mean()), 3),
            }
        )

    # FRS vs unfiltered rank (model macro)
    if "unfiltered_reasoning_mean_pct" in panel.columns:
        frs_macro = panel.groupby("model")["frs_pct"].mean()
        unf_macro = panel.groupby("model")["unfiltered_reasoning_mean_pct"].mean()
        common = sorted(set(frs_macro.index) & set(unf_macro.index))
        sp, _, _ = safe_spearman(
            frs_macro.loc[common].values.astype(float),
            unf_macro.loc[common].values.astype(float),
        )
        rows.append(
            {
                "summary_type": "frs_vs_unfiltered_rank",
                "model": "_MACRO_",
                "benchmark": "_ALL_",
                "metric": "spearman_model_rank",
                "value": round(sp, 4) if np.isfinite(sp) else sp,
            }
        )

    # Precomputed cross-benchmark (FRS)
    path_cb = repo / "analysis" / "cross_benchmark_generalization" / "frs_cross_benchmark_summary.csv"
    if path_cb.is_file():
        df_cb = pd.read_csv(path_cb)
        sub = df_cb[
            (df_cb["train_metric"] == "frs_pct")
            & (df_cb["test_target"] == "frs_pct")
            & (df_cb["corr_kind"] == "spearman")
            & (df_cb["agg_method"] == "mean")
        ]
        if len(sub):
            rows.append(
                {
                    "summary_type": "lobo_transfer_precomputed",
                    "model": "_MACRO_",
                    "benchmark": "_ALL_",
                    "metric": "mean_lobo_spearman_frs_to_frs",
                    "value": round(float(sub["mean_r"].iloc[0]), 4),
                }
            )

    return pd.DataFrame(rows)


def build_metric_specs(panel: pd.DataFrame) -> List[Dict[str, Any]]:
    specs = [
        {
            "metric_id": "pass1_pct",
            "metric_name": "pass@1 accuracy",
            "column": "pass1_pct",
            "unit": "percent",
            "is_reference": False,
            "incomplete_proxy": False,
            "incomplete_note": "",
        },
        {
            "metric_id": "pass16_pct",
            "metric_name": "pass@16 accuracy",
            "column": "pass16_pct",
            "unit": "percent",
            "is_reference": False,
            "incomplete_proxy": False,
            "incomplete_note": "",
        },
        {
            "metric_id": "high_conf_accuracy_pct",
            "metric_name": "Top-10% pool accuracy (confidence filter)",
            "column": "high_conf_accuracy_pct",
            "unit": "percent",
            "is_reference": False,
            "incomplete_proxy": False,
            "incomplete_note": "",
        },
        {
            "metric_id": "unfiltered_reasoning_pct",
            "metric_name": "Unfiltered mean reasoning score (100× judge 0–1)",
            "column": "unfiltered_reasoning_mean_pct",
            "unit": "percent",
            "is_reference": False,
            "incomplete_proxy": False,
            "incomplete_note": "100 questions/pair, random trace per question, no confidence filter",
        },
        {
            "metric_id": "frs_pct",
            "metric_name": "FRS (paper top-confidence bin, %)",
            "column": "frs_pct",
            "unit": "percent",
            "is_reference": True,
            "incomplete_proxy": False,
            "incomplete_note": "",
        },
        {
            "metric_id": "cum_top10_reasoning_pct",
            "metric_name": "Cumulative top-10% bin mean RS (filtered)",
            "column": "cum_top10_reasoning_pct",
            "unit": "percent",
            "is_reference": False,
            "incomplete_proxy": False,
            "incomplete_note": "Same judge family as FRS; filtered slice",
        },
        {
            "metric_id": "base_reasoning",
            "metric_name": "base_reasoning (pass@1-era eval API)",
            "column": "base_reasoning",
            "unit": "percent",
            "is_reference": False,
            "incomplete_proxy": True,
            "incomplete_note": "Not GPT judge; external pass@1-era reasoning metric",
        },
        {
            "metric_id": "trace0_rs_judged_pct",
            "metric_name": "Trace-0 mean RS (judged subset only)",
            "column": "trace0_rs_mean_pct",
            "unit": "percent",
            "is_reference": False,
            "incomplete_proxy": True,
            "incomplete_note": "~6% of judged traces are trace_idx==0; not full-corpus",
        },
        {
            "metric_id": "trace0_accuracy_pct",
            "metric_name": "Trace-0 accuracy (all problems, pass16 JSONL)",
            "column": "trace0_accuracy_pct",
            "unit": "percent",
            "is_reference": False,
            "incomplete_proxy": False,
            "incomplete_note": "Correctness only, not reasoning score",
        },
    ]
    return [s for s in specs if s["column"] in panel.columns or s["metric_id"] == "trace0_accuracy_pct"]


def unified_metric_comparison(
    panel: pd.DataFrame,
    specs: List[Dict[str, Any]],
    logger: logging.Logger,
) -> pd.DataFrame:
    ref_col = "frs_pct"
    frs_ranks = model_macro_ranks(panel, ref_col)
    rows: List[Dict[str, Any]] = []

    for spec in specs:
        col = spec["column"]
        if col not in panel.columns:
            logger.warning("Skipping metric %s: column %s missing", spec["metric_id"], col)
            continue
        sub = panel.dropna(subset=[col, ref_col])
        n_pairs = len(sub)
        disc = pairwise_discrimination(sub, col)
        ranks = model_macro_ranks(sub, col)
        common_models = sorted(set(frs_ranks.index) & set(ranks.index))
        sp_rank, sp_rank_p, _ = safe_spearman(
            frs_ranks.loc[common_models].values.astype(float),
            ranks.loc[common_models].values.astype(float),
        )
        if "mean_selection_gain" in sub.columns:
            sg_sub = sub.dropna(subset=["mean_selection_gain", col])
            sg_r, sg_p, n_sg = safe_pearson(
                sg_sub["mean_selection_gain"].values.astype(float),
                sg_sub[col].values.astype(float),
            )
        else:
            sg_r, sg_p, n_sg = float("nan"), float("nan"), 0
        lobo_r, lobo_folds = run_lobo_spearman(sub, col)
        unf_r, unf_p, _ = safe_pearson(
            sub["unfiltered_reasoning_mean_pct"].values.astype(float)
            if "unfiltered_reasoning_mean_pct" in sub.columns
            else np.array([]),
            sub[col].values.astype(float),
        ) if "unfiltered_reasoning_mean_pct" in sub.columns and col != "unfiltered_reasoning_mean_pct" else (float("nan"), float("nan"), 0)
        if col != "unfiltered_reasoning_mean_pct" and "unfiltered_reasoning_mean_pct" in sub.columns:
            u2 = sub.dropna(subset=["unfiltered_reasoning_mean_pct", col])
            unf_r, unf_p, _ = safe_pearson(
                u2["unfiltered_reasoning_mean_pct"].values.astype(float),
                u2[col].values.astype(float),
            )

        rows.append(
            {
                "metric_id": spec["metric_id"],
                "metric_name": spec["metric_name"],
                "is_frs_reference": spec["is_reference"],
                "incomplete_proxy": spec["incomplete_proxy"],
                "incomplete_note": spec["incomplete_note"],
                "n_model_benchmark_pairs": n_pairs,
                "coverage_fraction_of_54": round(n_pairs / 54.0, 3),
                "mean_value": round(float(sub[col].mean()), 2),
                "median_abs_delta_at_pass1_tie_2pp": round(disc["median_abs_delta_at_pass1_tie_2pp"], 2)
                if np.isfinite(disc["median_abs_delta_at_pass1_tie_2pp"])
                else disc["median_abs_delta_at_pass1_tie_2pp"],
                "spearman_abs_delta_vs_pass1": round(disc["spearman_abs_delta_vs_pass1"], 4)
                if np.isfinite(disc["spearman_abs_delta_vs_pass1"])
                else disc["spearman_abs_delta_vs_pass1"],
                "n_pairs_near_pass1_tie_2pp": disc["n_pairs_near_tie_2pp"],
                "spearman_model_rank_vs_frs": round(sp_rank, 4) if np.isfinite(sp_rank) else sp_rank,
                "pearson_vs_unfiltered_rs": round(unf_r, 4) if np.isfinite(unf_r) else unf_r,
                "pearson_vs_selection_gain": round(sg_r, 4) if np.isfinite(sg_r) else sg_r,
                "mean_lobo_transfer_spearman": round(lobo_r, 4) if np.isfinite(lobo_r) else lobo_r,
                "lobo_folds_used": lobo_folds,
            }
        )
        logger.info(
            "Metric %-28s n=%2d rank_sp=%.3f tie_med=%.2f lobo=%.3f",
            spec["metric_id"],
            n_pairs,
            sp_rank if np.isfinite(sp_rank) else float("nan"),
            disc["median_abs_delta_at_pass1_tie_2pp"]
            if np.isfinite(disc["median_abs_delta_at_pass1_tie_2pp"])
            else float("nan"),
            lobo_r if np.isfinite(lobo_r) else float("nan"),
        )
    return pd.DataFrame(rows)


def write_unified_md(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Unified metric comparison (zero-cost rebuttal)",
        "",
        "Discrimination at similar accuracy: among model pairs with |Δpass@1| ≤ 2 pp, "
        "median |Δmetric| (higher = metric still separates models when accuracy is tied). "
        "Low Spearman(|Δpass@1|, |Δmetric|) also indicates separation beyond accuracy.",
        "",
        "| Metric | n pairs | Coverage | Tie median |Δ| (pp) | Rank ρ vs FRS | Sel. gain r | LOBO ρ | Incomplete? |",
        "|:---|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for _, r in df.iterrows():
        inc = "yes" if r["incomplete_proxy"] else "no"
        lines.append(
            f"| {r['metric_name']} | {int(r['n_model_benchmark_pairs'])} | "
            f"{r['coverage_fraction_of_54']:.0%} | {r['median_abs_delta_at_pass1_tie_2pp']} | "
            f"{r['spearman_model_rank_vs_frs']} | {r['pearson_vs_selection_gain']} | "
            f"{r['mean_lobo_transfer_spearman']} | {inc} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def make_figures(
    out_dir: Path,
    unified: pd.DataFrame,
    loro_summary: pd.DataFrame,
    logger: logging.Logger,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping figures")
        return

    fig_dir = out_dir / "outputs/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    u = unified[~unified["is_frs_reference"]].copy()
    if len(u):
        fig, ax = plt.subplots(figsize=(8, 4))
        y = u["metric_id"]
        x = u["spearman_model_rank_vs_frs"].astype(float)
        colors = ["#c44" if inc else "#36a" for inc in u["incomplete_proxy"]]
        ax.barh(y, x, color=colors)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("Spearman(model rank) vs FRS")
        ax.set_title("Model ranking alignment with FRS")
        fig.tight_layout()
        fig.savefig(fig_dir / "rank_spearman_vs_frs.png", dpi=150)
        plt.close(fig)
        logger.info("Wrote %s", fig_dir / "rank_spearman_vs_frs.png")

    if len(loro_summary):
        sub = loro_summary[loro_summary["variant"].isin(
            ["full_4d", "drop_faithfulness", "drop_coherence", "drop_utility", "drop_factuality"]
        )]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(sub["variant"], sub["spearman_vs_full_4d_recomputed"].astype(float))
        ax.set_ylabel("Spearman vs full 4D FRS")
        ax.set_title("Leave-one-rubric-out (top bin 0–10)")
        plt.xticks(rotation=25, ha="right")
        fig.tight_layout()
        fig.savefig(fig_dir / "loro_rubric_vs_full.png", dpi=150)
        plt.close(fig)
        logger.info("Wrote %s", fig_dir / "loro_rubric_vs_full.png")


def write_key_numbers(
    path: Path,
    unified: pd.DataFrame,
    panel: pd.DataFrame,
    loro_summary: pd.DataFrame,
    trace0_df: pd.DataFrame,
    conf_df: pd.DataFrame,
) -> None:
    frs_row = unified[unified["is_frs_reference"]]
    pass1_row = unified[unified["metric_id"] == "pass1_pct"]
    unf_row = unified[unified["metric_id"] == "unfiltered_reasoning_pct"]
    t0_row = unified[unified["metric_id"] == "trace0_rs_judged_pct"]

    mean_t0_cov = float(trace0_df["trace0_coverage_fraction"].mean()) if len(trace0_df) else float("nan")

    lobo_frs = conf_df[
        (conf_df["summary_type"] == "lobo_transfer_precomputed")
        & (conf_df["metric"] == "mean_lobo_spearman_frs_to_frs")
    ]
    lobo_val = float(lobo_frs["value"].iloc[0]) if len(lobo_frs) else float("nan")

    lines = [
        "# Reviewer kp6q — key numbers (zero-cost)",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}_",
        "",
        "## One-sentence answer",
        "",
        "FRS is not redundant with pass@1 or a single unfiltered trace sample: at similar pass@1, "
        "FRS still separates models (larger |ΔFRS| when |Δpass@1|≤2 pp), rankings differ from unfiltered RS, "
        "and all four rubric dimensions contribute (LORO re-score from cached judges).",
        "",
        "## Headline statistics",
        "",
    ]

    def _g(row: pd.DataFrame, col: str, default="n/a") -> str:
        if row.empty or col not in row.columns:
            return default
        v = row[col].iloc[0]
        return f"{v}" if np.isfinite(v) or isinstance(v, (int, str)) else default

    lines.extend(
        [
            f"- **FRS vs pass@1 model-rank Spearman:** {_g(pass1_row, 'spearman_model_rank_vs_frs')} (pass@1 alone is not a substitute ranking).",
            f"- **FRS vs unfiltered RS model-rank Spearman:** {_g(unf_row, 'spearman_model_rank_vs_frs')} (single-trace-per-question unfiltered sample differs materially).",
            f"- **Median |ΔFRS| when |Δpass@1|≤2 pp:** {_g(frs_row, 'median_abs_delta_at_pass1_tie_2pp')} pp (similar-accuracy amplification).",
            f"- **Median |Δpass@1| when |Δpass@1|≤2 pp (baseline):** {_g(pass1_row, 'median_abs_delta_at_pass1_tie_2pp')} pp.",
            f"- **Median |Δunfiltered RS| when |Δpass@1|≤2 pp:** {_g(unf_row, 'median_abs_delta_at_pass1_tie_2pp')} pp.",
            f"- **FRS LOBO transfer (precomputed):** mean Spearman ≈ {lobo_val:.3f} (cross-benchmark).",
            f"- **FRS vs selection-gain Pearson:** {_g(frs_row, 'pearson_vs_selection_gain')} (deployment proxy; modest).",
            f"- **Trace-0 RS coverage (judged):** mean {mean_t0_cov:.1%} of judged traces — **do not claim full-corpus single-trace RS.**",
        ]
    )

    if len(loro_summary):
        drop = loro_summary[loro_summary["variant"].str.startswith("drop_")]
        lines.extend(["", "## Leave-one-rubric-out (Spearman vs full 4D, top bin)", ""])
        for _, r in drop.iterrows():
            lines.append(f"- **{r['variant']}:** {r['spearman_vs_full_4d_recomputed']}")

    lines.extend(
        [
            "",
            "## What we can claim",
            "",
            "- Confidence filtering changes accuracy monotonically (top-10% vs full pool) and concentrates traces (`confidence_filtering_necessity_summary.csv`).",
            "- FRS captures reasoning quality in high-confidence strata beyond pass@1 and beyond unfiltered one-trace-per-question judging.",
            "- Rubric dimensions are correlated but not interchangeable (LORO ρ < 1).",
            "",
            "## What we cannot claim",
            "",
            "- That one cheap trace (trace 0) reproduces FRS trends on full benchmarks (trace-0 RS is ~6% judged coverage).",
            "- That FRS is independent of the judge or confidence (same judge family; selection policy uses confidence).",
            "- Causal deployment gains from selection-gain alone (mean gain ≈ 0 at macro level).",
            "",
            "## Incomplete analyses (trace-0 RS)",
            "",
            f"- Judged trace-0 pairs: {int(trace0_df['n_judged_trace0'].sum()) if len(trace0_df) else 0} trace scores across 54 checkpoints.",
            f"- Use `trace0_accuracy_pct` for full-corpus **accuracy**; use `trace0_rs_judged_pct` only with coverage caveat.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_readme(path: Path, unified: pd.DataFrame, statuses: List[ArtifactStatus]) -> None:
    lines = [
        "# Zero-cost rebuttal outputs",
        "",
        "Produced by `analysis/run_rebuttal_zero_cost_ablations.py` from cached artifacts only.",
        "",
        "## Files",
        "",
        "| File | Description |",
        "|:---|:---|",
        "| `unified_metric_comparison.csv` | Main comparison table for Reviewer kp6q |",
        "| `unified_metric_comparison.md` | Markdown table |",
        "| `loro_rubric_*.csv` | Leave-one-rubric-out FRS from cached `judge_scores` |",
        "| `confidence_filtering_necessity_summary.csv` | Top-K spread, median-split gaps, concentration |",
        "| `reviewer_kp6q_key_numbers.md` | Copy-paste headline stats |",
        "| `figures/` | Optional bar charts |",
        "",
        "## Rebuttal claims (safe)",
        "",
        "1. **Why not one trace?** Unfiltered judging uses one random trace per question (100 q/pair) and yields a **different model ranking** than FRS; pass@1 ties still show **large FRS gaps**.",
        "2. **Why confidence filter?** Top-10% accuracy exceeds full-pool accuracy on average; median-split confidence gaps are positive for most pairs.",
        "3. **All rubric dims?** LORO shows dropping any dimension changes the top-bin score (ρ < 1 vs full 4D).",
        "",
        "## Rebuttal claims (avoid)",
        "",
        "- Full-benchmark **trace-0 reasoning score** parity with FRS (sparse judge coverage).",
        "- FRS optimizes selection-gain (weak macro correlation / near-zero mean gain).",
        "- Second-judge robustness (not in repo).",
        "",
        "## Artifact coverage",
        "",
    ]
    for st in statuses:
        lines.append(f"- `{st.path}`: exists={st.exists}, rows={st.n_rows}, {st.note}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Zero-cost rebuttal ablations from cached data")
    ap.add_argument("--repo-root", type=str, default=str(REPO_ROOT))
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR_DEFAULT))
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "outputs/figures").mkdir(parents=True, exist_ok=True)

    logger = setup_logging()
    logger.info("Repo root: %s", repo)
    logger.info("Output dir: %s", out_dir)
    t0 = time.perf_counter()

    panel, statuses = build_canonical_panel(repo, logger)

    trace0_df = trace0_sparse_metrics(repo, logger)
    trace0_acc = load_trace0_accuracy(repo, logger)
    if trace0_acc is not None:
        panel = panel.merge(trace0_acc, on=["model", "benchmark"], how="left")
    if len(trace0_df):
        panel = panel.merge(
            trace0_df[["model", "benchmark", "trace0_rs_mean_pct", "trace0_coverage_fraction", "n_judged_trace0"]],
            on=["model", "benchmark"],
            how="left",
        )

    specs = build_metric_specs(panel)
    unified = unified_metric_comparison(panel, specs, logger)
    unified.to_csv(out_dir / "unified_metric_comparison.csv", index=False)
    write_unified_md(unified, out_dir / "unified_metric_comparison.md")

    logger.info("Computing LORO rubric FRS from judge checkpoints...")
    loro_pair, loro_macro, loro_summary = compute_loro_rubric(repo, logger)
    if len(loro_pair):
        loro_pair.to_csv(out_dir / "loro_rubric_frs_pair_scores.csv", index=False)
        loro_macro.to_csv(out_dir / "loro_rubric_model_macro.csv", index=False)
        loro_summary.to_csv(out_dir / "loro_rubric_summary.csv", index=False)

        # Attach selection-gain correlation per variant
        if "mean_selection_gain" in panel.columns:
            sg_rows = []
            for variant, g in loro_pair.groupby("variant"):
                m = g.merge(panel[["model", "benchmark", "mean_selection_gain"]], on=["model", "benchmark"])
                r, p, n = safe_pearson(
                    m["mean_selection_gain"].values.astype(float),
                    m["frs_bin010_pct"].values.astype(float),
                )
                sg_rows.append({"variant": variant, "pearson_vs_selection_gain": r, "p_value": p, "n_pairs": n})
            sg_df = pd.DataFrame(sg_rows)
            loro_summary = loro_summary.merge(sg_df, on="variant", how="left")
            loro_summary.to_csv(out_dir / "loro_rubric_summary.csv", index=False)

    conf_df = confidence_filtering_summary(repo, panel, logger)
    conf_df.to_csv(out_dir / "confidence_filtering_necessity_summary.csv", index=False)

    make_figures(out_dir, unified, loro_summary, logger)
    write_key_numbers(
        out_dir / "reviewer_kp6q_key_numbers.md",
        unified,
        panel,
        loro_summary,
        trace0_df,
        conf_df,
    )
    write_readme(out_dir / "README.md", unified, statuses)

    # Save panel snapshot for audit
    panel.to_csv(out_dir / "canonical_panel_snapshot.csv", index=False)

    logger.info("Done in %.1fs. Outputs in %s", time.perf_counter() - t0, out_dir)


if __name__ == "__main__":
    main()
