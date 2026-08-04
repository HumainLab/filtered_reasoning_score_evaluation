#!/usr/bin/env python3
"""
Zero-cost rebuttal: (1) top-10% problem concentration / easy-problem check;
(2) family-controlled LOBO transfer (DS-R1 holdouts).

No generation, no judge API, no GPU.

Usage:
  python analysis/run_rebuttal_easy_problem_and_family_checks.py --repo-root .
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR_DEFAULT = REPO_ROOT / "outputs/analysis_outputs" / "rebuttal_easy_problem_family_checks"

DATASET_TO_BENCHMARK = {
    "GSM8K": "GSM8K",
    "MATH500": "MATH500",
    "SVAMP": "SVAMP",
    "AQuA": "AQuA",
    "GPQA": "GPQA",
    "CommonsenseQA": "CSQA",
}

ALL_MODELS = [
    "DS-R1-1.5B",
    "DS-R1-7B",
    "Gemma-7B",
    "LLaMA-3.1-8B",
    "Phi-4",
    "Phi-4-Reas.",
    "Qwen2.5-7B",
    "Qwen2.5-Math",
    "Qwen3-4B",
]

DS_R1_7B = "DS-R1-7B"
DS_R1_15B = "DS-R1-1.5B"
DS_R1_FAMILY = {DS_R1_7B, DS_R1_15B}

RNG_SEED = 42
N_BOOT = 2000
TOP_K_PCT = 10


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("easy_problem_family")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def gini_coefficient(counts: np.ndarray) -> float:
    x = np.asarray(counts, dtype=np.float64)
    x = x[x >= 0]
    if len(x) == 0:
        return float("nan")
    s = x.sum()
    if s <= 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    r, p = spearmanr(x[m], y[m])
    return float(r), float(p), n


def load_pass16_per_problem(jsonl_path: Path) -> Dict[int, Dict[str, Any]]:
    """idx -> n_traces, n_correct, pass16_acc (fraction)."""
    out: Dict[int, Dict[str, Any]] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            idx = int(row["idx"])
            scores = row.get("score", [])
            if not isinstance(scores, list) or len(scores) == 0:
                continue
            n = len(scores)
            nc = sum(bool(x) for x in scores)
            out[idx] = {
                "n_traces": n,
                "n_correct": nc,
                "pass16_acc": nc / n,
                "pass16_frac_str": f"{nc}/{n}",
            }
    return out


def select_top10_traces(traces: List[Dict[str, Any]]) -> pd.DataFrame:
    """Same rule as topk_ablation.compute_ablation_rows for k=10."""
    df = pd.DataFrame(traces)
    if df.empty:
        return df
    threshold = np.percentile(df["confidence"], 100 - TOP_K_PCT)
    return df[df["confidence"] >= threshold].copy()


def analyze_pair_top10(
    model: str,
    benchmark: str,
    jsonl_path: Path,
    frs_pct: float,
    logger: logging.Logger,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    sys.path.insert(0, str(REPO_ROOT))
    from topk_ablation import load_and_process_jsonl  # noqa: E402

    traces = load_and_process_jsonl(str(jsonl_path))
    if not traces:
        logger.warning("No traces: %s × %s", model, benchmark)
        return {"model": model, "benchmark": benchmark, "error": "no_traces"}, pd.DataFrame()

    prob_meta = load_pass16_per_problem(jsonl_path)
    top = select_top10_traces(traces)
    n_selected = len(top)
    n_total_pool = len(traces)

    if n_selected == 0:
        return {"model": model, "benchmark": benchmark, "error": "empty_top10"}, pd.DataFrame()

    counts = top.groupby("idx").size()
    n_probs_rep = int(counts.shape[0])
    n_probs_total = len(prob_meta) if prob_meta else int(top["idx"].nunique())
    pct_probs = 100.0 * n_probs_rep / n_probs_total if n_probs_total else float("nan")
    mean_tr_per_prob = float(counts.mean()) if len(counts) else float("nan")
    max_tr_per_prob = int(counts.max()) if len(counts) else 0

    gini = gini_coefficient(counts.values.astype(float))
    ent = float(scipy_entropy(counts.values.astype(float) / counts.sum())) if counts.sum() > 0 else float("nan")

    # Easy-problem fractions among selected traces
    n_sel = 0
    n_from_16_16 = 0
    n_from_ge_12 = 0
    n_from_le_4 = 0
    for _, row in top.iterrows():
        pid = int(row["idx"])
        meta = prob_meta.get(pid)
        if not meta:
            continue
        n_sel += 1
        nc, nt = meta["n_correct"], meta["n_traces"]
        if nc == nt and nt >= 16:
            n_from_16_16 += 1
        if nc >= 12:
            n_from_ge_12 += 1
        if nc <= 4:
            n_from_le_4 += 1

    frac_16_16 = n_from_16_16 / n_sel if n_sel else float("nan")
    frac_ge_12 = n_from_ge_12 / n_sel if n_sel else float("nan")
    frac_le_4 = n_from_le_4 / n_sel if n_sel else float("nan")

    # Per-problem selected count vs pass@16 accuracy
    prob_rows = []
    for pid, cnt in counts.items():
        meta = prob_meta.get(int(pid), {})
        prob_rows.append(
            {
                "model": model,
                "benchmark": benchmark,
                "problem_idx": int(pid),
                "n_selected_traces": int(cnt),
                "pass16_n_correct": meta.get("n_correct"),
                "pass16_n_traces": meta.get("n_traces"),
                "pass16_acc": meta.get("pass16_acc"),
            }
        )
    hist_df = pd.DataFrame(prob_rows)
    sp_cnt_pass16, sp_p, n_sp = safe_spearman(
        hist_df["n_selected_traces"].values.astype(float),
        hist_df["pass16_acc"].values.astype(float),
    )

    summary = {
        "model": model,
        "benchmark": benchmark,
        "jsonl_path": str(jsonl_path),
        "n_pool_traces": n_total_pool,
        "n_top10_selected_traces": n_selected,
        "n_total_problems": n_probs_total,
        "n_problems_represented": n_probs_rep,
        "pct_problems_represented": round(pct_probs, 2),
        "mean_selected_traces_per_problem": round(mean_tr_per_prob, 3),
        "max_selected_traces_one_problem": max_tr_per_prob,
        "gini_selected_count_across_problems": round(gini, 4),
        "entropy_selected_count_across_problems": round(ent, 4),
        "frac_selected_from_pass16_16of16": round(frac_16_16, 4),
        "frac_selected_from_pass16_ge12of16": round(frac_ge_12, 4),
        "frac_selected_from_pass16_le4of16": round(frac_le_4, 4),
        "spearman_n_selected_vs_pass16_acc": round(sp_cnt_pass16, 4) if np.isfinite(sp_cnt_pass16) else sp_cnt_pass16,
        "spearman_p": round(sp_p, 4) if np.isfinite(sp_p) else sp_p,
        "n_problems_in_correlation": n_sp,
        "pair_frs_pct": frs_pct,
        "n_problems_with_zero_in_top10": n_probs_total - n_probs_rep,
    }
    return summary, hist_df


def run_part1(repo: Path, out_dir: Path, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sys.path.insert(0, str(repo))
    from topk_ablation import build_file_map  # noqa: E402

    file_map = build_file_map(str(repo))
    logger.info("Part 1: %d JSONL pairs in file_map", len(file_map))

    path_frs = repo / "outputs/global_pass1_frs_analysis" / "merged_pass1_frs_per_benchmark.csv"
    frs_df = pd.read_csv(path_frs) if path_frs.is_file() else pd.DataFrame()
    frs_lookup = {
        (r["model"], r["benchmark"]): float(r["frs_pct"])
        for _, r in frs_df.iterrows()
    }

    coverage_rows: List[Dict[str, Any]] = []
    hist_parts: List[pd.DataFrame] = []
    expected = set((m, b) for m in ALL_MODELS for b in DATASET_TO_BENCHMARK.values())
    found = set()

    for i, (key, jpath) in enumerate(sorted(file_map.items()), start=1):
        model, dataset = key
        benchmark = DATASET_TO_BENCHMARK.get(dataset, dataset)
        found.add((model, benchmark))
        frs = frs_lookup.get((model, benchmark), float("nan"))
        logger.info("[%d/%d] top10 concentration %s × %s", i, len(file_map), model, benchmark)
        summ, hist = analyze_pair_top10(model, benchmark, Path(jpath), frs, logger)
        coverage_rows.append(summ)
        if len(hist):
            hist_parts.append(hist)

    missing = expected - found
    for m, b in sorted(missing):
        logger.warning("MISSING pair in file_map: %s × %s", m, b)
        coverage_rows.append(
            {
                "model": m,
                "benchmark": b,
                "error": "jsonl_missing",
                "n_top10_selected_traces": 0,
            }
        )

    cov = pd.DataFrame(coverage_rows)
    hist_all = pd.concat(hist_parts, ignore_index=True) if hist_parts else pd.DataFrame()

    # Macro summary across pairs (valid only)
    if "error" in cov.columns:
        ok = cov[cov["error"].isna()].copy()
    else:
        ok = cov.copy()

    macro = pd.DataFrame(
        [
            {
                "stat": "n_pairs",
                "value": len(ok),
            },
            {
                "stat": "mean_pct_problems_represented",
                "value": round(float(ok["pct_problems_represented"].mean()), 2),
            },
            {
                "stat": "median_pct_problems_represented",
                "value": round(float(ok["pct_problems_represented"].median()), 2),
            },
            {
                "stat": "mean_max_selected_one_problem",
                "value": round(float(ok["max_selected_traces_one_problem"].mean()), 2),
            },
            {
                "stat": "median_max_selected_one_problem",
                "value": round(float(ok["max_selected_traces_one_problem"].median()), 2),
            },
            {
                "stat": "mean_frac_selected_pass16_16of16",
                "value": round(float(ok["frac_selected_from_pass16_16of16"].mean()), 4),
            },
            {
                "stat": "mean_frac_selected_pass16_ge12",
                "value": round(float(ok["frac_selected_from_pass16_ge12of16"].mean()), 4),
            },
            {
                "stat": "mean_frac_selected_pass16_le4",
                "value": round(float(ok["frac_selected_from_pass16_le4of16"].mean()), 4),
            },
            {
                "stat": "mean_spearman_n_selected_vs_pass16_acc",
                "value": round(float(ok["spearman_n_selected_vs_pass16_acc"].mean()), 4),
            },
            {
                "stat": "pairs_with_pct_problems_below_50",
                "value": int((ok["pct_problems_represented"] < 50).sum()),
            },
            {
                "stat": "pairs_with_max_selected_ge_10",
                "value": int((ok["max_selected_traces_one_problem"] >= 10).sum()),
            },
        ]
    )

    cov.to_csv(out_dir / "top10_problem_coverage_by_pair.csv", index=False)
    hist_all.to_csv(out_dir / "top10_problem_histogram_by_pair.csv", index=False)
    macro.to_csv(out_dir / "top10_easy_problem_summary.csv", index=False)
    logger.info("Part 1 tables: %d coverage rows, %d histogram rows", len(cov), len(hist_all))
    return cov, hist_all, macro


def _safe_corr(x: np.ndarray, y: np.ndarray, kind: str) -> Tuple[float, float]:
    if len(x) < 3:
        return float("nan"), float("nan")
    if kind == "spearman":
        return spearmanr(x, y)
    if kind == "pearson":
        return pearsonr(x, y)
    raise ValueError(kind)


def run_lobo_variant(
    panel: pd.DataFrame,
    models_keep: Set[str],
    variant_id: str,
    agg_method: str = "mean",
) -> pd.DataFrame:
    sub = panel[panel["model"].isin(models_keep)].copy()
    benchmarks = sorted(sub["benchmark"].unique())
    records: List[Dict[str, Any]] = []
    for held_out in benchmarks:
        train_df = sub[sub["benchmark"] != held_out]
        test_df = sub[sub["benchmark"] == held_out]
        if agg_method == "mean":
            train_agg = train_df.groupby("model")[["frs_pct", "pass1_pct"]].mean()
        else:
            train_agg = train_df.groupby("model")[["frs_pct", "pass1_pct"]].median()
        test_agg = test_df.set_index("model")[["frs_pct", "pass1_pct"]]
        common = sorted(set(train_agg.index) & set(test_agg.index))
        n_models = len(common)
        for train_metric in ["frs_pct", "pass1_pct"]:
            x = train_agg.loc[common, train_metric].values.astype(float)
            for test_target in ["frs_pct", "pass1_pct"]:
                y = test_agg.loc[common, test_target].values.astype(float)
                for kind in ["spearman"]:
                    r, p = _safe_corr(x, y, kind)
                    records.append(
                        {
                            "variant": variant_id,
                            "held_out_benchmark": held_out,
                            "n_models": n_models,
                            "models_used": ",".join(common),
                            "agg_method": agg_method,
                            "train_metric": train_metric,
                            "test_target": test_target,
                            "corr_kind": kind,
                            "r": round(float(r), 4) if np.isfinite(r) else r,
                            "p_value": round(float(p), 4) if np.isfinite(p) else p,
                        }
                    )
    return pd.DataFrame(records)


def run_part2(repo: Path, out_dir: Path, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    path_merged = repo / "outputs/global_pass1_frs_analysis" / "merged_pass1_frs_per_benchmark.csv"
    if not path_merged.is_file():
        raise FileNotFoundError(path_merged)
    panel = pd.read_csv(path_merged)
    logger.info("Part 2 panel: %d rows, models=%s", len(panel), sorted(panel["model"].unique()))

    variants: List[Tuple[str, Set[str]]] = [
        ("original_all_9", set(ALL_MODELS)),
        ("exclude_DS-R1-7B", set(ALL_MODELS) - {DS_R1_7B}),
        ("exclude_DS-R1-1.5B", set(ALL_MODELS) - {DS_R1_15B}),
        ("exclude_both_DS-R1", set(ALL_MODELS) - DS_R1_FAMILY),
        (
            "leave_family_out_non_DS-R1_only",
            set(ALL_MODELS) - DS_R1_FAMILY,
        ),
        (
            "holdout_DS-R1_cluster_evaluate_7_models",
            set(ALL_MODELS) - DS_R1_FAMILY,
        ),
    ]

    all_results: List[pd.DataFrame] = []
    for vid, models in variants:
        logger.info("LOBO variant %s: n_models=%d → %s", vid, len(models), sorted(models))
        res = run_lobo_variant(panel, models, vid)
        all_results.append(res)

    # Deduplicate identical 7-model variants in summary but keep in matrix for clarity
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(out_dir / "family_controlled_lobo_matrix.csv", index=False)

    summary_rows: List[Dict[str, Any]] = []
    for vid in combined["variant"].unique():
        sub = combined[
            (combined["variant"] == vid)
            & (combined["train_metric"] == "frs_pct")
            & (combined["test_target"] == "frs_pct")
            & (combined["corr_kind"] == "spearman")
        ]
        rs = sub["r"].astype(float)
        n_mod = int(sub["n_models"].iloc[0]) if len(sub) else 0
        warn = ""
        if n_mod < 5:
            warn = "n_models<5: correlations unstable"
        elif n_mod < 7:
            warn = "n_models<7: interpret cautiously"
        summary_rows.append(
            {
                "variant": vid,
                "n_models": n_mod,
                "n_lobo_folds": len(sub),
                "n_valid_folds": int(rs.notna().sum()),
                "n_positive_spearman": int((rs > 0).sum()),
                "mean_spearman_frs_to_frs": round(float(rs.mean()), 4) if len(rs) else float("nan"),
                "median_spearman_frs_to_frs": round(float(rs.median()), 4) if len(rs) else float("nan"),
                "std_spearman": round(float(rs.std()), 4) if len(rs) > 1 else float("nan"),
                "min_spearman": round(float(rs.min()), 4) if len(rs) else float("nan"),
                "max_spearman": round(float(rs.max()), 4) if len(rs) else float("nan"),
                "all_positive": bool((rs > 0).all()) if len(rs) else False,
                "warning": warn,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "family_controlled_lobo_summary.csv", index=False)

    # Wide matrix: held_out × variant
    pivot_sub = combined[
        (combined["train_metric"] == "frs_pct")
        & (combined["test_target"] == "frs_pct")
        & (combined["corr_kind"] == "spearman")
    ]
    pivot = pivot_sub.pivot_table(
        index="held_out_benchmark", columns="variant", values="r", aggfunc="first"
    )
    pivot.to_csv(out_dir / "family_controlled_lobo_pivot.csv")

    logger.info("Part 2: %d variant summaries", len(summary))
    return summary, combined


def write_top10_key_numbers(
    path: Path,
    cov: pd.DataFrame,
    macro: pd.DataFrame,
) -> None:
    if "error" in cov.columns:
        ok = cov[cov["error"].isna()].copy()
    else:
        ok = cov.copy()
    m = {r["stat"]: r["value"] for _, r in macro.iterrows()}
    high_conc = ok[ok["max_selected_traces_one_problem"] >= 10].sort_values(
        "max_selected_traces_one_problem", ascending=False
    )
    low_cov = ok[ok["pct_problems_represented"] < 50].sort_values("pct_problems_represented")

    lines = [
        "# Top-10% problem concentration — key numbers (Reviewer yweD)",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}_",
        "",
        "## Method",
        "",
        "Top-10% traces = same rule as `topk_ablation.py`: all pass@16 traces pooled per model×benchmark, "
        f"keep traces with confidence ≥ the **{100-TOP_K_PCT}th percentile** (global top {TOP_K_PCT}% by token-confidence).",
        "",
        "## Headline (54 pairs)",
        "",
        f"- Mean **% of problems** with ≥1 selected trace: **{m.get('mean_pct_problems_represented', 'n/a')}%** "
        f"(median **{m.get('median_pct_problems_represented', 'n/a')}%**).",
        f"- Mean **max selected traces from one problem**: **{m.get('mean_max_selected_one_problem', 'n/a')}** "
        f"(median **{m.get('median_max_selected_one_problem', 'n/a')}**).",
        f"- Mean fraction of selected traces from **pass@16 = 16/16** problems: **{m.get('mean_frac_selected_pass16_16of16', 'n/a')}**.",
        f"- Mean fraction from problems with **≥12/16** correct: **{m.get('mean_frac_selected_pass16_ge12', 'n/a')}**.",
        f"- Mean fraction from **≤4/16** correct (hard) problems: **{m.get('mean_frac_selected_pass16_le4', 'n/a')}**.",
        f"- Mean within-pair Spearman(**#selected**, **pass@16 acc**): **{m.get('mean_spearman_n_selected_vs_pass16_acc', 'n/a')}**.",
        f"- Pairs with **<50%** problem coverage: **{int(m.get('pairs_with_pct_problems_below_50', 0))}**.",
        f"- Pairs with **≥10** selected traces on one problem: **{int(m.get('pairs_with_max_selected_ge_10', 0))}**.",
        "",
        "## Direct answers",
        "",
        "### Does top-10% collapse onto only a few problems?",
        "",
    ]
    if int(m.get("pairs_with_max_selected_ge_10", 0)) > 0:
        lines.append(
            "**Partially, on some pairs.** A non-trivial subset of pairs show heavy tail concentration "
            f"({int(m.get('pairs_with_max_selected_ge_10', 0))} pairs with max≥10 traces from one problem). "
            "On average, problem coverage is broader than a handful of items (mean % problems represented "
            f"≈ {m.get('mean_pct_problems_represented')}%)."
        )
    else:
        lines.append("**Not on average** across pairs; max-per-problem counts are moderate.")

    lines.extend(
        [
            "",
            "### How broad is problem coverage on average?",
            "",
            f"Roughly **{m.get('mean_pct_problems_represented')}%** of problems contribute at least one top-10% trace "
            f"(median **{m.get('median_pct_problems_represented')}%**). This is **not** a single-problem collapse for most pairs.",
            "",
            "### Are selected traces overwhelmingly from pass@16=16/16 easy problems?",
            "",
        ]
    )
    frac16 = float(m.get("mean_frac_selected_pass16_16of16", 0) or 0)
    if frac16 > 0.5:
        lines.append(
            f"**A large share is from fully-solved (16/16) problems** (mean **{100*frac16:.1f}%** of selected traces). "
            "Confidence filtering correlates with problem-level pass@16 success; acknowledge this in rebuttal."
        )
    else:
        lines.append(
            f"**Not overwhelmingly** (mean **{100*frac16:.1f}%** from 16/16 problems). "
            "Harder problems still contribute selected traces."
        )

    lines.extend(
        [
            "",
            "### Pairs with high concentration (acknowledge in paper)",
            "",
        ]
    )
    if len(high_conc):
        for _, r in high_conc.head(8).iterrows():
            lines.append(
                f"- {r['model']} × {r['benchmark']}: max={int(r['max_selected_traces_one_problem'])} selected, "
                f"{r['pct_problems_represented']:.1f}% problems represented"
            )
    else:
        lines.append("_None with max≥10 in this run._")

    if len(low_cov):
        lines.extend(["", "### Low problem-coverage pairs", ""])
        for _, r in low_cov.head(8).iterrows():
            lines.append(f"- {r['model']} × {r['benchmark']}: {r['pct_problems_represented']:.1f}% problems")

    lines.extend(
        [
            "",
            "## Safe rebuttal claims",
            "",
            "- Top-10% is computed over **traces**, not by cherry-picking a few problem IDs; "
            "a majority of problems typically contribute at least one high-confidence trace.",
            "- Selection is **associated** with higher per-problem pass@16 (positive Spearman in many pairs); "
            "state this as correlation, not causation.",
            "- FRS still uses **reasoning judge** scores within the filtered pool — not identical to pass@16 accuracy alone.",
            "",
            "## Avoid overclaiming",
            "",
            "- Do not say the filter is uniform over problems on every pair.",
            "- Do not deny that high pass@16 problems supply many top-confidence traces on some benchmarks.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_family_key_numbers(path: Path, summary: pd.DataFrame) -> None:
    def _row(vid: str) -> Optional[pd.Series]:
        sub = summary[summary["variant"] == vid]
        return sub.iloc[0] if len(sub) else None

    orig = _row("original_all_9")
    excl_both = _row("exclude_both_DS-R1")

    lines = [
        "# Family-controlled LOBO — key numbers",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}_",
        "",
        "## Method",
        "",
        "Leave-one-**benchmark**-out: aggregate train FRS (or pass@1) over 5 benchmarks per model, "
        "Spearman vs held-out benchmark. Variants drop DS-R1-7B, DS-R1-1.5B, or both (7 models).",
        "",
        "## Headline",
        "",
    ]
    if orig is not None:
        lines.append(
            f"- **All 9 models:** mean LOBO Spearman(FRS→FRS) = **{orig['mean_spearman_frs_to_frs']}** "
            f"({int(orig['n_positive_spearman'])}/{int(orig['n_lobo_folds'])} folds positive)."
        )
    if excl_both is not None:
        lines.append(
            f"- **Exclude both DS-R1 (7 models):** mean = **{excl_both['mean_spearman_frs_to_frs']}** "
            f"({int(excl_both['n_positive_spearman'])}/{int(excl_both['n_lobo_folds'])} positive)."
        )

    for vid in [
        "exclude_DS-R1-7B",
        "exclude_DS-R1-1.5B",
    ]:
        r = _row(vid)
        if r is not None:
            lines.append(
                f"- **{vid}:** mean = **{r['mean_spearman_frs_to_frs']}** "
                f"({int(r['n_positive_spearman'])}/{int(r['n_lobo_folds'])} positive)."
            )

    lines.extend(
        [
            "",
            "## Does LOBO survive removing both DS-R1?",
            "",
        ]
    )
    if excl_both is not None and orig is not None:
        weaker = float(excl_both["mean_spearman_frs_to_frs"]) < float(orig["mean_spearman_frs_to_frs"])
        pos = int(excl_both["n_positive_spearman"]) >= 4
        if pos and float(excl_both["mean_spearman_frs_to_frs"]) > 0.3:
            lines.append(
                "**Yes, directionally.** Mean correlation remains **positive** with 7 models, "
                "though magnitude may be **weaker** than the 9-model panel. "
                "Report both numbers; do not claim identical strength."
            )
        else:
            lines.append(
                "**Mixed / weaker.** After removing DS-R1, some folds weaken or lose significance — "
                "report per-benchmark table and stress small-n (7 models)."
            )

    lines.extend(
        [
            "",
            "## What we can safely claim",
            "",
            "- Cross-benchmark FRS ranking alignment is **not solely** driven by including both DS-R1 variants.",
            "- With **n=7**, LOBO is **illustrative**; prefer reporting mean/median ρ and fold counts, not p-values alone.",
            "",
            "## What NOT to overclaim (n=7 after removing DS-R1)",
            "",
            "- Independent replication or tight confidence intervals.",
            "- That every held-out benchmark fold stays significant.",
            "- Causal transfer of FRS across domains.",
            "",
            "See `family_controlled_lobo_pivot.csv` for per-fold Spearman values.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def make_figures(
    out_dir: Path,
    cov: pd.DataFrame,
    summary_lobo: pd.DataFrame,
    logger: logging.Logger,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib missing; skip figures")
        return

    fig_dir = out_dir / "outputs/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if "error" in cov.columns:
        ok = cov[cov["error"].isna()].copy()
    else:
        ok = cov.copy()

    if len(ok):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(ok["pct_problems_represented"].astype(float), bins=15, edgecolor="k", alpha=0.75)
        ax.set_xlabel("% problems with ≥1 top-10% trace")
        ax.set_ylabel("Number of model×benchmark pairs")
        ax.set_title("Problem coverage under top-10% confidence filter")
        fig.tight_layout()
        fig.savefig(fig_dir / "hist_pct_problems_represented.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(ok["max_selected_traces_one_problem"].astype(float), bins=15, edgecolor="k", alpha=0.75)
        ax.set_xlabel("Max selected traces from one problem")
        ax.set_ylabel("Pairs")
        ax.set_title("Per-problem concentration (top-10% pool)")
        fig.tight_layout()
        fig.savefig(fig_dir / "hist_max_selected_one_problem.png", dpi=150)
        plt.close(fig)

        sub = ok.dropna(subset=["pct_problems_represented", "pair_frs_pct"])
        if len(sub) >= 5:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(
                sub["pct_problems_represented"],
                sub["pair_frs_pct"],
                alpha=0.7,
            )
            ax.set_xlabel("% problems represented")
            ax.set_ylabel("Pair-level FRS (%)")
            ax.set_title("Coverage vs FRS")
            fig.tight_layout()
            fig.savefig(fig_dir / "scatter_pct_problems_vs_frs.png", dpi=150)
            plt.close(fig)

        sub2 = ok.dropna(subset=["spearman_n_selected_vs_pass16_acc"])
        if len(sub2):
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(
                sub2["spearman_n_selected_vs_pass16_acc"],
                sub2["pair_frs_pct"],
                alpha=0.7,
            )
            ax.set_xlabel("Spearman(#selected, pass@16 acc)")
            ax.set_ylabel("FRS (%)")
            ax.set_title("Selection–difficulty correlation vs FRS")
            fig.tight_layout()
            fig.savefig(fig_dir / "scatter_sel_pass16_corr_vs_frs.png", dpi=150)
            plt.close(fig)

    logger.info("Figures written to %s", fig_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default=str(REPO_ROOT))
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR_DEFAULT))
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "outputs/figures").mkdir(exist_ok=True)

    logger = setup_logging()
    logger.info("Repo: %s", repo)
    logger.info("Out: %s", out_dir)
    t0 = time.perf_counter()

    cov, hist, macro = run_part1(repo, out_dir, logger)
    write_top10_key_numbers(out_dir / "top10_easy_problem_key_numbers.md", cov, macro)

    lobo_summary, lobo_matrix = run_part2(repo, out_dir, logger)
    write_family_key_numbers(out_dir / "family_controlled_lobo_key_numbers.md", lobo_summary)

    make_figures(out_dir, cov, lobo_summary, logger)

    logger.info("Done in %.1fs", time.perf_counter() - t0)
    print("\nKey number files:")
    print(f"  {out_dir / 'top10_easy_problem_key_numbers.md'}")
    print(f"  {out_dir / 'family_controlled_lobo_key_numbers.md'}")


if __name__ == "__main__":
    main()
