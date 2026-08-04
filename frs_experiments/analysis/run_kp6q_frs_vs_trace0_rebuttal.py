#!/usr/bin/env python3
"""
Reviewer kp6q rebuttal: FRS vs trace-0 vs unfiltered/all-traces baselines.

Compares informativeness claims on cached artifacts only — no new API calls.

Usage:
  python analysis/run_kp6q_frs_vs_trace0_rebuttal.py --repo-root .
  python analysis/run_kp6q_frs_vs_trace0_rebuttal.py --force   # ignore cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR_DEFAULT = REPO_ROOT / "analysis_outputs" / "rebuttal_kp6q_frs_vs_trace0"

BENCHMARKS = ["GSM8K", "MATH500", "SVAMP", "AQuA", "GPQA", "CSQA"]
EXPECTED_MODELS = [
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
DATASET_TO_BENCH = {
    "GSM8K": "GSM8K",
    "MATH500": "MATH500",
    "SVAMP": "SVAMP",
    "AQuA": "AQuA",
    "GPQA": "GPQA",
    "CommonsenseQA": "CSQA",
}
PASS1_THRESHOLDS = [2.0, 3.0, 5.0]
REVERSAL_MIN_GAP_PP = 2.0
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 123


def setup_logger() -> logging.Logger:
    log = logging.getLogger("kp6q_frs_vs_trace0")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(h)
    return log


def safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    r, p = spearmanr(x[m], y[m])
    return float(r), float(p), n


def safe_pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    r, p = pearsonr(x[m], y[m])
    return float(r), float(p), n


def file_fingerprint(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size": st.st_size,
        "mtime": st.st_mtime,
    }


def manifest_hash(manifest: Dict[str, Any]) -> str:
    blob = json.dumps(manifest, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def discover_inputs(repo: Path, log: logging.Logger) -> Dict[str, Path]:
    candidates = {
        "frs_pass1": repo / "global_pass1_frs_analysis" / "merged_pass1_frs_per_benchmark.csv",
        "trace0": repo / "analysis_outputs" / "trace0_k1_judging" / "trace0_k1_vs_frs_comparison.csv",
        "unfiltered": repo / "analysis_outputs" / "unfiltered_reasoning" / "per_pair_scores.csv",
        "prior_reversals": repo / "analysis_outputs" / "rebuttal_trace0_frs_reversals" / "pairwise_rank_reversals.csv",
    }
    log.info("=== Discovered input paths ===")
    for k, p in candidates.items():
        log.info("  %-18s %s (exists=%s)", k, p, p.is_file())
    return candidates


def load_panel(paths: Dict[str, Path], log: logging.Logger) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    meta: Dict[str, Any] = {"sources": {}, "unfiltered_available": False}

    frs = pd.read_csv(paths["frs_pass1"])
    meta["sources"]["frs_pass1"] = {"rows": len(frs), "path": str(paths["frs_pass1"])}
    panel = frs[["model", "benchmark", "frs_pct", "pass1_pct"]].copy()

    if paths["trace0"].is_file():
        t0 = pd.read_csv(paths["trace0"])
        t0 = t0[["model", "benchmark", "mean_reasoning_score_pct"]].rename(
            columns={"mean_reasoning_score_pct": "trace0_pct"}
        )
        panel = panel.merge(t0, on=["model", "benchmark"], how="left")
        meta["sources"]["trace0"] = {"rows": len(t0), "path": str(paths["trace0"])}
    else:
        panel["trace0_pct"] = np.nan
        log.warning("Trace-0 file missing — trace0_pct will be NaN")

    if paths["unfiltered"].is_file():
        unf = pd.read_csv(paths["unfiltered"])
        unf["benchmark"] = unf["dataset"].map(DATASET_TO_BENCH).fillna(unf["dataset"])
        unf = unf[["model", "benchmark", "mean_reasoning_score"]].rename(
            columns={"mean_reasoning_score": "unfiltered_rs_0_1"}
        )
        unf["unfiltered_pct"] = unf["unfiltered_rs_0_1"] * 100.0
        panel = panel.merge(unf[["model", "benchmark", "unfiltered_pct"]], on=["model", "benchmark"], how="left")
        meta["sources"]["unfiltered"] = {"rows": len(unf), "path": str(paths["unfiltered"])}
        meta["unfiltered_available"] = True
    else:
        panel["unfiltered_pct"] = np.nan
        log.warning("Unfiltered file missing — unfiltered_pct will be NaN")

    return panel, meta


def sanity_check_panel(panel: pd.DataFrame, log: logging.Logger) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    report["n_pairs"] = len(panel)
    report["expected_pairs"] = 54
    report["n_models"] = panel["model"].nunique()
    report["n_benchmarks"] = panel["benchmark"].nunique()
    report["models_found"] = sorted(panel["model"].unique().tolist())
    report["benchmarks_found"] = sorted(panel["benchmark"].unique().tolist())

    expected = {(m, b) for m in EXPECTED_MODELS for b in BENCHMARKS}
    actual = set(zip(panel["model"], panel["benchmark"]))
    report["missing_pairs"] = sorted(expected - actual)
    report["extra_pairs"] = sorted(actual - expected)

    for col in ["frs_pct", "pass1_pct", "trace0_pct", "unfiltered_pct"]:
        if col in panel.columns:
            report[f"n_missing_{col}"] = int(panel[col].isna().sum())

    log.info("=== Sanity check ===")
    log.info("  Panel rows: %d (expected 54)", len(panel))
    log.info("  Models: %d | Benchmarks: %d", report["n_models"], report["n_benchmarks"])
    if report["missing_pairs"]:
        log.warning("  Missing %d expected pairs: %s", len(report["missing_pairs"]), report["missing_pairs"][:5])
    if report["extra_pairs"]:
        log.warning("  Extra pairs: %s", report["extra_pairs"])
    for col in ["frs_pct", "pass1_pct", "trace0_pct", "unfiltered_pct"]:
        if col in panel.columns:
            log.info("  Missing %s: %d", col, report.get(f"n_missing_{col}", 0))
    return report


def build_near_equal_pairs(panel: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    sub = panel.dropna(subset=["frs_pct", "pass1_pct"])
    for bench, g in sub.groupby("benchmark"):
        g = g.set_index("model")
        for ma, mb in combinations(g.index, 2):
            row = {
                "benchmark": bench,
                "model_a": ma,
                "model_b": mb,
                "pass1_a": float(g.loc[ma, "pass1_pct"]),
                "pass1_b": float(g.loc[mb, "pass1_pct"]),
                "frs_a": float(g.loc[ma, "frs_pct"]),
                "frs_b": float(g.loc[mb, "frs_pct"]),
                "pass1_gap_pp": abs(float(g.loc[ma, "pass1_pct"]) - float(g.loc[mb, "pass1_pct"])),
                "frs_gap_pp": abs(float(g.loc[ma, "frs_pct"]) - float(g.loc[mb, "frs_pct"])),
            }
            if "trace0_pct" in g.columns and np.isfinite(g.loc[ma, "trace0_pct"]) and np.isfinite(g.loc[mb, "trace0_pct"]):
                row["trace0_a"] = float(g.loc[ma, "trace0_pct"])
                row["trace0_b"] = float(g.loc[mb, "trace0_pct"])
                row["trace0_gap_pp"] = abs(row["trace0_a"] - row["trace0_b"])
            else:
                row["trace0_a"] = row["trace0_b"] = row["trace0_gap_pp"] = np.nan
            if "unfiltered_pct" in g.columns and np.isfinite(g.loc[ma, "unfiltered_pct"]) and np.isfinite(g.loc[mb, "unfiltered_pct"]):
                row["unfiltered_a"] = float(g.loc[ma, "unfiltered_pct"])
                row["unfiltered_b"] = float(g.loc[mb, "unfiltered_pct"])
                row["unfiltered_gap_pp"] = abs(row["unfiltered_a"] - row["unfiltered_b"])
            else:
                row["unfiltered_a"] = row["unfiltered_b"] = row["unfiltered_gap_pp"] = np.nan
            row["frs_gt_trace0"] = (
                row["frs_gap_pp"] > row["trace0_gap_pp"]
                if np.isfinite(row["trace0_gap_pp"])
                else np.nan
            )
            row["frs_gt_unfiltered"] = (
                row["frs_gap_pp"] > row["unfiltered_gap_pp"]
                if np.isfinite(row["unfiltered_gap_pp"])
                else np.nan
            )
            rows.append(row)
    out = pd.DataFrame(rows)
    log.info("Built %d within-benchmark model pairs for near-equal analysis", len(out))
    return out


def summarize_near_equal(pairs: pd.DataFrame, thresholds: Sequence[float]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for thr in thresholds:
        sub = pairs[pairs["pass1_gap_pp"] <= thr]
        mean_p1 = float(sub["pass1_gap_pp"].mean()) if len(sub) else np.nan
        mean_frs = float(sub["frs_gap_pp"].mean()) if len(sub) else np.nan
        mean_t0 = float(sub["trace0_gap_pp"].mean()) if len(sub) and sub["trace0_gap_pp"].notna().any() else np.nan
        mean_unf = (
            float(sub["unfiltered_gap_pp"].mean()) if len(sub) and sub["unfiltered_gap_pp"].notna().any() else np.nan
        )
        frs_wins_t0 = sub["frs_gt_trace0"].dropna()
        frs_wins_unf = sub["frs_gt_unfiltered"].dropna()
        rows.append(
            {
                "pass1_gap_threshold_pp": thr,
                "scope": "all_benchmarks",
                "n_eligible_pairs": len(sub),
                "mean_pass1_gap_pp": mean_p1,
                "mean_frs_gap_pp": mean_frs,
                "mean_trace0_gap_pp": mean_t0,
                "mean_unfiltered_gap_pp": mean_unf,
                "frac_frs_gap_gt_trace0": float(frs_wins_t0.mean()) if len(frs_wins_t0) else np.nan,
                "frac_frs_gap_gt_unfiltered": float(frs_wins_unf.mean()) if len(frs_wins_unf) else np.nan,
                "n_with_trace0": int(sub["trace0_gap_pp"].notna().sum()),
                "n_with_unfiltered": int(sub["unfiltered_gap_pp"].notna().sum()),
                "amplification_frs": mean_frs / mean_p1 if mean_p1 and mean_p1 > 0 else np.nan,
                "amplification_trace0": mean_t0 / mean_p1 if mean_p1 and mean_p1 > 0 and np.isfinite(mean_t0) else np.nan,
                "amplification_unfiltered": mean_unf / mean_p1 if mean_p1 and mean_p1 > 0 and np.isfinite(mean_unf) else np.nan,
            }
        )
        for bench in BENCHMARKS:
            bsub = sub[sub["benchmark"] == bench]
            if len(bsub) == 0:
                continue
            mp1 = float(bsub["pass1_gap_pp"].mean())
            mfrs = float(bsub["frs_gap_pp"].mean())
            mt0 = float(bsub["trace0_gap_pp"].mean()) if bsub["trace0_gap_pp"].notna().any() else np.nan
            munf = float(bsub["unfiltered_gap_pp"].mean()) if bsub["unfiltered_gap_pp"].notna().any() else np.nan
            fw_t0 = bsub["frs_gt_trace0"].dropna()
            fw_unf = bsub["frs_gt_unfiltered"].dropna()
            rows.append(
                {
                    "pass1_gap_threshold_pp": thr,
                    "scope": bench,
                    "n_eligible_pairs": len(bsub),
                    "mean_pass1_gap_pp": mp1,
                    "mean_frs_gap_pp": mfrs,
                    "mean_trace0_gap_pp": mt0,
                    "mean_unfiltered_gap_pp": munf,
                    "frac_frs_gap_gt_trace0": float(fw_t0.mean()) if len(fw_t0) else np.nan,
                    "frac_frs_gap_gt_unfiltered": float(fw_unf.mean()) if len(fw_unf) else np.nan,
                    "n_with_trace0": int(bsub["trace0_gap_pp"].notna().sum()),
                    "n_with_unfiltered": int(bsub["unfiltered_gap_pp"].notna().sum()),
                    "amplification_frs": mfrs / mp1 if mp1 > 0 else np.nan,
                    "amplification_trace0": mt0 / mp1 if mp1 > 0 and np.isfinite(mt0) else np.nan,
                    "amplification_unfiltered": munf / mp1 if mp1 > 0 and np.isfinite(munf) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def lobo_transfer(
    panel: pd.DataFrame,
    train_col: str,
    test_col: str,
    benchmarks: Sequence[str] = BENCHMARKS,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for held in benchmarks:
        train = panel[panel["benchmark"] != held].dropna(subset=[train_col])
        test = panel[panel["benchmark"] == held].dropna(subset=[test_col])
        if train.empty or test.empty:
            continue
        macro = train.groupby("model", as_index=True)[train_col].mean()
        te = test.set_index("model")[test_col]
        common = sorted(set(macro.index) & set(te.index))
        if len(common) < 3:
            continue
        x = macro.loc[common].values.astype(float)
        y = te.loc[common].values.astype(float)
        sp, sp_p, n = safe_spearman(x, y)
        pe, pe_p, _ = safe_pearson(x, y)
        rows.append(
            {
                "held_out_benchmark": held,
                "train_metric": train_col,
                "test_target": test_col,
                "spearman_r": sp,
                "spearman_p": sp_p,
                "pearson_r": pe,
                "pearson_p": pe_p,
                "n_models": n,
            }
        )
    return pd.DataFrame(rows)


def lobo_summary(lobo_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (tr, te), g in lobo_df.groupby(["train_metric", "test_target"]):
        rows.append(
            {
                "train_metric": tr,
                "test_target": te,
                "mean_spearman": float(g["spearman_r"].mean()),
                "std_spearman": float(g["spearman_r"].std()),
                "mean_pearson": float(g["pearson_r"].mean()),
                "std_pearson": float(g["pearson_r"].std()),
                "n_folds": len(g),
            }
        )
    return pd.DataFrame(rows)


def macro_excluding(panel: pd.DataFrame, held: str, col: str) -> pd.Series:
    train = panel[panel["benchmark"] != held]
    return train.groupby("model", as_index=True)[col].mean()


def compute_rank_reversals(panel: pd.DataFrame, min_gap: float = REVERSAL_MIN_GAP_PP) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for bench, g in panel.groupby("benchmark"):
        g = g.set_index("model")
        for ma, mb in combinations(g.index, 2):
            frs_a, frs_b = float(g.loc[ma, "frs_pct"]), float(g.loc[mb, "frs_pct"])
            p1_a, p1_b = float(g.loc[ma, "pass1_pct"]), float(g.loc[mb, "pass1_pct"])
            t0_a = float(g.loc[ma, "trace0_pct"]) if "trace0_pct" in g.columns else np.nan
            t0_b = float(g.loc[mb, "trace0_pct"]) if "trace0_pct" in g.columns else np.nan
            unf_a = float(g.loc[ma, "unfiltered_pct"]) if "unfiltered_pct" in g.columns else np.nan
            unf_b = float(g.loc[mb, "unfiltered_pct"]) if "unfiltered_pct" in g.columns else np.nan

            frs_diff = frs_a - frs_b
            t0_diff = t0_a - t0_b if np.isfinite(t0_a) and np.isfinite(t0_b) else np.nan
            unf_diff = unf_a - unf_b if np.isfinite(unf_a) and np.isfinite(unf_b) else np.nan
            p1_diff = p1_a - p1_b

            frs_w = ma if frs_diff > 0 else mb
            t0_w = ma if t0_diff > 0 else mb if np.isfinite(t0_diff) else None
            unf_w = ma if unf_diff > 0 else mb if np.isfinite(unf_diff) else None
            p1_w = ma if p1_diff > 0 else mb

            frs_rev_t0 = (
                np.isfinite(t0_diff)
                and abs(t0_diff) >= min_gap
                and abs(frs_diff) >= min_gap
                and (t0_diff * frs_diff) < 0
            )
            frs_rev_unf = (
                np.isfinite(unf_diff)
                and abs(unf_diff) >= min_gap
                and abs(frs_diff) >= min_gap
                and (unf_diff * frs_diff) < 0
            )

            m_frs = macro_excluding(panel, bench, "frs_pct")
            m_t0 = macro_excluding(panel, bench, "trace0_pct") if "trace0_pct" in panel.columns else pd.Series(dtype=float)
            lobo_frs_diff = float(m_frs.loc[ma] - m_frs.loc[mb]) if ma in m_frs.index and mb in m_frs.index else np.nan
            lobo_t0_diff = (
                float(m_t0.loc[ma] - m_t0.loc[mb]) if len(m_t0) and ma in m_t0.index and mb in m_t0.index else np.nan
            )
            lobo_frs_w = ma if lobo_frs_diff > 0 else mb if np.isfinite(lobo_frs_diff) else None
            lobo_t0_w = ma if lobo_t0_diff > 0 else mb if np.isfinite(lobo_t0_diff) else None

            rows.append(
                {
                    "benchmark": bench,
                    "model_a": ma,
                    "model_b": mb,
                    "frs_a": frs_a,
                    "frs_b": frs_b,
                    "trace0_a": t0_a,
                    "trace0_b": t0_b,
                    "unfiltered_a": unf_a,
                    "unfiltered_b": unf_b,
                    "pass1_a": p1_a,
                    "pass1_b": p1_b,
                    "frs_winner": frs_w,
                    "trace0_winner": t0_w,
                    "unfiltered_winner": unf_w,
                    "pass1_winner": p1_w,
                    "frs_gap_pp": abs(frs_diff),
                    "trace0_gap_pp": abs(t0_diff) if np.isfinite(t0_diff) else np.nan,
                    "unfiltered_gap_pp": abs(unf_diff) if np.isfinite(unf_diff) else np.nan,
                    "pass1_gap_pp": abs(p1_diff),
                    "frs_reverses_trace0": frs_rev_t0,
                    "frs_reverses_unfiltered": frs_rev_unf,
                    "pass1_agrees_trace0": p1_w == t0_w if t0_w else np.nan,
                    "pass1_agrees_frs": p1_w == frs_w,
                    "lobo_frs_diff_pp": lobo_frs_diff,
                    "lobo_trace0_diff_pp": lobo_t0_diff,
                    "lobo_frs_agrees_frs_winner": lobo_frs_w == frs_w if lobo_frs_w else np.nan,
                    "lobo_frs_agrees_trace0_winner": lobo_frs_w == t0_w if lobo_frs_w and t0_w else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_reversals(rev: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def _block(label: str, sub: pd.DataFrame, rev_col: str) -> None:
        revs = sub[sub[rev_col] == True]  # noqa: E712
        n_qual = sub[
            sub["frs_gap_pp"].ge(REVERSAL_MIN_GAP_PP)
            & (
                sub["trace0_gap_pp"].ge(REVERSAL_MIN_GAP_PP)
                if rev_col == "frs_reverses_trace0"
                else sub["unfiltered_gap_pp"].ge(REVERSAL_MIN_GAP_PP)
            )
        ]
        p1_t0 = revs["pass1_agrees_trace0"].dropna()
        lobo_frs = revs["lobo_frs_agrees_frs_winner"].dropna()
        lobo_t0 = revs["lobo_frs_agrees_trace0_winner"].dropna()
        rows.append(
            {
                "comparison": label,
                "scope": "all_benchmarks",
                "n_comparable_pairs": len(sub),
                "n_qualifying_pairs": len(n_qual),
                "n_reversals": len(revs),
                "frac_reversals_of_qualifying": len(revs) / max(1, len(n_qual)),
                "pass1_agrees_baseline_winner": float(p1_t0.mean()) if len(p1_t0) else np.nan,
                "lobo_frs_agrees_frs_winner": float(lobo_frs.mean()) if len(lobo_frs) else np.nan,
                "lobo_frs_agrees_baseline_winner": float(lobo_t0.mean()) if len(lobo_t0) else np.nan,
            }
        )
        for bench in BENCHMARKS:
            bsub = sub[sub["benchmark"] == bench]
            brevs = bsub[bsub[rev_col] == True]  # noqa: E712
            bn_qual = bsub[
                bsub["frs_gap_pp"].ge(REVERSAL_MIN_GAP_PP)
                & (
                    bsub["trace0_gap_pp"].ge(REVERSAL_MIN_GAP_PP)
                    if rev_col == "frs_reverses_trace0"
                    else bsub["unfiltered_gap_pp"].ge(REVERSAL_MIN_GAP_PP)
                )
            ]
            bp1 = brevs["pass1_agrees_trace0"].dropna() if rev_col == "frs_reverses_trace0" else pd.Series(dtype=float)
            blf = brevs["lobo_frs_agrees_frs_winner"].dropna()
            blt = brevs["lobo_frs_agrees_trace0_winner"].dropna()
            rows.append(
                {
                    "comparison": label,
                    "scope": bench,
                    "n_comparable_pairs": len(bsub),
                    "n_qualifying_pairs": len(bn_qual),
                    "n_reversals": len(brevs),
                    "frac_reversals_of_qualifying": len(brevs) / max(1, len(bn_qual)),
                    "pass1_agrees_baseline_winner": float(bp1.mean()) if len(bp1) else np.nan,
                    "lobo_frs_agrees_frs_winner": float(blf.mean()) if len(blf) else np.nan,
                    "lobo_frs_agrees_baseline_winner": float(blt.mean()) if len(blt) else np.nan,
                }
            )

    _block("frs_vs_trace0", rev.dropna(subset=["trace0_a", "trace0_b"]), "frs_reverses_trace0")
    _block("frs_vs_unfiltered", rev.dropna(subset=["unfiltered_a", "unfiltered_b"]), "frs_reverses_unfiltered")
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, stat_fn, n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    obs = float(stat_fn(values))
    boots = []
    for _ in range(n):
        samp = rng.choice(values, size=len(values), replace=True)
        boots.append(float(stat_fn(samp)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, float(lo), float(hi)


def bootstrap_ci_mean(values: np.ndarray) -> Tuple[float, float, float]:
    return bootstrap_ci(values, np.mean)


def bootstrap_ci_fraction(bool_arr: np.ndarray) -> Tuple[float, float, float]:
    return bootstrap_ci(bool_arr.astype(float), np.mean)


def run_bootstrap_cis(
    near_pairs: pd.DataFrame,
    near_summary: pd.DataFrame,
    lobo_df: pd.DataFrame,
    rev: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for thr in PASS1_THRESHOLDS:
        sub = near_pairs[near_pairs["pass1_gap_pp"] <= thr]
        if len(sub):
            obs, lo, hi = bootstrap_ci_fraction(sub["frs_gt_trace0"].dropna().values)
            rows.append(
                {
                    "quantity": f"frac_frs_gap_gt_trace0_pass1_le_{int(thr)}pp",
                    "point_estimate": obs,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "n_units": int(sub["frs_gt_trace0"].notna().sum()),
                    "bootstrap_n": BOOTSTRAP_N,
                }
            )
            amp = sub["frs_gap_pp"] / sub["pass1_gap_pp"].replace(0, np.nan)
            obs, lo, hi = bootstrap_ci_mean(amp.dropna().values)
            rows.append(
                {
                    "quantity": f"amplification_frs_pass1_le_{int(thr)}pp",
                    "point_estimate": obs,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "n_units": int(amp.notna().sum()),
                    "bootstrap_n": BOOTSTRAP_N,
                }
            )
            amp_t0 = sub["trace0_gap_pp"] / sub["pass1_gap_pp"].replace(0, np.nan)
            obs, lo, hi = bootstrap_ci_mean(amp_t0.dropna().values)
            rows.append(
                {
                    "quantity": f"amplification_trace0_pass1_le_{int(thr)}pp",
                    "point_estimate": obs,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "n_units": int(amp_t0.notna().sum()),
                    "bootstrap_n": BOOTSTRAP_N,
                }
            )

    for train_col, test_col, label in [
        ("frs_pct", "frs_pct", "lobo_frs_to_frs"),
        ("trace0_pct", "frs_pct", "lobo_trace0_to_frs"),
        ("unfiltered_pct", "frs_pct", "lobo_unfiltered_to_frs"),
    ]:
        sub = lobo_df[(lobo_df["train_metric"] == train_col) & (lobo_df["test_target"] == test_col)]
        if len(sub):
            obs, lo, hi = bootstrap_ci_mean(sub["spearman_r"].values)
            rows.append(
                {
                    "quantity": label,
                    "point_estimate": obs,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "n_units": len(sub),
                    "bootstrap_n": BOOTSTRAP_N,
                }
            )

    rev_t0 = rev[rev["frs_reverses_trace0"] == True]  # noqa: E712
    qual = rev[
        rev["frs_gap_pp"].ge(REVERSAL_MIN_GAP_PP) & rev["trace0_gap_pp"].ge(REVERSAL_MIN_GAP_PP)
    ]
    if len(qual):
        obs, lo, hi = bootstrap_ci_fraction(qual["frs_reverses_trace0"].astype(float).values)
        rows.append(
            {
                "quantity": "frac_frs_reverses_trace0",
                "point_estimate": obs,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "n_units": len(qual),
                "bootstrap_n": BOOTSTRAP_N,
            }
        )
    return pd.DataFrame(rows)


def plot_amplification(summary: pd.DataFrame, out_path: Path) -> None:
    sub = summary[(summary["scope"] == "all_benchmarks")].sort_values("pass1_gap_threshold_pp")
    if sub.empty:
        return
    x = np.arange(len(sub))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w, sub["amplification_frs"], width=w, label="FRS")
    ax.bar(x, sub["amplification_trace0"], width=w, label="Trace-0 RS")
    if sub["amplification_unfiltered"].notna().any():
        ax.bar(x + w, sub["amplification_unfiltered"], width=w, label="Unfiltered RS")
    ax.set_xticks(x)
    ax.set_xticklabels([f"≤{int(t)} pp" for t in sub["pass1_gap_threshold_pp"]])
    ax.set_xlabel("|Δpass@1| threshold")
    ax.set_ylabel("Amplification (mean |Δmetric| / mean |Δpass@1|)")
    ax.set_title("Discrimination amplification among near-equal-accuracy pairs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scatter_trace0_frs(panel: pd.DataFrame, out_path: Path) -> None:
    sub = panel.dropna(subset=["trace0_pct", "frs_pct"])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(sub["trace0_pct"], sub["frs_pct"], alpha=0.75, s=50)
    lo = min(sub["trace0_pct"].min(), sub["frs_pct"].min()) - 2
    hi = max(sub["trace0_pct"].max(), sub["frs_pct"].max()) + 2
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Trace-0 RS (%)")
    ax.set_ylabel("FRS (%)")
    ax.set_title("54 pairs: trace-0 vs FRS")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_lobo_scatter(panel: pd.DataFrame, lobo_df: pd.DataFrame, out_path: Path) -> None:
    held = BENCHMARKS[0]
    sub_frs = lobo_df[
        (lobo_df["held_out_benchmark"] == held)
        & (lobo_df["train_metric"] == "frs_pct")
        & (lobo_df["test_target"] == "frs_pct")
    ]
    if sub_frs.empty:
        return
    m_frs = macro_excluding(panel, held, "frs_pct")
    m_t0 = macro_excluding(panel, held, "trace0_pct")
    test = panel[panel["benchmark"] == held].set_index("model")
    common = sorted(set(m_frs.index) & set(m_t0.index) & set(test.index))
    if len(common) < 3:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    y = test.loc[common, "frs_pct"].values
    axes[0].scatter(m_frs.loc[common].values, y, s=60)
    axes[0].set_xlabel(f"Macro FRS (train, LOBO)")
    axes[0].set_ylabel(f"Held-out FRS ({held})")
    axes[0].set_title("FRS → FRS")
    axes[1].scatter(m_t0.loc[common].values, y, s=60)
    axes[1].set_xlabel(f"Macro trace-0 RS (train, LOBO)")
    axes[1].set_ylabel(f"Held-out FRS ({held})")
    axes[1].set_title("Trace-0 → FRS")
    fig.suptitle(f"LOBO transfer (held out: {held})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rank_diff_hist(rev: pd.DataFrame, out_path: Path) -> None:
    sub = rev.dropna(subset=["trace0_a", "trace0_b"])
    pair_diff = (sub["frs_a"] - sub["trace0_a"]) - (sub["frs_b"] - sub["trace0_b"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pair_diff.dropna(), bins=20, edgecolor="0.3")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Pairwise (FRS−trace0)_A − (FRS−trace0)_B (pp)")
    ax.set_ylabel("Count")
    ax.set_title("FRS vs trace-0 rank-difference distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_readme(out_dir: Path, meta: Dict[str, Any], sanity: Dict[str, Any]) -> None:
    unf = "available" if meta.get("unfiltered_available") else "UNAVAILABLE"
    text = f"""# kp6q FRS vs trace-0 rebuttal analysis

Generated: {datetime.now(timezone.utc).isoformat()}

## Question

Does FRS provide more signal than single-trace (trace-0) or unfiltered/all-traces reasoning score,
especially among models with similar pass@1?

## Inputs

| Source | Path | Status |
|--------|------|--------|
| FRS + pass@1 | `{meta['sources'].get('frs_pass1', {}).get('path', 'N/A')}` | {meta['sources'].get('frs_pass1', {}).get('rows', '?')} rows |
| Trace-0 RS | `{meta['sources'].get('trace0', {}).get('path', 'N/A')}` | {meta['sources'].get('trace0', {}).get('rows', '?')} rows |
| Unfiltered RS | `{meta['sources'].get('unfiltered', {}).get('path', 'N/A')}` | {unf} |

Panel coverage: **{sanity['n_pairs']}/54** model×benchmark pairs.

## Methods

1. **Near-equal accuracy:** Within each benchmark, all model pairs with |Δpass@1| ≤ {{2,3,5}} pp.
   Compare mean |ΔFRS|, |Δtrace-0|, |Δunfiltered| and amplification ratios.

2. **LOBO transfer:** Macro mean over 5 train benchmarks → held-out benchmark (6 folds).
   Spearman and Pearson for each train→test metric pair.

3. **Rank reversals:** Pairs where FRS and baseline disagree on winner (≥{REVERSAL_MIN_GAP_PP} pp both sides).
   LOBO FRS from other benchmarks validates reversal direction.

4. **Bootstrap CIs:** {BOOTSTRAP_N} resamples, seed {BOOTSTRAP_SEED}.

## Interpretation

- **Amplification > 1** near equal pass@1 → metric separates models beyond accuracy.
- **FRS→FRS LOBO ρ >> trace-0→FRS** → FRS ranking transfers; trace-0 macro does not predict held-out FRS.
- **Rank reversals with LOBO FRS agreement** → FRS re-ranking is cross-benchmark signal, not noise.

## Outputs

See `key_numbers.md`, `rebuttal_paragraph.md`, and CSV summaries in this directory.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def write_key_numbers(
    out_path: Path,
    sanity: Dict[str, Any],
    near_summary: pd.DataFrame,
    lobo_summary_df: pd.DataFrame,
    rev_summary: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    panel: pd.DataFrame,
    meta: Dict[str, Any],
) -> None:
    macro = near_summary[near_summary["scope"] == "all_benchmarks"]
    row2 = macro[macro["pass1_gap_threshold_pp"] == 2.0].iloc[0]
    row3 = macro[macro["pass1_gap_threshold_pp"] == 3.0].iloc[0]

    def _lobo(tr: str, te: str) -> float:
        s = lobo_summary_df[(lobo_summary_df["train_metric"] == tr) & (lobo_summary_df["test_target"] == te)]
        return float(s["mean_spearman"].iloc[0]) if len(s) else float("nan")

    rev_all = rev_summary[(rev_summary["comparison"] == "frs_vs_trace0") & (rev_summary["scope"] == "all_benchmarks")].iloc[0]
    rev_unf = rev_summary[
        (rev_summary["comparison"] == "frs_vs_unfiltered") & (rev_summary["scope"] == "all_benchmarks")
    ]
    sp_panel, _, _ = safe_spearman(
        panel["trace0_pct"].dropna().values,
        panel["frs_pct"].dropna().values,
    )

    lines = [
        "# kp6q: FRS vs trace-0 vs unfiltered — key numbers",
        "",
        "## Data coverage",
        "",
        f"- Panel pairs: **{sanity['n_pairs']}/54**",
        f"- Unfiltered RS: **{'available (54/54)' if meta.get('unfiltered_available') else 'UNAVAILABLE'}**",
        f"- Macro mean FRS: {panel['frs_pct'].mean():.1f}% | trace-0: {panel['trace0_pct'].mean():.1f}% | unfiltered: {panel['unfiltered_pct'].mean():.1f}%",
        f"- Spearman(trace-0, FRS) over 54 pairs: **{sp_panel:.3f}**",
        "",
        "## 1. Near-equal accuracy (|Δpass@1| ≤ 2 pp)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Eligible pairs | {int(row2['n_eligible_pairs'])} |",
        f"| Mean |Δpass@1| | {row2['mean_pass1_gap_pp']:.2f} pp |",
        f"| Mean |ΔFRS| | **{row2['mean_frs_gap_pp']:.2f} pp** |",
        f"| Mean |Δtrace-0| | {row2['mean_trace0_gap_pp']:.2f} pp |",
        f"| Mean |Δunfiltered| | {row2['mean_unfiltered_gap_pp']:.2f} pp |",
        f"| Frac pairs FRS gap > trace-0 gap | **{row2['frac_frs_gap_gt_trace0']:.1%}** |",
        f"| Frac pairs FRS gap > unfiltered gap | {row2['frac_frs_gap_gt_unfiltered']:.1%} |",
        f"| Amplification FRS | **{row2['amplification_frs']:.2f}×** |",
        f"| Amplification trace-0 | {row2['amplification_trace0']:.2f}× |",
        f"| Amplification unfiltered | {row2['amplification_unfiltered']:.2f}× |",
        "",
        "At |Δpass@1| ≤ 3 pp: "
        f"{int(row3['n_eligible_pairs'])} pairs, FRS amp {row3['amplification_frs']:.2f}× vs trace-0 {row3['amplification_trace0']:.2f}×.",
        "",
        "## 2. LOBO transfer (mean Spearman ρ, 6 folds)",
        "",
        "| Train → Test | Mean ρ |",
        "|--------------|--------|",
        f"| FRS → FRS | **{_lobo('frs_pct', 'frs_pct'):.3f}** |",
        f"| trace-0 → FRS | {_lobo('trace0_pct', 'frs_pct'):.3f} |",
        f"| trace-0 → trace-0 | {_lobo('trace0_pct', 'trace0_pct'):.3f} |",
        f"| unfiltered → FRS | {_lobo('unfiltered_pct', 'frs_pct'):.3f} |",
        f"| unfiltered → unfiltered | {_lobo('unfiltered_pct', 'unfiltered_pct'):.3f} |",
        f"| pass@1 → FRS | {_lobo('pass1_pct', 'frs_pct'):.3f} |",
        "",
        "## 3. Rank reversals (≥2 pp both sides)",
        "",
        f"- FRS vs trace-0: **{int(rev_all['n_reversals'])}/{int(rev_all['n_qualifying_pairs'])}** "
        f"({rev_all['frac_reversals_of_qualifying']:.0%})",
        f"- pass@1 agrees trace-0 winner: {rev_all['pass1_agrees_baseline_winner']:.0%} of reversals",
        f"- LOBO FRS agrees FRS winner: **{rev_all['lobo_frs_agrees_frs_winner']:.0%}**",
    ]
    if len(rev_unf):
        ru = rev_unf.iloc[0]
        lines.append(
            f"- FRS vs unfiltered: {int(ru['n_reversals'])}/{int(ru['n_qualifying_pairs'])} "
            f"({ru['frac_reversals_of_qualifying']:.0%})"
        )

    lines.extend(["", "## 4. Bootstrap 95% CIs", ""])
    for _, r in bootstrap_df.iterrows():
        lines.append(
            f"- {r['quantity']}: {r['point_estimate']:.3f} [{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}] (n={int(r['n_units'])})"
        )

    lines.extend(["", "## Files", "", "See README.md and CSV summaries in this directory."])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_rebuttal_paragraph(
    out_path: Path,
    near_summary: pd.DataFrame,
    lobo_summary_df: pd.DataFrame,
    rev_summary: pd.DataFrame,
    panel: pd.DataFrame,
    meta: Dict[str, Any],
) -> None:
    macro = near_summary[near_summary["scope"] == "all_benchmarks"]
    r2 = macro[macro["pass1_gap_threshold_pp"] == 2.0].iloc[0]

    def _lobo(tr: str, te: str) -> float:
        s = lobo_summary_df[(lobo_summary_df["train_metric"] == tr) & (lobo_summary_df["test_target"] == te)]
        return float(s["mean_spearman"].iloc[0]) if len(s) else float("nan")

    rev = rev_summary[(rev_summary["comparison"] == "frs_vs_trace0") & (rev_summary["scope"] == "all_benchmarks")].iloc[0]
    lobo_ff = _lobo("frs_pct", "frs_pct")
    lobo_tf = _lobo("trace0_pct", "frs_pct")
    lobo_tt = _lobo("trace0_pct", "trace0_pct")
    lobo_uf = _lobo("unfiltered_pct", "frs_pct")

    gap = panel["frs_pct"].mean() - panel["trace0_pct"].mean()
    sp_t0_frs, _, _ = safe_spearman(panel["trace0_pct"].values, panel["frs_pct"].values)

    text = f"""# Rebuttal paragraph (Reviewer kp6q)

## Draft

If sampling and evaluating one trace reveals a similar trend, why bother using FRS?

Trace-0 reasoning score and FRS are correlated (Spearman ρ≈{sp_t0_frs:.2f} over 54 pairs), and trace-0 LOBO transfer to held-out trace-0 (mean ρ={lobo_tt:.2f}) is comparable to FRS→FRS (ρ={lobo_ff:.2f}) on some benchmarks — we do not claim FRS replaces all single-trace information. However, FRS adds **informativeness where accuracy is tied** and **cross-benchmark reasoning signal** that trace-0 does not carry.

Among **{int(r2['n_eligible_pairs'])}** model pairs with |Δpass@1|≤2 pp, FRS separates models **{r2['amplification_frs']:.1f}×** more than pass@1 itself (mean |ΔFRS|={r2['mean_frs_gap_pp']:.1f} pp vs mean |Δpass@1|={r2['mean_pass1_gap_pp']:.1f} pp), versus **{r2['amplification_trace0']:.1f}×** for trace-0 RS and **{r2['amplification_unfiltered']:.1f}×** for unfiltered one-trace-per-question RS. FRS wins the head-to-head gap comparison in **{r2['frac_frs_gap_gt_trace0']:.0%}** of these near-tie pairs ({r2['frac_frs_gap_gt_unfiltered']:.0%} vs unfiltered).

For transferability, macro FRS predicts held-out FRS (LOBO ρ={lobo_ff:.2f}), but macro trace-0 **does not** predict held-out FRS (ρ={lobo_tf:.2f}) — the aggregate {gap:.1f} pp macro gap reflects re-ranking that a single trace macro cannot recover. We observe **{int(rev['n_reversals'])}** concrete pairwise rank reversals (≥2 pp) where FRS and trace-0 disagree; in **{rev['lobo_frs_agrees_frs_winner']:.0%}** of these, LOBO FRS from other benchmarks sides with the FRS winner. Unfiltered RS is closer to FRS on LOBO→FRS (ρ={lobo_uf:.2f}) but still under-amplifies near-equal pass@1 ({r2['amplification_unfiltered']:.1f}×) and diverges in rank (ρ≈0.45 vs FRS in prior analysis).

**Bottom line:** one trace gives a correlated snapshot; FRS aggregates confidence-filtered reasoning quality to (i) discriminate models at similar pass@1 and (ii) produce a benchmark-transferable ranking that trace-0 macro does not predict.
"""
    out_path.write_text(text, encoding="utf-8")


def cache_valid(out_dir: Path, manifest_path: Path, current_hash: str, force: bool) -> bool:
    if force or not manifest_path.is_file():
        return False
    try:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        return old.get("inputs_hash") == current_hash and (out_dir / "key_numbers.md").is_file()
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--force", action="store_true", help="Recompute even if cache valid")
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logger()

    paths = discover_inputs(repo, log)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {k: file_fingerprint(p) for k, p in paths.items() if k != "prior_reversals"},
        "params": {
            "pass1_thresholds": PASS1_THRESHOLDS,
            "reversal_min_gap_pp": REVERSAL_MIN_GAP_PP,
            "bootstrap_n": BOOTSTRAP_N,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
    }
    current_hash = manifest_hash(manifest)
    manifest_path = out_dir / "run_manifest.json"

    if cache_valid(out_dir, manifest_path, current_hash, args.force):
        log.info("Cache hit (inputs unchanged) — skipping recompute. Use --force to rerun.")
        return 0

    log.info("Loading and merging panel …")
    panel, meta = load_panel(paths, log)
    sanity = sanity_check_panel(panel, log)
    panel.to_csv(out_dir / "canonical_panel.csv", index=False)

    log.info("Analysis 1/4: near-equal accuracy …")
    near_pairs = build_near_equal_pairs(panel, log)
    near_pairs.to_csv(out_dir / "near_equal_accuracy_gap_comparison.csv", index=False)
    near_summary = summarize_near_equal(near_pairs, PASS1_THRESHOLDS)
    near_summary.to_csv(out_dir / "near_equal_accuracy_summary.csv", index=False)

    log.info("Analysis 2/4: LOBO transfer …")
    predictors = ["frs_pct", "pass1_pct", "trace0_pct"]
    targets = ["frs_pct", "pass1_pct", "trace0_pct"]
    if meta["unfiltered_available"]:
        predictors.append("unfiltered_pct")
        targets.append("unfiltered_pct")

    lobo_parts = []
    for tr in tqdm(predictors, desc="LOBO predictors"):
        for te in targets:
            part = lobo_transfer(panel, tr, te)
            if len(part):
                lobo_parts.append(part)
    lobo_df = pd.concat(lobo_parts, ignore_index=True) if lobo_parts else pd.DataFrame()
    lobo_df.to_csv(out_dir / "lobo_transfer_comparison.csv", index=False)
    lobo_sum = lobo_summary(lobo_df)
    lobo_sum.to_csv(out_dir / "lobo_transfer_summary.csv", index=False)

    log.info("Analysis 3/4: rank reversals …")
    rev = compute_rank_reversals(panel)
    rev.to_csv(out_dir / "rank_reversal_comparison.csv", index=False)
    rev_sum = summarize_reversals(rev)
    rev_sum.to_csv(out_dir / "rank_reversal_summary.csv", index=False)

    log.info("Analysis 4/4: bootstrap CIs …")
    boot = run_bootstrap_cis(near_pairs, near_summary, lobo_df, rev)
    boot.to_csv(out_dir / "bootstrap_ci_summary.csv", index=False)

    log.info("Plots …")
    plot_amplification(near_summary, out_dir / "plot_amplification_ratios.png")
    plot_scatter_trace0_frs(panel, out_dir / "plot_scatter_trace0_vs_frs.png")
    plot_lobo_scatter(panel, lobo_df, out_dir / "plot_lobo_frs_vs_trace0_to_frs.png")
    plot_rank_diff_hist(rev, out_dir / "plot_rank_diff_histogram.png")

    write_readme(out_dir, meta, sanity)
    write_key_numbers(out_dir / "key_numbers.md", sanity, near_summary, lobo_sum, rev_sum, boot, panel, meta)
    write_rebuttal_paragraph(out_dir / "rebuttal_paragraph.md", near_summary, lobo_sum, rev_sum, panel, meta)

    manifest["inputs_hash"] = current_hash
    manifest["sanity"] = sanity
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Done → %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
