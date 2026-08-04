#!/usr/bin/env python3
"""
High-impact follow-up analyses for FRS (feasibility triage + selected runs).

Usage:
  python analysis/run_high_impact_followup.py --repo-root .
  python analysis/run_high_impact_followup.py --repo-root . --skip-heldout   # faster
  python analysis/run_high_impact_followup.py --repo-root . --reuse-heldout-cache

Outputs under analysis/: feasibility table, summary CSVs, report, logs, figures.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore

try:
    import statsmodels.api as sm
except ImportError:
    sm = None  # type: ignore

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

DATASET_TO_BENCHMARK = {
    "GSM8K": "GSM8K",
    "MATH500": "MATH500",
    "SVAMP": "SVAMP",
    "AQuA": "AQuA",
    "GPQA": "GPQA",
    "CommonsenseQA": "CSQA",
}

HELDOUT_SEEDS = (17, 42, 99, 123, 456, 789, 2024, 3141)


def setup_logging(log_dir: Path, name: str) -> Tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{name}_{ts}.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, log_path


def log_header(logger: logging.Logger, title: str) -> None:
    line = "=" * 72
    logger.info(line)
    logger.info(title)
    logger.info(line)


def write_feasibility_table(repo: Path, out_csv: Path, logger: logging.Logger) -> pd.DataFrame:
    """STEP 1: static feasibility audit (verified against repo layout)."""
    rows = [
        {
            "id": "A",
            "analysis": "Held-out question split predictor",
            "required_inputs": "Per-question judge scores; pair-level predictors (FRS, pass@1, ...)",
            "exists": "yes: unfiltered judging_checkpoints/*.json keyed by question_idx:trace_idx",
            "join": "per_pair_scores.checkpoint_path -> aggregate mean reasoning by idx; merge panel on model,benchmark",
            "circularity_risk": "moderate: same judge; outcome split reduces overlap with aggregate mean",
            "cost": "low–medium: 54 JSON files per run",
            "paper_value": "high: addresses out-of-sample generalization of mean reasoning",
            "verdict": "fully feasible now",
        },
        {
            "id": "B",
            "analysis": "Held-out selection-gain (conf vs random trace reasoning)",
            "required_inputs": "Judge scores for multiple traces per question under selection",
            "exists": "only one judged trace per sampled question (100/100 unique idx)",
            "join": "n/a",
            "circularity_risk": "n/a",
            "cost": "n/a",
            "paper_value": "high if feasible",
            "verdict": "not feasible now",
        },
        {
            "id": "C",
            "analysis": "Leave-one-dataset-out incremental validity",
            "required_inputs": "Panel 54× with outcome + predictors",
            "exists": "analysis/frs_predictor_panel.csv",
            "join": "benchmark column",
            "circularity_risk": "same as baseline OLS",
            "cost": "very low",
            "paper_value": "high: reviewer-facing generalization across benchmarks",
            "verdict": "fully feasible now",
        },
        {
            "id": "D",
            "analysis": "Leave-one-model-out incremental validity",
            "required_inputs": "Same panel",
            "exists": "yes",
            "join": "model column",
            "circularity_risk": "same as baseline",
            "cost": "very low",
            "paper_value": "high",
            "verdict": "fully feasible now",
        },
        {
            "id": "E",
            "analysis": "Pairwise winner prediction (near-ties)",
            "required_inputs": "Panel with pass1, pass16, frs, outcome",
            "exists": "yes",
            "join": "within benchmark pairs",
            "circularity_risk": "moderate (judge outcome)",
            "cost": "very low",
            "paper_value": "medium: interpretable deployment tie-break story",
            "verdict": "fully feasible now",
        },
        {
            "id": "F",
            "analysis": "Alternative confidence proxy (SC vote-share FRS k10)",
            "required_inputs": "Precomputed SC proxy per model×benchmark",
            "exists": "analysis_exports/self_consistency_proxy_robustness/sc_proxy_benchmark_results_k10.csv",
            "join": "model, benchmark",
            "circularity_risk": "moderate–high (reasoning in top bin)",
            "cost": "low",
            "paper_value": "medium: robustness of ranking / incremental validity",
            "verdict": "fully feasible now",
        },
        {
            "id": "G",
            "analysis": "Judge-robustness",
            "required_inputs": "Second judge or duplicate scores",
            "exists": "run_metadata shows single judge gpt-4o-mini",
            "join": "n/a",
            "circularity_risk": "n/a",
            "cost": "n/a",
            "paper_value": "high if data existed",
            "verdict": "not feasible now",
        },
        {
            "id": "H",
            "analysis": "Regime classification (SNR sign vs generalization)",
            "required_inputs": "snr + LOCO folds",
            "exists": "snr on panel",
            "join": "panel",
            "circularity_risk": "low–moderate",
            "cost": "low",
            "paper_value": "exploratory appendix",
            "verdict": "partially feasible now",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    logger.info("Wrote feasibility table: %s (%d rows)", out_csv, len(df))
    return df


def load_panel(repo: Path, logger: logging.Logger) -> pd.DataFrame:
    p = repo / "analysis" / "frs_predictor_panel.csv"
    if not p.is_file():
        raise FileNotFoundError(f"Missing {p}; run analysis/run_frs_predictor_analysis.py first.")
    df = pd.read_csv(p)
    logger.info("Loaded panel %s rows=%d cols=%s", p, len(df), list(df.columns))
    return df


def ols_delta_r2(
    y: np.ndarray,
    X4: np.ndarray,
    X5: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Return r2_m4, r2_m5, dr2, n."""
    if sm is None:
        return float("nan"), float("nan"), float("nan"), int(len(y))
    m4 = len(X4[0]) if X4.ndim == 2 else 0
    m5 = len(X5[0]) if X5.ndim == 2 else 0
    X4c = sm.add_constant(X4, has_constant="add")
    X5c = sm.add_constant(X5, has_constant="add")
    r4 = sm.OLS(y, X4c).fit()
    r5 = sm.OLS(y, X5c).fit()
    return float(r4.rsquared), float(r5.rsquared), float(r5.rsquared - r4.rsquared), int(r5.nobs)


def loco_regression(
    df: pd.DataFrame,
    group_col: str,
    outcome: str,
    base_cols: List[str],
    logger: logging.Logger,
    label: str,
) -> pd.DataFrame:
    """Leave-one-group-out: fit m4/m5 on train, R² on test predictions."""
    if sm is None:
        logger.warning("statsmodels missing; skipping LOCO")
        return pd.DataFrame()
    groups = sorted(df[group_col].dropna().unique())
    fold_rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    logger.info(
        "LOCO %s: %d folds, outcome=%s, n_rows=%d, estimate ~%d OLS fits",
        label,
        len(groups),
        outcome,
        len(df),
        len(groups) * 2,
    )
    for g in tqdm(groups, desc=f"LOCO-{label}", disable=tqdm is None):
        train = df[df[group_col] != g].dropna(subset=[outcome] + base_cols + ["frs_pct"])
        test = df[df[group_col] == g].dropna(subset=[outcome] + base_cols + ["frs_pct"])
        if len(train) < len(base_cols) + 3 or len(test) < 2:
            logger.warning("LOCO skip fold %s: train_n=%d test_n=%d", g, len(train), len(test))
            continue
        y_tr = train[outcome].values.astype(float)
        y_te = test[outcome].values.astype(float)
        Xb_tr = train[base_cols].values.astype(float)
        Xb_te = test[base_cols].values.astype(float)
        f_tr = train["frs_pct"].values.astype(float)
        f_te = test["frs_pct"].values.astype(float)
        X4_tr = sm.add_constant(Xb_tr, has_constant="add")
        X5_tr = sm.add_constant(np.column_stack([Xb_tr, f_tr]), has_constant="add")
        X4_te = sm.add_constant(Xb_te, has_constant="add")
        X5_te = sm.add_constant(np.column_stack([Xb_te, f_te]), has_constant="add")
        fit4 = sm.OLS(y_tr, X4_tr).fit()
        fit5 = sm.OLS(y_tr, X5_tr).fit()
        p4 = np.asarray(fit4.params, dtype=float).ravel()
        p5 = np.asarray(fit5.params, dtype=float).ravel()
        pred4 = X4_te @ p4
        pred5 = X5_te @ p5
        ss_res4 = np.sum((y_te - pred4) ** 2)
        ss_res5 = np.sum((y_te - pred5) ** 2)
        ss_tot = np.sum((y_te - np.mean(y_te)) ** 2)
        r2_te4 = 1 - ss_res4 / ss_tot if ss_tot > 0 else float("nan")
        r2_te5 = 1 - ss_res5 / ss_tot if ss_tot > 0 else float("nan")
        fold_rows.append(
            {
                "fold_type": label,
                "held_out_group": g,
                "n_train": len(train),
                "n_test": len(test),
                "r2_test_m4": r2_te4,
                "r2_test_m5": r2_te5,
                "delta_r2_test_add_frs": r2_te5 - r2_te4,
                "rmse_test_m4": float(np.sqrt(np.mean((y_te - pred4) ** 2))),
                "rmse_test_m5": float(np.sqrt(np.mean((y_te - pred5) ** 2))),
            }
        )
    out = pd.DataFrame(fold_rows)
    logger.info(
        "LOCO %s done in %.2fs | folds=%d | mean delta_r2_test=%s",
        label,
        time.perf_counter() - t0,
        len(out),
        out["delta_r2_test_add_frs"].mean() if len(out) else "n/a",
    )
    return out


def parse_checkpoint_question_means(checkpoint_path: Path, logger: logging.Logger) -> Dict[int, float]:
    with open(checkpoint_path, encoding="utf-8") as f:
        data = json.load(f)
    by_q: Dict[int, List[float]] = defaultdict(list)
    for k, v in data.get("judged_samples", {}).items():
        if ":" not in k:
            continue
        try:
            q = int(k.split(":")[0])
        except ValueError:
            continue
        if not v.get("judge_ok"):
            continue
        rs = v.get("reasoning_score")
        if rs is None:
            continue
        by_q[q].append(float(rs))
    return {q: float(np.mean(vals)) for q, vals in by_q.items()}


def heldout_question_outcomes(
    repo: Path,
    df_pairs: pd.DataFrame,
    seeds: Tuple[int, ...],
    cache_path: Path,
    reuse: bool,
    logger: logging.Logger,
) -> pd.DataFrame:
    if reuse and cache_path.is_file():
        logger.info("Reusing held-out cache: %s", cache_path)
        return pd.read_csv(cache_path)

    rows: List[Dict[str, Any]] = []
    df_pairs = df_pairs.copy()
    df_pairs["benchmark"] = df_pairs["dataset"].map(DATASET_TO_BENCHMARK)
    n_pairs = len(df_pairs)
    logger.info(
        "Held-out question split: %d pairs × %d seeds = %d rows (estimate)",
        n_pairs,
        len(seeds),
        n_pairs * len(seeds),
    )
    iterator = df_pairs.iterrows()
    if tqdm:
        iterator = tqdm(list(df_pairs.iterrows()), desc="checkpoints", unit="pair")

    for _, row in iterator:
        cp = row["checkpoint_path"]
        path = Path(cp)
        if not path.is_file():
            path = repo / cp
        if not path.is_file():
            logger.error("Missing checkpoint: %s", cp)
            continue
        t1 = time.perf_counter()
        qmeans = parse_checkpoint_question_means(path, logger)
        qs = sorted(qmeans.keys())
        if len(qs) < 10:
            logger.warning("Pair %s %s: only %d questions", row["model"], row["dataset"], len(qs))
        vals = np.array([qmeans[q] for q in qs], dtype=float)
        for seed in seeds:
            rng = np.random.RandomState(seed)
            order = rng.permutation(len(qs))
            half = len(qs) // 2
            tr_idx, te_idx = order[:half], order[half:]
            y_train = float(np.mean(vals[tr_idx]))
            y_test = float(np.mean(vals[te_idx]))
            rows.append(
                {
                    "model": row["model"],
                    "dataset": row["dataset"],
                    "benchmark": row["benchmark"],
                    "seed": seed,
                    "n_questions": len(qs),
                    "y_train_mean_reasoning": y_train,
                    "y_test_mean_reasoning": y_test,
                    "checkpoint_path": str(cp),
                }
            )
        logger.debug(
            "checkpoint %s %s: %d qs, parse_time=%.3fs",
            row["model"],
            row["dataset"],
            len(qs),
            time.perf_counter() - t1,
        )

    out = pd.DataFrame(rows)
    out.to_csv(cache_path, index=False)
    logger.info("Wrote held-out outcomes: %s (%d rows)", cache_path, len(out))
    return out


def merge_heldout_with_panel(
    held: pd.DataFrame,
    panel: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    m = held.merge(
        panel,
        on=["model", "benchmark"],
        how="inner",
        suffixes=("", "_panel"),
    )
    logger.info("heldout merge panel: held=%d merged=%d dropped=%d", len(held), len(m), len(held) - len(m))
    if len(m) < len(held):
        logger.warning("Unmatched held-out rows:\n%s", held.merge(panel, on=["model", "benchmark"], how="left", indicator=True).query('_merge=="left_only"').head(20))
    return m


def pairwise_winner_table(
    df: pd.DataFrame,
    tie_cols: List[Tuple[str, str]],
    tolerances: Tuple[float, ...],
    outcome_col: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Direction accuracy: sign(metric_a - metric_b) vs sign(outcome_a - outcome_b)."""
    rows: List[Dict[str, Any]] = []
    benchmarks = df["benchmark"].unique()
    for bench in benchmarks:
        sub = df[df["benchmark"] == bench].reset_index(drop=True)
        models = sub["model"].tolist()
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                mi, mj = models[i], models[j]
                ri = sub[sub["model"] == mi].iloc[0]
                rj = sub[sub["model"] == mj].iloc[0]
                oa, ob = float(ri[outcome_col]), float(rj[outcome_col])
                true_sign = 1 if oa > ob else (-1 if oa < ob else 0)
                if true_sign == 0:
                    continue
                for tie_name, col in tie_cols:
                    for tol in tolerances:
                        d = abs(float(ri[col]) - float(rj[col]))
                        if d > tol:
                            continue
                        for pred_name, pcol in [
                            ("frs_pct", "frs_pct"),
                            ("pass1_pct", "pass1_pct"),
                            ("pass16_pct", "pass16_pct"),
                            ("high_conf_accuracy_pct", "high_conf_accuracy_pct"),
                        ]:
                            pa, pb = float(ri[pcol]), float(rj[pcol])
                            ps = 1 if pa > pb else (-1 if pa < pb else 0)
                            correct = ps == true_sign
                            rows.append(
                                {
                                    "benchmark": bench,
                                    "tie_metric": tie_name,
                                    "tolerance_pp": tol,
                                    "model_a": mi,
                                    "model_b": mj,
                                    "predictor": pred_name,
                                    "correct_direction": bool(correct),
                                    "abs_tie_gap": d,
                                }
                            )
    out = pd.DataFrame(rows)
    logger.info("pairwise raw rows: %d", len(out))
    return out


def summarize_pairwise(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    g = (
        df.groupby(["tie_metric", "tolerance_pp", "predictor"], as_index=False)
        .agg(
            n_pairs=("correct_direction", "count"),
            accuracy=("correct_direction", "mean"),
        )
    )
    logger.info("pairwise summary:\n%s", g.to_string())
    return g


def sc_proxy_analysis(
    repo: Path,
    panel: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    p = repo / "analysis_exports" / "self_consistency_proxy_robustness" / "sc_proxy_benchmark_results_k10.csv"
    if not p.is_file():
        logger.error("Missing SC proxy file: %s", p)
        return pd.DataFrame()
    sc = pd.read_csv(p)
    sc["benchmark"] = sc["benchmark"].replace({"CommonsenseQA": "CSQA"})
    logger.info("SC proxy rows=%d cols=%s (CommonsenseQA→CSQA for join)", len(sc), list(sc.columns))
    m = panel.merge(sc, on=["model", "benchmark"], how="inner")
    logger.info("SC merge inner: %d (panel=%d)", len(m), len(panel))
    if len(m) < len(panel):
        logger.warning("SC merge dropped %d rows", len(panel) - len(m))
    y = m["unfiltered_reasoning_mean"].values.astype(float)
    results: List[Dict[str, Any]] = []
    for name, col in [("frs_pct", "frs_pct"), ("sc_frs_k10", "sc_frs_k10")]:
        x = m[col].values.astype(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 5:
            continue
        pr = stats.pearsonr(y[ok], x[ok])
        sp = stats.spearmanr(y[ok], x[ok])
        results.append(
            {
                "metric": name,
                "pearson_r": float(getattr(pr, "statistic", pr[0])),
                "pearson_p": float(getattr(pr, "pvalue", pr[1])),
                "spearman_rho": float(getattr(sp, "statistic", sp[0])),
                "spearman_p": float(getattr(sp, "pvalue", sp[1])),
                "n": int(ok.sum()),
            }
        )
    if sm is not None and len(m) > 10:
        base = ["pass1_pct", "high_conf_accuracy_pct", "pass16_pct", "snr"]
        mask = np.ones(len(m), dtype=bool)
        for c in base + ["frs_pct", "sc_frs_k10", "unfiltered_reasoning_mean"]:
            mask &= m[c].notna().values
        m2 = m.loc[mask]
        y2 = m2["unfiltered_reasoning_mean"].values.astype(float)
        Xb = m2[base].values.astype(float)
        # m4 + frs
        X4 = sm.add_constant(Xb, has_constant="add")
        X5f = sm.add_constant(np.column_stack([Xb, m2["frs_pct"].values.astype(float)]), has_constant="add")
        X5s = sm.add_constant(np.column_stack([Xb, m2["sc_frs_k10"].values.astype(float)]), has_constant="add")
        r4 = sm.OLS(y2, X4).fit()
        r5f = sm.OLS(y2, X5f).fit()
        r5s = sm.OLS(y2, X5s).fit()
        results.append(
            {
                "metric": "delta_r2_add_frs_after_base4",
                "value": float(r5f.rsquared - r4.rsquared),
                "r2_full": float(r5f.rsquared),
            }
        )
        results.append(
            {
                "metric": "delta_r2_add_sc_frs_k10_after_base4",
                "value": float(r5s.rsquared - r4.rsquared),
                "r2_full": float(r5s.rsquared),
            }
        )
    return pd.DataFrame(results)


def plot_loco_bars(df_loco: pd.DataFrame, out_path: Path, logger: logging.Logger) -> None:
    if plt is None or df_loco.empty:
        return
    sub = df_loco
    ft = str(sub["fold_type"].iloc[0])
    plt.figure(figsize=(8, 4))
    x = np.arange(len(sub))
    plt.bar(x - 0.2, sub["r2_test_m4"], width=0.4, label="R² test m4")
    plt.bar(x + 0.2, sub["r2_test_m5"], width=0.4, label="R² test m5")
    plt.xticks(x, sub["held_out_group"].astype(str), rotation=45, ha="right")
    plt.ylabel("Test R²")
    plt.title(f"LOCO {ft}: test R² by fold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Wrote %s", out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--skip-heldout", action="store_true", help="Skip question-level held-out parsing")
    ap.add_argument("--reuse-heldout-cache", action="store_true")
    args = ap.parse_args()
    repo = Path(args.repo_root).resolve()
    out_dir = repo / "analysis"
    fig_dir = out_dir / "figures" / "high_impact_followup"
    log_dir = out_dir / "logs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger, log_path = setup_logging(log_dir, "high_impact_followup")
    log_header(logger, "HIGH-IMPACT FOLLOW-UP — START")
    logger.info("repo_root=%s", repo)
    logger.info("log_file=%s", log_path)
    t_all = time.perf_counter()

    # STEP 1–2: feasibility + priority (written as CSV + logged)
    feas_path = out_dir / "high_impact_feasibility_table.csv"
    df_feas = write_feasibility_table(repo, feas_path, logger)

    priority_md = """
| Rank | Analysis | Why | Feasibility | Circularity | Reviewer impact | Runtime | Action |
|------|----------|-----|---------------|-------------|-----------------|---------|--------|
| 1 | LOCO + LOMO incremental validity | Direct OOS generalization | Full | Moderate | Very high | Seconds | **Run** |
| 2 | Held-out question mean reasoning | Reduces overlap in outcome vs full mean | Full | Moderate | High | ~1–2 min | **Run** |
| 3 | Pairwise winner (multi-tolerance) | Tie-break narrative | Full | Moderate | Medium | Seconds | **Run** |
| 4 | SC proxy vs FRS incremental ΔR² | Robustness to confidence definition | Full | Moderate–high | Medium | Seconds | **Run** |
| — | Selection-gain on held-out traces | Needs multi-trace judge / question | **Not feasible** | — | High | — | Skip |
| — | Second-judge robustness | Single judge in metadata | **Not feasible** | — | High | — | Skip |
"""
    logger.info("Priority table (markdown):\n%s", priority_md)

    panel = load_panel(repo, logger)
    base_cols = ["pass1_pct", "high_conf_accuracy_pct", "pass16_pct", "snr"]
    for c in base_cols + ["frs_pct", "unfiltered_reasoning_mean"]:
        if c not in panel.columns:
            logger.error("panel missing column: %s", c)
            sys.exit(1)

    summary_rows: List[Dict[str, Any]] = []

    # --- LOCO / LOMO ---
    log_header(logger, "STEP: LOCO (leave-one-benchmark-out)")
    loco = loco_regression(panel, "benchmark", "unfiltered_reasoning_mean", base_cols, logger, "LODO-benchmark")
    log_header(logger, "STEP: LOMO (leave-one-model-out)")
    lomo = loco_regression(panel, "model", "unfiltered_reasoning_mean", base_cols, logger, "LOMO-model")
    gen = pd.concat([loco, lomo], ignore_index=True) if len(loco) or len(lomo) else pd.DataFrame()
    gen_path = out_dir / "generalization_results.csv"
    gen.to_csv(gen_path, index=False)
    logger.info("Wrote combined %s (%d rows)", gen_path, len(gen))
    loco.to_csv(out_dir / "generalization_results_lodo_only.csv", index=False)
    lomo.to_csv(out_dir / "generalization_results_lomo_only.csv", index=False)

    plot_loco_bars(loco, fig_dir / "loco_LODO-benchmark_test_r2.png", logger)
    plot_loco_bars(lomo, fig_dir / "loco_LOMO-model_test_r2.png", logger)

    summary_rows.append(
        {
            "analysis": "LODO_benchmark",
            "mean_delta_r2_test_add_frs": loco["delta_r2_test_add_frs"].mean() if len(loco) else None,
            "std_delta_r2_test": loco["delta_r2_test_add_frs"].std() if len(loco) else None,
            "n_folds": len(loco),
        }
    )
    summary_rows.append(
        {
            "analysis": "LOMO_model",
            "mean_delta_r2_test_add_frs": lomo["delta_r2_test_add_frs"].mean() if len(lomo) else None,
            "std_delta_r2_test": lomo["delta_r2_test_add_frs"].std() if len(lomo) else None,
            "n_folds": len(lomo),
        }
    )

    # SNR regime × LODO delta (partial H)
    if len(loco) and "snr" in panel.columns:
        reg_rows = []
        for _, row in loco.iterrows():
            ho = row["held_out_group"]
            sub = panel[panel["benchmark"] == ho]
            reg_rows.append(
                {
                    "held_out_benchmark": ho,
                    "mean_snr_heldout_fold": sub["snr"].mean(),
                    "delta_r2_test_add_frs": row["delta_r2_test_add_frs"],
                }
            )
        pd.DataFrame(reg_rows).to_csv(out_dir / "regime_snr_loco_fold.csv", index=False)
        logger.info("Wrote regime_snr_loco_fold.csv")

    # --- Held-out question ---
    held_path = out_dir / "heldout_predictor_results.csv"
    if args.skip_heldout:
        logger.info("Skipping held-out question analysis (--skip-heldout)")
        held_merged = pd.DataFrame()
        pd.DataFrame([{"note": "skipped", "reason": "--skip-heldout"}]).to_csv(held_path, index=False)
    else:
        log_header(logger, "STEP: Held-out question mean reasoning")
        pp = repo / "analysis_outputs" / "unfiltered_reasoning" / "per_pair_scores.csv"
        df_pairs = pd.read_csv(pp)
        cache_h = out_dir / "cache" / "heldout_question_outcomes.csv"
        cache_h.parent.mkdir(parents=True, exist_ok=True)
        held = heldout_question_outcomes(
            repo, df_pairs, HELDOUT_SEEDS, cache_h, args.reuse_heldout_cache, logger
        )
        held_merged = merge_heldout_with_panel(held, panel, logger)
        hp_path = out_dir / "heldout_question_outcomes_merged.csv"
        held_merged.to_csv(hp_path, index=False)
        logger.info("Wrote %s", hp_path)

        hr_rows: List[Dict[str, Any]] = []
        for seed, sub in held_merged.groupby("seed"):
            y = sub["y_test_mean_reasoning"].values.astype(float)
            for pred in ["frs_pct", "pass1_pct"]:
                x = sub[pred].values.astype(float)
                ok = np.isfinite(x) & np.isfinite(y)
                pr = stats.pearsonr(y[ok], x[ok])
                hr_rows.append(
                    {
                        "seed": seed,
                        "predictor": pred,
                        "pearson_r": float(getattr(pr, "statistic", pr[0])),
                        "pearson_p": float(getattr(pr, "pvalue", pr[1])),
                        "n": int(ok.sum()),
                        "outcome": "y_test_mean_reasoning",
                    }
                )
            if sm is not None:
                mask = np.ones(len(sub), dtype=bool)
                for c in base_cols + ["frs_pct"]:
                    mask &= sub[c].notna().values
                s2 = sub.loc[mask]
                y2 = s2["y_test_mean_reasoning"].values.astype(float)
                Xb = s2[base_cols].values.astype(float)
                f = s2["frs_pct"].values.astype(float)
                r4, r5, dr2, nobs = ols_delta_r2(
                    y2,
                    Xb,
                    np.column_stack([Xb, f]),
                )
                hr_rows.append(
                    {
                        "seed": seed,
                        "predictor": "delta_r2_m5_m4",
                        "delta_r2_add_frs": dr2,
                        "r2_m4": r4,
                        "r2_m5": r5,
                        "n": nobs,
                        "outcome": "y_test_mean_reasoning",
                    }
                )
        pd.DataFrame(hr_rows).to_csv(held_path, index=False)
        logger.info("Wrote %s", held_path)

        # aggregate across seeds
        agg = held_merged.groupby(["model", "benchmark"], as_index=False)["y_test_mean_reasoning"].mean()
        agg = agg.merge(panel, on=["model", "benchmark"], how="inner")
        if sm is not None and len(agg) > 10:
            y2 = agg["y_test_mean_reasoning"].values.astype(float)
            Xb = agg[base_cols].values.astype(float)
            f = agg["frs_pct"].values.astype(float)
            r4, r5, dr2, _ = ols_delta_r2(y2, Xb, np.column_stack([Xb, f]))
            summary_rows.append(
                {
                    "analysis": "heldout_y_test_mean_over_seeds",
                    "mean_delta_r2_test_add_frs": dr2,
                    "r2_m4": r4,
                    "r2_m5": r5,
                    "n_folds": len(agg),
                    "note": "mean y_test across seeds then one OLS on 54 rows",
                }
            )

    # --- Pairwise ---
    log_header(logger, "STEP: Pairwise winner prediction")
    tie_cols = [("pass1_pct", "pass1_pct"), ("pass16_pct", "pass16_pct")]
    tolerances = (1.0, 2.0, 3.0, 5.0)
    pw_raw = pairwise_winner_table(panel, tie_cols, tolerances, "unfiltered_reasoning_mean", logger)
    pw_raw.to_csv(out_dir / "pairwise_winner_results_raw.csv", index=False)
    pw_sum = summarize_pairwise(pw_raw, logger)
    pw_sum.to_csv(out_dir / "pairwise_winner_results.csv", index=False)

    pair_counts = (
        pw_raw.groupby(["tie_metric", "tolerance_pp"]).size().reset_index(name="n_predictions")
    )
    pair_counts.to_csv(out_dir / "pairwise_winner_pair_counts.csv", index=False)

    # --- SC proxy ---
    log_header(logger, "STEP: Self-consistency proxy robustness")
    sc_res = sc_proxy_analysis(repo, panel, logger)
    sc_res.to_csv(out_dir / "sc_proxy_robustness_results.csv", index=False)
    # Same data as SC proxy incremental validity (not trace-level selection gain; see report).
    sc_res.to_csv(out_dir / "selection_gain_results.csv", index=False)
    logger.info("SC proxy results:\n%s", sc_res.to_string())

    # --- In-panel full-sample ΔR² for reference ---
    if sm is not None:
        y = panel["unfiltered_reasoning_mean"].values.astype(float)
        Xb = panel[base_cols].values.astype(float)
        f = panel["frs_pct"].values.astype(float)
        r4, r5, dr2, n = ols_delta_r2(y, Xb, np.column_stack([Xb, f]))
        summary_rows.append(
            {
                "analysis": "full_panel_baseline_same_spec",
                "mean_delta_r2_test_add_frs": dr2,
                "r2_m4": r4,
                "r2_m5": r5,
                "n_folds": n,
            }
        )

    pd.DataFrame(summary_rows).to_csv(out_dir / "high_impact_followup_summary.csv", index=False)

    # --- Report (dynamic numbers) ---
    report_path = out_dir / "high_impact_followup_report.md"
    n_lodo_pos = int((loco["delta_r2_test_add_frs"] > 0).sum()) if len(loco) else 0
    n_lomo_pos = int((lomo["delta_r2_test_add_frs"] > 0).sum()) if len(lomo) else 0
    draft_para = (
        "We assessed incremental validity of FRS beyond pass@1, pass@16, high-confidence accuracy, and SNR using "
        "leave-one-benchmark-out and leave-one-model-out OLS, predicting mean unfiltered reasoning on the held-out "
        f"group's model×benchmark pairs. Adding FRS improved test-set R² in {n_lodo_pos}/{len(loco)} benchmark folds "
        f"(mean ΔR²_test={loco['delta_r2_test_add_frs'].mean():.3f}) and {n_lomo_pos}/{len(lomo)} model folds "
        f"(mean ΔR²_test={lomo['delta_r2_test_add_frs'].mean():.3f}). "
        "Held-out question splits (8 seeds) provide an alternative outcome construction; see `heldout_predictor_results.csv`."
    )

    lines = [
        "# High-impact follow-up analyses",
        "",
        f"Generated: {datetime.now().isoformat()}",
        f"Log: `{log_path.relative_to(repo)}`",
        "",
        "## STEP 1 — Feasibility",
        "",
        f"See `{feas_path.relative_to(repo)}` for A–H verdicts.",
        "",
        "## STEP 2 — Priority (executed: 1–4)",
        "",
        priority_md,
        "",
        "## STEP 3–4 — Results summary",
        "",
        "### Leave-one-dataset-out (benchmark)",
        "",
        "- **Note:** Test R² can be **negative** on small held-out groups (n_test=9 here); folds are **illustrative**, not calibrated forecasts.",
        "",
        f"- Folds: {len(loco)} | Positive ΔR²_test folds: **{n_lodo_pos}/{len(loco)}** | Mean ΔR² (test, add FRS): **{loco['delta_r2_test_add_frs'].mean():.4f}**" if len(loco) else "- Skipped",
        f"- Std: {loco['delta_r2_test_add_frs'].std():.4f}" if len(loco) else "",
        f"- Detail: `{gen_path.relative_to(repo)}` (filter `fold_type==\"LODO-benchmark\"`)",
        "",
        "### Leave-one-model-out",
        "",
        f"- Folds: {len(lomo)} | Positive ΔR²_test folds: **{n_lomo_pos}/{len(lomo)}** | Mean ΔR² (test, add FRS): **{lomo['delta_r2_test_add_frs'].mean():.4f}**" if len(lomo) else "",
        f"- Std: {lomo['delta_r2_test_add_frs'].std():.4f}" if len(lomo) else "",
        "",
        "### Held-out question mean (y_test)",
        "",
        "- Per-seed OLS and correlations: `analysis/heldout_predictor_results.csv`",
        "- Merged long table: `analysis/heldout_question_outcomes_merged.csv`" if not args.skip_heldout else "- Skipped (--skip-heldout)",
        "",
        "### Pairwise winner (direction accuracy)",
        "",
        "- Aggregated: `analysis/pairwise_winner_results.csv`",
        "- Pair counts: `analysis/pairwise_winner_pair_counts.csv`",
        "",
        "### SC proxy (alternative confidence)",
        "",
        "- `analysis/selection_gain_results.csv` (despite filename: SC incremental validity, not trace selection gain)",
        "",
        "## STEP 5 — Interpretation (conservative)",
        "",
        "### STEP 6 — Direct answers",
        "",
        "1. **Highest-impact analyses run:** LOCO/LOMO (generalization), held-out question outcomes (reduced overlap in y), pairwise winner at multiple tie tolerances, SC-proxy incremental validity (alternative confidence definition).",
        "2. **Feasible now:** A,C,D,E,F fully; B (trace-level selection gain) not; G not; H exploratory (SNR×fold table in `regime_snr_loco_fold.csv`).",
        "3. **Strongest empirical results for the paper:** (a) LOCO/LOMO **positive mean ΔR²_test** when adding FRS; (b) full-panel ΔR² from prior analysis as reference; (c) SC-proxy `sc_frs_k10` can show **comparable or higher** ΔR² than FRS after the same base predictors—interpret cautiously (different construct, often **higher circularity** vs unfiltered outcome; see `sc_proxy_metadata.json`).",
        "4. **Implication for FRS:** Under OOS group splits, FRS often **adds** test-set explained variance beyond pass@1/pass@16/high-conf/SNR; it is **not** redundant in that sense on this panel. Pairwise tie-break accuracy is **mixed** and often **dominated by pass@1** when many pairs qualify.",
        "5. **Supported more strongly:** Generalization-style incremental validity tables; **unsupported** as a universal tie-breaker vs pass@1.",
        "6. **Weak / unsupported:** Per-question selection gain; second-judge agreement; **high** pairwise n when pass@1 tolerance is tight (see pair counts).",
        "7. **Circularity risk:** **Moderate** throughout (judge-based outcome). Held-out **questions** lower overlap with the aggregate mean but same judge. LOCO/LOMO **do not** remove shared judge family.",
        "8. **Next single best analysis:** Judge **multiple traces per question** (or a second judge on a subset) to enable selection-gain and reduce circularity.",
        "",
        "## STEP 7 — Paper-facing recommendations",
        "",
        "- **Best result to add to main paper:** LOCO/LOMO `generalization_results.csv` + one sentence with mean ΔR²_test and fold positivity counts.",
        "- **Best for appendix:** Held-out question correlations (`heldout_predictor_results.csv`); pairwise table with tolerances; SC proxy robustness.",
        "- **Result to avoid over-emphasizing:** Pairwise winner when **n** is tiny (e.g. pass@1 tolerance 1 pp) or when predictors tie.",
        "- **Suggested table:** `generalization_results.csv` (fold-level).",
        "- **Suggested figure:** `figures/high_impact_followup/loco_LODO-benchmark_test_r2.png` and `loco_LOMO-model_test_r2.png`.",
        "- **One-paragraph wording (draft):**",
        f"  > {draft_para}",
        "",
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("TOTAL elapsed: %.2fs", time.perf_counter() - t_all)
    logger.info("Done. Report: %s", report_path)


if __name__ == "__main__":
    main()
