#!/usr/bin/env python3
"""
COLM rebuttal numbers from cached data only (no API / no inference).

Writes:
  results_for_rebuttal.md
  results_for_rebuttal.json

Usage:
  python analysis/run_results_for_rebuttal.py --repo-root .
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
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
DIMS = ["faithfulness", "coherence", "utility", "factuality"]
FIRST_BIN = "0-10"
REVERSAL_MIN_GAP_PP = 2.0
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 42
N_PERM = 5000
N_PERM_DIM_LOBO = 2000
EPS = 1e-9

PATHS_DOC = {
    "confidence_per_trace": (
        "NOT pre-exported as a flat array. Computed from pass@16 JSONL via "
        "topk_ablation.compute_trace_confidence (mean of lowest 10% token probs). "
        "Published FRS uses bin-sampled judged traces instead."
    ),
    "judge_4dim_subscores": (
        "reasoning_confidence_bins_results/judging_checkpoints/judged_*.json → "
        "judged_samples[].judge_scores.{faithfulness,coherence,utility,factuality} (int 1–5)"
    ),
    "frs_pass1_table5": "global_pass1_frs_analysis/merged_pass1_frs_per_benchmark.csv (frs_pct, pass1_pct)",
    "trace0_single": (
        "analysis_outputs/trace0_k1_judging/per_pair_scores.csv (mean_reasoning_score_pct)"
    ),
    "unfiltered": (
        "analysis_outputs/unfiltered_reasoning/per_pair_scores.csv (mean_reasoning_score ×100)"
    ),
}


def setup_logger() -> logging.Logger:
    log = logging.getLogger("rebuttal_results")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(h)
    return log


def rs_from_dims(scores: Dict[str, Any], dims: Sequence[str]) -> float:
    vals = []
    for d in dims:
        v = scores.get(d)
        if v is None:
            return float("nan")
        vals.append(float(v))
    n = len(vals)
    return (sum(vals) - n) / (4.0 * n)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    r, p = spearmanr(x[m], y[m])
    return float(r), float(p), n


def bootstrap_ci(values: np.ndarray, stat_fn, n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    obs = float(stat_fn(values))
    boots = [float(stat_fn(rng.choice(values, size=len(values), replace=True))) for _ in range(n)]
    return obs, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def permutation_pvalue(x: np.ndarray, y: np.ndarray, n_perm: int = N_PERM, seed: int = BOOTSTRAP_SEED) -> float:
    rng = np.random.default_rng(seed)
    r_obs, _, n = safe_spearman(x, y)
    if not np.isfinite(r_obs) or n < 3:
        return float("nan")
    count = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        rp, _, _ = safe_spearman(x, yp)
        if np.isfinite(rp) and rp >= r_obs:
            count += 1
    return count / n_perm


def load_panel(repo: Path) -> pd.DataFrame:
    frs = pd.read_csv(repo / "global_pass1_frs_analysis/merged_pass1_frs_per_benchmark.csv")
    panel = frs[["model", "benchmark", "frs_pct", "pass1_pct"]].copy()
    t0 = pd.read_csv(repo / "analysis_outputs/trace0_k1_judging/per_pair_scores.csv")
    t0 = t0[["model", "benchmark", "mean_reasoning_score_pct"]].rename(
        columns={"mean_reasoning_score_pct": "trace0_pct"}
    )
    panel = panel.merge(t0, on=["model", "benchmark"], how="left")
    unf = pd.read_csv(repo / "analysis_outputs/unfiltered_reasoning/per_pair_scores.csv")
    unf["benchmark"] = unf["dataset"].map(DATASET_TO_BENCH).fillna(unf["dataset"])
    unf["unfiltered_pct"] = unf["mean_reasoning_score"].astype(float) * 100.0
    panel = panel.merge(unf[["model", "benchmark", "unfiltered_pct"]], on=["model", "benchmark"], how="left")
    return panel


def build_model_pairs(panel: pd.DataFrame, metric_a: str, metric_b: str, scope: str) -> pd.DataFrame:
    """Pairwise comparisons within scope ('macro' or benchmark name)."""
    rows: List[Dict[str, Any]] = []
    if scope == "macro":
        g = panel.groupby("model", as_index=True)[[metric_a, metric_b]].mean()
        bench_label = "macro"
    else:
        g = panel[panel["benchmark"] == scope].set_index("model")[[metric_a, metric_b]]
        bench_label = scope
    for ma, mb in combinations(g.index, 2):
        va, vb = float(g.loc[ma, metric_a]), float(g.loc[mb, metric_a])
        ua, ub = float(g.loc[ma, metric_b]), float(g.loc[mb, metric_b])
        da, db = va - vb, ua - ub
        tie_a = abs(da) < EPS
        tie_b = abs(db) < EPS
        flip = (not tie_a) and (not tie_b) and (da * db < 0)
        rows.append(
            {
                "scope": bench_label,
                "model_a": ma,
                "model_b": mb,
                f"{metric_a}_diff": da,
                f"{metric_b}_diff": db,
                f"{metric_a}_gap_pp": abs(da),
                f"{metric_b}_gap_pp": abs(db),
                "tie_on_a": tie_a,
                "tie_on_b": tie_b,
                "any_tie": tie_a or tie_b,
                "flip": flip,
                "qualifying_2pp": abs(da) >= REVERSAL_MIN_GAP_PP and abs(db) >= REVERSAL_MIN_GAP_PP,
            }
        )
    return pd.DataFrame(rows)


def summarize_pair_flips(pairs: pd.DataFrame) -> Dict[str, Any]:
    n = len(pairs)
    n_tie = int(pairs["any_tie"].sum())
    n_non_tied = n - n_tie
    n_flip = int(pairs["flip"].sum())
    n_qual = int(pairs["qualifying_2pp"].sum())
    n_rev_2pp = int(pairs[pairs["qualifying_2pp"] & pairs["flip"]].shape[0])
    return {
        "n_pairs": n,
        "n_ties_any_metric": n_tie,
        "n_non_tied": n_non_tied,
        "n_flips_strict": n_flip,
        "n_qualifying_both_ge_2pp": n_qual,
        "n_reversals_among_qualifying_2pp": n_rev_2pp,
    }


def macro_ranking(panel: pd.DataFrame, col: str) -> pd.Series:
    return panel.groupby("model")[col].mean().sort_values(ascending=False)


def compute_reversals(panel: pd.DataFrame) -> Dict[str, Any]:
    macro_pairs = build_model_pairs(panel, "frs_pct", "trace0_pct", "macro")
    per_bench = []
    pooled_qual = []
    for b in BENCHMARKS:
        bp = build_model_pairs(panel, "frs_pct", "trace0_pct", b)
        per_bench.append({"benchmark": b, **summarize_pair_flips(bp)})
        pooled_qual.append(bp[bp["qualifying_2pp"]])
    pooled_df = pd.concat(pooled_qual, ignore_index=True) if pooled_qual else pd.DataFrame()
    out = {
        "aggregate_model_level": summarize_pair_flips(macro_pairs),
        "per_benchmark": per_bench,
        "pooled_qualifying_2pp_across_benchmarks": {
            "n_qualifying_pairs": len(pooled_df),
            "n_reversals": int(pooled_df["flip"].sum()) if len(pooled_df) else 0,
            "note": (
                "Denominator 181 = total per-benchmark model pairs (9 choose 2 = 36 per bench) "
                f"where BOTH |ΔFRS| and |Δtrace-0| ≥ {REVERSAL_MIN_GAP_PP} pp, pooled over 6 benchmarks. "
                "Ties (either metric exactly equal) are excluded from flip counts but included in n_pairs=216 per bench."
            ),
        },
        "tie_rule": (
            "Strict ordering flip: sign(FRS_a−FRS_b) ≠ sign(trace0_a−trace0_b) with neither diff ≈ 0 (|diff|<1e-9). "
            f"Paper-style reversal: same sign opposition among pairs with both gaps ≥ {REVERSAL_MIN_GAP_PP} pp."
        ),
    }
    macro_frs = macro_ranking(panel, "frs_pct")
    macro_t0 = macro_ranking(panel, "trace0_pct")
    r, p, n = safe_spearman(
        macro_frs.loc[EXPECTED_MODELS].values.astype(float),
        macro_t0.loc[EXPECTED_MODELS].values.astype(float),
    )
    out["macro_rank_spearman_frs_vs_trace0"] = {"rho": r, "p": p, "n": n}
    pair_vals = panel.dropna(subset=["frs_pct", "trace0_pct"])
    x = pair_vals["trace0_pct"].values.astype(float)
    y = pair_vals["frs_pct"].values.astype(float)
    r54, p54, _ = safe_spearman(x, y)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boots = []
    n54 = len(x)
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, n54, size=n54)
        boots.append(safe_spearman(x[idx], y[idx])[0])
    boots = np.array([b for b in boots if np.isfinite(b)])
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    out["pair_level_spearman_trace0_vs_frs_54"] = {
        "rho": r54,
        "p": p54,
        "bootstrap_95_ci": [lo, hi],
        "note": "54 pair-level cells (model×benchmark), not 9 macro-averaged models.",
    }
    return out


def build_within_bench_pairs(panel: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for bench, g in panel.groupby("benchmark"):
        g = g.set_index("model")
        for ma, mb in combinations(g.index, 2):
            row = {
                "benchmark": bench,
                "model_a": ma,
                "model_b": mb,
                "pass1_gap_pp": abs(float(g.loc[ma, "pass1_pct"]) - float(g.loc[mb, "pass1_pct"])),
                "frs_gap_pp": abs(float(g.loc[ma, "frs_pct"]) - float(g.loc[mb, "frs_pct"])),
            }
            for col, key in [("trace0_pct", "trace0"), ("unfiltered_pct", "unfiltered")]:
                if col in g.columns and np.isfinite(g.loc[ma, col]) and np.isfinite(g.loc[mb, col]):
                    row[f"{key}_gap_pp"] = abs(float(g.loc[ma, col]) - float(g.loc[mb, col]))
                else:
                    row[f"{key}_gap_pp"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def amplification_for_metric(pairs: pd.DataFrame, metric_col: str, thresh: float) -> Dict[str, Any]:
    gap = f"{metric_col}_gap_pp"
    if gap not in pairs.columns:
        metric_map = {"frs": "frs", "trace0": "trace0", "unfiltered": "unfiltered"}
        gap = f"{metric_map.get(metric_col, metric_col)}_gap_pp"
    sub = pairs[pairs["pass1_gap_pp"] <= thresh].dropna(subset=[gap])
    if len(sub) == 0:
        return {"threshold_pp": thresh, "metric": metric_col, "n_pairs": 0}
    mp = float(sub["pass1_gap_pp"].mean())
    mm = float(sub[gap].mean())
    ratio = mm / mp if mp > 0 else float("nan")
    win = (sub[gap] > sub["pass1_gap_pp"]).astype(float)
    win_frac = float(win.mean())

    def ratio_stat(idx):
        s = sub.iloc[idx]
        mp2 = float(s["pass1_gap_pp"].mean())
        mg = float(s[gap].mean())
        return mg / mp2 if mp2 > 0 else float("nan")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boots = []
    n = len(sub)
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        boots.append(ratio_stat(idx))
    boots = np.array([b for b in boots if np.isfinite(b)])
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if len(boots) else [np.nan, np.nan]
    return {
        "threshold_pp": thresh,
        "metric": metric_col,
        "n_close_accuracy_pairs": len(sub),
        "mean_pass1_gap_pp": mp,
        "mean_metric_gap_pp": mm,
        "amplification_ratio": ratio,
        "close_accuracy_win_fraction": win_frac,
        "bootstrap_95_ci_ratio": ci,
    }


def lobo_transfer(
    panel: pd.DataFrame, train_col: str, test_col: str, n_perm: int = N_PERM
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for held in BENCHMARKS:
        train = panel[panel["benchmark"] != held]
        test = panel[panel["benchmark"] == held]
        macro = train.groupby("model")[train_col].mean()
        te = test.set_index("model")[test_col]
        common = sorted(set(macro.index) & set(te.index))
        x = macro.loc[common].values.astype(float)
        y = te.loc[common].values.astype(float)
        r, p, n = safe_spearman(x, y)
        p_perm = permutation_pvalue(x, y, n_perm=n_perm)
        rows.append(
            {
                "held_out_benchmark": held,
                "train_metric": train_col,
                "test_target": test_col,
                "spearman_r": r,
                "p_parametric": p,
                "p_permutation": p_perm,
                "n_models": n,
            }
        )
    return pd.DataFrame(rows)


def lobo_summary(lobo_df: pd.DataFrame) -> Dict[str, Any]:
    r = lobo_df["spearman_r"]
    return {
        "mean_rho": float(r.mean()),
        "std_rho_across_folds": float(r.std()),
        "per_fold": lobo_df.to_dict(orient="records"),
    }


def discover_checkpoints(judge_dir: Path) -> Dict[Tuple[str, str], Path]:
    out: Dict[Tuple[str, str], Path] = {}
    for fp in judge_dir.glob("judged_*.json"):
        m = re.match(r"^judged_(.+)__(.+)\.json$", fp.name)
        if m:
            out[(m.group(1), m.group(2))] = fp
    return out


def load_dim_ablation_traces(repo: Path, log: logging.Logger) -> pd.DataFrame:
    judge_dir = repo / "reasoning_confidence_bins_results/judging_checkpoints"
    rows: List[Dict[str, Any]] = []
    for (model, dataset), jpath in sorted(discover_checkpoints(judge_dir).items()):
        benchmark = DATASET_TO_BENCH.get(dataset, dataset)
        with open(jpath, encoding="utf-8") as f:
            data = json.load(f)
        for s in data.get("judged_samples", []):
            if not s.get("judge_ok", True):
                continue
            if str(s.get("bin_label")) != FIRST_BIN:
                continue
            js = s.get("judge_scores") or {}
            if not js:
                continue
            base = {"model": model, "benchmark": benchmark}
            subsets: List[Tuple[str, ...]] = [tuple(DIMS)]
            for d in DIMS:
                subsets.append((d,))
            for drop in DIMS:
                subsets.append(tuple(x for x in DIMS if x != drop))
            seen: set[Tuple[str, ...]] = set()
            for dims in subsets:
                if dims in seen:
                    continue
                seen.add(dims)
                rs01 = rs_from_dims(js, dims)
                if not np.isfinite(rs01):
                    continue
                if len(dims) == 4:
                    label = "full"
                elif len(dims) == 1:
                    label = f"only-{dims[0]}"
                else:
                    dropped = set(DIMS) - set(dims)
                    label = f"drop-{next(iter(dropped))}"
                rows.append({**base, "subset_label": label, "subset_dims": dims, "rs_pct": rs01 * 100.0})
    df = pd.DataFrame(rows)
    log.info("Dim-ablation trace rows (bin %s): %d", FIRST_BIN, len(df))
    return df


def pair_frs_from_traces(trace_df: pd.DataFrame) -> pd.DataFrame:
    return (
        trace_df.groupby(["model", "benchmark", "subset_label"], as_index=False)
        .agg(frs_pct=("rs_pct", "mean"), n_traces=("rs_pct", "count"))
    )


def dim_amplification_and_lobo(
    pair_frs: pd.DataFrame, pass1_panel: pd.DataFrame, labels: Sequence[str]
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for lab in labels:
        sub = pair_frs[pair_frs["subset_label"] == lab].merge(
            pass1_panel[["model", "benchmark", "pass1_pct"]], on=["model", "benchmark"]
        )
        panel = sub.rename(columns={"frs_pct": "metric_pct"})
        rows = []
        for bench, g in panel.groupby("benchmark"):
            g = g.set_index("model")
            for ma, mb in combinations(g.index, 2):
                rows.append(
                    {
                        "benchmark": bench,
                        "pass1_gap_pp": abs(float(g.loc[ma, "pass1_pct"]) - float(g.loc[mb, "pass1_pct"])),
                        "metric_gap_pp": abs(float(g.loc[ma, "metric_pct"]) - float(g.loc[mb, "metric_pct"])),
                    }
                )
        pdf = pd.DataFrame(rows)
        amp = {}
        for thr in [3.0, 5.0]:
            s = pdf[pdf["pass1_gap_pp"] <= thr]
            mp, mm = float(s["pass1_gap_pp"].mean()), float(s["metric_gap_pp"].mean())
            amp[str(thr)] = {
                "n_pairs": len(s),
                "amplification_ratio": mm / mp if mp > 0 else float("nan"),
                "win_fraction_metric_gt_acc": float((s["metric_gap_pp"] > s["pass1_gap_pp"]).mean())
                if len(s)
                else float("nan"),
            }
        lobo_panel = panel.rename(columns={"metric_pct": "frs_pct"})
        lobo = lobo_transfer(lobo_panel, "frs_pct", "frs_pct", n_perm=N_PERM_DIM_LOBO)
        out[lab] = {"amplification": amp, "lobo_same_metric": lobo_summary(lobo)}
    return out


def single_dim_rank_correlation(pair_frs: pd.DataFrame) -> Dict[str, Any]:
    singles = [f"only-{d}" for d in DIMS]
    macro = {}
    for lab in singles:
        sub = pair_frs[pair_frs["subset_label"] == lab]
        macro[lab] = sub.groupby("model")["frs_pct"].mean()
    labels = singles
    mat = np.full((4, 4), np.nan)
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            common = sorted(set(macro[li].index) & set(macro[lj].index))
            r, _, _ = safe_spearman(
                macro[li].loc[common].values.astype(float),
                macro[lj].loc[common].values.astype(float),
            )
            mat[i, j] = r
    return {
        "labels": labels,
        "spearman_matrix": mat.tolist(),
    }


def md_table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def write_markdown(results: Dict[str, Any], path: Path) -> None:
    lines: List[str] = ["# COLM rebuttal — cached-data results", ""]

    lines += ["## STEP 0 — Discovery (schemas & paths)", ""]
    for k, v in results["step0"].items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append(
        f"**Models (n={len(EXPECTED_MODELS)})**: {', '.join(EXPECTED_MODELS)}"
    )
    lines.append(f"**Benchmarks (n={len(BENCHMARKS)})**: {', '.join(BENCHMARKS)}")
    lines.append(f"**Pairs**: {len(EXPECTED_MODELS)} × {len(BENCHMARKS)} = 54")
    lines.append("")

    r1 = results["section1_reversals"]
    lines += ["## (1) Reversal denominator (FRS vs single-trace / trace-0)", ""]
    agg = r1["aggregate_model_level"]
    lines.append("### Aggregate model-level (macro mean over 6 benchmarks, C(9,2)=36)")
    lines.append(
        md_table(
            ["Quantity", "Count"],
            [
                ["Total pairs", agg["n_pairs"]],
                ["Ties (either metric |Δ|<ε)", agg["n_ties_any_metric"]],
                ["Non-tied pairs", agg["n_non_tied"]],
                ["Strict flips (opposite sign, neither tied)", agg["n_flips_strict"]],
                [f"Qualifying (both ≥{REVERSAL_MIN_GAP_PP} pp)", agg["n_qualifying_both_ge_2pp"]],
                ["Reversals among qualifying", agg["n_reversals_among_qualifying_2pp"]],
            ],
        )
    )
    lines.append("")
    lines.append("### Per-benchmark (C(9,2)=36 each; 216 total)")
    pb_rows = []
    tot_flip = tot_tie = tot_pairs = 0
    for row in r1["per_benchmark"]:
        pb_rows.append(
            [
                row["benchmark"],
                row["n_pairs"],
                row["n_ties_any_metric"],
                row["n_non_tied"],
                row["n_flips_strict"],
                row["n_qualifying_both_ge_2pp"],
                row["n_reversals_among_qualifying_2pp"],
            ]
        )
        tot_pairs += row["n_pairs"]
        tot_tie += row["n_ties_any_metric"]
        tot_flip += row["n_flips_strict"]
    lines.append(
        md_table(
            ["Benchmark", "Pairs", "Ties", "Non-tied", "Flips", f"Qual≥{REVERSAL_MIN_GAP_PP}pp", "Rev@qual"],
            pb_rows,
        )
    )
    lines.append(f"\n**Totals across 6 benchmarks**: pairs={tot_pairs}, ties={tot_tie}, flips={tot_flip}")
    pq = r1["pooled_qualifying_2pp_across_benchmarks"]
    lines.append(
        f"\n**Denominator = {pq['n_qualifying_pairs']}** (pooled qualifying pairs); "
        f"**reversals = {pq['n_reversals']}**. {pq['note']}"
    )
    lines.append(f"\n**Tie / flip rule**: {r1['tie_rule']}")
    sp = r1["pair_level_spearman_trace0_vs_frs_54"]
    lines.append(
        f"\n**Spearman ρ (54 pair-level trace-0 vs FRS)**: {sp['rho']:.4f} "
        f"(bootstrap 95% CI [{sp['bootstrap_95_ci'][0]:.3f}, {sp['bootstrap_95_ci'][1]:.3f}])"
    )
    lines.append("")

    lines += ["## (2) LOBO transfer (mean ρ, std across 6 folds, per-fold permutation p)", ""]
    for key, title in [
        ("lobo_frs", "FRS → FRS"),
        ("lobo_trace0", "trace-0 → trace-0"),
        ("lobo_single_trace_to_frs", "trace-0 macro → held-out FRS"),
        ("lobo_unfiltered", "unfiltered → unfiltered"),
        ("lobo_unfiltered_to_frs", "unfiltered macro → held-out FRS"),
        ("lobo_pass1", "pass@1 → pass@1"),
        ("lobo_pass1_to_frs", "pass@1 macro → held-out FRS"),
    ]:
        block = results["section2_lobo"][key]
        lines.append(f"### {title}")
        lines.append(
            f"Mean ρ = **{block['mean_rho']:.4f}**, std = {block['std_rho_across_folds']:.4f}"
        )
        fold_rows = [
            [f["held_out_benchmark"], f"{f['spearman_r']:.4f}", f"{f['p_permutation']:.4f}"]
            for f in block["per_fold"]
        ]
        lines.append(md_table(["Held-out", "ρ", "p_perm"], fold_rows))
        lines.append("")

    lines += ["## (3) Amplification at 3 pp and 5 pp", ""]
    for thr in ["3.0", "5.0"]:
        lines.append(f"### |Δpass@1| ≤ {thr} pp")
        rows = []
        for m in ["frs", "trace0", "unfiltered"]:
            a = results["section3_amplification"][thr][m]
            ci = a.get("bootstrap_95_ci_ratio", [np.nan, np.nan])
            rows.append(
                [
                    m,
                    a["n_close_accuracy_pairs"],
                    f"{a['amplification_ratio']:.3f}",
                    f"{a['close_accuracy_win_fraction']:.3f}",
                    f"[{ci[0]:.3f}, {ci[1]:.3f}]",
                ]
            )
        lines.append(md_table(["Metric", "n_pairs", "ratio", "win_frac", "bootstrap 95% CI"], rows))
        lines.append("")

    lines += ["## (4) Leave-one-dimension-out FRS (bin 0–10 judged traces)", ""]
    dim_labels = ["full", "drop-faithfulness", "drop-coherence", "drop-utility", "drop-factuality"]
    for thr in ["3.0", "5.0"]:
        lines.append(f"### |Δpass@1| ≤ {thr} pp")
        rows = []
        for lab in dim_labels:
            a = results["section4_dim_ablation"][lab]["amplification"][thr]
            rows.append([lab, a["n_pairs"], f"{a['amplification_ratio']:.3f}", f"{a['win_fraction_metric_gt_acc']:.3f}"])
        lines.append(md_table(["Subset", "n_pairs", "amp ratio", "win_frac"], rows))
        lines.append("")

    lines += ["## (5) Inter-dimension redundancy (macro model rankings, 4×4 Spearman)", ""]
    mat = results["section5_single_dim_matrix"]
    hdr = [""] + [x.replace("only-", "") for x in mat["labels"]]
    mrows = []
    for i, li in enumerate(mat["labels"]):
        mrows.append([li.replace("only-", "")] + [f"{v:.3f}" for v in mat["spearman_matrix"][i]])
    lines.append(md_table(hdr, mrows))
    lines.append("")

    lines += ["## (6) LOBO transfer ρ by drop-one-dimension subset (optional)", ""]
    rows = []
    for lab, block in results["section6_lodo_dim"].items():
        rows.append([lab, f"{block['mean_rho']:.4f}", f"{block['std_rho_across_folds']:.4f}"])
    lines.append(md_table(["Subset", "mean ρ", "std"], rows))
    lines.append("")

    lines += ["## Provenance (files read)", ""]
    for p in results["provenance"]:
        lines.append(f"- `{p}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    log = setup_logger()

    panel = load_panel(repo)
    log.info("Panel loaded: %d rows", len(panel))

    results: Dict[str, Any] = {
        "step0": PATHS_DOC,
        "models": EXPECTED_MODELS,
        "benchmarks": BENCHMARKS,
        "provenance": [
            str(repo / "global_pass1_frs_analysis/merged_pass1_frs_per_benchmark.csv"),
            str(repo / "analysis_outputs/trace0_k1_judging/per_pair_scores.csv"),
            str(repo / "analysis_outputs/unfiltered_reasoning/per_pair_scores.csv"),
            str(repo / "reasoning_confidence_bins_results/judging_checkpoints/judged_*.json"),
        ],
    }

    results["section1_reversals"] = compute_reversals(panel)

    lobo_block: Dict[str, Any] = {}
    lobo_block["lobo_frs"] = lobo_summary(lobo_transfer(panel, "frs_pct", "frs_pct"))
    lobo_block["lobo_trace0"] = lobo_summary(lobo_transfer(panel, "trace0_pct", "trace0_pct"))
    lobo_block["lobo_single_trace_to_frs"] = lobo_summary(lobo_transfer(panel, "trace0_pct", "frs_pct"))
    lobo_block["lobo_unfiltered"] = lobo_summary(lobo_transfer(panel, "unfiltered_pct", "unfiltered_pct"))
    lobo_block["lobo_unfiltered_to_frs"] = lobo_summary(lobo_transfer(panel, "unfiltered_pct", "frs_pct"))
    lobo_block["lobo_pass1"] = lobo_summary(lobo_transfer(panel, "pass1_pct", "pass1_pct"))
    lobo_block["lobo_pass1_to_frs"] = lobo_summary(lobo_transfer(panel, "pass1_pct", "frs_pct"))
    results["section2_lobo"] = lobo_block

    pairs = build_within_bench_pairs(panel)
    amp_sec: Dict[str, Dict[str, Any]] = {}
    for thr in [3.0, 5.0]:
        amp_sec[str(thr)] = {
            "frs": amplification_for_metric(pairs, "frs", thr),
            "trace0": amplification_for_metric(pairs, "trace0", thr),
            "unfiltered": amplification_for_metric(pairs, "unfiltered", thr),
        }
    results["section3_amplification"] = amp_sec

    trace_df = load_dim_ablation_traces(repo, log)
    pair_frs = pair_frs_from_traces(trace_df)
    dim_labels = ["full", "drop-faithfulness", "drop-coherence", "drop-utility", "drop-factuality"]
    results["section4_dim_ablation"] = dim_amplification_and_lobo(pair_frs, panel, dim_labels)
    results["section5_single_dim_matrix"] = single_dim_rank_correlation(pair_frs)
    lodo_labels = dim_labels + [f"only-{d}" for d in DIMS]
    section6: Dict[str, Any] = {}
    for lab in lodo_labels:
        sub = pair_frs[pair_frs["subset_label"] == lab][["model", "benchmark", "frs_pct"]]
        lobo_panel = sub.rename(columns={"frs_pct": "frs_pct"})
        section6[lab] = lobo_summary(
            lobo_transfer(lobo_panel, "frs_pct", "frs_pct", n_perm=N_PERM_DIM_LOBO)
        )
    results["section6_lodo_dim"] = section6

    out_md = repo / "results_for_rebuttal.md"
    out_json = repo / "results_for_rebuttal.json"

    def _json_default(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o) if isinstance(o, np.floating) else int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    out_json.write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    write_markdown(results, out_md)
    log.info("Wrote %s and %s", out_md, out_json)


if __name__ == "__main__":
    main()
