#!/usr/bin/env python3
"""
COLM rebuttal — selection-gain follow-ups (cache only).

Writes:
  results_selectiongain_decisive.md
  results_selectiongain_decisive.json

Usage:
  python analysis/run_results_selectiongain_decisive.py --repo-root .
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
from scipy import stats

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
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 42

STEP0_PATHS = {
    "pair_level_54": "analysis/selection_gain_pair_level.csv",
    "question_level": "analysis/selection_gain_question_level.csv",
    "judge_outputs": "analysis/selection_gain_judge_outputs.csv",
    "predictor_panel": "analysis/selection_gain_predictor_panel_merged.csv",
    "prior_table14": "analysis/selection_gain_predictor_results.csv",
    "frs_predictor_panel": "analysis/frs_predictor_panel.csv",
    "trace0": "outputs/analysis_outputs/trace0_k1_judging/per_pair_scores.csv",
    "frs_pass1": "outputs/global_pass1_frs_analysis/merged_pass1_frs_per_benchmark.csv",
    "unfiltered": "outputs/analysis_outputs/unfiltered_reasoning/per_pair_scores.csv",
    "frs_judge_bins": "outputs/reasoning_confidence_bins_results/judging_checkpoints/judged_*.json",
    "trace0_checkpoints": "outputs/analysis_outputs/trace0_k1_judging/judging_checkpoints/trace0_judged_*.json",
}


def setup_logger() -> logging.Logger:
    log = logging.getLogger("sg_decisive")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(h)
    return log


def corr_row(x: np.ndarray, y: np.ndarray, predictor: str) -> Dict[str, Any]:
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 5:
        return {"predictor": predictor, "n": n}
    pr = stats.pearsonr(x[ok], y[ok])
    sp = stats.spearmanr(x[ok], y[ok])
    return {
        "predictor": predictor,
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_rho": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "n": n,
    }


def load_selection_gain_54(repo: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    pair = pd.read_csv(repo / "analysis/selection_gain_pair_level.csv")
    meta = {
        "source": str(repo / "analysis/selection_gain_pair_level.csv"),
        "variable": "mean_selection_gain",
        "definition": "mean over 50 questions of (RS(top_conf trace) - RS(random trace)), 0-1 scale",
        "n_pairs": len(pair),
        "n_questions_per_pair": int(pair["n_questions"].iloc[0]) if len(pair) else 0,
    }
    return pair, meta


def build_predictor_panel(repo: Path, log: logging.Logger) -> pd.DataFrame:
    pair = pd.read_csv(repo / "analysis/selection_gain_pair_level.csv")
    panel = pd.read_csv(repo / "analysis/selection_gain_predictor_panel_merged.csv")

    t0 = pd.read_csv(repo / "outputs/analysis_outputs/trace0_k1_judging/per_pair_scores.csv")
    t0 = t0[["model", "benchmark", "mean_reasoning_score_pct"]].rename(
        columns={"mean_reasoning_score_pct": "trace0_pct"}
    )
    t0["trace0_reasoning_0_1"] = t0["trace0_pct"] / 100.0

    if "mean_top_conf_reasoning" not in panel.columns:
        m = panel.merge(pair[["model", "benchmark", "mean_top_conf_reasoning"]], on=["model", "benchmark"], how="left")
    else:
        m = panel.copy()
    m = m.merge(t0[["model", "benchmark", "trace0_pct", "trace0_reasoning_0_1"]], on=["model", "benchmark"], how="left")
    log.info("Predictor panel: %d rows, cols=%s", len(m), list(m.columns))
    return m


def experiment1_table14(panel: pd.DataFrame) -> pd.DataFrame:
    y = panel["mean_selection_gain"].values.astype(float)
    predictors = [
        ("frs_pct", "FRS (%)"),
        ("high_conf_accuracy_pct", "high-conf accuracy (%)"),
        ("snr", "SNR"),
        ("unfiltered_reasoning_mean", "unfiltered reasoning (0-1)"),
        ("pass16_pct", "pass@16 (%)"),
        ("pass1_pct", "pass@1 (%)"),
        ("trace0_reasoning_0_1", "single-trace (trace-0, 0-1)"),
        ("mean_top_conf_reasoning", "top-1 conf. single trace (0-1 mean)"),
    ]
    rows = []
    for col, label in predictors:
        if col not in panel.columns:
            continue
        row = corr_row(panel[col].values.astype(float), y, label)
        row["predictor_col"] = col
        rows.append(row)
    return pd.DataFrame(rows)


def dependent_corr_bootstrap(
    x_a: np.ndarray,
    x_b: np.ndarray,
    y: np.ndarray,
    label_a: str,
    label_b: str,
    n_boot: int = BOOTSTRAP_N,
) -> Dict[str, Any]:
    ok = np.isfinite(x_a) & np.isfinite(x_b) & np.isfinite(y)
    xa, xb, yy = x_a[ok], x_b[ok], y[ok]
    n = len(yy)
    r_a, _ = stats.pearsonr(xa, yy)
    r_b, _ = stats.pearsonr(xb, yy)
    r_ab, _ = stats.pearsonr(xa, xb)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = stats.pearsonr(xa[idx], yy[idx]).statistic - stats.pearsonr(xb[idx], yy[idx]).statistic
    return {
        "comparison": f"{label_a} vs {label_b}",
        "label_a": label_a,
        "label_b": label_b,
        "n_pairs": n,
        "pearson_r_a": float(r_a),
        "pearson_r_b": float(r_b),
        "pearson_r_between_predictors": float(r_ab),
        "diff_r_point": float(r_a - r_b),
        "diff_r_bootstrap_mean": float(np.mean(diffs)),
        "diff_r_ci_95_lo": float(np.percentile(diffs, 2.5)),
        "diff_r_ci_95_hi": float(np.percentile(diffs, 97.5)),
        "bootstrap_p_diff_le_0": float(np.mean(diffs <= 0)),
        "bootstrap_p_diff_gt_0": float(np.mean(diffs > 0)),
    }


def build_close_accuracy_pairs(panel: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    cols = ["frs_pct", "trace0_pct", "unfiltered_reasoning_mean", "pass1_pct"]
    sub = panel.dropna(subset=["pass1_pct", "frs_pct"])
    sub = sub.copy()
    sub["unfiltered_pct"] = sub["unfiltered_reasoning_mean"] * 100.0
    for bench, g in sub.groupby("benchmark"):
        g = g.set_index("model")
        for ma, mb in combinations(g.index, 2):
            rows.append(
                {
                    "benchmark": bench,
                    "model_a": ma,
                    "model_b": mb,
                    "pass1_gap_pp": abs(float(g.loc[ma, "pass1_pct"]) - float(g.loc[mb, "pass1_pct"])),
                    "frs_gap_pp": abs(float(g.loc[ma, "frs_pct"]) - float(g.loc[mb, "frs_pct"])),
                    "trace0_gap_pp": abs(float(g.loc[ma, "trace0_pct"]) - float(g.loc[mb, "trace0_pct"])),
                    "unfiltered_gap_pp": abs(
                        float(g.loc[ma, "unfiltered_pct"]) - float(g.loc[mb, "unfiltered_pct"])
                    ),
                }
            )
    return pd.DataFrame(rows)


def paired_amp_bootstrap(
    pairs: pd.DataFrame,
    metric_a: str,
    metric_b: str,
    thresh: float,
    n_boot: int = BOOTSTRAP_N,
) -> Dict[str, Any]:
    ga, gb = f"{metric_a}_gap_pp", f"{metric_b}_gap_pp"
    sub = pairs[pairs["pass1_gap_pp"] <= thresh].dropna(subset=[ga, gb, "pass1_gap_pp"])
    if len(sub) < 2:
        return {"threshold_pp": thresh, "metric_a": metric_a, "metric_b": metric_b, "n_pairs": len(sub)}
    acc = sub["pass1_gap_pp"].values.astype(float)
    fa = sub[ga].values.astype(float)
    fb = sub[gb].values.astype(float)
    acc_m = float(np.mean(acc))
    point_a = float(np.mean(fa)) / acc_m
    point_b = float(np.mean(fb)) / acc_m
    point_diff = (float(np.mean(fa)) - float(np.mean(fb))) / acc_m
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    diffs = np.empty(n_boot)
    n = len(sub)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        am = float(np.mean(acc[idx]))
        diffs[i] = (float(np.mean(fa[idx])) - float(np.mean(fb[idx]))) / am if am > 0 else np.nan
    diffs = diffs[np.isfinite(diffs)]
    return {
        "threshold_pp": thresh,
        "comparison": f"FRS vs {metric_b}",
        "n_close_pairs": n,
        "amplification_frs": point_a,
        "amplification_comparator": point_b,
        "amplification_diff_point": point_diff,
        "amplification_diff_ci_95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "bootstrap_p_frs_amp_le_comparator": float(np.mean(diffs <= 0)),
        "bootstrap_p_frs_amp_gt_comparator": float(np.mean(diffs > 0)),
    }


def load_trace0_scores(repo: Path) -> Dict[Tuple[str, str], np.ndarray]:
    out: Dict[Tuple[str, str], List[float]] = {}
    ckpt_dir = repo / "outputs/analysis_outputs/trace0_k1_judging/judging_checkpoints"
    for fp in ckpt_dir.glob("trace0_judged_*.json"):
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        model = data["model"]
        bench = data.get("benchmark") or DATASET_TO_BENCH.get(data.get("dataset", ""), data.get("dataset"))
        scores = []
        js = data["judged_samples"]
        items = js.values() if isinstance(js, dict) else js
        for s in items:
            if s.get("judge_ok", True) and s.get("reasoning_score") is not None:
                scores.append(float(s["reasoning_score"]))
        out[(model, bench)] = np.array(scores, dtype=float)
    return {k: v for k, v in out.items()}


def load_frs_bin_scores(repo: Path) -> Dict[Tuple[str, str], np.ndarray]:
    out: Dict[Tuple[str, str], List[float]] = {}
    judge_dir = repo / "outputs/reasoning_confidence_bins_results/judging_checkpoints"
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
            if not s.get("judge_ok", True):
                continue
            if str(s.get("bin_label")) != FIRST_BIN:
                continue
            rs = s.get("reasoning_score")
            if rs is None and s.get("judge_scores"):
                js = s["judge_scores"]
                vals = [float(js[d]) for d in ["faithfulness", "coherence", "utility", "factuality"] if d in js]
                if len(vals) == 4:
                    rs = (sum(vals) - 4) / 16.0
            if rs is not None:
                scores.append(float(rs))
        out[(model, bench)] = np.array(scores, dtype=float)
    return {k: v for k, v in out.items()}


def bootstrap_mean_sd(scores: np.ndarray, n_boot: int = BOOTSTRAP_N) -> float:
    """SD of bootstrap distribution of the mean (estimator uncertainty)."""
    if len(scores) < 2:
        return float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(scores)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(scores[idx]))
    return float(np.std(means, ddof=1))


def experiment4_reliability(repo: Path, log: logging.Logger) -> Dict[str, Any]:
    t0_map = load_trace0_scores(repo)
    frs_map = load_frs_bin_scores(repo)
    rows = []
    for (model, bench), t0_sc in t0_map.items():
        frs_sc = frs_map.get((model, bench))
        rows.append(
            {
                "model": model,
                "benchmark": bench,
                "n_trace0": len(t0_sc),
                "n_frs_bin": len(frs_sc) if frs_sc is not None else 0,
                "bootstrap_sd_trace0_mean": bootstrap_mean_sd(t0_sc),
                "bootstrap_sd_frs_mean": bootstrap_mean_sd(frs_sc) if frs_sc is not None and len(frs_sc) >= 2 else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    log.info("Reliability rows: %d (trace0 pairs=%d, frs pairs=%d)", len(df), len(t0_map), len(frs_map))
    t0_sd = df["bootstrap_sd_trace0_mean"].dropna()
    frs_sd = df["bootstrap_sd_frs_mean"].dropna()
    return {
        "per_pair": df.to_dict(orient="records"),
        "median_bootstrap_sd_trace0": float(t0_sd.median()),
        "median_bootstrap_sd_frs": float(frs_sd.median()),
        "mean_bootstrap_sd_trace0": float(t0_sd.mean()),
        "mean_bootstrap_sd_frs": float(frs_sd.mean()),
        "ratio_median_sd_trace0_over_frs": float(t0_sd.median() / frs_sd.median()) if frs_sd.median() > 0 else np.nan,
        "note": "Bootstrap SD = std of resampled-with-replacement means (10k draws); measures estimator noise.",
    }


def md_table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def write_markdown(results: Dict[str, Any], path: Path) -> None:
    lines = ["# Selection-gain decisive analyses (COLM rebuttal)", ""]

    lines += ["## STEP 0 — Discovery", ""]
    s0 = results["step0"]
    lines.append(f"**Status:** {s0['status']}")
    lines.append(f"**54-cell selection gain:** `{s0['pair_level_path']}` → `{s0['variable']}`")
    lines.append(f"- {s0['definition']}")
    lines.append(f"- n_pairs={s0['n_pairs']}, n_questions/pair={s0['n_questions_per_pair']}")
    lines.append("")
    lines.append("**Supporting caches:**")
    for k, v in s0["supporting_paths"].items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")

    lines += ["## Experiment 1 — Table 14 predictors vs mean selection gain (54 pairs)", ""]
    lines.append("Sanity: FRS Pearson r ≈ 0.49; unfiltered ≈ 0.008.")
    rows = []
    for r in results["experiment1"]["rows"]:
        rows.append(
            [
                r.get("predictor", r.get("predictor_col")),
                f"{r.get('pearson_r', np.nan):.4f}" if "pearson_r" in r else "—",
                f"{r.get('pearson_p', np.nan):.4f}" if "pearson_p" in r else "—",
                f"{r.get('spearman_rho', np.nan):.4f}" if "spearman_rho" in r else "—",
                f"{r.get('spearman_p', np.nan):.4f}" if "spearman_p" in r else "—",
                r.get("n", ""),
            ]
        )
    lines.append(md_table(["Predictor", "Pearson r", "p", "Spearman ρ", "p", "n"], rows))
    if results["experiment1"].get("top_conf_note"):
        lines.append(f"\n*{results['experiment1']['top_conf_note']}*")
    lines.append("")

    lines += ["## Experiment 2 — Dependent correlation: FRS vs trace-0 / unfiltered", ""]
    lines.append("Paired bootstrap over 54 cells: Δr = r(predictor, gain) each resample.")
    e2_rows = []
    for block in results["experiment2"]:
        e2_rows.append(
            [
                block["comparison"],
                f"{block['pearson_r_a']:.4f}",
                f"{block['pearson_r_b']:.4f}",
                f"{block['pearson_r_between_predictors']:.4f}",
                f"{block['diff_r_point']:.4f}",
                f"[{block['diff_r_ci_95_lo']:.3f}, {block['diff_r_ci_95_hi']:.3f}]",
                f"{block['bootstrap_p_diff_le_0']:.4f}",
            ]
        )
    lines.append(
        md_table(
            ["Test", "r(A,gain)", "r(B,gain)", "r(A,B)", "Δr", "95% CI", "p(Δr≤0)"],
            e2_rows,
        )
    )
    lines.append("")

    lines += ["## Experiment 3 — Paired amplification difference (shared denominator)", ""]
    lines.append("Δamp = [mean(FRS_gap) − mean(comparator_gap)] / mean(acc_gap) on the same close-accuracy pairs.")
    e3_rows = []
    for block in results["experiment3"]:
        ci = block.get("amplification_diff_ci_95", [np.nan, np.nan])
        e3_rows.append(
            [
                block.get("comparison", ""),
                block.get("threshold_pp", ""),
                block.get("n_close_pairs", ""),
                f"{block.get('amplification_diff_point', np.nan):.3f}",
                f"[{ci[0]:.3f}, {ci[1]:.3f}]",
                f"{block.get('bootstrap_p_frs_amp_le_comparator', np.nan):.4f}",
            ]
        )
    lines.append(
        md_table(
            ["Comparison", "thresh (pp)", "n", "Δamp point", "95% CI", "p(FRS≤comp)"],
            e3_rows,
        )
    )
    lines.append("")

    e4 = results["experiment4"]
    lines += ["## Experiment 4 — Estimator reliability (bootstrap SD of mean)", ""]
    lines.append(
        f"Median per-pair bootstrap SD: **trace-0 = {e4['median_bootstrap_sd_trace0']:.4f}**, "
        f"**FRS = {e4['median_bootstrap_sd_frs']:.4f}** "
        f"(ratio trace0/FRS ≈ {e4['ratio_median_sd_trace0_over_frs']:.2f}×)"
    )
    lines.append("")

    lines += ["## Provenance", ""]
    for p in results["provenance"]:
        lines.append(f"- `{p}`")
    if results.get("skipped"):
        lines.append("\n**Skipped / not run:**")
        for s in results["skipped"]:
            lines.append(f"- {s}")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    log = setup_logger()

    pair, sg_meta = load_selection_gain_54(repo)
    panel = build_predictor_panel(repo, log)

    # STEP 0
    step0 = {
        "status": "cached — 54-cell means available",
        "pair_level_path": str(repo / "analysis/selection_gain_pair_level.csv"),
        "variable": "mean_selection_gain",
        "definition": sg_meta["definition"],
        "n_pairs": sg_meta["n_pairs"],
        "n_questions_per_pair": sg_meta["n_questions_per_pair"],
        "supporting_paths": {k: str(repo / v) for k, v in STEP0_PATHS.items()},
        "reconstruction": "Not needed; pair_level pre-aggregated from selection_gain_judge_outputs.csv (5400 rows).",
    }

    exp1_df = experiment1_table14(panel)
    prior = pd.read_csv(repo / "analysis/selection_gain_predictor_results.csv")
    sanity = {}
    for col, key in [("frs_pct", "frs"), ("unfiltered_reasoning_mean", "unfiltered")]:
        new = exp1_df[exp1_df["predictor_col"] == col]
        old = prior[prior["predictor"] == col]
        if len(new) and len(old):
            sanity[key] = {
                "recomputed_pearson": float(new["pearson_r"].iloc[0]),
                "prior_pearson": float(old["pearson_r"].iloc[0]),
            }

    top_conf_note = (
        "Predictor (b): `mean_top_conf_reasoning` from pair_level — mean RS of the top-confidence "
        "trace per question (50/50 judged in selection_gain_judge_outputs.csv, selection_type=top_conf)."
    )

    y = panel["mean_selection_gain"].values.astype(float)
    exp2 = [
        dependent_corr_bootstrap(
            panel["frs_pct"].values.astype(float),
            panel["trace0_reasoning_0_1"].values.astype(float),
            y,
            "FRS",
            "trace0",
        ),
        dependent_corr_bootstrap(
            panel["frs_pct"].values.astype(float),
            panel["unfiltered_reasoning_mean"].values.astype(float),
            y,
            "FRS",
            "unfiltered",
        ),
    ]

    pairs = build_close_accuracy_pairs(panel)
    exp3 = []
    for thr in [3.0, 5.0]:
        exp3.append(paired_amp_bootstrap(pairs, "frs", "trace0", thr))
        exp3.append(paired_amp_bootstrap(pairs, "frs", "unfiltered", thr))

    exp4 = experiment4_reliability(repo, log)

    results: Dict[str, Any] = {
        "step0": step0,
        "experiment1": {
            "rows": exp1_df.to_dict(orient="records"),
            "sanity_vs_prior_csv": sanity,
            "top_conf_note": top_conf_note,
        },
        "experiment2": exp2,
        "experiment3": exp3,
        "experiment4": exp4,
        "provenance": [str(repo / p) for p in STEP0_PATHS.values()],
        "skipped": [],
    }

    out_md = repo / "results_selectiongain_decisive.md"
    out_json = repo / "results_selectiongain_decisive.json"

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
