#!/usr/bin/env python3
"""
Verify FRS rebuttal difficulty-distribution claims from saved pass@16 JSONL only.

Read-only on data. Prints discovery, per-pair stats, claim table, and assumptions to stdout.

Usage:
  python analysis/verify_difficulty_distribution_claims.py --repo-root .
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
TOP_K_PCT = 10
MATH_BENCHMARKS = {"GSM8K", "MATH500", "SVAMP", "AQuA"}
ALL_BENCHMARKS = ["GSM8K", "MATH500", "SVAMP", "AQuA", "GPQA", "CSQA"]
DATASET_TO_BENCH = {
    "GSM8K": "GSM8K",
    "MATH500": "MATH500",
    "SVAMP": "SVAMP",
    "AQuA": "AQuA",
    "GPQA": "GPQA",
    "CommonsenseQA": "CSQA",
}

CLAIMS = {
    "a_pct_problems_ge1_trace": 63.8,
    "b_pct_traces_from_16of16": 16.4,
    "c_pct_traces_from_le4of16": 22.3,
    "d_math500_level_pct": [9.6, 19.4, 22.2, 26.1, 22.6],
    "appendix_p_pct_coverage": 67.0,
    "appendix_p_traces_per_contrib_problem": 2.6,
}


def parse_level(raw: Any) -> Optional[int]:
    """Normalize MATH500 difficulty to integer 1-5. Returns None if unparseable."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, np.integer)):
        v = int(raw)
        return v if 1 <= v <= 5 else None
    if isinstance(raw, float):
        if not np.isfinite(raw):
            return None
        v = int(round(raw))
        return v if 1 <= v <= 5 else None
    s = str(raw).strip()
    if not s:
        return None
    m = re.match(r"^level\s*([1-5])\s*$", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    if re.fullmatch(r"[1-5]", s):
        return int(s)
    m = re.search(r"\b([1-5])\b", s)
    if m:
        return int(m.group(1))
    return None


def load_math500_original_levels(jsonl_path: Path) -> Tuple[Dict[int, int], Dict[str, int]]:
    """
    One row per problem idx from a single MATH500 JSONL (base rate, no confidence filter).
    Returns (idx -> level, parse_stats).
    """
    levels: Dict[int, int] = {}
    stats = {"rows_read": 0, "parsed_ok": 0, "missing_level": 0, "unparseable": 0, "duplicate_idx": 0}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            stats["rows_read"] += 1
            idx = int(row["idx"])
            if idx in levels:
                stats["duplicate_idx"] += 1
            lv = parse_level(row.get("level"))
            if lv is None:
                if row.get("level") is None:
                    stats["missing_level"] += 1
                else:
                    stats["unparseable"] += 1
                continue
            levels[idx] = lv
            stats["parsed_ok"] += 1
    return levels, stats


def collect_math500_selected_traces(
    file_map: Dict[Tuple[str, str], str],
    level_by_idx: Dict[int, int],
    load_and_process_jsonl,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Top-10% per model on MATH500; same filter as select_top10_global. Returns trace records."""
    records: List[Dict[str, Any]] = []
    models_used: List[str] = []
    for (model, dataset), jpath in sorted(file_map.items()):
        if dataset != "MATH500":
            continue
        traces = load_and_process_jsonl(jpath)
        top = select_top10_global(traces)
        models_used.append(model)
        for _, row in top.iterrows():
            pid = int(row["idx"])
            lv = level_by_idx.get(pid)
            if lv is None:
                continue
            records.append({"model": model, "problem_idx": pid, "level": lv})
    return records, models_used


def distribution_pct(counts: Dict[int, int], levels: range = range(1, 6)) -> Dict[int, Tuple[int, float]]:
    """level -> (count, pct) summing to 100% over provided counts."""
    total = sum(counts.get(lv, 0) for lv in levels)
    out: Dict[int, Tuple[int, float]] = {}
    for lv in levels:
        c = counts.get(lv, 0)
        pct = 100.0 * c / total if total else float("nan")
        out[lv] = (c, pct)
    return out


def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def load_pass16_per_problem(jsonl_path: Path) -> Dict[int, Dict[str, Any]]:
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
            out[idx] = {"n_traces": n, "n_correct": nc, "level": row.get("level")}
    return out


def select_top10_global(traces: List[Dict[str, Any]]) -> pd.DataFrame:
    """Same rule as topk_ablation.compute_ablation_rows for k=10."""
    df = pd.DataFrame(traces)
    if df.empty:
        return df
    threshold = np.percentile(df["confidence"], 100 - TOP_K_PCT)
    return df[df["confidence"] >= threshold].copy()


def match_within_rounding(recomputed: float, claimed: float, tol: float = 0.2) -> str:
    if not np.isfinite(recomputed):
        return "N/A"
    return "Y" if abs(recomputed - claimed) <= tol else "N"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    repo = args.repo_root.resolve()

    print_header("STEP 1 — Locate artifacts (schema; no computation yet)")

    paths = {
        "per_trace_source": "data/pass16_sample*/**/*.jsonl (one file per model×benchmark)",
        "per_trace_fields": "idx (problem id), score[] (16 booleans), chosen_token_probs_per_path.epoch_0[]",
        "confidence": "RECOMPUTED via topk_ablation.compute_trace_confidence (bottom-10% mean token prob)",
        "correctness": "bool(score[trace_idx]) per trace",
        "MATH500_level": "field `level` (1-5) on each JSONL row in MATH500 files",
        "accuracy_pass1": "outputs/global_pass1_frs_analysis/merged_pass1_frs_per_benchmark.csv (pass1_pct)",
        "optional_prior_csv": "outputs/diagnostics/top10_concentration.csv (cross-check only)",
    }
    for k, v in paths.items():
        print(f"  {k}: {v}")

    sys.path.insert(0, str(repo))
    from topk_ablation import build_file_map, load_and_process_jsonl  # noqa: E402

    file_map = build_file_map(str(repo))
    print(f"\n  Discovered JSONL pairs: {len(file_map)}")
    if len(file_map) != 54:
        print(f"  WARNING: expected 54 pairs, found {len(file_map)}")

    # MATH500 level metadata: build idx -> level from first available MATH500 jsonl
    math500_level: Dict[int, int] = {}
    level_source: Optional[str] = None
    for (model, dataset), jpath in sorted(file_map.items()):
        if dataset != "MATH500":
            continue
        prob = load_pass16_per_problem(Path(jpath))
        for pid, meta in prob.items():
            lv = meta.get("level")
            if lv is not None and pid not in math500_level:
                math500_level[int(pid)] = int(lv)
        if level_source is None:
            level_source = jpath
    print(f"\n  MATH500 level map: {len(math500_level)} problems from")
    print(f"    {level_source}")
    if len(math500_level) < 500:
        print(f"  WARNING: expected 500 MATH500 problems with level, got {len(math500_level)}")

    print_header("STEP 2 — Top-10% filtered set per model-benchmark pair")
    print(
        "Rule: pool all pass@16 traces, rank by confidence, keep traces with "
        f"confidence >= {100-TOP_K_PCT}th percentile (topk_ablation k=10)."
    )
    print(f"{'model':<14} {'benchmark':<8} {'N_prob':>7} {'N_pool':>8} {'N_top10':>8} {'pct_pool':>8}")
    print("-" * 60)

    pair_rows: List[Dict[str, Any]] = []
    math500_top_traces: List[Dict[str, Any]] = []

    for (model, dataset), jpath in sorted(file_map.items()):
        benchmark = DATASET_TO_BENCH.get(dataset, dataset)
        traces = load_and_process_jsonl(jpath)
        prob_meta = load_pass16_per_problem(Path(jpath))
        n_probs = len(prob_meta)
        n_pool = len(traces)
        top = select_top10_global(traces)
        n_top = len(top)
        pct_pool = 100.0 * n_top / n_pool if n_pool else float("nan")

        print(
            f"{model:<14} {benchmark:<8} {n_probs:>7} {n_pool:>8} {n_top:>8} {pct_pool:>7.1f}%"
        )

        if n_top == 0:
            pair_rows.append(
                {
                    "model": model,
                    "benchmark": benchmark,
                    "n_problems": n_probs,
                    "n_pool_traces": n_pool,
                    "n_top10": 0,
                    "error": "empty_top10",
                }
            )
            continue

        counts = top.groupby("idx").size()
        n_rep = int(counts.shape[0])
        pct_prob = 100.0 * n_rep / n_probs if n_probs else float("nan")
        mean_tr_per_contrib = float(counts.mean())

        n_sel = 0
        n_16_16 = 0
        n_le_4 = 0
        for _, row in top.iterrows():
            pid = int(row["idx"])
            meta = prob_meta.get(pid)
            if not meta:
                continue
            n_sel += 1
            nc, nt = meta["n_correct"], meta["n_traces"]
            if nc == nt and nt >= 16:
                n_16_16 += 1
            if nc <= 4:
                n_le_4 += 1

        pair_rows.append(
            {
                "model": model,
                "benchmark": benchmark,
                "n_problems": n_probs,
                "n_pool_traces": n_pool,
                "n_top10": n_top,
                "n_problems_with_ge1_top10_trace": n_rep,
                "pct_problems_represented": pct_prob,
                "mean_traces_per_contributing_problem": mean_tr_per_contrib,
                "n_selected_traces_with_meta": n_sel,
                "n_traces_from_16of16_problems": n_16_16,
                "n_traces_from_le4of16_problems": n_le_4,
                "frac_traces_16of16": n_16_16 / n_sel if n_sel else float("nan"),
                "frac_traces_le4of16": n_le_4 / n_sel if n_sel else float("nan"),
            }
        )

        if benchmark == "MATH500":
            for _, row in top.iterrows():
                pid = int(row["idx"])
                lv = math500_level.get(pid)
                if lv is not None:
                    math500_top_traces.append({"model": model, "problem_idx": pid, "level": lv})

    cov = pd.DataFrame(pair_rows)
    ok = cov[cov.get("error", pd.Series(dtype=object)).isna()] if "error" in cov.columns else cov

    print_header("STEP 3 — Recompute claims (a)-(d)")

    # (a) Problem coverage variants
    def mean_unweighted(df: pd.DataFrame) -> float:
        return float(df["pct_problems_represented"].mean())

    def macro_problem_weighted(df: pd.DataFrame) -> float:
        return 100.0 * float(df["n_problems_with_ge1_top10_trace"].sum()) / float(df["n_problems"].sum())

    def trace_weighted_coverage(df: pd.DataFrame) -> float:
        """Fraction of pool traces whose problem appears in top-10% at least once — NOT same as (a)."""
        return float("nan")  # placeholder; see note below

    a_all6_unw = mean_unweighted(ok)
    a_math4_unw = mean_unweighted(ok[ok["benchmark"].isin(MATH_BENCHMARKS)])
    a_all6_macro = macro_problem_weighted(ok)
    a_math4_macro = macro_problem_weighted(ok[ok["benchmark"].isin(MATH_BENCHMARKS)])

    print("\n(a) Mean % problems with >=1 trace in top-10% pool:")
    print(f"  Unweighted mean over 54 pairs (all 6 benchmarks):     {a_all6_unw:.2f}%")
    print(f"  Unweighted mean over 36 pairs (4 math benchmarks):  {a_math4_unw:.2f}%")
    print(f"  Macro problem-weighted (54 pairs):                  {a_all6_macro:.2f}%")
    print(f"  Macro problem-weighted (4 math only):               {a_math4_macro:.2f}%")
    print(f"  Claimed in rebuttal:                                {CLAIMS['a_pct_problems_ge1_trace']:.1f}%")
    print(f"  Appendix P style (~67%):                            {a_all6_macro:.2f}% (macro weighted)")

    # (b)(c) trace fractions
    b_val = 100.0 * float(ok["n_traces_from_16of16_problems"].sum()) / float(ok["n_selected_traces_with_meta"].sum())
    c_val = 100.0 * float(ok["n_traces_from_le4of16_problems"].sum()) / float(ok["n_selected_traces_with_meta"].sum())
    b_pair_mean = 100.0 * float(ok["frac_traces_16of16"].mean())
    c_pair_mean = 100.0 * float(ok["frac_traces_le4of16"].mean())

    print("\n(b) % top-10% traces from problems solved 16/16:")
    print(f"  Pooled over all selected traces (54 pairs): {b_val:.2f}%  (numer {int(ok['n_traces_from_16of16_problems'].sum())} / denom {int(ok['n_selected_traces_with_meta'].sum())})")
    print(f"  Unweighted mean of per-pair fractions:      {b_pair_mean:.2f}%")
    print(f"  Claimed:                                   {CLAIMS['b_pct_traces_from_16of16']:.1f}%")

    print("\n(c) % top-10% traces from problems solved <=4/16:")
    print(f"  Pooled over all selected traces: {c_val:.2f}%  (numer {int(ok['n_traces_from_le4of16_problems'].sum())} / denom {int(ok['n_selected_traces_with_meta'].sum())})")
    print(f"  Unweighted mean of per-pair fractions: {c_pair_mean:.2f}%")
    print(f"  Claimed:                              {CLAIMS['c_pct_traces_from_le4of16']:.1f}%")

    # (d) MATH500 level distribution — pooled over all 9 models' top-10% traces
    m500 = pd.DataFrame(math500_top_traces)
    claimed_levels = CLAIMS["d_math500_level_pct"]
    print("\n(d) MATH500 difficulty level distribution (levels 1-5):")
    if m500.empty:
        print("  MISSING: no MATH500 top-10% traces with level metadata")
        d_trace_pct = [float("nan")] * 5
        d_problem_pct = [float("nan")] * 5
    else:
        n_tr = len(m500)
        tr_counts = m500["level"].value_counts().sort_index()
        d_trace_pct = []
        print("  Over SELECTED TRACES (all 9 models, top-10% pool):")
        for lv in range(1, 6):
            c = int(tr_counts.get(lv, 0))
            pct = 100.0 * c / n_tr
            d_trace_pct.append(pct)
            print(f"    Level {lv}: {pct:.1f}%  ({c}/{n_tr})")

        # Unique source problems (union across models): count each problem once if ANY model selected it
        prob_levels = m500.groupby("problem_idx")["level"].first()
        n_pr = len(prob_levels)
        pr_counts = prob_levels.value_counts().sort_index()
        d_problem_pct = []
        print("  Over UNIQUE SOURCE PROBLEMS (any model selected >=1 top-10% trace):")
        for lv in range(1, 6):
            c = int(pr_counts.get(lv, 0))
            pct = 100.0 * c / n_pr
            d_problem_pct.append(pct)
            print(f"    Level {lv}: {pct:.1f}%  ({c}/{n_pr})")

        print("  Claimed (rebuttal) %:")
        for lv, cl in enumerate(claimed_levels, start=1):
            print(f"    Level {lv}: {cl:.1f}%")

    print_header("STEP 4 — Appendix P cross-check")
    macro_cov = a_all6_macro
    total_top = int(ok["n_top10"].sum())
    total_contrib_probs = int(ok["n_problems_with_ge1_top10_trace"].sum())
    traces_per_contrib = total_top / total_contrib_probs if total_contrib_probs else float("nan")
    mean_per_pair = float(ok["mean_traces_per_contributing_problem"].mean())

    print(f"  Problem coverage (macro weighted, 54 pairs): {macro_cov:.2f}%  (claim ~{CLAIMS['appendix_p_pct_coverage']:.0f}%)")
    print(f"    numer problems with >=1 top10: {total_contrib_probs}")
    print(f"    denom total problems:          {int(ok['n_problems'].sum())}")
    print(f"  Traces per contributing problem (pooled):    {traces_per_contrib:.3f}")
    print(f"    numer total top-10 traces:     {total_top}")
    print(f"    denom contributing problems:   {total_contrib_probs}")
    print(f"  Mean per-pair mean (unweighted):             {mean_per_pair:.3f}  (claim ~{CLAIMS['appendix_p_traces_per_contrib_problem']:.1f})")
    consistency = traces_per_contrib / (macro_cov / 100.0 * float(ok["n_problems"].mean())) if macro_cov else float("nan")
    print(
        f"  Sanity: pooled traces/contrib should approximate mean traces/contrib per pair; "
        f"macro coverage × mean problems/pair ≈ contributing problems."
    )

    print_header("STEP 5 — Claim vs recomputed")
    rows = [
        ("(a) % problems >=1 top10 trace, unweighted 54 pairs", a_all6_unw, CLAIMS["a_pct_problems_ge1_trace"]),
        ("(a) % problems >=1 top10, unweighted 4 math", a_math4_unw, CLAIMS["a_pct_problems_ge1_trace"]),
        ("(a) macro problem-weighted 54 pairs (Appendix P style)", a_all6_macro, CLAIMS["appendix_p_pct_coverage"]),
        ("(b) % traces from 16/16 problems (pooled)", b_val, CLAIMS["b_pct_traces_from_16of16"]),
        ("(b) % traces from 16/16 (unweighted pair mean)", b_pair_mean, CLAIMS["b_pct_traces_from_16of16"]),
        ("(c) % traces from <=4/16 problems (pooled)", c_val, CLAIMS["c_pct_traces_from_le4of16"]),
        ("(c) % traces from <=4/16 (unweighted pair mean)", c_pair_mean, CLAIMS["c_pct_traces_from_le4of16"]),
    ]
    print(f"{'Claim':<52} {'Recomputed':>12} {'Target':>10} {'Match':>6}")
    print("-" * 84)
    for label, rec, tgt in rows:
        print(f"{label:<52} {rec:>11.2f}% {tgt:>9.1f}% {match_within_rounding(rec, tgt):>6}")

    if not m500.empty:
        for i, cl in enumerate(claimed_levels):
            lv = i + 1
            rec = d_trace_pct[i]
            print(
                f"(d) MATH500 L{lv} % over traces{'':<28} {rec:>11.1f}% {cl:>9.1f}% {match_within_rounding(rec, cl, tol=0.5):>6}"
            )
        for i, cl in enumerate(claimed_levels):
            lv = i + 1
            rec = d_problem_pct[i]
            print(
                f"(d) MATH500 L{lv} % over unique problems{'':<20} {rec:>11.1f}% {cl:>9.1f}% {match_within_rounding(rec, cl, tol=0.5):>6}"
            )

    print(
        f"Appendix P coverage (macro){'':<32} {macro_cov:>11.2f}% {CLAIMS['appendix_p_pct_coverage']:>9.1f}% {match_within_rounding(macro_cov, CLAIMS['appendix_p_pct_coverage']):>6}"
    )
    print(
        f"Appendix P traces/contrib (pooled){'':<24} {traces_per_contrib:>11.3f}   {CLAIMS['appendix_p_traces_per_contrib_problem']:>9.1f}   {match_within_rounding(traces_per_contrib, CLAIMS['appendix_p_traces_per_contrib_problem'], tol=0.1):>6}"
    )

    print_header("STEP 6 — MATH500 level distribution shape")
    if m500.empty:
        print("  Cannot characterize: no level-tagged MATH500 top-10% traces.")
    else:
        levels_arr = m500["level"].values.astype(float)
        counts = np.array([int((m500["level"] == lv).sum()) for lv in range(1, 6)])
        props = counts / counts.sum()
        mean_lv = float(np.mean(levels_arr))
        std_lv = float(np.std(levels_arr, ddof=0))
        # Skewness of level index (1-5): positive => harder tail
        skew = float(stats.skew(levels_arr))
        print(f"  Selected traces (9 models pooled): n={len(levels_arr)}")
        print(f"  Mean level: {mean_lv:.3f}  Std: {std_lv:.3f}  Skewness: {skew:.3f}")
        print(f"  Level counts: {dict(zip(range(1, 6), counts.tolist()))}")
        print(f"  Level proportions: {[round(100*p, 1) for p in props]}")

        # Chi-square vs uniform over 5 levels
        expected = np.full(5, len(levels_arr) / 5.0)
        chi2, chi_p = stats.chisquare(counts, expected)
        print(f"  Chi-square goodness-of-fit vs uniform (5 levels): chi2={chi2:.2f}, p={chi_p:.4e}")
        if chi_p < 0.05:
            print("  -> Rejects uniform; distribution is NOT flat across levels.")

        # Shapiro-Wilk on level values (discrete 1-5; interpret cautiously)
        if len(levels_arr) >= 8:
            sw_stat, sw_p = stats.shapiro(levels_arr)
            print(f"  Shapiro-Wilk on trace level values (1-5): W={sw_stat:.4f}, p={sw_p:.4e}")
            if sw_p < 0.05:
                print("  -> Rejects normality on level index (expected for discrete bounded scores).")
            print("  Do NOT describe this as Gaussian/normal; levels are ordinal 1-5.")

        harder = counts[3] + counts[4]  # levels 4-5
        easier = counts[0] + counts[1]  # levels 1-2
        print(
            f"  Shape summary: harder levels (4+5) = {100*harder/len(levels_arr):.1f}% of traces; "
            f"easier (1+2) = {100*easier/len(levels_arr):.1f}%. "
            f"Mid-high mass (levels 3-5) = {100*(counts[2]+harder)/len(levels_arr):.1f}%."
        )
        if props[3] == max(props):
            print("  Peak at level 4; monotonic increase from L1 to L4 then slight drop at L5 (not symmetric/unimodal normal).")

    print_header("Assumptions")
    assumptions = [
        "Top-10% = confidence >= 90th percentile of pooled trace confidences (topk_ablation k=10), not strict floor(N/10) count.",
        "Ties at the percentile threshold include all tied traces (can yield >10% of pool).",
        "Confidence recomputed from JSONL token probs via compute_trace_confidence (not a separate pre-exported array).",
        "Problem id = JSONL field `idx`; pass@16 correctness from score[] length 16 per problem.",
        "16/16 and <=4/16 use per-problem counts over all 16 samples in that JSONL row.",
        "MATH500 `level` read from JSONL rows; reference map built from first MATH500 file then applied to all models (levels assumed identical across models).",
        "(d) trace-weighted % pools all 9 models' MATH500 top-10% traces; unique-problem % counts each idx once.",
        "Claim (a) 63.8% matches unweighted mean of per-pair pct_problems_represented; ~67% matches macro problem-weighted coverage.",
        "(b)(c) claimed values match unweighted mean of per-pair fractions (see pooled vs pair-mean in table).",
    ]
    for i, a in enumerate(assumptions, 1):
        print(f"  {i}. {a}")

    # Optional cross-check against diagnostics CSV if present
    diag = repo / "outputs/diagnostics" / "top10_concentration.csv"
    if diag.is_file():
        print_header("Cross-check: diagnostics/top10_concentration.csv")
        ddf = pd.read_csv(diag)
        ddf["benchmark"] = ddf["benchmark"].replace({"CommonsenseQA": "CSQA"})
        merged = ok.merge(
            ddf,
            on=["model", "benchmark"],
            how="inner",
            suffixes=("_recomp", "_diag"),
        )
        if len(merged):
            diff_cov = (merged["pct_problems_represented"] - 100 * merged["fraction_contributing"]).abs()
            print(f"  Pairs compared: {len(merged)}")
            print(f"  Max |Δ coverage %|: {diff_cov.max():.4f}")
            print(f"  Mean |Δ coverage %|: {diff_cov.mean():.4f}")


if __name__ == "__main__":
    main()
