#!/usr/bin/env python3
"""
Self-consistency (vote-share) confidence → FRS-style metrics using **existing** pass@16 JSONL
and **existing** judge checkpoints only. No generation, no judge API calls.

Confidence (per trace t on a problem with k traces):
  conf_sc(t) = (# of j with normalize(pred[j]) == normalize(pred[t])) / k

Pipeline matches ``run_confidence_proxy_robustness.py``: rank by conf_sc ↓, top_pool_frac=0.5,
n_bins=5 equal-count bins, FRS@k10 = mean reasoning_score over **judged** traces in bin_id==0.

Usage:
  python analysis/run_self_consistency_frs_proxy.py --repo-root .
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from build_downstream_parquets import (  # noqa: E402
    discover_jsonl_groups,
    jsonl_from_judge_metadata,
    judge_path,
)

# Reuse binning / slice / cumulative from confidence proxy (single source of truth)
_CP_PATH = os.path.join(REPO_ROOT, "analysis", "run_confidence_proxy_robustness.py")
_spec = importlib.util.spec_from_file_location("_conf_proxy_sc", _CP_PATH)
assert _spec and _spec.loader
_cp = importlib.util.module_from_spec(_spec)
sys.modules["_conf_proxy_sc"] = _cp
_spec.loader.exec_module(_cp)

BinConfig = _cp.BinConfig
assign_disjoint_bins = _cp.assign_disjoint_bins
slice_top_pool = _cp.slice_top_pool
frs_cumulative_bins = _cp.frs_cumulative_bins
TOP_POOL_FRAC = _cp.TOP_POOL_FRAC
N_BINS = _cp.N_BINS
BENCHMARKS = _cp.BENCHMARKS

DEFAULT_FRS_CSV = os.path.join(REPO_ROOT, "outputs/global_pass1_frs_analysis", "paper_frs_by_benchmark.csv")

HEADLINE_MODELS = [
    "DS-R1-7B",
    "DS-R1-1.5B",
    "Qwen2.5-7B",
    "Phi-4",
    "Phi-4-Reas.",
]


def normalize_pred_for_agreement(s: Any) -> str:
    """
    Conservative normalization for self-consistency agreement:
    - If not a string, coerce with str() then strip.
    - Empty after strip → empty string (traces with empty preds agree only with each other).
    - No case-folding by default (documented); optional flag elsewhere if added.
    """
    if s is None:
        return ""
    t = str(s).strip()
    return t


def vote_share_confidence(preds_norm: List[str], t: int, k: int) -> float:
    """Fraction of traces j (0..k-1) with preds_norm[j] == preds_norm[t]."""
    if k <= 0 or t < 0 or t >= k:
        return float("nan")
    pt = preds_norm[t]
    matches = sum(1 for j in range(k) if preds_norm[j] == pt)
    return float(matches) / float(k)


def load_judge_reasoning_map_robust(path: str) -> Dict[Tuple[int, int], float]:
    """Like proxy script but tolerates ``judged_samples`` as list or dict of entries."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("judged_samples", [])
    if isinstance(raw, dict):
        samples = list(raw.values())
    else:
        samples = list(raw)
    out: Dict[Tuple[int, int], float] = {}
    for s in samples:
        if not isinstance(s, dict):
            continue
        if s.get("judge_ok") is False:
            continue
        rs = s.get("reasoning_score")
        if rs is None:
            continue
        v = float(rs)
        if not np.isfinite(v):
            continue
        rs100 = v * 100.0 if v <= 1.5 else v
        try:
            key = (int(s["idx"]), int(s["trace_idx"]))
        except (KeyError, TypeError, ValueError):
            continue
        out[key] = rs100
    return out


def load_jsonl_self_consistency_rows(
    filepath: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    One row per trace: idx, trace_idx, conf_sc, correct, has_empty_pred_in_problem.
    Skips lines with no pred list or length mismatch with score.
    """
    recs: List[Dict[str, Any]] = []
    stats = {
        "jsonl_lines": 0,
        "skipped_bad_pred": 0,
        "skipped_len_mismatch": 0,
        "problems_with_any_empty_pred": 0,
    }
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stats["jsonl_lines"] += 1
            row = json.loads(line)
            pred = row.get("pred", [])
            scores = row.get("score", [])
            if not isinstance(pred, list) or not isinstance(scores, list):
                stats["skipped_bad_pred"] += 1
                continue
            k = len(scores)
            if len(pred) != k:
                stats["skipped_len_mismatch"] += 1
                continue
            preds_norm = [normalize_pred_for_agreement(p) for p in pred]
            if any(p == "" for p in preds_norm):
                stats["problems_with_any_empty_pred"] += 1
            try:
                pid = row.get("idx")
                if pid is None:
                    continue
                pid_i = int(pid)
            except (TypeError, ValueError):
                continue
            for t in range(k):
                conf = vote_share_confidence(preds_norm, t, k)
                cor = bool(scores[t]) if t < len(scores) else False
                recs.append(
                    {
                        "idx": pid_i,
                        "trace_idx": t,
                        "conf_sc": conf,
                        "correct": cor,
                    }
                )
    df = pd.DataFrame(recs)
    return df, stats


def frs_sc_bin010_detailed(
    df_traces: pd.DataFrame,
    judge_rs: Dict[Tuple[int, int], float],
    bin_cfg: BinConfig,
    sort_col: str = "conf_sc",
) -> Tuple[float, int, int, int]:
    """
    Returns:
      mean_frs (nan if none),
      n_judged_in_bin0,
      n_traces_in_top_pool,
      n_traces_in_bin0_total,
    """
    if df_traces.empty:
        return float("nan"), 0, 0, 0
    top_slice, _ = slice_top_pool(df_traces, sort_col, bin_cfg.top_pool_frac)
    spec = bin_cfg.bin_spec
    binned = assign_disjoint_bins(top_slice, bin_cfg.n_bins, spec)
    in_bin0 = binned[binned["bin_id"] == 0]
    n_bin0 = len(in_bin0)
    scores: List[float] = []
    for _, r in in_bin0.iterrows():
        key = (int(r["idx"]), int(r["trace_idx"]))
        if key in judge_rs:
            scores.append(judge_rs[key])
    mean_frs = float(np.mean(scores)) if scores else float("nan")
    return mean_frs, len(scores), len(top_slice), n_bin0


def load_default_frs_table(path: str) -> Tuple[Dict[str, float], Dict[str, int]]:
    """model -> FRS_Avg and dense rank (1=best) from paper CSV."""
    by_model: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        import csv

        rdr = csv.DictReader(f)
        for row in rdr:
            m = row.get("model", "").strip()
            frs = row.get("FRS_Avg", "").strip()
            if not m or not frs:
                continue
            try:
                by_model[m] = float(frs)
            except ValueError:
                continue
    items = sorted(by_model.items(), key=lambda kv: kv[1], reverse=True)
    rank = {m: i + 1 for i, (m, _) in enumerate(items)}
    return by_model, rank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default=REPO_ROOT)
    ap.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(REPO_ROOT, "outputs/analysis_exports", "self_consistency_proxy_robustness"),
    )
    ap.add_argument("--frs-csv", type=str, default=DEFAULT_FRS_CSV)
    ap.add_argument(
        "--also-cumulative-k",
        action="store_true",
        help="Also compute cumulative FRS for top 20/30/40/50%% of slice (bins 1–5).",
    )
    ap.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="If >0, only first N model×benchmark pairs (smoke test).",
    )
    args = ap.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    bin_cfg = BinConfig(top_pool_frac=TOP_POOL_FRAC, n_bins=N_BINS)
    jsonl_map = discover_jsonl_groups(repo_root)
    pair_list = sorted((k for k in jsonl_map.keys() if k[1] in BENCHMARKS))
    if args.max_pairs > 0:
        pair_list = pair_list[: args.max_pairs]

    default_frs, default_rank = load_default_frs_table(args.frs_csv)

    rows_bench: List[Dict[str, Any]] = []
    rows_coverage: List[Dict[str, Any]] = []
    missing: List[str] = []
    cumulative_rows: List[Dict[str, Any]] = []

    for model, dataset in pair_list:
        jpath = judge_path(repo_root, model, dataset)
        if not os.path.isfile(jpath):
            missing.append(f"{model}__{dataset} (no judge)")
            continue
        jsonl_path = jsonl_from_judge_metadata(repo_root, jpath) or jsonl_map.get((model, dataset))
        if not jsonl_path or not os.path.isfile(jsonl_path):
            missing.append(f"{model}__{dataset} (no JSONL)")
            continue

        df_tr, line_stats = load_jsonl_self_consistency_rows(jsonl_path)
        judge_rs = load_judge_reasoning_map_robust(jpath)
        n_judged_avail = len(judge_rs)

        if df_tr.empty:
            missing.append(f"{model}__{dataset} (empty df_tr)")
            continue

        mean_frs, n_judged_bin0, n_pool, n_bin0_total = frs_sc_bin010_detailed(
            df_tr, judge_rs, bin_cfg, "conf_sc"
        )
        cov_frac = (n_judged_bin0 / n_bin0_total) if n_bin0_total > 0 else float("nan")

        rows_bench.append(
            {
                "model": model,
                "benchmark": dataset,
                "sc_frs_k10": mean_frs,
                "n_judged_traces_used": n_judged_bin0,
                "n_total_topbin_traces": n_bin0_total,
                "judged_coverage_fraction": cov_frac,
                "n_traces_in_top_pool_slice": n_pool,
                "n_judged_traces_available_checkpoint": n_judged_avail,
                "n_total_traces_in_jsonl": len(df_tr),
            }
        )

        has_empty = line_stats.get("problems_with_any_empty_pred", 0) > 0
        rows_coverage.append(
            {
                "model": model,
                "benchmark": dataset,
                "n_total_traces": len(df_tr),
                "n_judged_traces_available": n_judged_avail,
                "n_topbin_traces_under_sc": n_bin0_total,
                "n_judged_topbin_traces_under_sc": n_judged_bin0,
                "judged_coverage_fraction": cov_frac,
                "has_empty_preds": has_empty,
                "notes": "empty preds: traces agree only with other empty-normalized preds"
                if has_empty
                else "",
            }
        )

        if args.also_cumulative_k:
            for k_bins, label in [(2, "k20"), (3, "k30"), (4, "k40"), (5, "k50")]:
                cum = frs_cumulative_bins(df_tr, "conf_sc", judge_rs, bin_cfg, k_bins)
                cumulative_rows.append(
                    {
                        "model": model,
                        "benchmark": dataset,
                        "cumulative_label": label,
                        "n_cumulative_bins": k_bins,
                        "frs_cumulative": cum,
                    }
                )

    df_bench = pd.DataFrame(rows_bench)
    bench_path = os.path.join(out_dir, "sc_proxy_benchmark_results_k10.csv")
    df_bench.to_csv(bench_path, index=False)
    pd.DataFrame(rows_coverage).to_csv(os.path.join(out_dir, "sc_data_coverage.csv"), index=False)

    if cumulative_rows:
        pd.DataFrame(cumulative_rows).to_csv(
            os.path.join(out_dir, "sc_proxy_cumulative_k.csv"), index=False
        )

    # Model-level SC ranking (mean sc_frs_k10 over benchmarks with finite values)
    rows_rank: List[Dict[str, Any]] = []
    if not df_bench.empty:
        for model, g in df_bench.groupby("model"):
            v = g["sc_frs_k10"].values
            finite = v[np.isfinite(v)]
            covs = g["judged_coverage_fraction"].values
            cov_fin = covs[np.isfinite(covs)]
            rows_rank.append(
                {
                    "model": model,
                    "sc_frs_avg_k10": float(np.mean(finite)) if len(finite) else float("nan"),
                    "n_benchmarks_covered": int(len(finite)),
                    "mean_judged_coverage_fraction": float(np.mean(cov_fin)) if len(cov_fin) else float("nan"),
                }
            )

    df_rank = pd.DataFrame(rows_rank)
    if not df_rank.empty:
        df_rank["sc_frs_rank_k10"] = df_rank["sc_frs_avg_k10"].rank(ascending=False, method="min")
    df_rank = df_rank.sort_values("sc_frs_avg_k10", ascending=False)
    df_rank.to_csv(os.path.join(out_dir, "sc_proxy_rankings_k10.csv"), index=False)

    # sc vs default
    cmp_rows: List[Dict[str, Any]] = []
    corr_summary: List[Dict[str, Any]] = []
    models_union = sorted(set(df_rank["model"].tolist()) | set(default_frs.keys()))
    for m in models_union:
        sc_avg = df_rank[df_rank["model"] == m]["sc_frs_avg_k10"].values
        sc_r = df_rank[df_rank["model"] == m]["sc_frs_rank_k10"].values
        def_avg = default_frs.get(m)
        def_r = default_rank.get(m)
        sc_avg_v = float(sc_avg[0]) if len(sc_avg) else float("nan")
        sc_r_v = float(sc_r[0]) if len(sc_r) else float("nan")
        dr = (
            (sc_r_v - def_r)
            if (def_r is not None and np.isfinite(sc_r_v))
            else float("nan")
        )
        cmp_rows.append(
            {
                "model": m,
                "default_frs_rank": def_r if def_r is not None else float("nan"),
                "sc_frs_rank_k10": sc_r_v if np.isfinite(sc_r_v) else float("nan"),
                "rank_difference": dr,
                "absolute_rank_difference": abs(dr) if np.isfinite(dr) else float("nan"),
                "default_frs_avg": def_avg if def_avg is not None else float("nan"),
                "sc_frs_avg_k10": sc_avg_v,
            }
        )
    pd.DataFrame(cmp_rows).to_csv(os.path.join(out_dir, "sc_vs_default_rank_comparison.csv"), index=False)

    # Correlation: models with both finite
    sub = []
    for m in models_union:
        a = default_frs.get(m)
        scrow = df_rank[df_rank["model"] == m]
        b = float(scrow["sc_frs_avg_k10"].values[0]) if len(scrow) else float("nan")
        if a is not None and np.isfinite(b):
            sub.append((m, a, b))
    if len(sub) >= 3:
        xa = np.array([x[1] for x in sub], dtype=float)
        xb = np.array([x[2] for x in sub], dtype=float)
        sp_r, sp_p = spearmanr(xa, xb)
        pe_r, pe_p = pearsonr(xa, xb)
    else:
        sp_r = sp_p = pe_r = pe_p = float("nan")

    ra = np.array([default_rank[m] for m, _, _ in sub], dtype=float) if sub else np.array([])
    rb = np.array(
        [float(df_rank[df_rank["model"] == m]["sc_frs_rank_k10"].values[0]) for m, _, _ in sub],
        dtype=float,
    )
    if len(sub) >= 3:
        sr_rank, sr_p = spearmanr(ra, rb)
    else:
        sr_rank = sr_p = float("nan")

    corr_summary.append(
        {
            "metric": "model_level_frs_avg",
            "spearman_rho": float(sp_r) if np.isfinite(sp_r) else float("nan"),
            "spearman_p_value": float(sp_p) if np.isfinite(sp_p) else float("nan"),
            "pearson_r": float(pe_r) if np.isfinite(pe_r) else float("nan"),
            "pearson_p_value": float(pe_p) if np.isfinite(pe_p) else float("nan"),
            "n_models": len(sub),
            "caveat": "Sparse judged-in-bin0; SC re-ranking changes which judged traces enter bin0.",
        }
    )
    corr_summary.append(
        {
            "metric": "model_level_rank",
            "spearman_rho": float(sr_rank) if np.isfinite(sr_rank) else float("nan"),
            "spearman_p_value": float(sr_p) if np.isfinite(sr_p) else float("nan"),
            "pearson_r": float("nan"),
            "pearson_p_value": float("nan"),
            "n_models": len(sub),
            "caveat": "Ranks discrete; ties possible.",
        }
    )
    pd.DataFrame(corr_summary).to_csv(os.path.join(out_dir, "sc_vs_default_correlation_summary.csv"), index=False)

    # Headline summary
    hl: List[Dict[str, Any]] = []
    for m in HEADLINE_MODELS:
        scrow = df_rank[df_rank["model"] == m]
        sc_avg = float(scrow["sc_frs_avg_k10"].values[0]) if len(scrow) else float("nan")
        sc_rnk = float(scrow["sc_frs_rank_k10"].values[0]) if len(scrow) else float("nan")
        hl.append(
            {
                "model": m,
                "default_frs_avg": default_frs.get(m, float("nan")),
                "default_rank": float(default_rank.get(m, float("nan"))),
                "sc_frs_avg": sc_avg,
                "sc_rank": sc_rnk,
                "rank_difference": sc_rnk - default_rank[m]
                if m in default_rank and np.isfinite(sc_rnk)
                else float("nan"),
            }
        )
    pd.DataFrame(hl).to_csv(os.path.join(out_dir, "sc_key_models_summary.csv"), index=False)

    meta = {
        "confidence_definition": (
            "Per-trace vote share: conf_sc(t) = (# j with normalize(pred[j])==normalize(pred[t])) / k, "
            "where k=len(pred) on that JSONL line (typically 16)."
        ),
        "normalization": {
            "rule": "str(s).strip(); empty string after strip remains empty; agreement is exact on this normalized string.",
            "case_folding": "not applied (conservative).",
        },
        "join_keys": "(idx, trace_idx) between JSONL-derived rows and judged_samples",
        "judge_checkpoint_glob": "outputs/reasoning_confidence_bins_results/judging_checkpoints/judged_<Model>__<Dataset>.json",
        "jsonl_discovery": "build_downstream_parquets.discover_jsonl_groups; per-pair path from judge metadata when present",
        "top_pool_frac": TOP_POOL_FRAC,
        "n_bins": N_BINS,
        "frs_k10_definition": (
            "Mean reasoning_score (0-100) over judged (idx,trace_idx) in bin_id==0 after ranking all traces "
            "by conf_sc descending, keeping top 50% by count, 5 equal-count bins."
        ),
        "k10_only_additional_cumulative": bool(args.also_cumulative_k),
        "caveats": [
            "Judged traces are a fixed subset from the original logit-based run; SC re-ranking changes which of them fall in bin0 — not Table 2 reproduction.",
            "Exact-string match after strip only; no LaTeX equivalence.",
            "Empty preds participate in agreement as empty string.",
            "Self-consistency is problem-coupled across traces (unlike per-trace logits).",
        ],
        "missing_pairs": missing,
        "n_benchmark_rows": len(df_bench),
        "outputs": {
            "sc_proxy_benchmark_results_k10.csv": bench_path,
            "sc_proxy_rankings_k10.csv": os.path.join(out_dir, "sc_proxy_rankings_k10.csv"),
            "sc_vs_default_rank_comparison.csv": os.path.join(out_dir, "sc_vs_default_rank_comparison.csv"),
            "sc_vs_default_correlation_summary.csv": os.path.join(out_dir, "sc_vs_default_correlation_summary.csv"),
            "sc_key_models_summary.csv": os.path.join(out_dir, "sc_key_models_summary.csv"),
            "sc_data_coverage.csv": os.path.join(out_dir, "sc_data_coverage.csv"),
            **(
                {"sc_proxy_cumulative_k.csv": os.path.join(out_dir, "sc_proxy_cumulative_k.csv")}
                if args.also_cumulative_k
                else {}
            ),
        },
    }
    with open(os.path.join(out_dir, "sc_proxy_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote outputs under {out_dir}/", flush=True)
    print(f"Benchmark rows: {len(df_bench)}  Models: {df_rank['model'].nunique() if len(df_rank) else 0}")
    print(f"Spearman (default FRS avg vs SC FRS avg): {sp_r:.4f}  n_models={len(sub)}")
    if missing:
        print(f"Missing/skipped ({len(missing)}):", *missing[:6], "..." if len(missing) > 6 else "", sep="\n  ")


if __name__ == "__main__":
    main()
