#!/usr/bin/env python3
"""
FRS robustness under alternative trace-level confidence proxies (no new inference).

Pipeline (aligned with reasoning_confidence_bins.py + paper metadata):
  - Pool all traces from pass@16 JSONL (chosen_token_probs_per_path.epoch_0).
  - Sort by confidence proxy descending.
  - Keep top top_pool_frac (0.5) of traces by count.
  - Split that slice into n_bins=5 equal-count bins (labels 0-10 … 40-50 within the slice).
  - FRS@“10%” / first bin: mean GPT-4o-mini reasoning_score (0–100) over **judged** traces
    whose (idx, trace_idx) falls in bin "0-10" under that proxy’s ranking.

Uses existing judge checkpoints only for reasoning_score; proxies are recomputed from JSONL.

Usage:
  python analysis/run_confidence_proxy_robustness.py --repo-root . --out-dir analysis_exports/confidence_proxy_robustness
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Repo imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataclasses import dataclass

from build_downstream_parquets import (  # noqa: E402
    discover_jsonl_groups,
    jsonl_from_judge_metadata,
    judge_path,
)
from topk_ablation import compute_trace_confidence, load_jsonl_raw  # noqa: E402


@dataclass(frozen=True)
class BinConfig:
    top_pool_frac: float = 0.5
    n_bins: int = 5

    def __post_init__(self) -> None:
        if 100 % self.n_bins != 0:
            raise ValueError("n_bins must divide 100")

    @property
    def bin_spec(self) -> List[Tuple[str, float, float]]:
        n = self.n_bins
        return [
            (
                f"{int(round(i * 100 / n))}-{int(round((i + 1) * 100 / n))}",
                i * (100.0 / n),
                (i + 1) * (100.0 / n),
            )
            for i in range(n)
        ]


def assign_disjoint_bins(
    top_slice: pd.DataFrame, n_bins: int, bin_spec: List[Tuple[str, float, float]]
) -> pd.DataFrame:
    """Equal-count bins on row order (caller sorts slice by confidence descending)."""
    m = len(top_slice)
    out = top_slice.copy().reset_index(drop=True)
    if m == 0:
        out["bin_id"] = np.array([], dtype=int)
        out["bin_label"] = np.array([], dtype=object)
        return out
    bin_idx = (np.arange(m, dtype=int) * n_bins) // m
    bin_idx = np.minimum(bin_idx, n_bins - 1)
    out["bin_id"] = bin_idx
    out["bin_label"] = [bin_spec[int(b)][0] for b in bin_idx]
    return out

EPS_LOG = 1e-12

# Paper run (reasoning_sampling_metadata.json)
TOP_POOL_FRAC = 0.5
N_BINS = 5

PROXIES = {
    "bottom10_mean_prob": "conf_bottom10_mean_prob",
    "full_trace_mean_logp": "conf_full_trace_mean_logp",
    "bottom20_mean_prob": "conf_bottom20_mean_prob",
}

BENCHMARKS = ["GSM8K", "MATH500", "SVAMP", "AQuA", "CommonsenseQA", "GPQA"]

HEADLINE_MODELS = [
    "DS-R1-7B",
    "DS-R1-1.5B",
    "Qwen2.5-7B",
    "Phi-4",
    "Phi-4-Reas.",
]


def tail_mean_prob(probs: List[float], frac_tail: float) -> float:
    """Mean of lowest frac_tail fraction of token probabilities (same spirit as compute_trace_confidence)."""
    if not probs:
        return float("nan")
    arr = np.asarray(probs, dtype=np.float64)
    if not np.any(np.isfinite(arr)):
        return float("nan")
    n_low = max(1, int(len(arr) * frac_tail))
    kth = min(n_low - 1, len(arr) - 1)
    lowest = np.partition(arr, kth)[:n_low]
    return float(np.mean(lowest))


def full_trace_mean_logp(probs: List[float], eps: float = EPS_LOG) -> float:
    """C_full_logp = mean_j log(max(p_j, eps))."""
    if not probs:
        return float("nan")
    arr = np.maximum(np.asarray(probs, dtype=np.float64), eps)
    return float(np.mean(np.log(arr)))


def traces_from_raw_records(raw: List[dict]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """One row per trace with three proxy columns (uses load_jsonl_raw-style rows)."""
    recs: List[Dict[str, Any]] = []
    bad_empty_probs = 0
    for row in raw:
        scores = row["scores"]
        probs_all = row["token_probs_all"]
        n = len(scores)
        idx = row.get("idx")
        for ti in range(n):
            probs = probs_all[ti] if ti < len(probs_all) else []
            if not probs or not isinstance(probs, list):
                bad_empty_probs += 1
                continue
            c10 = compute_trace_confidence(probs)
            c20 = tail_mean_prob(probs, 0.20)
            clf = full_trace_mean_logp(probs, EPS_LOG)
            if np.isnan(c10) and np.isnan(c20) and np.isnan(clf):
                continue
            recs.append(
                {
                    "idx": idx,
                    "trace_idx": ti,
                    "conf_bottom10_mean_prob": c10,
                    "conf_bottom20_mean_prob": c20,
                    "conf_full_trace_mean_logp": clf,
                    "correct": bool(scores[ti]) if ti < len(scores) else False,
                }
            )
    return pd.DataFrame(recs), {"bad_empty_probs": bad_empty_probs}


def slice_top_pool(df: pd.DataFrame, sort_col: str, frac: float) -> Tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0
    n = len(df)
    n_keep = max(1, int(np.floor(n * frac)))
    if n_keep >= n:
        n_keep = n
    ranked = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return ranked.iloc[:n_keep].copy(), n_keep


def load_judge_reasoning_map(path: str) -> Dict[Tuple[int, int], float]:
    """idx, trace_idx -> reasoning_score on 0-1 scale (converted to 0-100 for FRS)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[Tuple[int, int], float] = {}
    for s in data.get("judged_samples", []):
        if s.get("judge_ok") is False:
            continue
        rs = s.get("reasoning_score")
        if rs is None:
            continue
        v = float(rs)
        if not np.isfinite(v):
            continue
        rs100 = v * 100.0 if v <= 1.5 else v
        out[(int(s["idx"]), int(s["trace_idx"]))] = rs100
    return out


def frs_bin010(
    df_traces: pd.DataFrame,
    sort_col: str,
    judge_rs: Dict[Tuple[int, int], float],
    bin_cfg: BinConfig,
) -> Tuple[float, int, int]:
    """
    Mean reasoning score (0-100) for judged traces falling in bin 0-10 under this proxy.
    Returns (mean_frs, n_judged_in_bin0, n_pool_after_slice).
    """
    top_slice, _ = slice_top_pool(df_traces, sort_col, bin_cfg.top_pool_frac)
    spec = bin_cfg.bin_spec
    binned = assign_disjoint_bins(top_slice, bin_cfg.n_bins, spec)
    # First equal-count bin (highest confidence); do not match on string — paper checkpoints
    # use labels like "0-10" while BinConfig(5) may emit "0-20" style names.
    in_bin0 = binned[binned["bin_id"] == 0]
    scores: List[float] = []
    for _, r in in_bin0.iterrows():
        key = (int(r["idx"]), int(r["trace_idx"]))
        if key in judge_rs:
            scores.append(judge_rs[key])
    if not scores:
        return float("nan"), 0, len(top_slice)
    return float(np.mean(scores)), len(scores), len(top_slice)


def frs_cumulative_bins(
    df_traces: pd.DataFrame,
    sort_col: str,
    judge_rs: Dict[Tuple[int, int], float],
    bin_cfg: BinConfig,
    n_cumulative_bins: int,
) -> float:
    """Mean reasoning over judged traces in the first n_cumulative_bins bins (within top slice)."""
    top_slice, _ = slice_top_pool(df_traces, sort_col, bin_cfg.top_pool_frac)
    spec = bin_cfg.bin_spec
    binned = assign_disjoint_bins(top_slice, bin_cfg.n_bins, spec)
    allowed_ids = set(range(min(n_cumulative_bins, bin_cfg.n_bins)))
    scores: List[float] = []
    for _, r in binned.iterrows():
        if int(r["bin_id"]) not in allowed_ids:
            continue
        key = (int(r["idx"]), int(r["trace_idx"]))
        if key in judge_rs:
            scores.append(judge_rs[key])
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def rank_series(s: pd.Series, ascending: bool = False) -> pd.Series:
    return s.rank(ascending=ascending, method="min")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default=REPO_ROOT)
    ap.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(REPO_ROOT, "analysis_exports", "confidence_proxy_robustness"),
    )
    ap.add_argument(
        "--also-cumulative-k",
        action="store_true",
        help="Also write frs for cumulative top 20/30/40/50%% of slice (2-5 bins).",
    )
    ap.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="If >0, only process the first N (model,benchmark) pairs after sorting (smoke test).",
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

    rows_bench: List[Dict[str, Any]] = []
    coverage: List[Dict[str, Any]] = []
    missing: List[str] = []

    for (model, dataset) in pair_list:
        jpath = judge_path(repo_root, model, dataset)
        if not os.path.isfile(jpath):
            missing.append(f"{model}__{dataset} (no judge)")
            continue
        jsonl_path = jsonl_from_judge_metadata(repo_root, jpath) or jsonl_map.get((model, dataset))
        if not jsonl_path or not os.path.isfile(jsonl_path):
            missing.append(f"{model}__{dataset} (no JSONL)")
            continue

        raw = load_jsonl_raw(jsonl_path)
        df_tr, stats = traces_from_raw_records(raw)
        print(f"  {model} × {dataset}: {len(df_tr)} traces", flush=True)
        judge_rs = load_judge_reasoning_map(jpath)

        coverage.append(
            {
                "model": model,
                "benchmark": dataset,
                "n_traces_parsed": len(df_tr),
                "bad_empty_prob_lists_count": stats["bad_empty_probs"],
                "n_judged_with_score": len(judge_rs),
                "jsonl": os.path.relpath(jsonl_path, repo_root),
                "judge": os.path.relpath(jpath, repo_root),
            }
        )

        if df_tr.empty:
            continue

        for proxy_name, col in PROXIES.items():
            m, nj, nslice = frs_bin010(df_tr, col, judge_rs, bin_cfg)
            row = {
                "model": model,
                "benchmark": dataset,
                "proxy_name": proxy_name,
                "frs_k10_bin010": m,
                "n_judged_in_bin010": nj,
                "n_traces_in_top_pool_slice": nslice,
            }
            if args.also_cumulative_k:
                for k_bins, label in [(2, "k20"), (3, "k30"), (4, "k40"), (5, "k50")]:
                    row[f"frs_cumulative_{label}_of_slice"] = frs_cumulative_bins(
                        df_tr, col, judge_rs, bin_cfg, k_bins
                    )
            rows_bench.append(row)

    df_bench = pd.DataFrame(rows_bench)
    p_bench = os.path.join(out_dir, "proxy_benchmark_results_k10.csv")
    df_bench.to_csv(p_bench, index=False)
    pd.DataFrame(coverage).to_csv(os.path.join(out_dir, "proxy_data_coverage.csv"), index=False)

    # Model-level average FRS (mean over benchmarks), per proxy
    rows_rank: List[Dict[str, Any]] = []
    if not df_bench.empty:
        for (model, proxy), g in df_bench.groupby(["model", "proxy_name"]):
            v = g["frs_k10_bin010"].values
            finite = v[np.isfinite(v)]
            rows_rank.append(
                {
                    "model": model,
                    "proxy_name": proxy,
                    "frs_avg_k10": float(np.mean(finite)) if len(finite) else float("nan"),
                    "n_benchmarks": int(len(finite)),
                }
            )

    df_rank = pd.DataFrame(rows_rank)
    if not df_rank.empty:
        for proxy in df_rank["proxy_name"].unique():
            sub = df_rank[df_rank["proxy_name"] == proxy].copy()
            sub["frs_rank_k10"] = sub["frs_avg_k10"].rank(ascending=False, method="min")
            df_rank.loc[sub.index, "frs_rank_k10"] = sub["frs_rank_k10"]

    p_rank = os.path.join(out_dir, "proxy_rankings_k10.csv")
    df_rank.to_csv(p_rank, index=False)

    # Rank comparison wide
    if not df_rank.empty:
        wide = df_rank.pivot(index="model", columns="proxy_name", values="frs_rank_k10")
        wide = wide.rename(
            columns={
                "bottom10_mean_prob": "rank_bottom10_mean_prob",
                "full_trace_mean_logp": "rank_full_trace_mean_logp",
                "bottom20_mean_prob": "rank_bottom20_mean_prob",
            }
        )
        if "rank_bottom10_mean_prob" in wide.columns:
            r0 = wide["rank_bottom10_mean_prob"]
            if "rank_full_trace_mean_logp" in wide.columns:
                wide["delta_rank_full_logp_minus_current"] = wide["rank_full_trace_mean_logp"] - r0
            if "rank_bottom20_mean_prob" in wide.columns:
                wide["delta_rank_bottom20_minus_current"] = wide["rank_bottom20_mean_prob"] - r0
        wide.to_csv(os.path.join(out_dir, "proxy_rank_comparison.csv"))

    # Spearman between model-level frs_avg vectors
    summ: List[Dict[str, Any]] = []
    if not df_rank.empty:
        piv = df_rank.pivot(index="model", columns="proxy_name", values="frs_avg_k10")
        pairs = [
            ("bottom10_mean_prob", "full_trace_mean_logp"),
            ("bottom10_mean_prob", "bottom20_mean_prob"),
            ("full_trace_mean_logp", "bottom20_mean_prob"),
        ]
        for a, b in pairs:
            if a not in piv.columns or b not in piv.columns:
                continue
            xa = piv[a].values
            xb = piv[b].values
            m = np.isfinite(xa) & np.isfinite(xb)
            rho, p = spearmanr(xa[m], xb[m]) if m.sum() >= 3 else (float("nan"), float("nan"))
            summ.append(
                {
                    "proxy_a": a,
                    "proxy_b": b,
                    "spearman_rho": float(rho) if np.isfinite(rho) else float("nan"),
                    "p_value": float(p) if np.isfinite(p) else float("nan"),
                    "n_models": int(m.sum()),
                }
            )
    pd.DataFrame(summ).to_csv(os.path.join(out_dir, "proxy_spearman_summary.csv"), index=False)

    # Headline reversals
    hl_rows: List[Dict[str, Any]] = []
    if not df_rank.empty:
        for m in HEADLINE_MODELS:
            sub = df_rank[df_rank["model"] == m]
            if sub.empty:
                continue
            d = {"model": m}
            for _, r in sub.iterrows():
                d[f"frs_avg_{r['proxy_name']}"] = r["frs_avg_k10"]
                d[f"rank_{r['proxy_name']}"] = r["frs_rank_k10"]
            hl_rows.append(d)
    pd.DataFrame(hl_rows).to_csv(os.path.join(out_dir, "proxy_key_reversals_summary.csv"), index=False)

    meta = {
        "repo_root": repo_root,
        "top_pool_frac": TOP_POOL_FRAC,
        "n_bins": N_BINS,
        "bin_labels_emitted_by_BinConfig_n5": [b[0] for b in bin_cfg.bin_spec],
        "first_bin_selector": "bin_id == 0",
        "eps_log": EPS_LOG,
        "formulas": {
            "bottom10_mean_prob": "Same as topk_ablation.compute_trace_confidence: mean of lowest 10% of chosen-token probabilities (partition).",
            "full_trace_mean_logp": f"mean_j log(max(p_j, {EPS_LOG}))",
            "bottom20_mean_prob": "Mean of lowest 20% of chosen-token probabilities (same structure as bottom 10%).",
        },
        "frs_definition": (
            "Mean reasoning_score (0-100) over judged (idx,trace_idx) that fall in bin_id==0 "
            "(highest-confidence equal-count bin) after ranking all traces by the proxy, "
            "keeping top 50% by count, 5 equal-count bins. (Judge files may label this bin '0-10'; "
            "string labels from BinConfig can differ — we key on bin_id.)"
        ),
        "alignment_note": (
            "Judge JSON aligns to JSONL via metadata.input_file in build_downstream_parquets.jsonl_from_judge_metadata. "
            "idx and trace_idx are the join keys."
        ),
        "missing_or_skipped_pairs": missing,
        "n_benchmark_rows_written": len(df_bench),
        "also_cumulative_k": bool(args.also_cumulative_k),
        "outputs": {
            "proxy_benchmark_results_k10.csv": "Per model × benchmark × proxy.",
            "proxy_rankings_k10.csv": "Per model × proxy: avg FRS and rank.",
            "proxy_rank_comparison.csv": "Wide rank deltas.",
            "proxy_spearman_summary.csv": "Spearman between proxy-level model rankings.",
            "proxy_key_reversals_summary.csv": "Headline models only.",
        },
    }
    with open(os.path.join(out_dir, "proxy_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote outputs under {out_dir}/", flush=True)
    print(f"Benchmark rows: {len(df_bench)}  Models in rankings: {df_rank['model'].nunique() if len(df_rank) else 0}")
    if missing:
        print(f"Skipped/missing ({len(missing)}):", *missing[:8], "..." if len(missing) > 8 else "", sep="\n  ")


if __name__ == "__main__":
    main()
