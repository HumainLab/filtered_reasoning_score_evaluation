#!/usr/bin/env python3
"""
Compare self-consistency FRS at k=16 vs simulated k=8 from the **same** saved pass@16 JSONL
and **existing** judge checkpoints. No generation, no judge API.

k=16: all traces on each JSONL line (typically trace_idx 0..15).

k=8 (default subset **first8**): only trace_idx 0..7; vote-share confidence uses only those 8
normalized preds: conf_sc(t) = (# j in subset matching pred[t]) / 8.

Optional: ``--extra-subsets`` adds last8, even8, odd8 for variability summary.

Pipeline matches ``run_self_consistency_frs_proxy.py`` / ``run_confidence_proxy_robustness.py``:
rank by conf_sc ↓, top_pool_frac=0.5, n_bins=5, FRS@k10 = mean reasoning_score in bin_id==0.

Usage:
  python analysis/run_self_consistency_k_sensitivity.py --repo-root .
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

_CP_PATH = os.path.join(REPO_ROOT, "analysis", "run_confidence_proxy_robustness.py")
_spec = importlib.util.spec_from_file_location("_conf_proxy_k", _CP_PATH)
assert _spec and _spec.loader
_cp = importlib.util.module_from_spec(_spec)
sys.modules["_conf_proxy_k"] = _cp
_spec.loader.exec_module(_cp)

BinConfig = _cp.BinConfig
TOP_POOL_FRAC = _cp.TOP_POOL_FRAC
N_BINS = _cp.N_BINS
BENCHMARKS = _cp.BENCHMARKS

_SC_PATH = os.path.join(REPO_ROOT, "analysis", "run_self_consistency_frs_proxy.py")
_scs = importlib.util.spec_from_file_location("_sc_fr", _SC_PATH)
assert _scs and _scs.loader
_sc = importlib.util.module_from_spec(_scs)
sys.modules["_sc_fr_k"] = _sc
_scs.loader.exec_module(_sc)

normalize_pred_for_agreement = _sc.normalize_pred_for_agreement
vote_share_confidence = _sc.vote_share_confidence
load_judge_reasoning_map_robust = _sc.load_judge_reasoning_map_robust
frs_sc_bin010_detailed = _sc.frs_sc_bin010_detailed


def trace_indices_for_subset(name: str) -> List[int]:
    """Exactly 8 global trace indices into the pass@16 pool."""
    if name == "first8":
        return list(range(8))
    if name == "last8":
        return list(range(8, 16))
    if name == "even8":
        return [2 * i for i in range(8)]
    if name == "odd8":
        return [2 * i + 1 for i in range(8)]
    raise ValueError(f"unknown subset {name}")


def load_jsonl_self_consistency_subset(
    filepath: str,
    subset_name: Optional[str],
    k_pool: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    If subset_name is None and k_pool==16: all traces (same as full proxy).
    If subset_name is set (e.g. first8): only those trace_idx rows; conf computed within subset only.
    """
    recs: List[Dict[str, Any]] = []
    stats = {
        "jsonl_lines": 0,
        "skipped_bad_pred": 0,
        "skipped_len_mismatch": 0,
        "skipped_too_short_for_subset": 0,
    }
    if subset_name is not None:
        idx_sel = trace_indices_for_subset(subset_name)
        need_len = max(idx_sel) + 1
    else:
        idx_sel = None
        need_len = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stats["jsonl_lines"] += 1
            row = json.loads(line)
            pred = row.get("pred", [])
            scores = row.get("score", [])
            if not isinstance(pred, list) or not isinstance(scores, list):
                stats["skipped_bad_pred"] += 1
                continue
            nline = len(scores)
            if len(pred) != nline:
                stats["skipped_len_mismatch"] += 1
                continue

            if subset_name is not None:
                if nline < need_len:
                    stats["skipped_too_short_for_subset"] += 1
                    continue
                preds_norm = [normalize_pred_for_agreement(pred[i]) for i in idx_sel]
                k_sub = len(idx_sel)
                try:
                    pid_i = int(row["idx"])
                except (TypeError, ValueError, KeyError):
                    continue
                for local_t, global_t in enumerate(idx_sel):
                    conf = vote_share_confidence(preds_norm, local_t, k_sub)
                    cor = bool(scores[global_t]) if global_t < len(scores) else False
                    recs.append(
                        {
                            "idx": pid_i,
                            "trace_idx": global_t,
                            "conf_sc": conf,
                            "correct": cor,
                        }
                    )
            else:
                preds_norm = [normalize_pred_for_agreement(p) for p in pred]
                k = nline
                try:
                    pid_i = int(row["idx"])
                except (TypeError, ValueError, KeyError):
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

    return pd.DataFrame(recs), stats


def load_jsonl_bundle(
    filepath: str,
    include_extra_k8_subsets: bool,
) -> Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    One pass over JSONL: build ``full`` (all traces) and k=8 subset DataFrames.
    Avoids re-reading large files once per subset (major speedup vs separate loads).
    """
    recs_full: List[Dict[str, Any]] = []
    recs_first: List[Dict[str, Any]] = []
    recs_last: List[Dict[str, Any]] = []
    recs_even: List[Dict[str, Any]] = []
    recs_odd: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "jsonl_lines": 0,
        "skipped_bad_pred": 0,
        "skipped_len_mismatch": 0,
        "skipped_too_short_first8": 0,
        "skipped_too_short_last8": 0,
        "skipped_too_short_even8": 0,
        "skipped_too_short_odd8": 0,
    }
    idx_first = trace_indices_for_subset("first8")
    idx_last = trace_indices_for_subset("last8")
    idx_even = trace_indices_for_subset("even8")
    idx_odd = trace_indices_for_subset("odd8")
    need_first = max(idx_first) + 1
    need_last = max(idx_last) + 1
    need_even = max(idx_even) + 1
    need_odd = max(idx_odd) + 1

    def append_subset(
        recs: List[Dict[str, Any]],
        pred: List[Any],
        scores: List[Any],
        pid_i: int,
        idx_sel: List[int],
        stat_key: str,
        nline: int,
        need_len: int,
    ) -> None:
        if nline < need_len:
            stats.setdefault(stat_key, 0)
            stats[stat_key] += 1
            return
        preds_norm = [normalize_pred_for_agreement(pred[i]) for i in idx_sel]
        k_sub = len(idx_sel)
        for local_t, global_t in enumerate(idx_sel):
            conf = vote_share_confidence(preds_norm, local_t, k_sub)
            cor = bool(scores[global_t]) if global_t < len(scores) else False
            recs.append(
                {
                    "idx": pid_i,
                    "trace_idx": global_t,
                    "conf_sc": conf,
                    "correct": cor,
                }
            )

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stats["jsonl_lines"] += 1
            row = json.loads(line)
            pred = row.get("pred", [])
            scores = row.get("score", [])
            if not isinstance(pred, list) or not isinstance(scores, list):
                stats["skipped_bad_pred"] += 1
                continue
            nline = len(scores)
            if len(pred) != nline:
                stats["skipped_len_mismatch"] += 1
                continue
            try:
                pid_i = int(row["idx"])
            except (TypeError, ValueError, KeyError):
                continue

            preds_norm = [normalize_pred_for_agreement(p) for p in pred]
            k = nline
            for t in range(k):
                conf = vote_share_confidence(preds_norm, t, k)
                cor = bool(scores[t]) if t < len(scores) else False
                recs_full.append(
                    {
                        "idx": pid_i,
                        "trace_idx": t,
                        "conf_sc": conf,
                        "correct": cor,
                    }
                )

            append_subset(recs_first, pred, scores, pid_i, idx_first, "skipped_too_short_first8", nline, need_first)
            if include_extra_k8_subsets:
                append_subset(recs_last, pred, scores, pid_i, idx_last, "skipped_too_short_last8", nline, need_last)
                append_subset(recs_even, pred, scores, pid_i, idx_even, "skipped_too_short_even8", nline, need_even)
                append_subset(recs_odd, pred, scores, pid_i, idx_odd, "skipped_too_short_odd8", nline, need_odd)

    out: Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]] = {
        "full": (pd.DataFrame(recs_full), dict(stats)),
        "first8": (pd.DataFrame(recs_first), dict(stats)),
    }
    if include_extra_k8_subsets:
        out["last8"] = (pd.DataFrame(recs_last), dict(stats))
        out["even8"] = (pd.DataFrame(recs_even), dict(stats))
        out["odd8"] = (pd.DataFrame(recs_odd), dict(stats))
    return out


def run_pair(
    model: str,
    dataset: str,
    bundle: Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]],
    jpath: str,
    bin_cfg: BinConfig,
    subset_name: Optional[str],
    k_label: str,
    judge_cache: Optional[Dict[str, Dict[Tuple[int, int], float]]] = None,
) -> Dict[str, Any]:
    sk = "full" if subset_name is None else subset_name
    if sk not in bundle:
        raise KeyError(f"missing subset {sk} in bundle (available: {list(bundle.keys())})")
    df_tr, st = bundle[sk]
    if judge_cache is not None:
        if jpath not in judge_cache:
            judge_cache[jpath] = load_judge_reasoning_map_robust(jpath)
        judge_rs = judge_cache[jpath]
    else:
        judge_rs = load_judge_reasoning_map_robust(jpath)
    mean_frs, n_judged, n_pool, n_bin0 = frs_sc_bin010_detailed(df_tr, judge_rs, bin_cfg, "conf_sc")
    cov = (n_judged / n_bin0) if n_bin0 > 0 else float("nan")
    return {
        "model": model,
        "benchmark": dataset,
        "k": k_label,
        "subset": subset_name or "full16",
        "sc_frs_k10": mean_frs,
        "n_judged_traces_used": n_judged,
        "n_total_topbin_traces": n_bin0,
        "judged_coverage_fraction": cov,
        "n_traces_in_top_pool_slice": n_pool,
        "n_total_traces_available": len(df_tr),
        "n_judged_traces_available": len(judge_rs),
        "line_stats": st,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default=REPO_ROOT)
    ap.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(REPO_ROOT, "analysis_exports", "self_consistency_k_sensitivity"),
    )
    ap.add_argument(
        "--extra-subsets",
        action="store_true",
        help="Also compute last8, even8, odd8 and write subset variation summary.",
    )
    ap.add_argument("--max-pairs", type=int, default=0)
    args = ap.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    bin_cfg = BinConfig(top_pool_frac=TOP_POOL_FRAC, n_bins=N_BINS)
    jsonl_map = discover_jsonl_groups(repo_root)
    pair_list = sorted((k for k in jsonl_map.keys() if k[1] in BENCHMARKS))
    if args.max_pairs > 0:
        pair_list = pair_list[: args.max_pairs]

    bench_rows: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []
    subset_rankings: List[Dict[str, Any]] = []
    judge_cache: Dict[str, Dict[Tuple[int, int], float]] = {}

    pairs_valid: List[Tuple[str, str, str, str]] = []
    for model, dataset in pair_list:
        jpath = judge_path(repo_root, model, dataset)
        if not os.path.isfile(jpath):
            continue
        jsonl_path = jsonl_from_judge_metadata(repo_root, jpath) or jsonl_map.get((model, dataset))
        if not jsonl_path or not os.path.isfile(jsonl_path):
            continue
        pairs_valid.append((model, dataset, jsonl_path, jpath))

    bundles_by_abs: Dict[str, Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]]] = {}
    seen_abs: set = set()
    for _m, _d, jsonl_path, _j in pairs_valid:
        ap = os.path.abspath(jsonl_path)
        if ap in seen_abs:
            continue
        seen_abs.add(ap)
        bundles_by_abs[ap] = load_jsonl_bundle(jsonl_path, args.extra_subsets)

    # k=16 full
    for model, dataset, jsonl_path, jpath in pairs_valid:
        bundle = bundles_by_abs[os.path.abspath(jsonl_path)]
        r = run_pair(model, dataset, bundle, jpath, bin_cfg, None, "16", judge_cache)
        bench_rows.append(
            {
                "model": r["model"],
                "benchmark": r["benchmark"],
                "k": "16",
                "sc_frs_k10": r["sc_frs_k10"],
                "n_judged_traces_used": r["n_judged_traces_used"],
                "judged_coverage_fraction": r["judged_coverage_fraction"],
            }
        )
        coverage_rows.append(
            {
                "model": model,
                "benchmark": dataset,
                "k": "16",
                "n_total_traces_available": r["n_total_traces_available"],
                "n_total_traces_used": r["n_total_traces_available"],
                "n_judged_traces_available": r["n_judged_traces_available"],
                "n_judged_topbin_traces": r["n_judged_traces_used"],
                "judged_coverage_fraction": r["judged_coverage_fraction"],
                "notes": "full pass@16 pool",
            }
        )

    # k=8 first8 (primary)
    for model, dataset, jsonl_path, jpath in pairs_valid:
        bundle = bundles_by_abs[os.path.abspath(jsonl_path)]
        r = run_pair(model, dataset, bundle, jpath, bin_cfg, "first8", "8", judge_cache)
        bench_rows.append(
            {
                "model": r["model"],
                "benchmark": r["benchmark"],
                "k": "8",
                "sc_frs_k10": r["sc_frs_k10"],
                "n_judged_traces_used": r["n_judged_traces_used"],
                "judged_coverage_fraction": r["judged_coverage_fraction"],
            }
        )
        coverage_rows.append(
            {
                "model": model,
                "benchmark": dataset,
                "k": "8",
                "n_total_traces_available": r["n_total_traces_available"],
                "n_total_traces_used": r["n_total_traces_available"],
                "n_judged_traces_available": r["n_judged_traces_available"],
                "n_judged_topbin_traces": r["n_judged_traces_used"],
                "judged_coverage_fraction": r["judged_coverage_fraction"],
                "notes": "first8: trace_idx 0-7 only; conf from 8-way vote share",
            }
        )

    df_bench = pd.DataFrame(bench_rows)

    # Model-level rankings for k=8 and k=16
    model_rank_rows: List[Dict[str, Any]] = []
    for k_val in ["16", "8"]:
        sub = df_bench[df_bench["k"] == k_val]
        for model, g in sub.groupby("model"):
            v = g["sc_frs_k10"].values
            finite = v[np.isfinite(v)]
            cov = g["judged_coverage_fraction"].values
            cov_f = cov[np.isfinite(cov)]
            model_rank_rows.append(
                {
                    "model": model,
                    "k": k_val,
                    "sc_frs_avg_k10": float(np.mean(finite)) if len(finite) else float("nan"),
                    "mean_judged_coverage_fraction": float(np.mean(cov_f)) if len(cov_f) else float("nan"),
                }
            )

    df_mr = pd.DataFrame(model_rank_rows)
    if not df_mr.empty:
        df_mr["sc_frs_rank"] = np.nan
        for k_val in ["16", "8"]:
            m = df_mr["k"] == k_val
            sub = df_mr[m].copy()
            sub["sc_frs_rank"] = sub["sc_frs_avg_k10"].rank(ascending=False, method="min")
            df_mr.loc[sub.index, "sc_frs_rank"] = sub["sc_frs_rank"]

    df_mr.to_csv(os.path.join(out_dir, "sc_k_sensitivity_model_rankings.csv"), index=False)
    df_bench.to_csv(os.path.join(out_dir, "sc_k_sensitivity_benchmark_results.csv"), index=False)
    pd.DataFrame(coverage_rows).to_csv(os.path.join(out_dir, "sc_k_sensitivity_coverage.csv"), index=False)

    # k8 vs k16 comparison
    cmp_rows: List[Dict[str, Any]] = []
    models = sorted(df_mr["model"].unique())
    for m in models:
        r16 = df_mr[(df_mr["model"] == m) & (df_mr["k"] == "16")]
        r8 = df_mr[(df_mr["model"] == m) & (df_mr["k"] == "8")]
        rank16 = float(r16["sc_frs_rank"].values[0]) if len(r16) else float("nan")
        rank8 = float(r8["sc_frs_rank"].values[0]) if len(r8) else float("nan")
        avg16 = float(r16["sc_frs_avg_k10"].values[0]) if len(r16) else float("nan")
        avg8 = float(r8["sc_frs_avg_k10"].values[0]) if len(r8) else float("nan")
        dr = rank8 - rank16 if np.isfinite(rank8) and np.isfinite(rank16) else float("nan")
        cmp_rows.append(
            {
                "model": m,
                "sc_rank_k8": rank8,
                "sc_rank_k16": rank16,
                "rank_difference": dr,
                "absolute_rank_difference": abs(dr) if np.isfinite(dr) else float("nan"),
                "sc_frs_avg_k8": avg8,
                "sc_frs_avg_k16": avg16,
            }
        )
    pd.DataFrame(cmp_rows).to_csv(os.path.join(out_dir, "sc_k8_vs_k16_rank_comparison.csv"), index=False)

    # Key models (user list includes Qwen3-4B, not Phi-4-Reas.)
    key_names = [
        "DS-R1-7B",
        "DS-R1-1.5B",
        "Qwen2.5-7B",
        "Phi-4",
        "Qwen3-4B",
    ]
    hl: List[Dict[str, Any]] = []
    for m in key_names:
        row = next((r for r in cmp_rows if r["model"] == m), None)
        if row:
            hl.append(row)
    pd.DataFrame(hl).to_csv(os.path.join(out_dir, "sc_k8_vs_k16_key_models.csv"), index=False)

    # Correlation: same models, aligned order (alphabetical by model name)
    xs8: List[float] = []
    xs16: List[float] = []
    r8l: List[float] = []
    r16l: List[float] = []
    for m in models:
        r16 = df_mr[(df_mr["model"] == m) & (df_mr["k"] == "16")]
        r8 = df_mr[(df_mr["model"] == m) & (df_mr["k"] == "8")]
        if not len(r16) or not len(r8):
            continue
        a16 = float(r16["sc_frs_avg_k10"].values[0])
        a8 = float(r8["sc_frs_avg_k10"].values[0])
        rk16 = float(r16["sc_frs_rank"].values[0])
        rk8 = float(r8["sc_frs_rank"].values[0])
        if not (
            np.isfinite(a16)
            and np.isfinite(a8)
            and np.isfinite(rk16)
            and np.isfinite(rk8)
        ):
            continue
        xs16.append(a16)
        xs8.append(a8)
        r16l.append(rk16)
        r8l.append(rk8)

    n_corr = len(xs16)
    if n_corr >= 3:
        sp_s, sp_p = spearmanr(r16l, r8l)
        pe_r, pe_p = pearsonr(xs16, xs8)
    else:
        sp_s = sp_p = pe_r = pe_p = float("nan")

    corr_summary = {
        "spearman_rho_model_rank_k16_vs_k8": float(sp_s) if np.isfinite(sp_s) else float("nan"),
        "spearman_p_value": float(sp_p) if np.isfinite(sp_p) else float("nan"),
        "pearson_r_model_level_sc_frs_avg_k16_vs_k8": float(pe_r) if np.isfinite(pe_r) else float("nan"),
        "pearson_p_value": float(pe_p) if np.isfinite(pe_p) else float("nan"),
        "n_models": n_corr,
        "notes": (
            "Macro mean sc_frs_k10 over benchmarks per model; Spearman on discrete ranks. "
            "Sparse judged-in-bin0 coverage can move ranks; not absolute score replication."
        ),
    }
    pd.DataFrame([corr_summary]).to_csv(
        os.path.join(out_dir, "sc_k8_vs_k16_correlation_summary.csv"), index=False
    )

    # Optional extra subsets (k=8 only)
    if args.extra_subsets:
        for sn in ["last8", "even8", "odd8"]:
            for model, dataset, jsonl_path, jpath in pairs_valid:
                bundle = bundles_by_abs[os.path.abspath(jsonl_path)]
                r = run_pair(model, dataset, bundle, jpath, bin_cfg, sn, "8", judge_cache)
                subset_rankings.append(
                    {
                        "subset": sn,
                        "model": model,
                        "benchmark": dataset,
                        "sc_frs_k10": r["sc_frs_k10"],
                    }
                )
        df_ss = pd.DataFrame(subset_rankings)
        if not df_ss.empty:
            rows_var: List[Dict[str, Any]] = []
            for subset in ["first8", "last8", "even8", "odd8"]:
                sub = df_ss[df_ss["subset"] == subset] if subset != "first8" else None
                # first8 from df_bench
                if subset == "first8":
                    b = df_bench[df_bench["k"] == "8"]
                    for model, g in b.groupby("model"):
                        v = g["sc_frs_k10"].values
                        finite = v[np.isfinite(v)]
                        rows_var.append(
                            {
                                "subset": subset,
                                "model": model,
                                "sc_frs_avg_k10": float(np.mean(finite)) if len(finite) else float("nan"),
                            }
                        )
                else:
                    ssub = df_ss[df_ss["subset"] == subset]
                    for model, g in ssub.groupby("model"):
                        v = g["sc_frs_k10"].values
                        finite = v[np.isfinite(v)]
                        rows_var.append(
                            {
                                "subset": subset,
                                "model": model,
                                "sc_frs_avg_k10": float(np.mean(finite)) if len(finite) else float("nan"),
                            }
                        )
            dfv = pd.DataFrame(rows_var)
            if not dfv.empty:
                for subset in dfv["subset"].unique():
                    sub = dfv[dfv["subset"] == subset].copy()
                    sub["sc_frs_rank"] = sub["sc_frs_avg_k10"].rank(ascending=False, method="min")
                    dfv.loc[sub.index, "sc_frs_rank"] = sub["sc_frs_rank"]
                dfv.to_csv(os.path.join(out_dir, "sc_k8_subset_variation_summary.csv"), index=False)

    meta = {
        "k16": "All traces on JSONL line; conf_sc = vote share / len(pred) (typically 16).",
        "k8_primary": "first8: trace_idx 0..7 only; vote share among preds at those indices / 8.",
        "normalization": "str(pred).strip(); same as run_self_consistency_frs_proxy.py",
        "join_keys": "(idx, trace_idx) to judge checkpoints",
        "top_pool_frac": TOP_POOL_FRAC,
        "n_bins": N_BINS,
        "caveats": [
            "Judged traces are fixed subset; sparse in bin0.",
            "k=8 uses only half the samples — ranking may differ from k=16.",
            "Purpose: rank stability check, not score replication.",
        ],
        "n_pairs": len(pairs_valid),
        "jsonl_load": "One pass per unique JSONL via load_jsonl_bundle (full + k=8 subsets in one read).",
    }
    with open(os.path.join(out_dir, "sc_k_sensitivity_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {out_dir}/", flush=True)
    if np.isfinite(sp_s):
        print(f"Spearman rank k16 vs k8: {sp_s:.4f}  n={n_corr}", flush=True)
    else:
        print(f"Spearman rank k16 vs k8: nan  n={n_corr}", flush=True)
    if np.isfinite(pe_r):
        print(f"Pearson score k16 vs k8: {pe_r:.4f}", flush=True)
    else:
        print("Pearson score k16 vs k8: nan", flush=True)


if __name__ == "__main__":
    main()
