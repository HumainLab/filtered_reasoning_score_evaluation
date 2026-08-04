#!/usr/bin/env python3
"""
FRS sensitivity to k: top-10% pooled trace set overlap (k=16 baseline vs subsampled k).

Pools traces across all problems, ranks by precomputed confidence (mean of lowest 10%
token probs). No new judge calls; overlaps judge-scored traces from checkpoints.

Data layout (this repo):
  - Pass@16 JSONL under ``source_pass16_jsonl_by_model*`` (see ``topk_ablation.build_file_map``).
  - Judge scores: ``reasoning_confidence_bins_results/judging_checkpoints/judged_<Model>__<Bench>.json``,
    0–10% bin samples only for partial FRS.
  - FRS@10% (k=16): ``reasoning_by_confidence_bin.csv``, ``bin_label == "0-10"``.

CSV columns:
  - ``snr_k8`` / ``mean_snr_k8`` hold SNR on the *subsampled* trace pool; use the ``k`` column
    for the actual subsample size (8 or 4), not the column name.

Run (unbuffered logs): ``python -u k_sensitivity_analysis.py --data_dir .``
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Reuse proven loaders from topk_ablation
from topk_ablation import (
    build_file_map,
    compute_trace_confidence,
    load_jsonl_raw,
)

LOG = logging.getLogger("k_sensitivity_analysis")

# Paper regime taxonomy (for plot coloring)
RELIABLE = {"DS-R1-1.5B", "DS-R1-7B", "Qwen3-4B"}
DECEPTIVE = {"Qwen2.5-7B", "Phi-4", "Phi-4-Reas."}
UNRELIABLE = {"LLaMA-3.1-8B", "Qwen2.5-Math", "Gemma-7B"}

REGIME_COLOR = {
    "reliable": "#2ca02c",
    "deceptive": "#d62728",
    "unreliable": "#888888",
    "unknown": "#bbbbbb",
}


def regime_for_model(model: str) -> str:
    if model in RELIABLE:
        return "reliable"
    if model in DECEPTIVE:
        return "deceptive"
    if model in UNRELIABLE:
        return "unreliable"
    return "unknown"


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(ch)


def top_frac_count(n: int, frac: float) -> int:
    """Number of traces in the top `frac` of the pool (at least 1 when n > 0)."""
    if n == 0:
        return 0
    return max(1, int(np.ceil(n * frac)))


def snr_cohen_style(correct: np.ndarray, incorrect: np.ndarray) -> float:
    """SNR = (mean_c - mean_i) / sqrt(0.5 * (var_c + var_i))."""
    if len(correct) == 0 or len(incorrect) == 0:
        return float("nan")
    m_c = float(np.mean(correct))
    m_i = float(np.mean(incorrect))
    v_c = float(np.var(correct, ddof=0))
    v_i = float(np.var(incorrect, ddof=0))
    denom = np.sqrt(0.5 * (v_c + v_i))
    if denom == 0.0 or not np.isfinite(denom):
        return float("nan")
    return (m_c - m_i) / denom


def build_problems_from_raw(
    raw_records: List[Dict[str, Any]],
    expected_k: int = 16,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Each problem: list of trace dicts with trace_id, confidence, correct, trace_idx, problem_id.
    Returns (problems, warnings).
    """
    warnings: List[str] = []
    problems: List[Dict[str, Any]] = []
    for row in raw_records:
        pid = row.get("idx")
        scores = row["scores"]
        token_probs_all = row["token_probs_all"]
        n = len(scores)
        if n != expected_k:
            warnings.append(
                f"problem idx={pid}: expected {expected_k} traces, got {n} (skipped)"
            )
            continue
        traces: List[Dict[str, Any]] = []
        for trace_idx in range(n):
            probs = token_probs_all[trace_idx] if trace_idx < len(token_probs_all) else []
            conf = compute_trace_confidence(probs)
            if np.isnan(conf):
                warnings.append(f"problem idx={pid} trace {trace_idx}: nan confidence (skipped trace)")
                continue
            correct = bool(scores[trace_idx]) if trace_idx < len(scores) else False
            tid = f"{pid}::{trace_idx}"
            traces.append(
                {
                    "trace_id": tid,
                    "problem_id": pid,
                    "trace_idx": trace_idx,
                    "confidence": float(conf),
                    "correct": correct,
                }
            )
        if len(traces) != expected_k:
            warnings.append(
                f"problem idx={pid}: after nan filter {len(traces)} != {expected_k} (skipped problem)"
            )
            continue
        problems.append({"problem_id": pid, "traces": traces})
    return problems, warnings


def pool_traces(problems: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in problems:
        out.extend(p["traces"])
    return out


def top_frac_set(
    traces: Sequence[Dict[str, Any]],
    frac: float,
) -> Set[str]:
    """Trace ids in the top `frac` by confidence (global pool)."""
    if not traces:
        return set()
    # Sort: confidence desc, then stable tie-break
    sorted_t = sorted(
        traces,
        key=lambda t: (-t["confidence"], t["problem_id"], t["trace_idx"]),
    )
    k = top_frac_count(len(traces), frac)
    return {sorted_t[i]["trace_id"] for i in range(k)}


def subsample_problems(
    problems: Sequence[Dict[str, Any]],
    k_sub: int,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """Each problem: k_sub traces chosen uniformly without replacement."""
    sub: List[Dict[str, Any]] = []
    for p in problems:
        tr = p["traces"]
        n = len(tr)
        if n < k_sub:
            continue
        idxs = rng.choice(n, size=k_sub, replace=False)
        chosen = [tr[int(i)] for i in sorted(idxs)]
        sub.append({"problem_id": p["problem_id"], "traces": chosen})
    return sub


def load_frs_at_10_csv(path: str) -> Dict[Tuple[str, str], float]:
    """mean_reasoning_score for bin 0-10 → scale to 0–100."""
    df = pd.read_csv(path)
    sub = df[df["bin_label"].astype(str) == "0-10"]
    out: Dict[Tuple[str, str], float] = {}
    for _, r in sub.iterrows():
        m = str(r["model"])
        d = str(r["dataset"])
        v = float(r["mean_reasoning_score"])
        out[(m, d)] = v * 100.0
    return out


def discover_judge_files(judging_dir: str) -> Dict[Tuple[str, str], str]:
    """Map (display_model, display_dataset) -> path to judged_*.json."""
    pattern = os.path.join(judging_dir, "judged_*.json")
    files = glob.glob(pattern)
    out: Dict[Tuple[str, str], str] = {}
    for fp in files:
        base = os.path.basename(fp)
        m = re.match(r"^judged_(.+)__(.+)\.json$", base)
        if not m:
            continue
        out[(m.group(1), m.group(2))] = fp
    return out


def load_judge_scores_top_bin(
    path: str,
) -> Dict[str, float]:
    """trace_id -> reasoning_score (0–1) for samples judged in bin 0-10 only."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, float] = {}
    for s in data.get("judged_samples", []):
        if str(s.get("bin_label")) != "0-10":
            continue
        idx = s.get("idx")
        tix = s.get("trace_idx")
        tid = f"{idx}::{tix}"
        out[tid] = float(s.get("reasoning_score", float("nan")))
    return out


def run_analysis(
    data_dir: str,
    judging_dir: str,
    frs_csv: str,
    k_values: Sequence[int],
    num_resamples: int,
    seed: int,
    output_dir: str,
    figures_dir: str,
    top_frac: float = 0.10,
    expected_k: int = 16,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    file_map = build_file_map(data_dir)
    judge_map = discover_judge_files(judging_dir)
    frs_k16_map = load_frs_at_10_csv(frs_csv)

    overlap_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for (model, benchmark), jsonl_path in sorted(file_map.items()):
        judge_path = judge_map.get((model, benchmark))
        judge_top: Dict[str, float] = {}
        if judge_path:
            judge_top = load_judge_scores_top_bin(judge_path)
        else:
            LOG.warning("No judge checkpoint for %s × %s — partial FRS disabled", model, benchmark)

        raw = load_jsonl_raw(jsonl_path)
        problems, warns = build_problems_from_raw(raw, expected_k=expected_k)
        for w in warns[:20]:
            LOG.warning("%s | %s", model, w)
        if len(warns) > 20:
            LOG.warning("%s | ... %d more problem warnings", model, len(warns) - 20)

        if not problems:
            LOG.warning("Skip %s × %s — no valid problems", model, benchmark)
            continue

        LOG.info("Pair: %s × %s — %d problems, %s", model, benchmark, len(problems), jsonl_path)

        pool = pool_traces(problems)
        n_traces = len(pool)
        n_prob = len(problems)
        baseline_top = top_frac_set(pool, top_frac)
        n_base = len(baseline_top)

        conf_all = np.array([t["confidence"] for t in pool], dtype=np.float64)
        corr_mask = np.array([t["correct"] for t in pool], dtype=bool)
        snr_k16 = snr_cohen_style(conf_all[corr_mask], conf_all[~corr_mask])
        frs_paper = frs_k16_map.get((model, benchmark), float("nan"))

        for kv in k_values:
            if kv > expected_k:
                LOG.warning("%s × %s: k=%d > expected_k=%d, skip", model, benchmark, kv, expected_k)
                continue
            recalls: List[float] = []
            precs: List[float] = []
            jacs: List[float] = []
            snr_ks: List[float] = []
            flips: List[bool] = []
            scored_ns: List[int] = []
            partial_frs: List[float] = []

            print(
                f"\n=== Model: {model} | Benchmark: {benchmark} | k_sub={kv} ===\n"
                f"Total problems: {n_prob} | Total traces (k=16): {n_traces}\n"
                f"Baseline top-{int(top_frac*100)}% size: {n_base} traces"
            )

            for res_id in range(num_resamples):
                pair_h = int(hashlib.md5(f"{model}::{benchmark}".encode()).hexdigest()[:8], 16)
                rng = np.random.default_rng(seed + res_id * 10007 + kv * 131 + (pair_h % 100000))
                sub_problems = subsample_problems(problems, kv, rng)
                if not sub_problems:
                    continue
                sub_pool = pool_traces(sub_problems)
                sub_top = top_frac_set(sub_pool, top_frac)
                inter = baseline_top & sub_top
                uni = baseline_top | sub_top
                rec = len(inter) / len(sub_top) if sub_top else float("nan")
                prec = len(inter) / len(baseline_top) if baseline_top else float("nan")
                jac = len(inter) / len(uni) if uni else float("nan")

                conf_s = np.array([t["confidence"] for t in sub_pool], dtype=np.float64)
                cm = np.array([t["correct"] for t in sub_pool], dtype=bool)
                snr_sub = snr_cohen_style(conf_s[cm], conf_s[~cm])

                flip = False
                if np.isfinite(snr_k16) and np.isfinite(snr_sub):
                    flip = np.sign(snr_k16) != np.sign(snr_sub)

                # Partial FRS: judged 0–10 bin traces ∩ subsample top set
                scored_in = 0
                rs_vals: List[float] = []
                for tid in sub_top:
                    if tid in judge_top:
                        scored_in += 1
                        rs_vals.append(judge_top[tid] * 100.0)
                pfrs = float(np.mean(rs_vals)) if rs_vals else float("nan")

                overlap_rows.append(
                    {
                        "model": model,
                        "benchmark": benchmark,
                        "resample_id": res_id,
                        "k": kv,
                        "recall": rec,
                        "precision": prec,
                        "jaccard": jac,
                        "snr_k16": snr_k16,
                        "snr_k8": snr_sub,
                        "regime_flip": flip,
                        "num_scored_in_overlap": scored_in,
                        "partial_frs_estimate": pfrs,
                    }
                )

                recalls.append(rec)
                precs.append(prec)
                jacs.append(jac)
                snr_ks.append(snr_sub)
                flips.append(flip)
                scored_ns.append(scored_in)
                partial_frs.append(pfrs)

            if not recalls:
                continue

            mr, sr = float(np.mean(recalls)), float(np.std(recalls, ddof=0))
            mj, sj = float(np.mean(jacs)), float(np.std(jacs, ddof=0))
            mp, sp = float(np.mean(precs)), float(np.std(precs, ddof=0))
            ms_snr = float(np.mean(snr_ks))
            std_snr = float(np.std(snr_ks, ddof=0))
            flip_rate = sum(flips) / len(flips)
            m_scored = float(np.mean(scored_ns))
            std_scored = float(np.std(scored_ns, ddof=0))
            m_pfrs = float(np.nanmean(partial_frs))

            summary_rows.append(
                {
                    "model": model,
                    "benchmark": benchmark,
                    "k": kv,
                    "mean_recall": mr,
                    "std_recall": sr,
                    "mean_jaccard": mj,
                    "std_jaccard": sj,
                    "mean_precision": mp,
                    "std_precision": sp,
                    "mean_snr_k8": ms_snr,
                    "std_snr_k8": std_snr,
                    "snr_k16": snr_k16,
                    "regime_flip_rate": flip_rate,
                    "mean_scored_overlap": m_scored,
                    "std_scored_overlap": std_scored,
                    "mean_partial_frs": m_pfrs,
                    "frs_k16": frs_paper,
                }
            )

            pfrs_std = float(np.nanstd(partial_frs, ddof=0))
            frs_tail = (
                f"{frs_paper:.2f}"
                if np.isfinite(frs_paper)
                else "n/a"
            )
            print(
                f"\nSubsample k={kv} ({num_resamples} resamples):\n"
                f"  Avg recall:    {mr:.4f} ± {sr:.4f}\n"
                f"  Avg jaccard:   {mj:.4f} ± {sj:.4f}\n"
                f"  Avg precision: {mp:.4f} ± {sp:.4f}\n"
                f"\n  SNR (k=16): {snr_k16:.4f} | SNR (k={kv} avg): {ms_snr:.4f} ± {std_snr:.4f} | "
                f"Regime flips: {sum(flips)}/{len(flips)}\n"
                f"\n  Scored traces in k={kv} top-{int(top_frac*100)}% "
                f"(from 0–10% bin judge set): {m_scored:.1f} ± {std_scored:.1f} of {len(judge_top)}\n"
                f"  Partial FRS estimate (k={kv}): {m_pfrs:.2f} ± {pfrs_std:.2f} vs FRS@10% (k=16): {frs_tail}"
            )

    df_overlap = pd.DataFrame(overlap_rows)
    df_summary = pd.DataFrame(summary_rows)

    overlap_path = os.path.join(output_dir, "k_sensitivity_overlap.csv")
    summary_path = os.path.join(output_dir, "k_sensitivity_summary.csv")
    df_overlap.to_csv(overlap_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    LOG.info("Wrote %s (%d rows)", overlap_path, len(df_overlap))
    LOG.info("Wrote %s (%d rows)", summary_path, len(df_summary))

    # Aggregate per k
    agg_rows: List[Dict[str, Any]] = []
    for kv in k_values:
        sub = df_summary[df_summary["k"] == kv]
        if sub.empty:
            continue
        gmr = float(sub["mean_recall"].mean())
        gsr = float(sub["mean_recall"].std(ddof=0))
        gmj = float(sub["mean_jaccard"].mean())
        # regime flips: from overlap df
        sub_o = df_overlap[df_overlap["k"] == kv]
        flip_rate = float(sub_o["regime_flip"].mean()) if len(sub_o) else float("nan")
        # mean FRS shift: pairs with enough scored overlap
        thresh = 20
        eligible = sub[sub["mean_scored_overlap"] >= thresh]
        shifts = eligible["mean_partial_frs"] - eligible["frs_k16"]
        mean_shift = float(np.nanmean(shifts)) if len(eligible) else np.nan

        agg_rows.append(
            {
                "k": kv,
                "grand_mean_recall": gmr,
                "grand_std_recall": gsr,
                "grand_mean_jaccard": gmj,
                "regime_flip_rate": flip_rate,
                "mean_frs_shift": mean_shift,
                "n_pairs": len(sub),
                "n_pairs_frs_shift": len(eligible),
            }
        )

    df_agg = pd.DataFrame(agg_rows)
    agg_path = os.path.join(output_dir, "k_sensitivity_aggregate.csv")
    df_agg.to_csv(agg_path, index=False)
    LOG.info("Wrote %s", agg_path)

    # --- Figures ---
    _set_pub_style()
    primary_k = 8 if 8 in k_values else (k_values[0] if k_values else 8)

    # Heatmap: mean recall
    sub_h = df_summary[df_summary["k"] == primary_k]
    if not sub_h.empty:
        pivot = sub_h.pivot(index="model", columns="benchmark", values="mean_recall")
        pivot = pivot.reindex(sorted(pivot.index))
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            ax=ax,
            cbar_kws={"label": "Mean recall"},
        )
        ax.set_title(f"Top-10% overlap recall (k={primary_k} vs k=16 baseline)")
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "overlap_heatmap.png"), dpi=200)
        plt.close(fig)
        LOG.info("Wrote overlap heatmap (k=%s)", primary_k)

    # SNR scatter
    sub_s = df_summary[df_summary["k"] == primary_k]
    if not sub_s.empty:
        fig, ax = plt.subplots(figsize=(7, 7))
        xs = sub_s["snr_k16"].values
        ys = sub_s["mean_snr_k8"].values
        yerr = sub_s["std_snr_k8"].values
        colors = [REGIME_COLOR[regime_for_model(m)] for m in sub_s["model"].values]
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            fmt="none",
            ecolor="#333333",
            alpha=0.5,
            zorder=1,
        )
        ax.scatter(xs, ys, c=colors, s=55, zorder=2, edgecolors="black", linewidths=0.4)
        lim = max(
            float(np.nanmax(np.abs(np.concatenate([xs, ys])))) if len(xs) else 1.0,
            1.0,
        )
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.35, label="y = x")
        ax.axhline(0.0, color="gray", lw=0.8, alpha=0.5)
        ax.axvline(0.0, color="gray", lw=0.8, alpha=0.5)
        ax.set_xlabel("SNR at k=16 (full pool)")
        ax.set_ylabel(f"SNR at k={primary_k} (mean ± std over resamples)")
        ax.set_title("Regime stability: SNR under subsampling")
        from matplotlib.patches import Patch

        leg = [
            Patch(facecolor=REGIME_COLOR["reliable"], label="reliable"),
            Patch(facecolor=REGIME_COLOR["deceptive"], label="deceptive"),
            Patch(facecolor=REGIME_COLOR["unreliable"], label="unreliable"),
        ]
        ax.legend(handles=leg, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "regime_stability.png"), dpi=200)
        plt.close(fig)
        LOG.info("Wrote regime_stability.png (k=%s)", primary_k)

    # FRS shift histogram (pairs with mean_scored_overlap >= 20)
    for kv in k_values:
        sub = df_summary[(df_summary["k"] == kv) & (df_summary["mean_scored_overlap"] >= 20)]
        if sub.empty:
            LOG.warning("No pairs with scored overlap ≥20 for k=%s — skip FRS shift histogram", kv)
            continue
        shifts = sub["mean_partial_frs"] - sub["frs_k16"]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(shifts.dropna(), bins=20, color="steelblue", edgecolor="white")
        ax.axvline(0.0, color="black", ls="--", lw=1)
        ax.set_xlabel(f"Partial FRS (k={kv}) − FRS@10% (k=16)")
        ax.set_ylabel("Count (model × benchmark)")
        ax.set_title(f"Distribution of FRS shifts (pairs with mean scored overlap ≥ 20, k={kv})")
        fig.tight_layout()
        multi = len(k_values) > 1
        suf = f"_k{kv}" if multi else ""
        out_path = os.path.join(figures_dir, f"frs_shift_distribution{suf}.png")
        fig.savefig(out_path, dpi=200)
        if multi and kv == primary_k:
            fig.savefig(os.path.join(figures_dir, "frs_shift_distribution.png"), dpi=200)
        plt.close(fig)
        LOG.info("Wrote FRS shift histogram for k=%s", kv)


def _set_pub_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 120,
        }
    )


def main() -> None:
    p = argparse.ArgumentParser(description="k-sensitivity top-10% overlap for FRS")
    p.add_argument(
        "--data_dir",
        type=str,
        default=".",
        help="Project root containing source_pass16_jsonl_by_model* trees",
    )
    p.add_argument(
        "--judging_dir",
        type=str,
        default="outputs/reasoning_confidence_bins_results/judging_checkpoints",
        help="Directory with judged_*.json checkpoints",
    )
    p.add_argument(
        "--frs_csv",
        type=str,
        default="outputs/reasoning_confidence_bins_results/reasoning_by_confidence_bin.csv",
        help="CSV with FRS@10% (bin 0-10 mean_reasoning_score)",
    )
    p.add_argument("--k_values", type=int, nargs="+", default=[8, 4])
    p.add_argument("--num_resamples", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="outputs/results")
    p.add_argument("--figures_dir", type=str, default="outputs/figures")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    setup_logging(args.verbose)
    data_dir = os.path.abspath(args.data_dir)
    judging_dir = os.path.join(data_dir, args.judging_dir) if not os.path.isabs(args.judging_dir) else args.judging_dir
    frs_csv = os.path.join(data_dir, args.frs_csv) if not os.path.isabs(args.frs_csv) else args.frs_csv
    out_dir = os.path.join(data_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    fig_dir = os.path.join(data_dir, args.figures_dir) if not os.path.isabs(args.figures_dir) else args.figures_dir

    LOG.info("data_dir=%s", data_dir)
    run_analysis(
        data_dir=data_dir,
        judging_dir=judging_dir,
        frs_csv=frs_csv,
        k_values=args.k_values,
        num_resamples=args.num_resamples,
        seed=args.seed,
        output_dir=out_dir,
        figures_dir=fig_dir,
    )


if __name__ == "__main__":
    main()
