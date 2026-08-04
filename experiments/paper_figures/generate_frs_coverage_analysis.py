#!/usr/bin/env python3
"""
Coverage-aware FRS / accuracy analysis (threshold sweeps & equal-coverage tables).

Uses per-sample:
  - answer_confidence from *_filtered_p1_only.jsonl
  - fused_scores.overall (reasoning) and correctness from *_filtered_p1_only_results.json

Pooled curves (default): for each model, merge all dataset splits that have judge results,
then sweep confidence thresholds τ ∈ [0,1]: keep samples with conf >= τ (and conf not null).
  - coverage(τ) = (# kept) / N_pool
  - FRS(τ)      = mean(fused overall) on kept × 100
  - Acc(τ)      = mean(correct) on kept × 100

Equal-coverage table: sort pooled samples by confidence descending; take the top ⌊c·N⌋
points for c ∈ {0.40, 0.60, 0.80} (same coverage across models; no abstention advantage).

If results JSON is incomplete vs JSONL (common for partial math500 runs), only merged idx
are used and a warning is printed.

Outputs (under experiments/paper_figures/judge_validation/):
  - frs_coverage_curves_long.csv
  - frs_coverage_curves_pooled.png
  - frs_at_fixed_coverage.csv
  - frs_coverage_metadata.json
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover
    raise SystemExit("matplotlib required for plotting") from e


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "experiments" / "paper_figures" / "judge_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILTERED_DIRS: Dict[str, Path] = {
    "gsm8k": REPO / "results" / "filtered_cot" / "gsm8k",
    "math500": REPO / "results" / "filtered_cot" / "math500",
    "svamp": REPO / "results" / "filtered_cot" / "svamp",
    "aqua": REPO / "results" / "filtered_cot" / "aqua",
    "gpqa": REPO / "results" / "filtered_cot" / "gpqa",
    "commonsense_qa": REPO / "results" / "filtered_cot" / "commonsense",
}

STEM_TO_SHORT = {
    "DeepSeek_R1_Distill_Qwen_1.5B": "DS-R1-1.5B",
    "DeepSeek_R1_Distill_Qwen_7B": "DS-R1-7B",
    "Llama_3.1_8B_Instruct": "LLaMA-3.1-8B",
    "Qwen2.5_7B_Instruct": "Qwen2.5-7B",
    "Qwen2.5_Math_7B": "Qwen2.5-Math-7B",
    "gemma_7b": "Gemma-7B",
    "phi_4": "Phi-4",
    "Phi_4_reasoning": "Phi-4-Reasoning",
    "Qwen3_4B_Thinking_2507": "Qwen3-4B",
}

SHORT_ORDER = [
    "DS-R1-1.5B",
    "DS-R1-7B",
    "LLaMA-3.1-8B",
    "Qwen2.5-7B",
    "Qwen2.5-Math-7B",
    "Gemma-7B",
    "Phi-4",
    "Phi-4-Reasoning",
    "Qwen3-4B",
]

COLORS = {
    "DS-R1-1.5B": "#1b9e77",
    "DS-R1-7B": "#d95f02",
    "LLaMA-3.1-8B": "#7570b3",
    "Qwen2.5-7B": "#e7298a",
    "Qwen2.5-Math-7B": "#66a61e",
    "Gemma-7B": "#a6761d",
    "Phi-4": "#e41a1c",
    "Phi-4-Reasoning": "#984ea3",
    "Qwen3-4B": "#377eb8",
}


def _score_to_bool(score) -> Optional[bool]:
    if score is None:
        return None
    if isinstance(score, list):
        return bool(score[0]) if score else None
    return bool(score)


def _correct_from_results(rec: dict) -> Optional[bool]:
    ev = rec.get("evidence") or {}
    if "final_correct" in ev and ev["final_correct"] is not None:
        return bool(ev["final_correct"])
    oc = rec.get("original_correct")
    if oc is not None:
        return bool(oc)
    return None


def load_jsonl_by_idx(path: Path) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[int(r["idx"])] = r
    return out


def merge_combo(
    jsonl_path: Path, results_path: Path
) -> Tuple[List[dict], int, int, List[str]]:
    """
    Returns (records, n_jsonl, n_results_matched, warnings).
    Each record: conf, fused_overall, correct, idx, dataset (caller adds dataset).
    """
    warns: List[str] = []
    if not jsonl_path.is_file() or not results_path.is_file():
        return [], 0, 0, ["missing_file"]

    jl = load_jsonl_by_idx(jsonl_path)
    with results_path.open(encoding="utf-8") as f:
        res_list = json.load(f).get("results", [])
    by_idx = {int(r["idx"]): r for r in res_list}

    n_j = len(jl)
    n_r = len(by_idx)
    if n_j != n_r:
        warns.append(f"jsonl_n={n_j} vs results_n={n_r}")

    merged: List[dict] = []
    for idx in jl:
        if idx not in by_idx:
            continue
        j = jl[idx]
        b = by_idx[idx]
        conf = j.get("answer_confidence")
        fs = (b.get("fused_scores") or {}).get("overall")
        if fs is None:
            continue
        cor = _correct_from_results(b)
        if cor is None:
            sc = j.get("score")
            cor = _score_to_bool(sc)
        merged.append(
            {
                "idx": idx,
                "conf": float(conf) if conf is not None else np.nan,
                "fused": float(fs),
                "correct": bool(cor) if cor is not None else np.nan,
            }
        )

    if n_j and len(merged) / n_j < 0.95:
        warns.append(f"only {len(merged)}/{n_j} idx merged (<95%)")

    return merged, n_j, len(merged), warns


def discover_stems() -> List[str]:
    stems: set = set()
    for d in FILTERED_DIRS.values():
        if not d.is_dir():
            continue
        for p in d.glob("*_filtered_p1_only.jsonl"):
            stem = p.name.replace("_filtered_p1_only.jsonl", "")
            stems.add(stem)
    return sorted(stems)


def pool_model_records(stem: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Returns conf, fused, correct (float nan for missing), warn list."""
    warns: List[str] = []
    rows: List[Tuple[float, float, float]] = []
    for ds, folder in FILTERED_DIRS.items():
        jl = folder / f"{stem}_filtered_p1_only.jsonl"
        rs = folder / "results" / f"{stem}_filtered_p1_only_results.json"
        recs, n_j, n_m, w = merge_combo(jl, rs)
        if w:
            warns.append(f"{ds}::{stem}: " + "; ".join(w))
        for r in recs:
            rows.append((r["conf"], r["fused"], r["correct"]))

    if not rows:
        return np.array([]), np.array([]), np.array([]), warns

    arr = np.array(rows, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2], warns


def curve_threshold_sweep(
    conf: np.ndarray, fused: np.ndarray, correct: np.ndarray, n_tau: int = 401
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep τ from 0 to 1. Keep samples with conf >= tau and finite conf.
    Returns tau_grid, coverage, frs_pct, acc_pct (nan where no samples).
    """
    m = np.isfinite(conf)
    conf = conf[m]
    fused = fused[m]
    correct = correct[m]
    n = conf.size
    if n == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    taus = np.linspace(0.0, 1.0, n_tau)
    cov = np.zeros_like(taus)
    frs = np.zeros_like(taus)
    acc = np.zeros_like(taus)

    for i, tau in enumerate(taus):
        sel = conf >= tau
        k = int(np.sum(sel))
        cov[i] = k / n
        if k == 0:
            frs[i] = np.nan
            acc[i] = np.nan
        else:
            frs[i] = 100.0 * float(np.mean(fused[sel]))
            acc[i] = 100.0 * float(np.mean(correct[sel]))

    return taus, cov, frs, acc


def equal_coverage_metrics(
    conf: np.ndarray, fused: np.ndarray, correct: np.ndarray, levels: List[float]
) -> Dict[float, dict]:
    """Sort by conf descending; take top fraction for each coverage level."""
    m = np.isfinite(conf)
    conf = conf[m]
    fused = fused[m]
    correct = correct[m]
    n = conf.size
    out: Dict[float, dict] = {}
    if n == 0:
        for c in levels:
            out[c] = {"n": 0, "frs": np.nan, "acc": np.nan, "tau_implied": np.nan}
        return out

    order = np.argsort(-conf)
    conf = conf[order]
    fused = fused[order]
    correct = correct[order]

    for c in levels:
        k = max(1, int(np.floor(c * n)))
        sl = slice(0, k)
        tau_min = float(conf[k - 1]) if k else np.nan
        out[c] = {
            "n": k,
            "frs": 100.0 * float(np.mean(fused[sl])),
            "acc": 100.0 * float(np.nanmean(correct[sl])),
            "tau_implied": tau_min,
            "coverage_empirical": k / n,
        }
    return out


def plot_pooled(coverage_by_model: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))

    for short in SHORT_ORDER:
        if short not in coverage_by_model:
            continue
        cov, frs, acc = coverage_by_model[short]
        c = COLORS.get(short, "#333333")
        axes[0].plot(cov, frs, label=short, color=c, linewidth=2)
        axes[1].plot(cov, acc, label=short, color=c, linewidth=2)

    for ax, title, ylab in [
        (axes[0], "FRS vs coverage (pooled across datasets)", "FRS (mean fused × 100)"),
        (axes[1], "Accuracy vs coverage (same retained set)", "Accuracy (%)"),
    ]:
        ax.set_xlabel("Coverage (fraction with conf ≥ τ)", fontweight="bold")
        ax.set_ylabel(ylab, fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.set_xlim(0, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="lower left", ncol=2)

    fig.suptitle(
        "Confidence threshold sweep — samples with judge scores only\n"
        "(higher τ → fewer kept → lower coverage; incomplete splits are omitted from pool)",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    out = OUT_DIR / "frs_coverage_curves_pooled.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    stems = discover_stems()
    meta: dict = {"stems": stems, "datasets": list(FILTERED_DIRS.keys()), "combos": []}

    coverage_by_model: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    table_rows: List[dict] = []
    long_rows: List[dict] = []

    for stem in stems:
        short = STEM_TO_SHORT.get(stem, stem)
        conf, fused, correct, warns = pool_model_records(stem)
        meta["combos"].append({"stem": stem, "short": short, "n_pooled": int(conf.size), "warns": warns})

        if conf.size == 0:
            continue

        taus, cov, frs, acc = curve_threshold_sweep(conf, fused, correct)
        coverage_by_model[short] = (cov, frs, acc)

        for tau, c, f, a in zip(taus, cov, frs, acc):
            long_rows.append(
                {
                    "model": short,
                    "tau": round(float(tau), 5),
                    "coverage": round(float(c), 6),
                    "frs_percent": float(f) if np.isfinite(f) else "",
                    "accuracy_percent": float(a) if np.isfinite(a) else "",
                }
            )

        eq = equal_coverage_metrics(conf, fused, correct, [0.40, 0.60, 0.80])
        for c_level, d in eq.items():
            table_rows.append(
                {
                    "model": short,
                    "target_coverage": c_level,
                    "n_kept": d["n"],
                    "coverage_empirical": round(d["coverage_empirical"], 4),
                    "frs_percent": round(d["frs"], 2) if np.isfinite(d["frs"]) else "",
                    "accuracy_percent": round(d["acc"], 2) if np.isfinite(d["acc"]) else "",
                    "min_conf_in_set": round(d["tau_implied"], 6) if np.isfinite(d["tau_implied"]) else "",
                }
            )

    # CSV long
    long_path = OUT_DIR / "frs_coverage_curves_long.csv"
    with long_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model", "tau", "coverage", "frs_percent", "accuracy_percent"],
        )
        w.writeheader()
        w.writerows(long_rows)
    print(f"Wrote {long_path} ({len(long_rows)} rows)")

    # Table
    tab_path = OUT_DIR / "frs_at_fixed_coverage.csv"
    with tab_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "target_coverage",
                "n_kept",
                "coverage_empirical",
                "frs_percent",
                "accuracy_percent",
                "min_conf_in_set",
            ],
        )
        w.writeheader()
        w.writerows(table_rows)
    print(f"Wrote {tab_path} ({len(table_rows)} rows)")

    meta_path = OUT_DIR / "frs_coverage_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {meta_path}")

    if coverage_by_model:
        plot_pooled(coverage_by_model)


if __name__ == "__main__":
    main()
