"""
Top-K Confidence Ablation for FRS Paper
========================================
For each model-benchmark pair, computes per-trace confidence (mean of lowest 10%
token probabilities), then reports accuracy at different confidence percentile
thresholds (top 10%, 20%, 30%, 50%, 70%, 100%).

If confidence is informative, accuracy should degrade monotonically as you include
less confident traces.

Usage:
    python topk_ablation.py --data_root /path/to/threshold

    python topk_ablation.py --data_root /path/to/threshold \\
        --multiseed 1 2 4 8 --n-seeds 30 --seed-base 42

    # Verbose: DEBUG logs; full per-pair lines during multiseed (or use --multiseed-verbose-pairs)
    python topk_ablation.py --data_root /path/to/threshold --log-file topk_run.log -v

Output:
    - CSV with all results (full pass@16 pool)
    - Per-benchmark plots; `topk_ablation_all_datasets_accuracy.png` (all 6 panels)
    - With --multiseed: long + aggregated CSVs; per-k figures with mean ± std
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────
PERCENTILES = [10, 20, 30, 50, 70, 100]  # top K% most confident
LOW_PROB_CUTOFF = 0.10  # bottom 10% of token probs for confidence

LOG = logging.getLogger("topk_ablation")


def setup_logging(log_file: Optional[str], verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt, datefmt))
        root.addHandler(fh)
        LOG.info("Logging also to %s", os.path.abspath(log_file))


def _short_path(path: str, max_len: int = 96) -> str:
    p = path.replace("\\", "/")
    if len(p) <= max_len:
        return p
    return "…" + p[-(max_len - 1) :]


# Panel order for multi-dataset figures (math benchmarks first)
DATASET_PANEL_ORDER = [
    "GSM8K",
    "MATH500",
    "SVAMP",
    "AQuA",
    "CommonsenseQA",
    "GPQA",
]


def compute_trace_confidence(token_probs: list) -> float:
    """Confidence = mean probability of the lowest 10% of tokens in a trace."""
    if not token_probs or len(token_probs) == 0:
        return float("nan")
    arr = np.asarray(token_probs, dtype=np.float64)
    n_low = max(1, int(len(arr) * LOW_PROB_CUTOFF))
    kth = min(n_low - 1, len(arr) - 1)
    lowest = np.partition(arr, kth)[:n_low]
    return float(np.mean(lowest))


def extract_model_dataset(filepath: str) -> Tuple[str, str]:
    """Extract model name and dataset from filepath."""
    parts = filepath.replace("\\", "/").split("/")

    model = None
    for i, p in enumerate(parts):
        if p.startswith("source_pass16"):
            if i + 1 < len(parts):
                model = parts[i + 1]
            break

    fname = parts[-1]
    dataset_match = re.match(r"^([a-z_0-9]+?)__", fname)
    dataset = dataset_match.group(1) if dataset_match else "unknown"

    dataset_map = {
        "gsm8k": "GSM8K",
        "math500": "MATH500",
        "svamp": "SVAMP",
        "aqua": "AQuA",
        "gpqa": "GPQA",
        "commonsense_qa": "CommonsenseQA",
    }
    dataset = dataset_map.get(dataset, dataset)

    model_map = {
        "DeepSeek_R1_Distill_Qwen_1.5B": "DS-R1-1.5B",
        "DeepSeek_R1_Distill_Qwen_7B": "DS-R1-7B",
        "Llama_3.1_8B_Instruct": "LLaMA-3.1-8B",
        "Qwen2.5_7B_Instruct": "Qwen2.5-7B",
        "Qwen2.5_Math_7B": "Qwen2.5-Math",
        "gemma_7b": "Gemma-7B",
        "phi_4": "Phi-4",
        "Phi_4_reasoning": "Phi-4-Reas.",
        "Qwen3_4B_Thinking_2507": "Qwen3-4B",
    }
    model = model_map.get(model, model)

    return model, dataset


def build_file_map(data_root: str) -> Dict[Tuple[str, str], str]:
    patterns = [
        os.path.join(data_root, "data/pass16_sample*", "**", "*.jsonl"),
    ]
    all_files: List[str] = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern, recursive=True))

    file_map: Dict[Tuple[str, str], str] = {}
    for fp in all_files:
        model, dataset = extract_model_dataset(fp)
        key = (model, dataset)
        if key not in file_map or "_processed" in fp:
            file_map[key] = fp
    LOG.info(
        "Glob: %d JSONL paths scanned → %d unique (model, dataset) pairs",
        len(all_files),
        len(file_map),
    )
    return file_map


def load_jsonl_raw(filepath: str) -> List[Dict[str, Any]]:
    """Per-question rows with scores and token probs (no subsampling)."""
    records: List[Dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            scores = row.get("score", [])
            token_probs_all = row.get("chosen_token_probs_per_path", {}).get("epoch_0", [])
            if not isinstance(token_probs_all, list):
                continue
            if len(scores) == 0:
                continue
            records.append(
                {
                    "idx": row.get("idx"),
                    "scores": scores,
                    "token_probs_all": token_probs_all,
                }
            )
    return records


def records_to_trace_list(
    raw_records: List[Dict[str, Any]],
    subsample_k: Optional[int],
    rng: Optional[np.random.Generator],
) -> List[Dict[str, Any]]:
    """Flatten to per-trace records with confidence; optionally subsample traces per question."""
    out: List[Dict[str, Any]] = []
    for row in raw_records:
        scores = row["scores"]
        token_probs_all = row["token_probs_all"]
        if not isinstance(token_probs_all, list):
            continue
        n_traces = len(scores)
        if n_traces == 0:
            continue
        if subsample_k is None:
            idxs = np.arange(n_traces)
        else:
            assert rng is not None
            k = min(subsample_k, n_traces)
            if k < 1:
                continue
            idxs = np.sort(rng.choice(n_traces, size=k, replace=False))
        for i in idxs:
            ii = int(i)
            correct = bool(scores[ii]) if ii < len(scores) else False
            probs = token_probs_all[ii] if ii < len(token_probs_all) else []
            conf = compute_trace_confidence(probs)
            if np.isnan(conf):
                continue
            out.append(
                {
                    "idx": row.get("idx"),
                    "trace_idx": ii,
                    "correct": correct,
                    "confidence": conf,
                }
            )
    return out


def load_and_process_jsonl(filepath: str) -> list:
    """Load a JSONL file and compute per-trace confidence + correctness (full trace pool)."""
    raw = load_jsonl_raw(filepath)
    return records_to_trace_list(raw, None, None)


def compute_ablation_rows(model: str, dataset: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if df.empty:
        return rows
    n_total = len(df)
    for k in PERCENTILES:
        if k == 100:
            subset = df
        else:
            threshold = np.percentile(df["confidence"], 100 - k)
            subset = df[df["confidence"] >= threshold]

        n_subset = len(subset)
        acc = subset["correct"].mean() * 100 if n_subset > 0 else float("nan")
        mean_conf = subset["confidence"].mean() if n_subset > 0 else float("nan")

        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "top_k_pct": k,
                "accuracy": round(acc, 2),
                "n_traces": n_subset,
                "n_total": n_total,
                "mean_confidence": round(mean_conf, 4),
            }
        )
    return rows


def run_ablation_with_cache(
    file_map: Dict[Tuple[str, str], str],
    raw_cache: Dict[str, List[Dict[str, Any]]],
    subsample_k: Optional[int],
    rng: Optional[np.random.Generator],
    *,
    phase_label: str = "top-k",
    log_each_pair: bool = True,
) -> pd.DataFrame:
    """Compute top-K% ablation using pre-loaded raw JSONL."""
    results: List[Dict[str, Any]] = []
    items = sorted(file_map.items())
    n_items = len(items)
    for i, ((model, dataset), filepath) in enumerate(items, 1):
        t_pair = time.perf_counter()
        raw = raw_cache[filepath]
        traces = records_to_trace_list(raw, subsample_k, rng)
        if not traces:
            LOG.warning(
                "[%s] [%d/%d] %s × %s — no valid traces, skip (%s)",
                phase_label,
                i,
                n_items,
                model,
                dataset,
                _short_path(filepath),
            )
            continue
        df = pd.DataFrame(traces)
        results.extend(compute_ablation_rows(model, dataset, df))
        dt = time.perf_counter() - t_pair
        msg = "[%s] [%d/%d] %s × %s — pooled traces=%d, %.2fs" % (
            phase_label,
            i,
            n_items,
            model,
            dataset,
            len(df),
            dt,
        )
        if log_each_pair:
            LOG.info(msg)
        else:
            LOG.debug(msg)
    return pd.DataFrame(results)


def run_subsample_multiseed_from_cache(
    file_map: Dict[Tuple[str, str], str],
    raw_cache: Dict[str, List[Dict[str, Any]]],
    k_list: List[int],
    seeds: List[int],
    *,
    log_each_pair: bool = False,
) -> pd.DataFrame:
    """Random k traces per question; one global RNG per (subsample_k, seed)."""
    parts: List[pd.DataFrame] = []
    total_runs = len(k_list) * len(seeds)
    run_i = 0
    t_multi = time.perf_counter()
    for subsample_k in k_list:
        for seed in seeds:
            run_i += 1
            LOG.info(
                "Multiseed [%d/%d] starting — subsample_k=%d seed=%d",
                run_i,
                total_runs,
                subsample_k,
                seed,
            )
            t_run = time.perf_counter()
            rng = np.random.default_rng(seed)
            df = run_ablation_with_cache(
                file_map,
                raw_cache,
                subsample_k,
                rng,
                phase_label=f"k{subsample_k}/s{seed}",
                log_each_pair=log_each_pair,
            )
            df["subsample_k"] = subsample_k
            df["seed"] = seed
            parts.append(df)
            LOG.info(
                "Multiseed [%d/%d] finished — k=%d seed=%d in %.1fs (%d result rows)",
                run_i,
                total_runs,
                subsample_k,
                seed,
                time.perf_counter() - t_run,
                len(df),
            )
    LOG.info("All multiseed runs completed in %.1fs", time.perf_counter() - t_multi)
    return pd.concat(parts, ignore_index=True)


def aggregate_multiseed(long_df: pd.DataFrame) -> pd.DataFrame:
    g = long_df.groupby(["model", "dataset", "top_k_pct", "subsample_k"], as_index=False)
    agg = g.agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        n_seeds=("seed", "nunique"),
    )
    agg["accuracy_std"] = agg["accuracy_std"].fillna(0.0)
    return agg


def run_pipeline(
    data_root: str,
    multiseed_k: Optional[List[int]] = None,
    n_seeds: int = 30,
    seed_base: int = 42,
    *,
    multiseed_log_each_pair: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Returns (full_results, multiseed_long_or_none, multiseed_agg_or_none)."""
    t0 = time.perf_counter()
    file_map = build_file_map(data_root)

    unique_paths = sorted(set(file_map.values()))
    LOG.info(
        "Caching %d unique JSONL files (one read per file; shared by all phases)",
        len(unique_paths),
    )
    raw_cache: Dict[str, List[Dict[str, Any]]] = {}
    for fi, fp in enumerate(unique_paths, 1):
        t_load = time.perf_counter()
        raw_cache[fp] = load_jsonl_raw(fp)
        n_q = len(raw_cache[fp])
        LOG.info(
            "  Raw load [%d/%d] %s — %d questions, %.2fs",
            fi,
            len(unique_paths),
            _short_path(fp),
            n_q,
            time.perf_counter() - t_load,
        )
    LOG.info("Raw cache complete in %.1fs", time.perf_counter() - t0)

    t_ab = time.perf_counter()
    LOG.info("Phase: full pass@16 pool — top-k ablation (all traces per question, INFO per pair)")
    results_full = run_ablation_with_cache(
        file_map,
        raw_cache,
        None,
        None,
        phase_label="full",
        log_each_pair=True,
    )
    LOG.info(
        "Phase: full pool finished in %.1fs (%d CSV rows)",
        time.perf_counter() - t_ab,
        len(results_full),
    )

    long_df: Optional[pd.DataFrame] = None
    agg_df: Optional[pd.DataFrame] = None
    if multiseed_k:
        seeds = [seed_base + i for i in range(n_seeds)]
        LOG.info(
            "Phase: subsample multiseed — k in %s, %d seeds (%d … %d); "
            "per-pair detail at DEBUG unless --verbose multiseed pairs",
            multiseed_k,
            n_seeds,
            seeds[0],
            seeds[-1],
        )
        t_ms = time.perf_counter()
        long_df = run_subsample_multiseed_from_cache(
            file_map,
            raw_cache,
            multiseed_k,
            seeds,
            log_each_pair=multiseed_log_each_pair,
        )
        LOG.info("Multiseed long table: %d rows in %.1fs", len(long_df), time.perf_counter() - t_ms)
        t_agg = time.perf_counter()
        agg_df = aggregate_multiseed(long_df)
        LOG.info(
            "Aggregated multiseed: %d rows in %.2fs",
            len(agg_df),
            time.perf_counter() - t_agg,
        )

    LOG.info("Pipeline total elapsed: %.1fs", time.perf_counter() - t0)
    return results_full, long_df, agg_df


def check_monotonicity(results_df: pd.DataFrame) -> pd.DataFrame:
    """Check if accuracy is monotonically decreasing as K increases (more traces included)."""
    mono_results = []
    for (model, dataset), group in results_df.groupby(["model", "dataset"]):
        group = group.sort_values("top_k_pct")
        accs = group["accuracy"].values
        violations = sum(1 for i in range(1, len(accs)) if accs[i] > accs[i - 1])
        is_monotonic = violations == 0
        spread = accs[0] - accs[-1]
        mono_results.append(
            {
                "model": model,
                "dataset": dataset,
                "monotonic": is_monotonic,
                "violations": violations,
                "spread_pp": round(spread, 2),
                "top10_acc": accs[0],
                "full_acc": accs[-1],
            }
        )
    return pd.DataFrame(mono_results)


def _model_colors(models: List[str]) -> Dict[str, Any]:
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(len(models), 1))
    return {m: cmap(i) for i, m in enumerate(models)}


def _ordered_datasets(datasets_in_df: List[str]) -> List[str]:
    have = set(datasets_in_df)
    ordered = [d for d in DATASET_PANEL_ORDER if d in have]
    for d in sorted(datasets_in_df):
        if d not in ordered:
            ordered.append(d)
    return ordered


def plot_all_datasets_accuracy_grid(
    results_df: pd.DataFrame,
    output_path: str,
    *,
    suptitle: str = "Accuracy @ Top-K% Confidence Pool (all traces per question)",
    accuracy_col: str = "accuracy",
    std_col: Optional[str] = None,
    x_axis_ascending: bool = True,
) -> None:
    """2×3 grid: accuracy vs top-K% for every benchmark (same content as six per-dataset PNGs)."""
    datasets = _ordered_datasets(list(results_df["dataset"].unique()))
    models = sorted(results_df["model"].unique())
    colors = _model_colors(models)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)
    axes_flat = axes.ravel()

    for ax, dataset in zip(axes_flat, datasets):
        ds_data = results_df[results_df["dataset"] == dataset]
        for model in models:
            md = ds_data[ds_data["model"] == model].sort_values("top_k_pct")
            if md.empty:
                continue
            x = md["top_k_pct"].values
            y = md[accuracy_col].values
            ax.plot(x, y, marker="o", label=model, color=colors[model], linewidth=1.8)
            if std_col and std_col in md.columns:
                err = md[std_col].values
                ax.fill_between(
                    x,
                    np.maximum(0, y - err),
                    np.minimum(100, y + err),
                    color=colors[model],
                    alpha=0.12,
                    linewidth=0,
                )

        ax.set_title(dataset, fontsize=11)
        ax.set_xlabel("Top-K% confidence pool (10% → 100%)", fontsize=9)
        ax.set_ylabel("Accuracy (%)", fontsize=9)
        ax.set_xticks(PERCENTILES)
        ax.set_xticklabels([f"{k}%" for k in PERCENTILES])
        ax.grid(True, alpha=0.3)
        if x_axis_ascending:
            ax.set_xlim(10, 100)
        else:
            ax.invert_xaxis()
        ax.set_ylim(0, 100)

    # Hide any unused axes (if <6 datasets)
    for j in range(len(datasets), len(axes_flat)):
        axes_flat[j].set_visible(False)

    leg_handles = [
        mlines.Line2D([0], [0], color=colors[m], marker="o", linewidth=1.8, label=m)
        for m in models
    ]
    fig.legend(
        leg_handles,
        models,
        loc="lower center",
        ncol=min(5, len(models)),
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(suptitle, fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved figure: %s", output_path)


def plot_results(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create per-benchmark plots of accuracy vs. top-K% for each model."""
    LOG.info("Plotting per-dataset line charts (%d datasets)...", len(results_df["dataset"].unique()))
    datasets = sorted(results_df["dataset"].unique())
    models = sorted(results_df["model"].unique())
    colors = _model_colors(models)

    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(8, 5))
        ds_data = results_df[results_df["dataset"] == dataset]

        for model in models:
            md = ds_data[ds_data["model"] == model].sort_values("top_k_pct")
            if md.empty:
                continue
            ax.plot(
                md["top_k_pct"],
                md["accuracy"],
                marker="o",
                label=model,
                color=colors[model],
                linewidth=2,
            )

        ax.set_xlabel("Top K% Most Confident Traces Included", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title(f"{dataset}: Accuracy vs. Confidence Percentile", fontsize=14)
        ax.set_xticks(PERCENTILES)
        ax.set_xticklabels([f"{k}%" for k in PERCENTILES])
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

        plt.tight_layout()
        outpath = os.path.join(output_dir, f"topk_ablation_{dataset}.png")
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        LOG.info("Saved per-dataset plot: %s", outpath)

    LOG.info("Plotting spread heatmap...")
    mono_df = check_monotonicity(results_df)
    pivot = mono_df.pivot(index="model", columns="dataset", values="spread_pp")

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black" if abs(val) < 15 else "white",
                )

    ax.set_title("Accuracy Spread: Top 10% - Full Set (pp)", fontsize=14)
    plt.colorbar(im, ax=ax, label="Spread (pp)")
    plt.tight_layout()
    outpath = os.path.join(output_dir, "topk_ablation_spread_heatmap.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved heatmap: %s", outpath)


def print_summary(results_df: pd.DataFrame) -> None:
    """Log a formatted summary table."""
    LOG.info("%s", "\n" + "=" * 80)
    LOG.info("TOP-K CONFIDENCE ABLATION RESULTS")
    LOG.info("%s", "=" * 80)

    datasets = sorted(results_df["dataset"].unique())
    for dataset in datasets:
        LOG.info("%s", "\n" + "─" * 60)
        LOG.info("  %s", dataset)
        LOG.info("%s", "─" * 60)

        ds_data = results_df[results_df["dataset"] == dataset]
        pivot = ds_data.pivot(index="model", columns="top_k_pct", values="accuracy")
        pivot = pivot[PERCENTILES]
        pivot.columns = [f"Top {k}%" for k in PERCENTILES]
        LOG.info("\n%s", pivot.to_string())

    mono_df = check_monotonicity(results_df)
    n_mono = mono_df["monotonic"].sum()
    n_total = len(mono_df)
    LOG.info("%s", "\n" + "=" * 60)
    LOG.info("MONOTONICITY CHECK: %d/%d pairs are monotonically decreasing", n_mono, n_total)
    LOG.info("%s", "=" * 60)
    non_mono = mono_df[~mono_df["monotonic"]]
    if len(non_mono) > 0:
        LOG.info("Non-monotonic pairs:\n%s", non_mono[["model", "dataset", "violations", "spread_pp"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Top-K Confidence Ablation for FRS")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root directory containing source_pass16_jsonl_by_model* folders",
    )
    parser.add_argument(
        "--from-csv",
        type=str,
        default=None,
        dest="from_csv",
        help="Load existing topk_ablation_results.csv and only regenerate plots (skips JSONL)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/topk_ablation_results",
        help="Output directory for plots and CSV",
    )
    parser.add_argument(
        "--multiseed",
        type=int,
        nargs="*",
        default=None,
        help="Subsample k traces per question; repeat with many RNG seeds (e.g. 1 2 4 8)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=30,
        dest="n_seeds",
        help="Number of random seeds: seed_base, seed_base+1, ...",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=42,
        dest="seed_base",
        help="First seed for multiseed subsampling",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logging (includes per-pair lines during multiseed runs)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Append full log to this file (timestamps on every line)",
    )
    parser.add_argument(
        "--multiseed-verbose-pairs",
        action="store_true",
        dest="multiseed_verbose_pairs",
        help="During multiseed, log every model×dataset at INFO (default: one line per k×seed)",
    )
    args = parser.parse_args()

    if not args.data_root and not args.from_csv:
        parser.error("Provide --data_root (run pipeline) or --from-csv (plot only)")
    if args.from_csv and args.multiseed:
        parser.error("--from-csv cannot be combined with --multiseed (run full pipeline for multiseed)")

    setup_logging(args.log_file, args.verbose)
    LOG.info(
        "Starting topk_ablation — data_root=%s from_csv=%s output_dir=%s",
        args.data_root,
        args.from_csv,
        args.output_dir,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    if args.from_csv:
        LOG.info("Plot-only mode: reading %s", os.path.abspath(args.from_csv))
        results_df = pd.read_csv(args.from_csv)
        long_df, agg_df = None, None
    else:
        results_df, long_df, agg_df = run_pipeline(
            args.data_root,
            multiseed_k=list(args.multiseed) if args.multiseed else None,
            n_seeds=args.n_seeds,
            seed_base=args.seed_base,
            multiseed_log_each_pair=args.verbose or args.multiseed_verbose_pairs,
        )

    if not args.from_csv:
        csv_path = os.path.join(args.output_dir, "topk_ablation_results.csv")
        results_df.to_csv(csv_path, index=False)
        LOG.info("Saved results CSV: %s (%d rows)", csv_path, len(results_df))

    print_summary(results_df)

    LOG.info("Generating per-dataset and heatmap figures...")
    plot_results(results_df, args.output_dir)

    LOG.info("Generating combined 6-panel all-datasets figure...")
    plot_all_datasets_accuracy_grid(
        results_df,
        os.path.join(args.output_dir, "topk_ablation_all_datasets_accuracy.png"),
        suptitle="Accuracy @ Top-K% Confidence Pool — all datasets (full pass@16 traces per question)",
    )

    if long_df is not None and agg_df is not None:
        long_path = os.path.join(args.output_dir, "topk_ablation_subsample_multiseed_long.csv")
        agg_path = os.path.join(args.output_dir, "topk_ablation_subsample_multiseed_agg.csv")
        long_df.to_csv(long_path, index=False)
        agg_df.to_csv(agg_path, index=False)
        LOG.info("Saved multiseed long CSV: %s (%d rows)", long_path, len(long_df))
        LOG.info("Saved multiseed aggregate CSV: %s (%d rows)", agg_path, len(agg_df))

        n_k = len(list(agg_df["subsample_k"].unique()))
        LOG.info("Generating %d subsample multiseed grid figures (mean ± std)...", n_k)
        for k in sorted(agg_df["subsample_k"].unique()):
            sub = agg_df[agg_df["subsample_k"] == k]
            out_png = os.path.join(
                args.output_dir,
                f"topk_ablation_all_datasets_subsample{k}_meanstd.png",
            )
            plot_all_datasets_accuracy_grid(
                sub,
                out_png,
                suptitle=(
                    f"Accuracy @ Top-K% Confidence Pool — random k={k} traces/question "
                    f"(mean ± std over {args.n_seeds} seeds)"
                ),
                accuracy_col="accuracy_mean",
                std_col="accuracy_std",
            )

    LOG.info("Done.")
