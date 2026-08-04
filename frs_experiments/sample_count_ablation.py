#!/usr/bin/env python3
"""
Sample-count ablation for FRS-style median-split accuracy.

Reads pass@16 JSONL (same layout as topk_ablation.py / correctness_conditioned.py).
For each problem, subsamples k ∈ {4,8,12,16} traces, splits at median confidence
(ceil(k/2) high-conf, floor(k/2) low-conf), records mean accuracy on each half.

Checkpointing: each (model, dataset) is saved to checkpoints/chunks/*.csv when done.
Use --resume (default) to skip completed pairs. Ctrl+C saves state; re-run to continue.

Usage:
  python sample_count_ablation.py --data_root /path/to/threshold \\
      --output_dir ./sample_count_ablation_results --log-file run.log

  # Rebuild CSVs + figures from existing chunks only:
  python sample_count_ablation.py --aggregate-only --output_dir ./sample_count_ablation_results
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import signal
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Publication style
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Nimbus Roman", "serif"],
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    }
)

LOW_PROB_CUTOFF = 0.10
K_VALUES = [4, 8, 12, 16]
N_BOOTSTRAP = 50
BASE_SEED = 42
STATE_VERSION = 2

LOG = logging.getLogger("sample_count_ablation")

# Regime labels (paper narrative) — used for heatmap row grouping only
REGIME_ORDER = [
    ("reliable", ["DS-R1-1.5B", "DS-R1-7B", "Qwen3-4B"]),
    ("deceptive", ["Qwen2.5-7B", "Phi-4", "Phi-4-Reas."]),
    ("unreliable", ["LLaMA-3.1-8B", "Qwen2.5-Math", "Gemma-7B"]),
]
MODEL_ORDER = [m for _, ms in REGIME_ORDER for m in ms]

NEUTRAL_GAP_PP = 2.0

# Global flag for graceful shutdown
_INTERRUPTED = False


def _on_signal(signum: int, frame: Any) -> None:
    global _INTERRUPTED
    _INTERRUPTED = True
    LOG.warning(
        "Received signal %s — will finish current (model,dataset) pair then exit. "
        "Re-run with the same --output_dir to resume.",
        signum,
    )


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


def compute_trace_confidence(token_probs: list) -> float:
    if not token_probs or len(token_probs) == 0:
        return float("nan")
    arr = np.asarray(token_probs, dtype=np.float64)
    n_low = max(1, int(len(arr) * LOW_PROB_CUTOFF))
    kth = min(n_low - 1, len(arr) - 1)
    lowest = np.partition(arr, kth)[:n_low]
    return float(np.mean(lowest))


def extract_model_dataset(filepath: str) -> Tuple[str, str]:
    parts = filepath.replace("\\", "/").split("/")
    model = None
    for i, p in enumerate(parts):
        if p.startswith("source_pass16"):
            if i + 1 < len(parts):
                model = parts[i + 1]
            break
    fname = parts[-1]
    m = re.match(r"^([a-z_0-9]+?)__", fname)
    dataset = m.group(1) if m else "unknown"
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


def dedupe_files(data_root: str) -> Dict[Tuple[str, str], str]:
    pattern = os.path.join(data_root, "source_pass16_jsonl_by_model*", "**", "*.jsonl")
    all_files = glob.glob(pattern, recursive=True)
    file_map: Dict[Tuple[str, str], str] = {}
    for fp in all_files:
        model, dataset = extract_model_dataset(fp)
        key = (model, dataset)
        if key not in file_map or "_processed" in fp:
            file_map[key] = fp
    return file_map


def pair_key(model: str, dataset: str) -> str:
    return f"{model}||{dataset}"


def chunk_filename(model: str, dataset: str) -> str:
    safe = f"{model}__{dataset}".replace("/", "_").replace(" ", "_")
    return f"{safe}.csv"


def load_problems_from_jsonl(
    filepath: str, max_problems: Optional[int] = None
) -> List[Dict[str, Any]]:
    problems: List[Dict[str, Any]] = []
    with open(filepath, encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            row = json.loads(line)
            scores = row.get("score", [])
            token_probs_all = row.get("chosen_token_probs_per_path", {}).get("epoch_0", [])
            if not isinstance(token_probs_all, list) or not scores:
                continue
            traces = []
            for i in range(len(scores)):
                probs = token_probs_all[i] if i < len(token_probs_all) else []
                conf = compute_trace_confidence(probs)
                if np.isnan(conf):
                    continue
                traces.append(
                    {
                        "correct": bool(scores[i]),
                        "confidence": float(conf),
                        "trace_idx": i,
                    }
                )
            if len(traces) < 16:
                continue
            pid = row.get("idx", line_num)
            problems.append({"problem_id": pid, "traces": traces})
            if max_problems is not None and len(problems) >= max_problems:
                break
    return problems


def median_split_accuracies(
    traces: List[Dict[str, Any]], k_sub: int, rng: np.random.RandomState
) -> Optional[Tuple[float, float]]:
    if len(traces) < k_sub:
        return None
    idx = rng.choice(len(traces), size=k_sub, replace=False)
    sub = [traces[i] for i in idx]
    sub.sort(key=lambda t: (t["confidence"], t["trace_idx"]))
    n_lo = k_sub // 2
    lo = sub[:n_lo]
    hi = sub[n_lo:]
    if not lo or not hi:
        return None
    acc_lo = float(np.mean([t["correct"] for t in lo]))
    acc_hi = float(np.mean([t["correct"] for t in hi]))
    return acc_hi, acc_lo


def sign_bucket(gap_pp: float) -> int:
    if gap_pp > NEUTRAL_GAP_PP:
        return 1
    if gap_pp < -NEUTRAL_GAP_PP:
        return -1
    return 0


def compute_pair_rows(
    model: str,
    dataset: str,
    problems: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, int, int], Tuple[float, float]]]:
    rows: List[Dict[str, Any]] = []
    cache: Dict[Tuple[str, str, int, int], Tuple[float, float]] = {}
    for k_sub in K_VALUES:
        for bi in range(N_BOOTSTRAP):
            seed = BASE_SEED + bi
            rng = np.random.RandomState(seed)
            hi_accs, lo_accs = [], []
            for p in problems:
                out = median_split_accuracies(p["traces"], k_sub, rng)
                if out is None:
                    continue
                acc_hi, acc_lo = out
                hi_accs.append(acc_hi)
                lo_accs.append(acc_lo)
            if not hi_accs:
                continue
            frs_acc = float(np.mean(hi_accs)) * 100.0
            mean_lo = float(np.mean(lo_accs)) * 100.0
            gap_pp = frs_acc - mean_lo
            cache[(model, dataset, k_sub, seed)] = (frs_acc, gap_pp)
            rows.append(
                {
                    "bootstrap_idx": bi,
                    "seed": seed,
                    "model": model,
                    "dataset": dataset,
                    "k_sub": k_sub,
                    "frs_accuracy_proxy_pct": round(frs_acc, 4),
                    "acc_unconfident_half_pct": round(mean_lo, 4),
                    "gap_pp": round(gap_pp, 4),
                }
            )
    return rows, cache


def _atomic_write_json(path: str, obj: Any) -> None:
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".state_", suffix=".json", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


STATE_FILENAME = "state.json"


def load_state(checkpoint_dir: str) -> Dict[str, Any]:
    path = os.path.join(checkpoint_dir, STATE_FILENAME)
    if not os.path.isfile(path):
        return {
            "version": STATE_VERSION,
            "completed_pairs": [],
            "config": {},
        }
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_state(checkpoint_dir: str, state: Dict[str, Any]) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, STATE_FILENAME)
    state = dict(state)
    state["version"] = STATE_VERSION
    _atomic_write_json(path, state)
    LOG.debug("Wrote state: %d pairs done", len(state.get("completed_pairs", [])))


def config_fingerprint(
    data_root: str,
    max_problems: Optional[int],
    min_problems: int,
) -> Dict[str, Any]:
    return {
        "data_root": os.path.abspath(data_root),
        "max_problems": max_problems,
        "min_problems": min_problems,
        "n_bootstrap": N_BOOTSTRAP,
        "base_seed": BASE_SEED,
        "k_values": list(K_VALUES),
    }


def merge_chunks_to_dataframe(chunks_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(chunks_dir, "*.csv")))
    if not paths:
        return pd.DataFrame()
    dfs = [pd.read_csv(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)


def aggregate_and_plot(
    detail_df: pd.DataFrame,
    output_dir: str,
) -> None:
    if detail_df.empty:
        LOG.error("No detail rows — cannot aggregate.")
        return

    detail_path = os.path.join(output_dir, "ablation_per_bootstrap_detail.csv")
    detail_df.to_csv(detail_path, index=False)
    LOG.info("Wrote merged detail: %s (%d rows)", detail_path, len(detail_df))

    datasets = sorted(detail_df["dataset"].unique())
    models = sorted(detail_df["model"].unique())

    cache: Dict[Tuple[str, str, int, int], Tuple[float, float]] = {}
    for _, r in detail_df.iterrows():
        cache[(r["model"], r["dataset"], int(r["k_sub"]), int(r["seed"]))] = (
            float(r["frs_accuracy_proxy_pct"]),
            float(r["gap_pp"]),
        )

    summary_m = (
        detail_df.groupby(["model", "dataset", "k_sub"])["frs_accuracy_proxy_pct"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_m.columns = ["model", "dataset", "k_sub", "frs_acc_mean", "frs_acc_std"]
    summary_m["frs_acc_std"] = summary_m["frs_acc_std"].fillna(0.0)

    rows_rho: List[Dict[str, Any]] = []
    for dataset in datasets:
        for k_sub in K_VALUES:
            if k_sub == 16:
                continue
            for bi in range(N_BOOTSTRAP):
                seed = BASE_SEED + bi
                vec_k, vec_16 = [], []
                for model in models:
                    key_k = (model, dataset, k_sub, seed)
                    key_16 = (model, dataset, 16, seed)
                    if key_k in cache and key_16 in cache:
                        vec_k.append(cache[key_k][0])
                        vec_16.append(cache[key_16][0])
                if (
                    len(vec_k) < 3
                    or np.std(vec_k) < 1e-12
                    or np.std(vec_16) < 1e-12
                ):
                    rho = np.nan
                else:
                    rho, _ = spearmanr(vec_k, vec_16)
                rows_rho.append(
                    {
                        "dataset": dataset,
                        "k_sub": k_sub,
                        "bootstrap_idx": bi,
                        "seed": seed,
                        "spearman_vs_k16": rho,
                        "n_models": len(vec_k),
                    }
                )

    rho_df = pd.DataFrame(rows_rho)
    rho_summary = (
        rho_df.groupby(["k_sub", "dataset"])["spearman_vs_k16"]
        .agg(["mean", "std"])
        .reset_index()
    )
    rho_summary.columns = ["k_sub", "dataset", "spearman_mean", "spearman_std"]

    global_rho = (
        rho_df.groupby(["k_sub"])["spearman_vs_k16"].agg(["mean", "std"]).reset_index()
    )
    global_rho.columns = ["k_sub", "spearman_mean_all_ds", "spearman_std_all_ds"]

    rankings_merge = summary_m.merge(
        rho_summary.rename(
            columns={
                "spearman_mean": "spearman_vs_k16_mean",
                "spearman_std": "spearman_vs_k16_std",
            }
        ),
        on=["k_sub", "dataset"],
        how="left",
    )
    rankings_merge.to_csv(os.path.join(output_dir, "ablation_rankings.csv"), index=False)
    global_rho.to_csv(
        os.path.join(output_dir, "ablation_rankings_global_spearman.csv"), index=False
    )

    gap_mean = (
        detail_df.groupby(["model", "dataset", "k_sub"])["gap_pp"]
        .mean()
        .reset_index()
    )
    pivot_ref = gap_mean[gap_mean["k_sub"] == 16].set_index(["model", "dataset"])[
        "gap_pp"
    ]
    rows_regime: List[Dict[str, Any]] = []
    for k_sub in K_VALUES:
        sub = gap_mean[gap_mean["k_sub"] == k_sub].set_index(["model", "dataset"])
        for (model, dataset) in sub.index:
            g_k = float(sub.loc[(model, dataset), "gap_pp"])
            g_16 = float(pivot_ref.get((model, dataset), np.nan))
            if np.isnan(g_16):
                continue
            rows_regime.append(
                {
                    "k_sub": k_sub,
                    "model": model,
                    "dataset": dataset,
                    "mean_gap_pp": round(g_k, 4),
                    "ref_gap_pp_k16": round(g_16, 4),
                    "sign_k": sign_bucket(g_k),
                    "sign_k16": sign_bucket(g_16),
                    "sign_agrees": int(sign_bucket(g_k) == sign_bucket(g_16)),
                }
            )

    regime_df = pd.DataFrame(rows_regime)
    regime_df.to_csv(
        os.path.join(output_dir, "ablation_regime_separation.csv"), index=False
    )

    agree_frac = (
        regime_df[regime_df["k_sub"] != 16]
        .groupby("k_sub")["sign_agrees"]
        .mean()
        .reset_index()
    )
    agree_frac.columns = ["k_sub", "fraction_sign_agrees_with_k16"]

    # Figures
    ds_colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(datasets), 1)))
    plot_k = [k for k in K_VALUES if k != 16]
    xpos = np.arange(len(plot_k))

    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    for i, ds in enumerate(datasets):
        means, stds = [], []
        for k in plot_k:
            row = rho_summary[(rho_summary["dataset"] == ds) & (rho_summary["k_sub"] == k)]
            if row.empty:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(row["spearman_mean"].values[0])
                stds.append(row["spearman_std"].values[0])
        ax1.errorbar(
            xpos + (i - len(datasets) / 2) * 0.06,
            means,
            yerr=stds,
            label=ds,
            marker="o",
            capsize=2,
            linewidth=1.2,
            markersize=4,
            color=ds_colors[i],
        )
    avg_means, avg_stds = [], []
    for k in plot_k:
        r = rho_df[rho_df["k_sub"] == k]["spearman_vs_k16"]
        avg_means.append(r.mean())
        avg_stds.append(r.std())
    ax1.errorbar(
        xpos,
        avg_means,
        yerr=avg_stds,
        label="Mean ± std (all dataset×bootstrap ρ)",
        color="black",
        linewidth=2.5,
        marker="s",
        capsize=3,
        zorder=10,
    )
    ax1.set_xticks(xpos)
    ax1.set_xticklabels([str(k) for k in plot_k])
    ax1.set_xlabel("Subsample size $k$")
    ax1.set_ylabel(r"Spearman $\rho$ vs. $k{=}16$ ranking")
    ax1.set_title("Ranking stability of FRS-accuracy proxy vs. reference $k{=}16$")
    ax1.legend(loc="lower right", ncol=2, framealpha=0.92)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, "ablation_ranking_stability.pdf"))
    plt.close(fig1)

    gap_model_k = gap_mean.groupby(["model", "k_sub"])["gap_pp"].mean().reset_index()
    pivot = gap_model_k.pivot(index="model", columns="k_sub", values="gap_pp")
    pivot = pivot.reindex([m for m in MODEL_ORDER if m in pivot.index])
    fig2, ax2 = plt.subplots(figsize=(6.5, 5))
    im = ax2.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-30, vmax=30)
    ax2.set_xticks(np.arange(len(K_VALUES)))
    ax2.set_xticklabels([str(k) for k in K_VALUES])
    ax2.set_yticks(np.arange(len(pivot.index)))
    ax2.set_yticklabels(pivot.index)
    ax2.set_xlabel("Subsample size $k$")
    ax2.set_ylabel("Model")
    ax2.set_title(
        "Mean accuracy gap (confident $-$ unconfident half), pp\n(averaged over benchmarks)"
    )
    plt.colorbar(im, ax=ax2, label="Gap (pp)")
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "ablation_regime_heatmap.pdf"))
    plt.close(fig2)

    var_rows = []
    for ds in datasets:
        for k in K_VALUES:
            sub = detail_df[(detail_df["dataset"] == ds) & (detail_df["k_sub"] == k)]
            std_per_model = sub.groupby("model")["frs_accuracy_proxy_pct"].std()
            var_rows.append(
                {
                    "dataset": ds,
                    "k_sub": k,
                    "mean_bootstrap_std_across_models": float(std_per_model.mean()),
                }
            )
    var_df = pd.DataFrame(var_rows)
    var_df.to_csv(os.path.join(output_dir, "ablation_frs_variance_table.csv"), index=False)

    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    xpos3 = np.arange(len(K_VALUES))
    for i, ds in enumerate(datasets):
        ys = [
            var_df[(var_df["dataset"] == ds) & (var_df["k_sub"] == k)][
                "mean_bootstrap_std_across_models"
            ].values[0]
            for k in K_VALUES
        ]
        ax3.plot(xpos3, ys, marker="o", label=ds, linewidth=1.2, color=ds_colors[i])
    ax3.set_xticks(xpos3)
    ax3.set_xticklabels([str(k) for k in K_VALUES])
    ax3.set_xlabel("Subsample size $k$")
    ax3.set_ylabel("Mean bootstrap SD of FRS-accuracy proxy (across models)")
    ax3.set_title("Subsampling variance cost (50 bootstrap draws)")
    ax3.legend(loc="upper right", ncol=2, framealpha=0.92)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, "ablation_frs_variance.pdf"))
    plt.close(fig3)

    k_target = 8
    gr = global_rho[global_rho["k_sub"] == k_target]
    rho_m = gr["spearman_mean_all_ds"].values[0] if not gr.empty else float("nan")
    rho_s = gr["spearman_std_all_ds"].values[0] if not gr.empty else float("nan")
    ag = agree_frac[agree_frac["k_sub"] == k_target]
    frac = ag["fraction_sign_agrees_with_k16"].values[0] if not ag.empty else float("nan")

    LOG.info("=== Summary ===")
    LOG.info(
        "At k=%s, mean Spearman ρ vs k=16 (pooled): %.4f (± %.4f)",
        k_target,
        rho_m,
        rho_s,
    )
    LOG.info(
        "Sign-bucket agreement with k=16 (|gap|≤%.1f pp → neutral): %.1f%% of pairs",
        NEUTRAL_GAP_PP,
        100 * frac,
    )
    LOG.info("Outputs in %s", os.path.abspath(output_dir))


def run_ablation(
    data_root: str,
    output_dir: str,
    checkpoint_dir: str,
    max_problems: Optional[int],
    min_problems: int,
    resume: bool,
    force_reset: bool,
) -> None:
    global _INTERRUPTED
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    chunks_dir = os.path.join(checkpoint_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    cfg = config_fingerprint(data_root, max_problems, min_problems)
    state = load_state(checkpoint_dir)

    if force_reset:
        state = {"version": STATE_VERSION, "completed_pairs": [], "config": {}}
        for f in glob.glob(os.path.join(chunks_dir, "*.csv")):
            os.unlink(f)
        save_state(checkpoint_dir, state)
        LOG.info("Reset checkpoint state and removed chunk files.")

    stored_cfg = state.get("config") or {}
    if state.get("completed_pairs") and stored_cfg and stored_cfg != cfg:
        LOG.warning(
            "Checkpoint config differs from this run.\n  Stored: %s\n  Current: %s\n"
            "Use --force-reset to clear, or match arguments.",
            stored_cfg,
            cfg,
        )
        if resume:
            raise SystemExit(2)

    if not stored_cfg:
        state["config"] = cfg
        save_state(checkpoint_dir, state)

    file_map = dedupe_files(data_root)
    if not file_map:
        raise SystemExit(
            "No JSONL files found under source_pass16_jsonl_by_model*. "
            "Cannot run without per-problem trace data."
        )

    # Data probe (first file only, small read)
    sample_fp = next(iter(file_map.values()))
    sample_problems = load_problems_from_jsonl(sample_fp, max_problems=3)
    LOG.info("=== Data probe ===")
    LOG.info("Sample file: %s", sample_fp)
    if sample_problems:
        ex = sample_problems[0]
        LOG.info("problem_id (example): %s", ex["problem_id"])
        LOG.info("n_traces (example): %s", len(ex["traces"]))
        df_demo = pd.DataFrame(ex["traces"])
        LOG.info("Per-trace columns: %s", list(df_demo.columns))
        LOG.debug("Head:\n%s", df_demo.head(8).to_string(index=False))

    completed = set(state.get("completed_pairs", []))
    total_pairs = len(file_map)
    done_n = len(completed)

    for idx, (key, fp) in enumerate(sorted(file_map.items()), start=1):
        model, dataset = key
        pk = pair_key(model, dataset)
        if resume and pk in completed:
            LOG.info(
                "[%d/%d] SKIP (already checkpointed): %s %s",
                idx,
                total_pairs,
                model,
                dataset,
            )
            continue

        if _INTERRUPTED:
            LOG.warning("Interrupted before starting pair %s — exiting.", pk)
            break

        LOG.info(
            "[%d/%d] START %s × %s | load %s",
            idx,
            total_pairs,
            model,
            dataset,
            fp,
        )
        t0 = time.perf_counter()
        problems = load_problems_from_jsonl(fp, max_problems=max_problems)
        t_load = time.perf_counter()
        if len(problems) < min_problems:
            LOG.warning(
                "SKIP %s × %s: only %d problems (min_problems=%d)",
                model,
                dataset,
                len(problems),
                min_problems,
            )
            continue

        t1 = time.perf_counter()
        rows, _ = compute_pair_rows(model, dataset, problems)
        t_comp = time.perf_counter()
        if not rows:
            LOG.warning("No rows computed for %s × %s — skip", model, dataset)
            continue

        chunk_path = os.path.join(chunks_dir, chunk_filename(model, dataset))
        pd.DataFrame(rows).to_csv(chunk_path, index=False)
        completed.add(pk)
        state["completed_pairs"] = sorted(completed)
        state["config"] = cfg
        save_state(checkpoint_dir, state)

        LOG.info(
            "DONE %s × %s | problems=%d | load=%.1fs compute=%.1fs | chunk=%s",
            model,
            dataset,
            len(problems),
            t_load - t0,
            t_comp - t1,
            chunk_path,
        )

        if _INTERRUPTED:
            LOG.warning("Interrupted after saving %s — resume with same --output_dir", pk)
            break

    # Merge all chunks and aggregate
    detail_df = merge_chunks_to_dataframe(chunks_dir)
    if detail_df.empty:
        LOG.error("No chunk CSVs in %s — nothing to aggregate.", chunks_dir)
        return

    if len(completed) < total_pairs:
        LOG.warning(
            "Partial run: %d / %d (model,dataset) pairs in checkpoint. "
            "Figures use available data only.",
            len(completed),
            total_pairs,
        )

    aggregate_and_plot(detail_df, output_dir)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sample count ablation for median-split accuracy (checkpointed)"
    )
    p.add_argument("--data_root", type=str, default=".", help="Root with JSONL trees")
    p.add_argument(
        "--output_dir",
        type=str,
        default="./sample_count_ablation_results",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint dir (default: OUTPUT_DIR/checkpoints)",
    )
    p.add_argument("--max_problems", type=int, default=None)
    p.add_argument("--min_problems", type=int, default=10)
    p.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip (model,dataset) pairs already in checkpoint (default: on)",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore checkpoint and re-process all pairs (still writes new chunks)",
    )
    p.add_argument(
        "--force-reset",
        action="store_true",
        help="Delete checkpoint state and chunk CSVs, start fresh",
    )
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only merge checkpoints/chunks/*.csv and rebuild tables + figures",
    )
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Append human-readable log to this path",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    args = p.parse_args()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    out_dir = os.path.abspath(args.output_dir)
    ckpt = args.checkpoint_dir or os.path.join(out_dir, "checkpoints")

    setup_logging(
        args.log_file or os.path.join(out_dir, "sample_count_ablation.log"),
        args.verbose,
    )

    if args.aggregate_only:
        chunks_dir = os.path.join(ckpt, "chunks")
        detail_df = merge_chunks_to_dataframe(chunks_dir)
        if detail_df.empty:
            LOG.error("No chunks in %s — run without --aggregate-only first.", chunks_dir)
            sys.exit(1)
        aggregate_and_plot(detail_df, out_dir)
        return

    resume = args.resume and not args.no_resume
    if not resume:
        LOG.info("--no-resume: will recompute all pairs (chunks overwritten per pair).")

    run_ablation(
        args.data_root,
        out_dir,
        ckpt,
        args.max_problems,
        args.min_problems,
        resume,
        args.force_reset,
    )


if __name__ == "__main__":
    main()
