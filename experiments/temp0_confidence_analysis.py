#!/usr/bin/env python3
"""
Temperature-0 confidence analyses for FRS paper (4 analyses, shared loaders).

T=0 JSONL discovery: prefers ``*_processed.jsonl`` under ``temp0/`` trees, but also accepts
main run files such as ``test_*_t0.0_*.jsonl`` when the first row has ``score`` and token
prob fields. Skips ``*_prob.jsonl`` sidecars and any path under a ``humaneval`` directory;
only the six ``CORE_DATASETS`` benchmarks are analyzed.

Requires numpy, pandas, matplotlib, scipy, sklearn.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, roc_auc_score

# --- Regimes (paper) ---
RELIABLE = ["DS-R1-1.5B", "DS-R1-7B", "Qwen3-4B"]
DECEPTIVE = ["Qwen2.5-7B", "Phi-4", "Phi-4-Reas."]
UNRELIABLE = ["LLaMA-3.1-8B", "Qwen2.5-Math", "Gemma-7B"]

REGIME_MAP: Dict[str, str] = {}
for m in RELIABLE:
    REGIME_MAP[m] = "reliable"
for m in DECEPTIVE:
    REGIME_MAP[m] = "deceptive"
for m in UNRELIABLE:
    REGIME_MAP[m] = "unreliable"

# Dataset short names (aligned with sample_count_ablation / topk_ablation)
DATASET_SHORT = {
    "gsm8k": "GSM8K",
    "math500": "MATH500",
    "svamp": "SVAMP",
    "aqua": "AQuA",
    "gpqa": "GPQA",
    "commonsense_qa": "CommonsenseQA",
}

# Six paper benchmarks (exclude humaneval / extras)
CORE_DATASETS = {"AQuA", "CommonsenseQA", "GPQA", "GSM8K", "MATH500", "SVAMP"}


def _path_has_humaneval(path: str) -> bool:
    return any(p.lower() == "humaneval" for p in Path(path.replace("\\", "/")).parts)


def is_core_benchmark_dataset(ds: str) -> bool:
    """Allowed dataset label for analyses (never HumanEval)."""
    return ds in CORE_DATASETS and "humaneval" not in ds.lower()

# Folder / pass16 names -> short model labels
FOLDER_TO_MODEL = {
    "DeepSeek-R1-Distill-Qwen-1.5B": "DS-R1-1.5B",
    "DeepSeek-R1-Distill-Qwen-7B": "DS-R1-7B",
    "Llama-3.1-8B-Instruct": "LLaMA-3.1-8B",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "Qwen2.5-Math-7B": "Qwen2.5-Math",
    "Qwen3-4B-Thinking-2507": "Qwen3-4B",
    "gemma-7b": "Gemma-7B",
    "phi-4": "Phi-4",
    "phi-4-reasoning": "Phi-4-Reas.",
    "Phi-4-reasoning": "Phi-4-Reas.",
}

PASS16_TO_MODEL = {
    "DeepSeek_R1_Distill_Qwen_1.5B": "DS-R1-1.5B",
    "DeepSeek_R1_Distill_Qwen_7B": "DS-R1-7B",
    "Llama_3.1_8B_Instruct": "LLaMA-3.1-8B",
    "Qwen2.5_7B_Instruct": "Qwen2.5-7B",
    "Qwen2.5_Math_7B": "Qwen2.5-Math",
    "Qwen3_4B_Thinking_2507": "Qwen3-4B",
    "gemma_7b": "Gemma-7B",
    "phi_4": "Phi-4",
    "Phi_4_reasoning": "Phi-4-Reas.",
}

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

LOG = logging.getLogger("temp0_confidence_analysis")
NEUTRAL_GAP_PP = 2.0
K16 = 16


class _FlushFileHandler(logging.FileHandler):
    """Flush after every record so `tail -f` shows live progress."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def setup_logging(verbose: bool, log_file: Optional[str]) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if log_file:
        path = os.path.abspath(log_file)
        log_dir = os.path.dirname(path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        fh = _FlushFileHandler(path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        root.addHandler(fh)
        LOG.info("Logging also to %s", path)


def confidence_low10(token_probs: Sequence[float]) -> float:
    """Mean probability of lowest 10% of tokens (paper recipe; full sort)."""
    if not token_probs:
        return float("nan")
    sorted_probs = sorted(float(x) for x in token_probs)
    cutoff = max(1, int(len(sorted_probs) * 0.10))
    return float(np.mean(sorted_probs[:cutoff]))


def confidence_full_mean(token_probs: Sequence[float]) -> float:
    if not token_probs:
        return float("nan")
    return float(np.mean(np.asarray(token_probs, dtype=np.float64)))


def _maybe_logprobs_to_probs(vals: List[float]) -> List[float]:
    if not vals:
        return vals
    a = np.asarray(vals, dtype=np.float64)
    if np.nanmax(a) <= 0.0 and np.nanmedian(a) < -0.05:
        return np.exp(np.clip(a, -80, 80)).tolist()
    return vals


def get_trace_token_prob_lists(row: Dict[str, Any]) -> List[List[float]]:
    """Return one list of token probabilities per trace (length may be 1 for greedy)."""
    ctpp = row.get("chosen_token_probs_per_path")
    if isinstance(ctpp, dict) and "epoch_0" in ctpp:
        epoch = ctpp["epoch_0"]
        if epoch and isinstance(epoch[0], list):
            return [_maybe_logprobs_to_probs(list(t)) for t in epoch]
    ctp = row.get("chosen_token_probs")
    if isinstance(ctp, dict) and "epoch_0" in ctp:
        ep = ctp["epoch_0"]
        if isinstance(ep, list) and ep and not isinstance(ep[0], list):
            return [_maybe_logprobs_to_probs(list(ep))]
    pl = row.get("probability_log", {})
    if isinstance(pl, dict) and "epoch_0" in pl:
        ep = pl["epoch_0"]
        if isinstance(ep, list) and ep and not isinstance(ep[0], list):
            return [_maybe_logprobs_to_probs(list(ep))]
    return []


def get_score_list(row: Dict[str, Any]) -> List[bool]:
    s = row.get("score")
    if isinstance(s, list):
        return [bool(x) for x in s]
    if isinstance(s, bool):
        return [s]
    if s is None:
        return []
    return [bool(s)]


def extract_model_dataset_pass16(filepath: str) -> Tuple[str, str]:
    """Same logic as sample_count_ablation / topk_ablation."""
    if _path_has_humaneval(filepath):
        return "unknown", "unknown"
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
    if dataset.lower() == "humaneval":
        return "unknown", "unknown"
    dataset = DATASET_SHORT.get(dataset, dataset)
    model = PASS16_TO_MODEL.get(model, model)
    return model, dataset


def extract_model_dataset_temp0(filepath: str) -> Tuple[str, str]:
    """Parse temp0 tree: .../temp0/<dataset>/<model_folder>/..."""
    parts = Path(filepath.replace("\\", "/")).parts
    if "temp0" not in parts:
        return "unknown", "unknown"
    i = parts.index("temp0")
    if i + 2 >= len(parts):
        return "unknown", "unknown"
    dataset_raw = parts[i + 1]
    if dataset_raw.lower() == "humaneval":
        return "unknown", "unknown"
    model_folder = parts[i + 2]
    if dataset_raw.startswith("_"):
        return "unknown", "unknown"
    ds = DATASET_SHORT.get(dataset_raw, dataset_raw)
    model = FOLDER_TO_MODEL.get(model_folder, FOLDER_TO_MODEL.get(model_folder.replace("_", "-"), model_folder))
    return model, ds


def discover_t07_files(data_root: str) -> Dict[Tuple[str, str], str]:
    pattern = os.path.join(data_root, "data/pass16_sample*", "**", "*.jsonl")
    all_files = glob.glob(pattern, recursive=True)
    LOG.info("T=0.7 discovery: %d pass16 jsonl files", len(all_files))
    file_map: Dict[Tuple[str, str], str] = {}

    def better_pass16(a: str, b: str) -> str:
        """Prefer *_processed.jsonl."""
        ap = "_processed" in os.path.basename(a)
        bp = "_processed" in os.path.basename(b)
        if ap != bp:
            return a if ap else b
        return a if len(a) <= len(b) else b

    for fp in all_files:
        if _path_has_humaneval(fp):
            continue
        model, dataset = extract_model_dataset_pass16(fp)
        if model == "unknown" or dataset == "unknown":
            continue
        if not is_core_benchmark_dataset(dataset):
            continue
        key = (model, dataset)
        if key not in file_map:
            file_map[key] = fp
        else:
            file_map[key] = better_pass16(fp, file_map[key])
    LOG.info("T=0.7 discovery finished: %d (model, dataset) paths (core filter applied in main)", len(file_map))
    return file_map


def _skip_t0_sidecar_jsonl(path: str) -> bool:
    """Broad *.jsonl globs also hit prob-only sidecars; never use those as the main run."""
    if _path_has_humaneval(path):
        return True
    n = os.path.basename(path).lower()
    if n.endswith("_prob.jsonl") or n.endswith("_custom_prob.jsonl"):
        return True
    return False


def _jsonl_first_row_has_run_fields(path: str) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    break
            else:
                return False
        d = json.loads(line)
    except (OSError, json.JSONDecodeError):
        return False
    has_prob = (
        isinstance(d.get("chosen_token_probs_per_path"), dict)
        or isinstance(d.get("chosen_token_probs"), dict)
        or isinstance(d.get("probability_log"), dict)
    )
    return has_prob and "score" in d


def _t0_file_rank(path: str) -> Tuple[int, int, int, int, int]:
    """Lower tuple = higher priority. Prefer *_processed.jsonl, then t0.0 tag, shorter path."""
    name = os.path.basename(path)
    tier = 0 if name.endswith("_processed.jsonl") else 1
    t0 = 0 if ("t0.0" in name or "t0_0" in name) else 1
    p = path.replace("\\", "/")
    ev = 0 if "evaluation" in p else 1
    sr = 0 if "standard_runs" in p else 1
    return (tier, t0, ev, sr, len(p))


def discover_t0_files(data_root: str) -> Dict[Tuple[str, str], str]:
    """Find greedy / T=0 JSONL: *_processed.jsonl and alternate names (e.g. test_*_t0.0_*.jsonl)."""
    patterns = [
        os.path.join(data_root, "t0_temp0_extracted", "temp0", "**", "*_processed.jsonl"),
        os.path.join(data_root, "temp0", "**", "*_processed.jsonl"),
        os.path.join(data_root, "source_pass1_jsonl_by_model*", "**", "*_processed.jsonl"),
        os.path.join(data_root, "greedy*", "**", "*_processed.jsonl"),
        os.path.join(data_root, "t0_temp0_extracted", "temp0", "**", "*.jsonl"),
        os.path.join(data_root, "temp0", "**", "*.jsonl"),
    ]
    candidates: List[str] = []
    for pat in patterns:
        candidates.extend(glob.glob(pat, recursive=True))

    unique = list(dict.fromkeys(candidates))
    LOG.info("T=0 discovery: %d glob hits -> %d unique paths", len(candidates), len(unique))

    file_map: Dict[Tuple[str, str], str] = {}
    n_u = len(unique)
    for idx, fp in enumerate(unique, start=1):
        if idx % 300 == 0 or idx == n_u:
            LOG.info("T=0 discovery: scanned %d/%d candidates, %d pairs mapped", idx, n_u, len(file_map))
        if not os.path.isfile(fp):
            continue
        if _skip_t0_sidecar_jsonl(fp):
            continue
        if not os.path.basename(fp).endswith("_processed.jsonl") and not _jsonl_first_row_has_run_fields(fp):
            continue
        model, dataset = extract_model_dataset_temp0(fp)
        if model == "unknown" or dataset == "unknown":
            continue
        if not is_core_benchmark_dataset(dataset):
            continue
        key = (model, dataset)
        if key not in file_map:
            file_map[key] = fp
        else:
            if _t0_file_rank(fp) < _t0_file_rank(file_map[key]):
                file_map[key] = fp
    LOG.info("T=0 discovery finished: %d (model, dataset) paths", len(file_map))
    return file_map


def load_jsonl_rows(path: str, max_problems: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(row)
            if max_problems is not None and len(rows) >= max_problems:
                break
    return rows


def build_problems_k16(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Problems with >=16 traces for T=0.7 analysis."""
    problems: List[Dict[str, Any]] = []
    for line_num, row in enumerate(rows):
        scores = get_score_list(row)
        traces_probs = get_trace_token_prob_lists(row)
        if len(scores) < K16 or len(traces_probs) < K16:
            continue
        traces = []
        for i in range(K16):
            probs = traces_probs[i] if i < len(traces_probs) else []
            conf = confidence_low10(probs)
            if np.isnan(conf):
                continue
            traces.append(
                {
                    "correct": bool(scores[i]) if i < len(scores) else False,
                    "confidence": float(conf),
                    "trace_idx": i,
                }
            )
        if len(traces) < K16:
            continue
        pid = row.get("idx", line_num)
        problems.append({"problem_id": pid, "traces": traces})
    return problems


def gap_median_split_k16(problems: List[Dict[str, Any]]) -> Optional[float]:
    """Mean (acc confident half - acc unconfident half) in percentage points."""
    if not problems:
        return None
    gaps: List[float] = []
    for p in problems:
        tr = sorted(p["traces"], key=lambda t: (t["confidence"], t["trace_idx"]))
        n_lo = K16 // 2
        lo = tr[:n_lo]
        hi = tr[n_lo:]
        if not lo or not hi:
            continue
        acc_lo = float(np.mean([t["correct"] for t in lo]))
        acc_hi = float(np.mean([t["correct"] for t in hi]))
        gaps.append((acc_hi - acc_lo) * 100.0)
    if not gaps:
        return None
    return float(np.mean(gaps))


def load_t0_problem_confidences(
    rows: List[Dict[str, Any]],
) -> List[Tuple[float, bool]]:
    """One (confidence_low10, problem_correct) per row for greedy runs."""
    out: List[Tuple[float, bool]] = []
    for row in rows:
        tpl = get_trace_token_prob_lists(row)
        scores = get_score_list(row)
        if not tpl or not scores:
            continue
        probs = tpl[0]
        conf = confidence_low10(probs)
        if np.isnan(conf):
            continue
        # single trace correctness
        correct = bool(scores[0]) if scores else False
        out.append((float(conf), correct))
    return out


def gap_median_split_problems(conf_pairs: List[Tuple[float, bool]]) -> Optional[float]:
    """Split problems at median confidence; gap = acc(top half) - acc(bottom half), in pp."""
    if len(conf_pairs) < 2:
        return None
    sorted_pairs = sorted(conf_pairs, key=lambda x: x[0])
    n = len(sorted_pairs)
    n_lo = n // 2
    lo = sorted_pairs[:n_lo]
    hi = sorted_pairs[n_lo:]
    if not lo or not hi:
        return None
    acc_lo = float(np.mean([c for _, c in lo]))
    acc_hi = float(np.mean([c for _, c in hi]))
    return (acc_hi - acc_lo) * 100.0


def snr_binary(
    confidences: np.ndarray,
    correct: np.ndarray,
) -> float:
    """SNR from prompt; undefined if only one class or degenerate."""
    if confidences.size < 2 or len(np.unique(correct)) < 2:
        return float("nan")
    m1 = confidences[correct].mean()
    m0 = confidences[~correct].mean()
    v1 = confidences[correct].var()
    v0 = confidences[~correct].var()
    denom = np.sqrt(0.5 * (v1 + v0))
    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    return float((m1 - m0) / denom)


def safe_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def regime_from_gap(gap_pp: float) -> str:
    if gap_pp > NEUTRAL_GAP_PP:
        return "reliable"
    if gap_pp < -NEUTRAL_GAP_PP:
        return "deceptive"
    return "unreliable"


def savefig_both(fig: plt.Figure, out_base: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", bbox_inches="tight")


def analysis1(
    t0_map: Dict[Tuple[str, str], str],
    t07_map: Dict[Tuple[str, str], str],
    max_problems: Optional[int],
    output_dir: str,
) -> pd.DataFrame:
    rows = []
    items = sorted(t0_map.items())
    n = len(items)
    LOG.info("Analysis 1: single-trace FRS (%d pairs)", n)
    for i, ((model, dataset), t0_path) in enumerate(items, start=1):
        LOG.info("[%d/%d] analysis1: %s @ %s", i, n, model, dataset)
        rows_t0 = load_jsonl_rows(t0_path, max_problems)
        pairs = load_t0_problem_confidences(rows_t0)
        gap0 = gap_median_split_problems(pairs)

        gap07 = None
        if (model, dataset) in t07_map:
            p16 = build_problems_k16(load_jsonl_rows(t07_map[(model, dataset)], max_problems))
            gap07 = gap_median_split_k16(p16)

        preserved = None
        if gap0 is not None and gap07 is not None:
            preserved = (gap0 > 0) == (gap07 > 0)

        LOG.info(
            "[%d/%d] analysis1 done: %s @ %s — rows=%d usable_pairs=%d gap0=%s gap07=%s",
            i,
            n,
            model,
            dataset,
            len(rows_t0),
            len(pairs),
            None if gap0 is None else f"{gap0:.2f}",
            None if gap07 is None else f"{gap07:.2f}",
        )

        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "regime": REGIME_MAP.get(model, "unknown"),
                "gap_t0_k1": gap0,
                "gap_t07_k16": gap07,
                "gap_preserved": preserved,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "analysis1_single_trace_frs.csv"), index=False)

    # scatter
    sub = df.dropna(subset=["gap_t0_k1", "gap_t07_k16"])
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for regime in ["reliable", "deceptive", "unreliable", "unknown"]:
        m = sub[sub["regime"] == regime]
        if m.empty:
            continue
        ax.scatter(m["gap_t07_k16"], m["gap_t0_k1"], label=regime, alpha=0.75, s=38)
    lims = [
        np.nanmin([sub["gap_t07_k16"].min(), sub["gap_t0_k1"].min()]),
        np.nanmax([sub["gap_t07_k16"].max(), sub["gap_t0_k1"].max()]),
    ]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.6)
    ax.set_xlabel(r"Gap (pp) T=0.7, k=16")
    ax.set_ylabel(r"Gap (pp) T=0, k=1")
    ax.set_title("Median-split accuracy gap: single greedy trace vs pass@16")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig_both(fig, os.path.join(output_dir, "analysis1_gap_comparison"))
    plt.close(fig)
    LOG.info("Analysis 1 complete: %s", os.path.join(output_dir, "analysis1_single_trace_frs.csv"))
    return df


def analysis2(
    t0_map: Dict[Tuple[str, str], str],
    max_problems: Optional[int],
    output_dir: str,
) -> pd.DataFrame:
    rows = []
    items = sorted(t0_map.items())
    n = len(items)
    LOG.info("Analysis 2: estimator comparison (%d pairs)", n)
    for i, ((model, dataset), path) in enumerate(items, start=1):
        LOG.info("[%d/%d] analysis2: %s @ %s", i, n, model, dataset)
        rows_json = load_jsonl_rows(path, max_problems)
        low10: List[float] = []
        fullm: List[float] = []
        corr: List[bool] = []
        for row in rows_json:
            tpl = get_trace_token_prob_lists(row)
            sc = get_score_list(row)
            if not tpl or not sc:
                continue
            probs = tpl[0]
            c = confidence_low10(probs)
            f = confidence_full_mean(probs)
            if np.isnan(c) or np.isnan(f):
                continue
            low10.append(c)
            fullm.append(f)
            corr.append(bool(sc[0]))
        y = np.asarray(corr)
        a_low = np.asarray(low10)
        a_full = np.asarray(fullm)

        snr_l = snr_binary(a_low, y)
        snr_f = snr_binary(a_full, y)
        auc_l = safe_auroc(a_low, y.astype(int))
        auc_f = safe_auroc(a_full, y.astype(int))

        LOG.info(
            "[%d/%d] analysis2 done: %s @ %s — jsonl_rows=%d scored=%d SNR_low10=%.4f SNR_full=%.4f",
            i,
            n,
            model,
            dataset,
            len(rows_json),
            len(low10),
            snr_l,
            snr_f,
        )

        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "regime": REGIME_MAP.get(model, "unknown"),
                "snr_low10": snr_l,
                "snr_full_mean": snr_f,
                "auroc_low10": auc_l,
                "auroc_full_mean": auc_f,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "analysis2_estimator_comparison.csv"), index=False)

    # grouped bars: per model, mean |SNR| across datasets
    agg = (
        df.groupby("model", sort=False)
        .agg(
            abs_snr_low10=("snr_low10", lambda s: np.nanmean(np.abs(s))),
            abs_snr_full=("snr_full_mean", lambda s: np.nanmean(np.abs(s))),
        )
        .reset_index()
    )
    models = agg["model"].tolist()
    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w / 2, agg["abs_snr_low10"], width=w, label=r"Low-10% $|\mathrm{SNR}|$")
    ax.bar(x + w / 2, agg["abs_snr_full"], width=w, label=r"Full-mean $|\mathrm{SNR}|$")
    adv = np.nanmean(agg["abs_snr_low10"] - agg["abs_snr_full"])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right")
    for tick in ax.get_xticklabels():
        m = tick.get_text()
        reg = REGIME_MAP.get(m, "unknown")
        if reg == "reliable":
            tick.set_color("darkgreen")
        elif reg == "deceptive":
            tick.set_color("darkred")
        elif reg == "unreliable":
            tick.set_color("darkgoldenrod")
    ax.set_ylabel(r"Mean $|\mathrm{SNR}|$ over datasets")
    ax.set_title(f"T=0: low-10% vs full-mean (mean advantage low10−full = {adv:.3f})")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    savefig_both(fig, os.path.join(output_dir, "analysis2_estimator_comparison"))
    plt.close(fig)
    LOG.info("Analysis 2 complete: %s", os.path.join(output_dir, "analysis2_estimator_comparison.csv"))
    return df


def analysis3(
    t0_map: Dict[Tuple[str, str], str],
    t07_map: Dict[Tuple[str, str], str],
    max_problems: Optional[int],
    output_dir: str,
) -> pd.DataFrame:
    rows = []
    items = sorted(t0_map.items())
    n = len(items)
    LOG.info("Analysis 3: temperature SNR (%d pairs)", n)
    for i, ((model, dataset), t0_path) in enumerate(items, start=1):
        LOG.info("[%d/%d] analysis3: %s @ %s", i, n, model, dataset)
        r0 = load_jsonl_rows(t0_path, max_problems)
        pairs = load_t0_problem_confidences(r0)
        conf0 = np.asarray([p[0] for p in pairs])
        y0 = np.asarray([p[1] for p in pairs], dtype=bool)
        snr0 = snr_binary(conf0, y0)

        snr_tr = float("nan")
        snr_pr = float("nan")
        if (model, dataset) in t07_map:
            jrows = load_jsonl_rows(t07_map[(model, dataset)], max_problems)
            # trace-level
            tr_conf: List[float] = []
            tr_y: List[bool] = []
            pr_conf: List[float] = []
            pr_y: List[bool] = []
            for row in jrows:
                tpl = get_trace_token_prob_lists(row)
                sc = get_score_list(row)
                if len(tpl) < K16 or len(sc) < K16:
                    continue
                probs_tr = [confidence_low10(p) for p in tpl[:K16]]
                for i in range(K16):
                    if np.isnan(probs_tr[i]):
                        continue
                    tr_conf.append(probs_tr[i])
                    tr_y.append(bool(sc[i]))
                mconf = float(np.mean(probs_tr))
                pr_conf.append(mconf)
                pr_y.append(sum(sc[:K16]) > K16 / 2)

            if tr_conf and len(np.unique(tr_y)) > 1:
                snr_tr = snr_binary(np.asarray(tr_conf), np.asarray(tr_y))
            if pr_conf and len(np.unique(pr_y)) > 1:
                snr_pr = snr_binary(np.asarray(pr_conf), np.asarray(pr_y))

        LOG.info(
            "[%d/%d] analysis3 done: %s @ %s — SNR_t0=%.4f SNR_t07_prob=%.4f",
            i,
            n,
            model,
            dataset,
            snr0,
            snr_pr,
        )

        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "regime": REGIME_MAP.get(model, "unknown"),
                "snr_t0": snr0,
                "snr_t07_trace": snr_tr,
                "snr_t07_problem": snr_pr,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "analysis3_temperature_comparison.csv"), index=False)

    sub = df.dropna(subset=["snr_t0", "snr_t07_problem"])
    fig, ax = plt.subplots(figsize=(6.5, 6))
    if len(sub) > 0:
        for regime in ["reliable", "deceptive", "unreliable", "unknown"]:
            m = sub[sub["regime"] == regime]
            if m.empty:
                continue
            ax.scatter(
                m["snr_t07_problem"],
                m["snr_t0"],
                label=regime,
                alpha=0.8,
                s=42,
            )
        lims = [
            float(
                np.nanmin(
                    [sub["snr_t07_problem"].min(), sub["snr_t0"].min()]
                )
            ),
            float(
                np.nanmax(
                    [sub["snr_t07_problem"].max(), sub["snr_t0"].max()]
                )
            ),
        ]
        if np.isfinite(lims[0]) and np.isfinite(lims[1]):
            ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
    ax.set_xlabel(r"SNR T=0.7 (problem-mean, low-10%)")
    ax.set_ylabel(r"SNR T=0 (greedy, low-10%)")
    ax.set_title("Temperature comparison: SNR scatter")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig_both(fig, os.path.join(output_dir, "analysis3_temperature_snr"))
    plt.close(fig)
    LOG.info("Analysis 3 complete: %s", os.path.join(output_dir, "analysis3_temperature_comparison.csv"))
    return df


def analysis4(
    df1: pd.DataFrame,
    output_dir: str,
) -> pd.DataFrame:
    rows = []
    disagreements: List[str] = []
    for _, r in df1.iterrows():
        g0 = r["gap_t0_k1"]
        g7 = r["gap_t07_k16"]
        if g0 is None or np.isnan(g0) or g7 is None or np.isnan(g7):
            continue
        rt7 = regime_from_gap(float(g7))
        rt0 = regime_from_gap(float(g0))
        agree = rt7 == rt0
        if not agree:
            disagreements.append(f"{r['model']} @ {r['dataset']}: T07={rt7} vs T0={rt0}")
        rows.append(
            {
                "model": r["model"],
                "dataset": r["dataset"],
                "regime_t07": rt7,
                "regime_t0": rt0,
                "gap_t07": g7,
                "gap_t0": g0,
                "agreement": agree,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "analysis4_regime_stability.csv"), index=False)

    # confusion matrix on regime strings
    labels = ["reliable", "unreliable", "deceptive"]
    y_t = df["regime_t07"].map({k: i for i, k in enumerate(labels)})
    y_0 = df["regime_t0"].map({k: i for i, k in enumerate(labels)})
    mask = y_t.notna() & y_0.notna()
    cm = confusion_matrix(y_t[mask], y_0[mask], labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_ylabel("T=0.7 regime (from gap)")
    ax.set_xlabel("T=0 regime (from gap)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="k")
    tot = float(cm.sum())
    pct = float(np.diag(cm).sum() / tot) if tot > 0 else 0.0
    ax.set_title(f"Regime agreement (diag frac ≈ {pct:.2%})")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    savefig_both(fig, os.path.join(output_dir, "analysis4_regime_stability"))
    plt.close(fig)

    # heatmap two-tone: face = t07, edge = t0
    pivot_ds = sorted(df["dataset"].unique())
    pivot_m = sorted(df["model"].unique())
    color_t07 = {"reliable": "#2ca02c", "unreliable": "#ffbb78", "deceptive": "#d62728"}
    edge_t0 = {"reliable": "#1f77b4", "unreliable": "#888888", "deceptive": "#9467bd"}
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, m in enumerate(pivot_m):
        for j, d in enumerate(pivot_ds):
            sub = df[(df["model"] == m) & (df["dataset"] == d)]
            if sub.empty:
                continue
            rt7 = sub["regime_t07"].iloc[0]
            rt0 = sub["regime_t0"].iloc[0]
            face = color_t07.get(rt7, "#dddddd")
            ec = edge_t0.get(rt0, "#333333")
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.45, i - 0.45),
                    0.9,
                    0.9,
                    facecolor=face,
                    edgecolor=ec,
                    linewidth=3,
                )
            )
    ax.set_xlim(-0.5, len(pivot_ds) - 0.5)
    ax.set_ylim(-0.5, len(pivot_m) - 0.5)
    ax.set_xticks(range(len(pivot_ds)))
    ax.set_xticklabels(pivot_ds, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot_m)))
    ax.set_yticklabels(pivot_m)
    ax.set_title("Fill=T=0.7 regime, border=T=0 regime")
    plt.tight_layout()
    savefig_both(fig, os.path.join(output_dir, "analysis4_regime_heatmap"))
    plt.close(fig)

    LOG.info("Analysis 4 complete: regime stability CSV + figures under %s", output_dir)
    return df, disagreements


def print_summary(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df3: pd.DataFrame,
    df4: pd.DataFrame,
    disagreements: List[str],
) -> None:
    n = len(df1)
    sub = df1.dropna(subset=["gap_t0_k1", "gap_t07_k16", "gap_preserved"])
    n_p = int(sub["gap_preserved"].sum())
    mean_abs0 = float(np.nanmean(np.abs(sub["gap_t0_k1"])))
    mean_abs7 = float(np.nanmean(np.abs(sub["gap_t07_k16"])))

    d2 = df2.copy()
    d2["abs_l"] = d2["snr_low10"].abs()
    d2["abs_f"] = d2["snr_full_mean"].abs()
    n_better = int((d2["abs_l"] > d2["abs_f"]).sum())
    mean_l = float(np.nanmean(d2["abs_l"]))
    mean_f = float(np.nanmean(d2["abs_f"]))
    auc_l = float(np.nanmean(d2["auroc_low10"]))
    auc_f = float(np.nanmean(d2["auroc_full_mean"]))

    d3 = df3.dropna(subset=["snr_t0", "snr_t07_problem"])
    if len(d3) > 2:
        r_sn = float(pearsonr(d3["snr_t0"], d3["snr_t07_problem"])[0])
    else:
        r_sn = float("nan")

    n_ag = int(df4["agreement"].sum()) if len(df4) else 0
    n_tot = len(df4)

    lines = [
        "",
        "=== ANALYSIS SUMMARY ===",
        "",
        "1. Single-trace FRS (k=1, T=0):",
        f"   - Regime gap sign preserved in {n_p}/{len(sub)} pairs ({100*n_p/max(len(sub),1):.1f}%) vs T=0.7 k=16",
        f"   - Mean |gap| at T=0: {mean_abs0:.2f} pp vs T=0.7: {mean_abs7:.2f} pp",
        "",
        "2. Confidence estimator (low-10% vs full-mean, T=0):",
        f"   - Low-10% achieves higher |SNR| in {n_better}/{len(d2)} pairs ({100*n_better/max(len(d2),1):.1f}%)",
        f"   - Average |SNR|: low-10% = {mean_l:.3f}, full-mean = {mean_f:.3f}",
        f"   - Average AUROC: low-10% = {auc_l:.3f}, full-mean = {auc_f:.3f}",
        "",
        "3. Temperature comparison (T=0 vs T=0.7):",
        f"   - Average |SNR| at T=0: {np.nanmean(d3['snr_t0'].abs()):.3f}, at T=0.7 (problem): {np.nanmean(d3['snr_t07_problem'].abs()):.3f}",
        f"   - Correlation of SNR across pairs: r = {r_sn:.3f}",
        "",
        "4. Regime stability:",
        f"   - T=0 and T=0.7 agree on regime in {n_ag}/{n_tot} pairs ({100*n_ag/max(n_tot,1):.1f}%)",
        "   - Disagreements:",
    ]
    for line in lines:
        print(line)
        LOG.info(line)
    if disagreements:
        for s in disagreements:
            t = f"     - {s}"
            print(t)
            LOG.info(t)
    else:
        print("     (none)")
        LOG.info("     (none)")


def main() -> None:
    parser = argparse.ArgumentParser(description="T=0 confidence analyses for FRS paper")
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Project root (contains source_pass16*, t0_temp0_extracted/, ...)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/temp0_analysis_results",
        help="Where to write CSVs and figures",
    )
    parser.add_argument("--max_problems", type=int, default=None, help="Max problems per JSONL (debug)")
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Append INFO/DEBUG logs here (flushed every line; use tail -f)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose, args.log_file)

    data_root = os.path.abspath(args.data_root)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # --- Discovery ---
    t0_patterns = [
        os.path.join(data_root, "t0_temp0_extracted", "temp0", "**", "*_processed.jsonl"),
        os.path.join(data_root, "temp0", "**", "*_processed.jsonl"),
        os.path.join(data_root, "source_pass1_jsonl_by_model*", "**", "*_processed.jsonl"),
        os.path.join(data_root, "greedy*", "**", "*_processed.jsonl"),
        os.path.join(data_root, "t0_temp0_extracted", "temp0", "**", "*.jsonl"),
        os.path.join(data_root, "temp0", "**", "*.jsonl"),
    ]
    t0_files: List[str] = []
    for p in t0_patterns:
        t0_files.extend(glob.glob(p, recursive=True))
    LOG.info("T=0 glob patterns:")
    for p in t0_patterns:
        LOG.info("  %s", p)
    LOG.info("T=0 candidate files found: %d", len(t0_files))

    t07_pattern = os.path.join(data_root, "data/pass16_sample*", "**", "*.jsonl")
    t07_files = glob.glob(t07_pattern, recursive=True)
    LOG.info("T=0.7 glob: %s", t07_pattern)
    LOG.info("T=0.7 candidate files found: %d", len(t07_files))

    t0_map_all = discover_t0_files(data_root)
    t07_map_all = discover_t07_files(data_root)

    t0_map = {k: v for k, v in t0_map_all.items() if is_core_benchmark_dataset(k[1])}
    t07_map = {k: v for k, v in t07_map_all.items() if is_core_benchmark_dataset(k[1])}

    LOG.info("Unique (model, dataset) pairs T=0 (core benchmarks): %d", len(t0_map))
    LOG.info("Unique (model, dataset) pairs T=0.7 (core): %d", len(t07_map))

    # First record introspection (prefer GSM8K if present)
    sample_path = None
    for key in sorted(t0_map.keys()):
        if key[1] == "GSM8K":
            sample_path = t0_map[key]
            break
    if sample_path is None:
        sample_path = next(iter(t0_map.values()), None)
    if sample_path:
        rows = load_jsonl_rows(sample_path, 1)
        if rows:
            LOG.info("Example T=0 file: %s", sample_path)
            LOG.info("First record keys: %s", sorted(rows[0].keys()))
    sample07 = next(iter(t07_map.values()), None)
    if sample07:
        r = load_jsonl_rows(sample07, 1)
        if r:
            LOG.info("Example T=0.7 file: %s", sample07)
            LOG.info("First record keys: %s", sorted(r[0].keys()))

    LOG.info(
        "Confidence: confidence_low10(token_probs) on chosen_token_probs_per_path epoch_0 "
        "or chosen_token_probs / probability_log epoch_0 for greedy."
    )
    LOG.info("Correctness: score[i] per trace; greedy uses score[0].")

    print("\n--- Discovered paths (T=0) ---")
    for k, v in sorted(t0_map.items()):
        print(f"  {k[0]:20} {k[1]:16} -> {v}")
    print("\n--- Discovered paths (T=0.7 pass16) ---")
    for k, v in sorted(t07_map.items()):
        print(f"  {k[0]:20} {k[1]:16} -> {v}")

    # Run analyses
    df1 = analysis1(t0_map, t07_map, args.max_problems, output_dir)
    df2 = analysis2(t0_map, args.max_problems, output_dir)
    df3 = analysis3(t0_map, t07_map, args.max_problems, output_dir)
    df4, disagreements = analysis4(df1, output_dir)

    print_summary(df1, df2, df3, df4, disagreements)
    LOG.info("Wrote outputs under %s", output_dir)


if __name__ == "__main__":
    main()
