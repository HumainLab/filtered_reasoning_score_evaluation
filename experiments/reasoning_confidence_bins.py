#!/usr/bin/env python3
"""
Reasoning quality by pooled confidence bins (rank traces, optional top-fraction slice,
then disjoint equal-count bins).

Default: full ranked pool (``--top-pool-frac 1``) with 10 bins → cumulative reasoning
at top 10%% … top 100%% (including 70%% and 100%%), aligned with ``topk_ablation.PERCENTILES``.

Mirrors the pooled top-K accuracy setup (rank all traces, slice, bin), then judges a
random subset per bin with the same GPT-4o-mini rubric as topk_judge_eval.py.

Usage:
  export PORTKEY_API_KEY=...
  python reasoning_confidence_bins.py run --data-root /path/to/threshold --output-dir ./reasoning_confidence_bins_results

  # Previous paper default: top 50%% of traces only, five bins (cumulative top10–top50):
  # python reasoning_confidence_bins.py run ... --top-pool-frac 0.5 --n-bins 5

  # Plots only from existing CSVs + judge JSON:
  python reasoning_confidence_bins.py plot --output-dir ./reasoning_confidence_bins_results

  # No judge: recompute top-K%% accuracy (by confidence rank) for K in 10,20,… — writes topk_trace_accuracy.csv
  python reasoning_confidence_bins.py topk-accuracy --data-root . --output-dir ./reasoning_confidence_bins_results

Does not modify topk_ablation.py or topk_judge_eval.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
import subprocess
import sys
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Reuse pooled confidence + file discovery
from topk_ablation import DATASET_PANEL_ORDER, build_file_map, compute_trace_confidence

# Reuse judge client + score normalization
from topk_judge_eval import (
    DEFAULT_JUDGE_MODEL,
    Judge,
    reasoning_score_from_judge,
)

LOG = logging.getLogger("reasoning_confidence_bins")

# Publication style (aligned with sample_count_ablation.py)
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

# ── Experiment constants (spec) ─────────────────────────────────────────────
DEFAULT_SAMPLES_PER_BIN = 50


@dataclass(frozen=True)
class BinConfig:
    """
    After ranking all traces by confidence (descending), keep the first
    ``top_pool_frac`` fraction by count, then split that slice into ``n_bins``
    equal-count disjoint bins. Cumulative rows use labels top{step}, top{2*step}, …
    through top100, where step = 100 // n_bins (so ``n_bins`` must divide 100).

    Default ``top_pool_frac=1.0``, ``n_bins=10`` yields deciles of the full pool
    and cumulative top10 … top100 (including 70 and 100), comparable to
    ``topk_ablation.PERCENTILES``.
    """

    top_pool_frac: float = 1.0
    n_bins: int = 10

    def __post_init__(self) -> None:
        if not 0 < self.top_pool_frac <= 1.0:
            raise ValueError("top_pool_frac must be in (0, 1]")
        if self.n_bins < 1:
            raise ValueError("n_bins must be >= 1")
        if 100 % self.n_bins != 0:
            raise ValueError("n_bins must divide 100 (e.g. 5, 10, 20, 25, 50).")

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

    @property
    def cumulative_labels(self) -> List[Tuple[str, int]]:
        step = 100 // self.n_bins
        return [(f"top{k}", k) for k in range(step, 101, step)]


def setup_logging(verbose: bool, log_file: Optional[str]) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S"))
    root.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S"))
        root.addHandler(fh)
        LOG.info("Also logging to %s", os.path.abspath(log_file))
    # Portkey / judge internals (topk_judge_eval.py loggers)
    for name in ("topk_judge", "topk_judge.api", "topk_judge.run"):
        lg = logging.getLogger(name)
        lg.setLevel(level)
        lg.propagate = True


def _safe_name(s: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", s)


def _stable_seed_int(model: str, dataset: str, bin_label: str, base: int) -> int:
    h = hashlib.sha256(f"{model}|{dataset}|{bin_label}|{base}".encode()).hexdigest()
    return (int(h[:12], 16) % (2**31)) + base


def _git_head(data_root: str) -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", data_root, "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except Exception:
        return None


def load_jsonl_rows(filepath: str) -> List[dict]:
    rows: List[dict] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def traces_to_dataframe(
    rows: List[dict],
) -> pd.DataFrame:
    """One row per trace with confidence, correctness, CoT text."""
    recs: List[Dict[str, Any]] = []
    for row in rows:
        idx = row.get("idx")
        question = row.get("question", "")
        gt = row.get("gt", row.get("answer", ""))
        scores = row.get("score", [])
        code = row.get("code", [])
        pred = row.get("pred", [])
        probs_all = row.get("chosen_token_probs_per_path", {}).get("epoch_0", [])
        if not isinstance(probs_all, list) or not isinstance(code, list):
            continue
        n = min(len(scores), len(code), len(probs_all))
        for ti in range(n):
            probs = probs_all[ti] if ti < len(probs_all) else []
            conf = compute_trace_confidence(probs)
            if np.isnan(conf):
                continue
            cot = code[ti] if isinstance(code[ti], str) else str(code[ti])
            recs.append(
                {
                    "idx": idx,
                    "trace_idx": ti,
                    "confidence": float(conf),
                    "correct": bool(scores[ti]) if ti < len(scores) else False,
                    "pred": pred[ti] if isinstance(pred, list) and ti < len(pred) else None,
                    "cot": cot,
                    "question": question,
                    "gt": gt,
                }
            )
    return pd.DataFrame(recs)


def slice_top_fraction(df: pd.DataFrame, frac: float) -> Tuple[pd.DataFrame, int]:
    """Sort descending by confidence; keep top `frac` of pooled traces (by count)."""
    if df.empty:
        return df, 0
    n = len(df)
    n_keep = max(1, int(np.floor(n * frac)))
    if n_keep >= n:
        n_keep = n
    ranked = df.sort_values("confidence", ascending=False).reset_index(drop=True)
    return ranked.iloc[:n_keep].copy(), n_keep


def assign_disjoint_bins(top_slice: pd.DataFrame, n_bins: int, bin_spec: List[Tuple[str, float, float]]) -> pd.DataFrame:
    """
    ``n_bins`` disjoint equal-count bins along rank order within the kept slice
    (bin 0 = highest-confidence 1/n_bins of this slice).
    """
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


def sample_bin(
    bin_df: pd.DataFrame,
    target_n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n = len(bin_df)
    if n == 0:
        return bin_df.iloc[:0]
    k = min(target_n, n)
    idx = rng.choice(n, size=k, replace=False)
    return bin_df.iloc[np.sort(idx)].reset_index(drop=True)


def mean_std_stderr_ci(scores: Sequence[float]) -> Tuple[float, float, float, float, float]:
    arr = np.asarray([s for s in scores if s is not None and not (isinstance(s, float) and np.isnan(s))], dtype=float)
    n = len(arr)
    if n == 0:
        return (float("nan"),) * 5
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    stderr = float(std / np.sqrt(n)) if n > 0 else float("nan")
    if n < 2:
        return mean, std, stderr, float("nan"), float("nan")
    t_crit = stats.t.ppf(0.975, df=n - 1)
    lo = mean - t_crit * stderr
    hi = mean + t_crit * stderr
    return mean, std, stderr, lo, hi


def bootstrap_mean_ci(
    scores: Sequence[float],
    n_boot: int = 2000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    arr = np.asarray([s for s in scores if s is not None], dtype=float)
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        means[b] = np.mean(sample)
    return (
        float(np.mean(arr)),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def weighted_cumulative_stats(
    bin_stats: List[Dict[str, Any]],
    n_cumulative_bins: int,
) -> Tuple[float, int, float, float]:
    """Combine first n_cumulative_bins bins (1-indexed: 1 = bin1 only)."""
    parts = bin_stats[:n_cumulative_bins]
    all_scores: List[float] = []
    w = 0
    for p in parts:
        for s in p.get("judged_scores", []):
            if s is not None and not (isinstance(s, float) and np.isnan(s)):
                all_scores.append(float(s))
        w += int(p.get("sampled_traces", 0))
    if not all_scores:
        return float("nan"), 0, float("nan"), float("nan")
    mean, _, _, lo, hi = mean_std_stderr_ci(all_scores)
    _, blo, bhi = bootstrap_mean_ci(all_scores, seed=42 + n_cumulative_bins)
    return mean, len(all_scores), blo, bhi


def population_accuracy_cumulative(bin_stats: List[Dict[str, Any]], n_cumulative_bins: int) -> float:
    """
    Weighted mean **population** accuracy over the first ``n_cumulative_bins`` bins
    (same bin boundaries and weights as reasoning). Uses ``total_traces_in_bin`` ×
    ``mean_accuracy_population`` per bin.
    """
    parts = bin_stats[:n_cumulative_bins]
    if not parts:
        return float("nan")
    w = np.array([float(p.get("total_traces_in_bin") or 0) for p in parts], dtype=float)
    acc = np.array([float(p.get("mean_accuracy_population", np.nan)) for p in parts], dtype=float)
    wsum = float(np.nansum(w))
    if wsum <= 0:
        return float("nan")
    return float(np.nansum(acc * w) / wsum)


def judge_one_sample(
    judge: Judge,
    row: pd.Series,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Run GPT-4o-mini judge on one trace; same rubric as topk_judge_eval."""
    judge_scores = judge.score(
        problem=str(row["question"]),
        cot=str(row["cot"]),
        gold=str(row["gt"]),
        flags_summary="No automated flags available.",
        evidence={"final_correct": row["correct"]},
        log_ctx={
            "eval_model": meta.get("model"),
            "dataset": meta.get("dataset"),
            "idx": row["idx"],
            "trace_idx": int(row["trace_idx"]),
            "bin": row.get("bin_label"),
        },
    )
    rs = reasoning_score_from_judge(judge_scores)
    return {
        "idx": row["idx"],
        "trace_idx": int(row["trace_idx"]),
        "bin_label": row["bin_label"],
        "confidence": float(row["confidence"]),
        "correct": bool(row["correct"]),
        "judge_scores": judge_scores,
        "reasoning_score": None if rs is None else float(rs),
        "judge_ok": rs is not None,
    }


def run_pair(
    filepath: str,
    model: str,
    dataset: str,
    judge: Judge,
    output_dir: str,
    samples_per_bin: int,
    base_seed: int,
    parallel: int,
    bin_cfg: BinConfig,
    progress_log_every: int = 5,
) -> Dict[str, Any]:
    """Process one (model, dataset): bin, sample, judge, return summary dict."""
    t_pair = time.perf_counter()
    checkpoint_dir = os.path.join(output_dir, "judging_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ck_path = os.path.join(checkpoint_dir, f"judged_{_safe_name(model)}__{_safe_name(dataset)}.json")

    LOG.info(
        "PAIR START | %s × %s | input=%s",
        model,
        dataset,
        filepath,
    )

    rows_json = load_jsonl_rows(filepath)
    df = traces_to_dataframe(rows_json)
    if df.empty:
        LOG.warning("No traces | %s × %s | %s", model, dataset, filepath)
        return {
            "model": model,
            "dataset": dataset,
            "error": "no_traces",
            "input_file": filepath,
        }

    n_total = len(df)
    top_slice, n_keep = slice_top_fraction(df, bin_cfg.top_pool_frac)
    spec = bin_cfg.bin_spec
    binned = assign_disjoint_bins(top_slice, bin_cfg.n_bins, spec)

    LOG.info(
        "%s × %s | pooled_traces=%d | kept_top_frac=%.2f → n=%d | checkpoint=%s",
        model,
        dataset,
        n_total,
        bin_cfg.top_pool_frac,
        n_keep,
        ck_path,
    )

    # Resume: load existing judged keys
    done_keys: Set[Tuple[Any, int, str]] = set()
    existing_samples: List[dict] = []
    if os.path.exists(ck_path):
        with open(ck_path, encoding="utf-8") as f:
            ck = json.load(f)
        existing_samples = ck.get("judged_samples", [])
        for s in existing_samples:
            done_keys.add((s["idx"], int(s["trace_idx"]), s["bin_label"]))
        LOG.info(
            "%s × %s | resume: %d judged samples already in checkpoint",
            model,
            dataset,
            len(existing_samples),
        )

    bin_summaries: List[Dict[str, Any]] = []
    all_tasks: List[Tuple[pd.Series, int]] = []

    for bin_id, (blabel, _p0, _p1) in enumerate(spec):
        bin_df = binned[binned["bin_id"] == bin_id]
        n_in_bin = len(bin_df)
        seed_i = _stable_seed_int(model, dataset, blabel, base_seed)
        rng = np.random.default_rng(seed_i)
        sampled = sample_bin(bin_df, samples_per_bin, rng)
        to_schedule = 0
        for _, r in sampled.iterrows():
            key = (r["idx"], int(r["trace_idx"]), r["bin_label"])
            if key in done_keys:
                continue
            all_tasks.append((r, bin_id))
            to_schedule += 1
        LOG.info(
            "%s × %s | bin %s | population=%d | sample_target=%d | new_API_tasks=%d | seed=%d",
            model,
            dataset,
            blabel,
            n_in_bin,
            min(samples_per_bin, n_in_bin),
            to_schedule,
            seed_i,
        )

    meta = {"model": model, "dataset": dataset, "input_file": filepath}

    def worker(item: Tuple[pd.Series, int]) -> Dict[str, Any]:
        row, _bid = item
        return judge_one_sample(judge, row, meta)

    new_results: List[dict] = []
    if all_tasks:
        LOG.info(
            "%s × %s | submitting %d judge API calls (parallel=%d)",
            model,
            dataset,
            len(all_tasks),
            parallel,
        )
        t_judge = time.perf_counter()
        done_n = 0
        with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
            futs = {ex.submit(worker, t): t for t in all_tasks}
            for fut in as_completed(futs):
                try:
                    new_results.append(fut.result())
                except Exception as e:
                    LOG.exception("Judge failed: %s", e)
                    warnings.warn(str(e))
                done_n += 1
                if progress_log_every > 0 and (
                    done_n % progress_log_every == 0 or done_n == len(all_tasks)
                ):
                    elapsed = time.perf_counter() - t_judge
                    rate = done_n / elapsed if elapsed > 0 else 0.0
                    rem = len(all_tasks) - done_n
                    eta = rem / rate if rate > 0 else 0.0
                    LOG.info(
                        "%s × %s | judge progress %d/%d | %.2f calls/s | elapsed=%.1fs | ETA~%.1fs",
                        model,
                        dataset,
                        done_n,
                        len(all_tasks),
                        rate,
                        elapsed,
                        eta,
                    )
        LOG.info(
            "%s × %s | judge phase done in %.1fs | new_results=%d",
            model,
            dataset,
            time.perf_counter() - t_judge,
            len(new_results),
        )

    judged_samples = list(existing_samples)
    judged_samples.extend(new_results)
    # Dedup by (idx, trace_idx, bin_label) — keep last
    dedup: Dict[Tuple[Any, int, str], dict] = {}
    for s in judged_samples:
        k = (s["idx"], int(s["trace_idx"]), s["bin_label"])
        dedup[k] = s
    judged_samples = list(dedup.values())

    payload = {
        "metadata": {
            "model": model,
            "dataset": dataset,
            "input_file": filepath,
            "n_pooled_traces": n_total,
            "n_top_pool_kept": n_keep,
            "top_pool_fraction": bin_cfg.top_pool_frac,
            "samples_per_bin_target": samples_per_bin,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "judged_samples": judged_samples,
    }
    fd, tmp = tempfile.mkstemp(prefix=".rcb_", suffix=".json.tmp", dir=checkpoint_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, ck_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    # Per-bin aggregates from judged samples
    for bin_id, (blabel, _a, _b) in enumerate(spec):
        bin_df = binned[binned["bin_id"] == bin_id]
        n_pop = len(bin_df)
        acc_pop = float(bin_df["correct"].mean()) if n_pop else float("nan")
        js = [s for s in judged_samples if s.get("bin_label") == blabel]
        scores = [s["reasoning_score"] for s in js if s.get("judge_ok")]
        mean, std, se, lo, hi = mean_std_stderr_ci(scores)
        _, blo, bhi = bootstrap_mean_ci(scores, seed=base_seed + bin_id)
        bin_summaries.append(
            {
                "model": model,
                "dataset": dataset,
                "bin_label": blabel,
                "bin_start_pct": spec[bin_id][1],
                "bin_end_pct": spec[bin_id][2],
                "total_traces_in_bin": n_pop,
                "sampled_traces": len(js),
                "random_seed": _stable_seed_int(model, dataset, blabel, base_seed),
                "mean_reasoning_score": mean,
                "std_reasoning_score": std,
                "stderr_reasoning_score": se,
                "ci_lower": lo if not np.isnan(lo) else blo,
                "ci_upper": hi if not np.isnan(hi) else bhi,
                "ci_method": "t_95" if len(scores) > 1 else "bootstrap_only",
                "bootstrap_ci_lower": blo,
                "bootstrap_ci_upper": bhi,
                "mean_accuracy_population": acc_pop,
                "judged_scores": scores,
            }
        )

    LOG.info(
        "PAIR END | %s × %s | wall_s=%.1f | checkpoint=%s",
        model,
        dataset,
        time.perf_counter() - t_pair,
        ck_path,
    )

    return {
        "model": model,
        "dataset": dataset,
        "input_file": filepath,
        "n_pooled_traces": n_total,
        "n_top_pool_kept": n_keep,
        "bin_summaries": bin_summaries,
        "checkpoint_path": ck_path,
    }


def aggregate_csvs(
    all_pair_results: List[Dict[str, Any]], output_dir: str, bin_cfg: BinConfig
) -> None:
    bin_rows: List[dict] = []
    cum_rows: List[dict] = []

    for pr in all_pair_results:
        if pr.get("error"):
            continue
        bs = pr.get("bin_summaries", [])
        for b in bs:
            bin_rows.append(
                {
                    "model": b["model"],
                    "dataset": b["dataset"],
                    "bin_label": b["bin_label"],
                    "bin_start_pct": b["bin_start_pct"],
                    "bin_end_pct": b["bin_end_pct"],
                    "total_traces_in_bin": b["total_traces_in_bin"],
                    "sampled_traces": b["sampled_traces"],
                    "random_seed": b["random_seed"],
                    "mean_reasoning_score": b["mean_reasoning_score"],
                    "std_reasoning_score": b["std_reasoning_score"],
                    "stderr_reasoning_score": b["stderr_reasoning_score"],
                    "ci_lower": b["ci_lower"],
                    "ci_upper": b["ci_upper"],
                    "bootstrap_ci_lower": b["bootstrap_ci_lower"],
                    "bootstrap_ci_upper": b["bootstrap_ci_upper"],
                    "mean_accuracy_population": b["mean_accuracy_population"],
                }
            )

        # cumulative
        for cum_i, (label, pct) in enumerate(bin_cfg.cumulative_labels, start=1):
            mean_r, n_j, blo, bhi = weighted_cumulative_stats(bs, cum_i)
            cum_acc = population_accuracy_cumulative(bs, cum_i)
            cum_rows.append(
                {
                    "model": pr["model"],
                    "dataset": pr["dataset"],
                    "topk_pct": pct,
                    "topk_label": label,
                    "judged_samples_used": n_j,
                    "mean_reasoning_score": mean_r,
                    "ci_lower": blo,
                    "ci_upper": bhi,
                    "cumulative_accuracy_population": cum_acc,
                }
            )

    pd.DataFrame(bin_rows).to_csv(
        os.path.join(output_dir, "reasoning_by_confidence_bin.csv"), index=False
    )
    pd.DataFrame(cum_rows).to_csv(
        os.path.join(output_dir, "reasoning_cumulative_topk.csv"), index=False
    )


def load_bin_config_from_metadata(output_dir: str) -> BinConfig:
    meta_path = os.path.join(output_dir, "reasoning_sampling_metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            m = json.load(f)
        return BinConfig(
            float(m.get("top_pool_fraction", 1.0)),
            int(m.get("n_bins", 10)),
        )
    return BinConfig(1.0, 10)


def infer_bin_order(df: pd.DataFrame) -> List[str]:
    g = df.groupby("bin_label", sort=False)["bin_start_pct"].min().reset_index()
    return list(g.sort_values("bin_start_pct")["bin_label"])


def backfill_cumulative_accuracy_if_missing(
    output_dir: str, df: pd.DataFrame, dfc: pd.DataFrame
) -> pd.DataFrame:
    """
    Add ``cumulative_accuracy_population`` to the cumulative CSV when missing:
    weighted population accuracy over the first k bins (same definition as plots).
    """
    if (
        "cumulative_accuracy_population" in dfc.columns
        and dfc["cumulative_accuracy_population"].notna().any()
    ):
        return dfc

    bin_order = infer_bin_order(df)
    vals: List[float] = []
    for _, row in dfc.iterrows():
        model, dataset = row["model"], row["dataset"]
        sub = df[(df["model"] == model) & (df["dataset"] == dataset)]
        if sub.empty:
            vals.append(float("nan"))
            continue
        ms = sub.set_index("bin_label").reindex(bin_order).reset_index()
        pop = ms["mean_accuracy_population"].values
        w = ms["total_traces_in_bin"].values.astype(float)
        pct = int(row["topk_pct"])
        cum_pcts = sorted(
            dfc[(dfc["model"] == model) & (dfc["dataset"] == dataset)]["topk_pct"].unique()
        )
        try:
            j = cum_pcts.index(pct)
        except ValueError:
            vals.append(float("nan"))
            continue
        cum_i = j + 1
        wsum = w[:cum_i].sum()
        if wsum <= 0:
            vals.append(float("nan"))
        else:
            vals.append(float(np.dot(pop[:cum_i], w[:cum_i]) / wsum))
    dfc = dfc.copy()
    dfc["cumulative_accuracy_population"] = vals
    cum_path = os.path.join(output_dir, "reasoning_cumulative_topk.csv")
    dfc.to_csv(cum_path, index=False)
    LOG.info("Backfilled cumulative_accuracy_population in %s", cum_path)
    return dfc


def plot_figures(output_dir: str, data_root: str, bin_cfg: Optional[BinConfig] = None) -> None:
    bin_csv = os.path.join(output_dir, "reasoning_by_confidence_bin.csv")
    cum_csv = os.path.join(output_dir, "reasoning_cumulative_topk.csv")
    if not os.path.isfile(bin_csv):
        LOG.error("Missing %s — run `run` first", bin_csv)
        return

    df = pd.read_csv(bin_csv)
    dfc = pd.read_csv(cum_csv) if os.path.isfile(cum_csv) else None

    if bin_cfg is None:
        bin_cfg = load_bin_config_from_metadata(output_dir)

    if dfc is not None:
        dfc = backfill_cumulative_accuracy_if_missing(output_dir, df, dfc)

    have = set(df["dataset"].unique())
    datasets = [d for d in DATASET_PANEL_ORDER if d in have] + sorted(have - set(DATASET_PANEL_ORDER))
    models = sorted(df["model"].unique())
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(len(models), 1))
    colors = {m: cmap(i) for i, m in enumerate(models)}

    bin_order = infer_bin_order(df)
    n_bins_eff = len(bin_order)
    pool_desc = (
        "full ranked pool"
        if bin_cfg.top_pool_frac >= 1.0 - 1e-9
        else f"top {bin_cfg.top_pool_frac:.0%} of traces by confidence"
    )
    x_pos = np.arange(len(bin_order))

    # 1) Reasoning vs bin per dataset (+ population accuracy on same bins, right axis)
    for ds in datasets:
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax2 = ax1.twinx()
        sub = df[df["dataset"] == ds]
        for mi, model in enumerate(models):
            msub = sub[sub["model"] == model]
            if msub.empty:
                continue
            msub = msub.set_index("bin_label").reindex(bin_order).reset_index()
            y = msub["mean_reasoning_score"].values
            yerr_lo = y - msub["ci_lower"].values
            yerr_hi = msub["ci_upper"].values - y
            offset = (mi - len(models) / 2) * 0.02
            ax1.errorbar(
                x_pos + offset,
                y,
                yerr=[yerr_lo, yerr_hi],
                label=f"{model} (reasoning)",
                color=colors[model],
                marker="o",
                capsize=2,
                linewidth=1.2,
            )
            acc_pop = msub["mean_accuracy_population"].values
            ax2.plot(
                x_pos + offset + 0.008,
                acc_pop,
                linestyle="--",
                marker="s",
                markersize=4,
                color=colors[model],
                alpha=0.85,
                label=f"{model} (acc, pop.)",
            )
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f"{b} ({pool_desc})" for b in bin_order], rotation=15, ha="right")
        ax1.set_ylabel("Mean reasoning score (judged, 0–1)", color="tab:blue")
        ax2.set_ylabel("Accuracy — all traces in bin (population, 0–1)", color="tab:orange")
        ax1.set_xlabel(f"Confidence bin (% rank within {pool_desc})")
        ax1.set_title(f"{ds}: reasoning vs population accuracy (identical bins)")
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        lines1, lab1 = ax1.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, lab1 + lab2, loc="best", fontsize=6)
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, f"reasoning_bins_{_safe_name(ds)}.png"),
            bbox_inches="tight",
        )
        plt.close(fig)

    # 2) Cumulative reasoning (+ same-bin cumulative population accuracy, right axis)
    if dfc is not None:
        cum_pcts = sorted(dfc["topk_pct"].unique())
        xp_c = np.arange(len(cum_pcts))
        for ds in datasets:
            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax2 = ax1.twinx()
            sub = dfc[dfc["dataset"] == ds]
            for mi, model in enumerate(models):
                msub = sub[sub["model"] == model].set_index("topk_pct").reindex(cum_pcts).reset_index()
                if msub["mean_reasoning_score"].isna().all():
                    continue
                ax1.plot(
                    xp_c,
                    msub["mean_reasoning_score"].values,
                    marker="o",
                    label=f"{model} (reasoning)",
                    color=colors[model],
                )
                if "cumulative_accuracy_population" in msub.columns:
                    acc_c = msub["cumulative_accuracy_population"].values.astype(float)
                    if not np.all(np.isnan(acc_c)):
                        ax2.plot(
                            xp_c,
                            acc_c,
                            linestyle="--",
                            marker="s",
                            markersize=4,
                            color=colors[model],
                            alpha=0.85,
                            label=f"{model} (acc, pop.)",
                        )
            ax1.set_xticks(xp_c)
            ax1.set_xticklabels([str(int(p)) for p in cum_pcts])
            ax1.set_ylabel("Mean reasoning score (judged)", color="tab:blue")
            ax2.set_ylabel("Cumulative population accuracy (same bins)", color="tab:orange")
            ax1.set_xlabel(f"Cumulative % of {pool_desc} (by rank)")
            ax1.set_title(f"{ds}: cumulative reasoning vs cumulative accuracy")
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis="y", labelcolor="tab:blue")
            ax2.tick_params(axis="y", labelcolor="tab:orange")
            lines1, lab1 = ax1.get_legend_handles_labels()
            lines2, lab2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, lab1 + lab2, loc="best", fontsize=6)
            fig.tight_layout()
            fig.savefig(
                os.path.join(output_dir, f"reasoning_cumulative_{_safe_name(ds)}.png"),
                bbox_inches="tight",
            )
            plt.close(fig)

    # 3) Comparison: population accuracy cumulative vs reasoning cumulative (same bin structure)
    acc_cum: Dict[Tuple[str, str], List[float]] = {}
    for ds in datasets:
        sub = df[df["dataset"] == ds]
        for model in models:
            key = (ds, model)
            ms = sub[sub["model"] == model].set_index("bin_label").reindex(bin_order).reset_index()
            pop = ms["mean_accuracy_population"].values
            w = ms["total_traces_in_bin"].values.astype(float)
            cum_acc: List[float] = []
            for k in range(1, n_bins_eff + 1):
                wsum = w[:k].sum()
                if wsum <= 0:
                    cum_acc.append(float("nan"))
                else:
                    cum_acc.append(float(np.dot(pop[:k], w[:k]) / wsum))
            acc_cum[key] = cum_acc

    if dfc is not None:
        cum_pcts3 = sorted(dfc["topk_pct"].unique())
        xp3 = np.arange(len(cum_pcts3))
        for ds in datasets:
            fig, ax1 = plt.subplots(figsize=(9, 5))
            ax2 = ax1.twinx()
            for mi, model in enumerate(models):
                subr = (
                    dfc[(dfc["dataset"] == ds) & (dfc["model"] == model)]
                    .set_index("topk_pct")
                    .reindex(cum_pcts3)
                    .reset_index()
                )
                if subr["mean_reasoning_score"].isna().all():
                    continue
                ax1.plot(
                    xp3,
                    subr["mean_reasoning_score"].values,
                    marker="o",
                    linestyle="-",
                    color=colors[model],
                    label=f"{model} (reasoning)",
                )
                if "cumulative_accuracy_population" in subr.columns:
                    acc_y = subr["cumulative_accuracy_population"].values.astype(float)
                else:
                    acc_line = acc_cum.get((ds, model), [])
                    acc_y = (
                        np.asarray(acc_line[: len(cum_pcts3)], dtype=float)
                        if acc_line and len(acc_line) >= len(cum_pcts3)
                        else np.full(len(cum_pcts3), np.nan)
                    )
                if len(acc_y) == len(cum_pcts3) and not np.all(np.isnan(acc_y)):
                    ax2.plot(
                        xp3,
                        acc_y,
                        marker="s",
                        linestyle="--",
                        color=colors[model],
                        alpha=0.7,
                        label=f"{model} (acc pop.)",
                    )
            ax1.set_xticks(xp3)
            ax1.set_xticklabels([str(int(p)) for p in cum_pcts3])
            ax1.set_ylabel("Reasoning score (judged)", color="tab:blue")
            ax2.set_ylabel("Accuracy (population, same bins)", color="tab:orange")
            ax1.set_xlabel(f"Cumulative % of {pool_desc}")
            ax1.set_title(f"{ds}: reasoning vs accuracy (aligned bins)")
            ax1.grid(True, alpha=0.3)
            lines1, lab1 = ax1.get_legend_handles_labels()
            lines2, lab2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, lab1 + lab2, loc="best", fontsize=6)
            fig.tight_layout()
            fig.savefig(
                os.path.join(output_dir, f"reasoning_vs_accuracy_bins_{_safe_name(ds)}.png"),
                bbox_inches="tight",
            )
            plt.close(fig)

    LOG.info("Figures written to %s", output_dir)


def write_metadata(
    output_dir: str,
    data_root: str,
    base_seed: int,
    samples_per_bin: int,
    all_results: List[Dict[str, Any]],
    failures: List[str],
    bin_cfg: BinConfig,
) -> None:
    meta = {
        "seed_base": base_seed,
        "sample_size_target": samples_per_bin,
        "top_pool_fraction": bin_cfg.top_pool_frac,
        "n_bins": bin_cfg.n_bins,
        "bin_definitions": bin_cfg.bin_spec,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": os.path.abspath(data_root),
        "git_commit_at_data_root": _git_head(data_root),
        "judge_model_default": DEFAULT_JUDGE_MODEL,
        "paths": {
            "checkpoints_dir": os.path.join(output_dir, "judging_checkpoints"),
            "reasoning_by_confidence_bin_csv": "reasoning_by_confidence_bin.csv",
            "reasoning_cumulative_topk_csv": "reasoning_cumulative_topk.csv",
        },
        "pairs": [
            {
                "model": r.get("model"),
                "dataset": r.get("dataset"),
                "n_pooled_traces": r.get("n_pooled_traces"),
                "n_top_pool_kept": r.get("n_top_pool_kept"),
                "checkpoint": r.get("checkpoint_path"),
                "error": r.get("error"),
            }
            for r in all_results
        ],
        "failures": failures,
        "caveats": [
            "Bins are disjoint equal-count slices of the ranked trace list AFTER optionally restricting "
            "to the top fraction by confidence (see top_pool_fraction).",
            "Reasoning scores use the same GPT-4o-mini 4-dimension rubric as topk_judge_eval.py (normalized to 0–1).",
            "Cumulative rows weight bin means by the number of successfully judged traces in each bin.",
            "cumulative_accuracy_population is weighted population accuracy over the same bins as reasoning "
            "(not topk_ablation.py; that uses a different pooled top-K% definition).",
        ],
    }
    with open(os.path.join(output_dir, "reasoning_sampling_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def cmd_run(args: argparse.Namespace) -> None:
    setup_logging(args.verbose, args.log_file)
    os.makedirs(args.output_dir, exist_ok=True)
    fm = build_file_map(args.data_root)
    if not fm:
        LOG.error("No JSONL found under data_root")
        sys.exit(1)
    if not args.portkey_key:
        LOG.error("Set PORTKEY_API_KEY or pass --portkey-key")
        sys.exit(1)

    try:
        judge = Judge(portkey_api_key=args.portkey_key, model=args.judge_model)
    except Exception as e:
        LOG.error("Cannot init judge: %s", e)
        sys.exit(1)

    failures: List[str] = []
    all_results: List[Dict[str, Any]] = []

    try:
        bin_cfg = BinConfig(args.top_pool_frac, args.n_bins)
    except ValueError as e:
        LOG.error("%s", e)
        sys.exit(2)

    items = sorted(fm.items())
    if getattr(args, "max_pairs", 0) and args.max_pairs > 0:
        items = items[: args.max_pairs]
        LOG.info("Limiting to %d (model,dataset) pairs", len(items))

    est_calls = len(items) * bin_cfg.n_bins * min(args.samples_per_bin, 50)
    LOG.info(
        "Run plan | pairs=%d | samples_per_bin=%d | top_pool_frac=%.2f | n_bins=%d | ~max API calls=%d | parallel=%d | judge=%s",
        len(items),
        args.samples_per_bin,
        bin_cfg.top_pool_frac,
        bin_cfg.n_bins,
        est_calls,
        args.parallel,
        args.judge_model,
    )
    t_run = time.perf_counter()

    for pair_i, ((model, dataset), fp) in enumerate(items, start=1):
        LOG.info(
            "GLOBAL [%d/%d] next pair: %s × %s",
            pair_i,
            len(items),
            model,
            dataset,
        )
        try:
            pr = run_pair(
                fp,
                model,
                dataset,
                judge,
                args.output_dir,
                args.samples_per_bin,
                args.seed,
                args.parallel,
                bin_cfg,
                progress_log_every=getattr(args, "progress_log_every", 5),
            )
            all_results.append(pr)
            if pr.get("error"):
                failures.append(f"{model}|{dataset}: {pr['error']}")
        except Exception as e:
            LOG.exception("Pair failed %s %s", model, dataset)
            failures.append(f"{model}|{dataset}: {e}")
            all_results.append({"model": model, "dataset": dataset, "error": str(e)})

    aggregate_csvs(all_results, args.output_dir, bin_cfg)
    write_metadata(
        args.output_dir,
        args.data_root,
        args.seed,
        args.samples_per_bin,
        all_results,
        failures,
        bin_cfg,
    )
    try:
        plot_figures(args.output_dir, args.data_root, bin_cfg)
    except Exception:
        LOG.exception("plot_figures failed (CSVs are still valid); run: python reasoning_confidence_bins.py plot ...")
    LOG.info(
        "Done. wall_s=%.1f | outputs in %s | failures=%d",
        time.perf_counter() - t_run,
        os.path.abspath(args.output_dir),
        len(failures),
    )


def cmd_plot(args: argparse.Namespace) -> None:
    setup_logging(args.verbose, args.log_file)
    plot_figures(args.output_dir, args.data_root or ".", None)


def cmd_topk_accuracy(args: argparse.Namespace) -> None:
    """
    Judge-free: accuracy on the top K%% of traces by confidence (count-based),
    after optionally keeping only the top ``top_pool_frac`` slice (same as ``run``).
    """
    setup_logging(args.verbose, args.log_file)
    fm = build_file_map(args.data_root)
    if not fm:
        LOG.error("No JSONL found under data_root")
        sys.exit(1)
    k_pcts = sorted(
        {int(x.strip()) for x in args.k_pcts.replace(",", " ").split() if x.strip()}
    )
    if not k_pcts or any(k <= 0 or k > 100 for k in k_pcts):
        LOG.error("--k-pcts must be integers in 1..100")
        sys.exit(2)

    rows: List[Dict[str, Any]] = []
    for (model, dataset), fp in sorted(fm.items()):
        rows_json = load_jsonl_rows(fp)
        df = traces_to_dataframe(rows_json)
        if df.empty:
            LOG.warning("No traces | %s × %s | %s", model, dataset, fp)
            continue
        top_slice, n_keep = slice_top_fraction(df, args.top_pool_frac)
        ranked = top_slice.sort_values("confidence", ascending=False).reset_index(drop=True)
        n = len(ranked)
        for k in k_pcts:
            if k == 100:
                n_take = n
            else:
                n_take = max(1, int(np.floor(n * k / 100.0)))
            sub = ranked.iloc[:n_take]
            acc = float(sub["correct"].mean()) if len(sub) else float("nan")
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "top_k_pct": k,
                    "accuracy": acc,
                    "n_traces_used": int(len(sub)),
                    "n_traces_in_pool": int(n),
                    "top_pool_frac": float(args.top_pool_frac),
                }
            )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "topk_trace_accuracy.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    LOG.info(
        "Wrote %s (%d rows) | pool=top %.0f%% by conf, then top K%% of that pool by count",
        os.path.abspath(out_path),
        len(rows),
        args.top_pool_frac * 100.0,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Reasoning by pooled confidence bins")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Bin, sample, judge, aggregate, plot")
    pr.add_argument("--data-root", required=True)
    pr.add_argument("--output-dir", default="outputs/reasoning_confidence_bins_results")
    pr.add_argument("--samples-per-bin", type=int, default=DEFAULT_SAMPLES_PER_BIN)
    pr.add_argument(
        "--top-pool-frac",
        type=float,
        default=1.0,
        help="Keep only this fraction of highest-confidence traces before binning (1.0 = full pool; needed for top-100%% cumulative).",
    )
    pr.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of equal-count disjoint bins within the kept slice; must divide 100 (default 10 → top10…top100 cumulative, including 70 and 100).",
    )
    pr.add_argument("--seed", type=int, default=42)
    pr.add_argument("--parallel", type=int, default=8)
    pr.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="If >0, only process the first N (model,dataset) pairs after sorting (smoke test)",
    )
    pr.add_argument("--portkey-key", default=os.environ.get("PORTKEY_API_KEY"))
    pr.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    pr.add_argument("-v", "--verbose", action="store_true")
    pr.add_argument("--log-file", default=None)
    pr.add_argument(
        "--progress-log-every",
        type=int,
        default=5,
        metavar="N",
        help="Log judge progress every N completed calls (default 5; 0 = only final)",
    )
    pr.set_defaults(func=cmd_run)

    pp = sub.add_parser("plot", help="Regenerate plots from CSVs")
    pp.add_argument("--output-dir", default="outputs/reasoning_confidence_bins_results")
    pp.add_argument("--data-root", default=".", help="For loading topk_ablation_results.csv")
    pp.add_argument("-v", "--verbose", action="store_true")
    pp.add_argument("--log-file", default=None)
    pp.set_defaults(func=cmd_plot)

    pt = sub.add_parser(
        "topk-accuracy",
        help="No judge: recompute accuracy on top-K%% traces (by confidence rank) for chosen K values",
    )
    pt.add_argument("--data-root", required=True)
    pt.add_argument("--output-dir", default="outputs/reasoning_confidence_bins_results")
    pt.add_argument(
        "--top-pool-frac",
        type=float,
        default=0.5,
        help="Same as ``run``: keep this fraction of highest-confidence traces before taking top-K%% (default 0.5 = match original 5-bin slice). Use 1.0 for full pool.",
    )
    pt.add_argument(
        "--k-pcts",
        type=str,
        default="10,20,30,40,50",
        help="Comma or space-separated K values (each: include the ⌊K%%⌋ most confident traces in the pool).",
    )
    pt.add_argument("-v", "--verbose", action="store_true")
    pt.add_argument("--log-file", default=None)
    pt.set_defaults(func=cmd_topk_accuracy)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
