#!/usr/bin/env python3
"""
Build per–model-benchmark parquet files for ``downstream_validation.py``.

Data sources (this repo):
  - ``source_pass16_jsonl_by_model*/**/*.jsonl`` — per-problem pass@k traces, token probs, correctness
  - ``reasoning_confidence_bins_results/judging_checkpoints/judged_<Model>__<Dataset>.json`` —
    LLM judge ``reasoning_score`` (and bin metadata) for a **fixed sample** of traces per pair

Rows are the **intersection**: only traces that appear in the judge file get a ``reasoning_score``.
Typically ~250 rows per parquet (not 16 × n_problems). ``downstream_validation.py`` still runs:
problems may have 0–few judged traces; BoN / random use whatever rows exist per problem.

Usage:
  python build_downstream_parquets.py --repo_root . --output_dir results/
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from topk_ablation import compute_trace_confidence, extract_model_dataset, load_jsonl_raw

# Parquet basename slugs (must match downstream_validation.MODEL_ALIASES / BENCHMARK_ALIASES)
MODEL_SLUG: Dict[str, str] = {
    "DS-R1-7B": "ds_r1_7b",
    "DS-R1-1.5B": "ds_r1_1_5b",
    "Qwen3-4B": "qwen3_4b",
    "LLaMA-3.1-8B": "llama_3_1_8b",
    "Qwen2.5-7B": "qwen2_5_7b",
    "Qwen2.5-Math": "qwen2_5_math",
    "Gemma-7B": "gemma_7b",
    "Phi-4": "phi_4",
    "Phi-4-Reas.": "phi_4_reasoning",
}

BENCH_SLUG: Dict[str, str] = {
    "GSM8K": "gsm8k",
    "MATH500": "math500",
    "SVAMP": "svamp",
    "AQuA": "aqua",
    "GPQA": "gpqa",
    "CommonsenseQA": "commonsenseqa",
}


def filter_paths_for_dataset(paths: List[str], canonical_dataset: str) -> List[str]:
    """Drop stray runs (e.g. ``test_custom_custom``) that share the same filename prefix."""
    hints = {
        "GSM8K": "test_gsm8k",
        "MATH500": "test_math500",
        "SVAMP": "test_svamp",
        "AQuA": "test_aqua",
        "GPQA": "test_gpqa",
        "CommonsenseQA": "test_commonsense_qa",
    }
    h = hints.get(canonical_dataset)
    if not h:
        return paths
    good = [p for p in paths if h in p.replace("\\", "/").lower()]
    return good if good else paths


def pick_best_jsonl(paths: List[str]) -> str:
    """Prefer *_processed* paths, then primary ``source_pass16_jsonl_by_model`` folder."""
    if len(paths) == 1:
        return paths[0]

    def score(p: str) -> Tuple[bool, int, int]:
        pnorm = p.replace("\\", "/")
        processed = "_processed" in pnorm
        if "/source_pass16_jsonl_by_model/" in pnorm and "by_model " not in pnorm:
            pri = 3
        elif "data/pass16_sample 2/" in pnorm:
            pri = 2
        elif "data/pass16_sample 3/" in pnorm:
            pri = 1
        else:
            pri = 0
        return (processed, pri, -len(p))

    return sorted(paths, key=score, reverse=True)[0]


def discover_jsonl_groups(repo_root: str) -> Dict[Tuple[str, str], str]:
    pattern = os.path.join(repo_root, "data/pass16_sample*", "**", "*.jsonl")
    all_files = glob.glob(pattern, recursive=True)
    groups: Dict[Tuple[str, str], List[str]] = {}
    for fp in all_files:
        model, ds = extract_model_dataset(fp)
        groups.setdefault((model, ds), []).append(fp)
    out: Dict[Tuple[str, str], str] = {}
    for k, paths in groups.items():
        filtered = filter_paths_for_dataset(paths, k[1])
        out[k] = pick_best_jsonl(filtered)
    return out


def load_judge_map(judge_path: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
    with open(judge_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for s in data.get("judged_samples", []):
        if not s.get("judge_ok", True):
            continue
        rs = s.get("reasoning_score")
        if rs is None:
            continue
        key = (int(s["idx"]), int(s["trace_idx"]))
        out[key] = s
    return out


def traces_from_jsonl_merged(
    jsonl_path: str,
    judge_map: Dict[Tuple[int, int], Dict[str, Any]],
    model: str,
    benchmark_slug: str,
) -> List[Dict[str, Any]]:
    raw = load_jsonl_raw(jsonl_path)
    rows: List[Dict[str, Any]] = []
    for rec in raw:
        pid = rec["idx"]
        scores = rec["scores"]
        token_probs_all = rec["token_probs_all"]
        n = len(scores)
        for trace_idx in range(n):
            key = (int(pid), int(trace_idx))
            if key not in judge_map:
                continue
            probs = token_probs_all[trace_idx] if trace_idx < len(token_probs_all) else []
            conf = compute_trace_confidence(probs)
            if np.isnan(conf):
                continue
            j = judge_map[key]
            rs = float(j["reasoning_score"])
            # 0–1 from judge → store as 0–100 for downstream (explicit)
            rs_100 = rs * 100.0 if rs <= 1.5 else rs
            rows.append(
                {
                    "model": model,
                    "benchmark": benchmark_slug,
                    "problem_id": int(pid),
                    "trace_id": int(trace_idx),
                    "confidence": float(conf),
                    "correct": bool(scores[trace_idx]) if trace_idx < len(scores) else False,
                    "reasoning_score": float(rs_100),
                }
            )
    return rows


def judge_path(repo_root: str, model: str, dataset: str) -> str:
    return os.path.join(
        repo_root,
        "outputs/reasoning_confidence_bins_results",
        "judging_checkpoints",
        f"judged_{model}__{dataset}.json",
    )


def jsonl_from_judge_metadata(repo_root: str, judge_json_path: str) -> Optional[str]:
    """Use ``metadata.input_file`` so rows align with the judge run (required for correct idx/trace_idx)."""
    try:
        with open(judge_json_path, "r", encoding="utf-8") as f:
            meta = json.load(f).get("metadata") or {}
        inp = meta.get("input_file")
        if not inp:
            return None
        cand = os.path.normpath(os.path.join(repo_root, inp.lstrip("./")))
        return cand if os.path.isfile(cand) else None
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Build downstream_validation parquet files from JSONL + judges.")
    ap.add_argument("--repo_root", type=str, default=".", help="Repository root (contains source_pass16_* and judging_checkpoints)")
    ap.add_argument("--output_dir", type=str, default="outputs/results", help="Directory for *.parquet files")
    args = ap.parse_args()
    repo_root = os.path.abspath(args.repo_root)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    jsonl_map = discover_jsonl_groups(repo_root)
    print(f"Discovered {len(jsonl_map)} JSONL groups (fallback map).", flush=True)

    written = 0
    for (model, dataset) in sorted(jsonl_map.keys()):
        slug_m = MODEL_SLUG.get(model)
        slug_b = BENCH_SLUG.get(dataset)
        if not slug_m or not slug_b:
            print(f"SKIP (unknown slug): {model} / {dataset}", flush=True)
            continue
        jpath = judge_path(repo_root, model, dataset)
        if not os.path.isfile(jpath):
            print(f"SKIP (no judge file): {jpath}", flush=True)
            continue
        meta_path = jsonl_from_judge_metadata(repo_root, jpath)
        jsonl_path = meta_path or jsonl_map.get((model, dataset))
        if not jsonl_path or not os.path.isfile(jsonl_path):
            print(f"SKIP (no JSONL): {model} / {dataset}", flush=True)
            continue
        src = "metadata" if meta_path else "glob"
        judge_map = load_judge_map(jpath)
        rows = traces_from_jsonl_merged(jsonl_path, judge_map, model, slug_b)
        if not rows:
            print(f"WARN: no merged rows for {model} / {dataset}", flush=True)
            continue
        df = pd.DataFrame(rows)
        fname = f"{slug_m}__{slug_b}.parquet"
        fpath = os.path.join(out_dir, fname)
        df.to_parquet(fpath, index=False)
        written += 1
        print(
            f"Wrote {fname}  ({len(df)} rows)  [{src}] ← {os.path.basename(jsonl_path)}",
            flush=True,
        )

    print(f"Done. {written} parquet files in {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
