#!/usr/bin/env python3
"""
Generate a stratified manifest for *post-filtered* samples (results/filtered_cot/*/).

Goal: choose a ~500-sized subset that covers all model×dataset combos while keeping
judge-call costs bounded.

Sampling strategy (per combo/file):
  - cap K samples per combo (default K=11), but take all if N<K
  - stratify by:
      (A) original_correct (True/False)
      (B) mini-judge overall fused score quantiles (low/med/high) within that combo
  - allocate roughly evenly across strata when possible, otherwise backfill from
    available strata.

Outputs:
  - experiments/paper_figures/judge_validation/validation_manifest_postfiltered.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "experiments" / "paper_figures" / "judge_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILTERED_DIRS = {
    "gsm8k": REPO / "results" / "filtered_cot" / "gsm8k",
    "svamp": REPO / "results" / "filtered_cot" / "svamp",
    "aqua": REPO / "results" / "filtered_cot" / "aqua",
    "gpqa": REPO / "results" / "filtered_cot" / "gpqa",
    "commonsense_qa": REPO / "results" / "filtered_cot" / "commonsense",
}


@dataclass(frozen=True)
class Item:
    idx: int
    correct: bool
    overall: float


def load_items(results_path: Path) -> List[Item]:
    with results_path.open() as f:
        data = json.load(f)
    out: List[Item] = []
    for r in data.get("results", []):
        if r.get("status") != "ok":
            continue
        fs = r.get("fused_scores") or {}
        overall = fs.get("overall", None)
        if overall is None:
            continue
        out.append(Item(idx=int(r["idx"]), correct=bool(r.get("original_correct", False)), overall=float(overall)))
    return out


def quantile_bucket(values: np.ndarray) -> np.ndarray:
    """
    Map each value to {0,1,2} by 1/3 and 2/3 quantiles (low/med/high).
    If all values equal, bucket everything to 1 (med).
    """
    if values.size == 0:
        return np.array([], dtype=int)
    if float(values.min()) == float(values.max()):
        return np.ones(values.shape[0], dtype=int)
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    b = np.zeros(values.shape[0], dtype=int)
    b[values > q1] = 1
    b[values > q2] = 2
    return b


def pick_stratified(items: List[Item], k: int) -> List[int]:
    if len(items) <= k:
        return sorted([it.idx for it in items])

    vals = np.array([it.overall for it in items], dtype=float)
    buckets = quantile_bucket(vals)  # 0/1/2

    # Build strata: correct(0/1) × bucket(0/1/2)
    strata: Dict[Tuple[int, int], List[Item]] = {}
    for it, b in zip(items, buckets):
        key = (1 if it.correct else 0, int(b))
        strata.setdefault(key, []).append(it)

    # Stable-ish ordering within stratum: take extremes first
    for key in strata:
        # sort by overall: low->high; we'll later alternate from ends
        strata[key].sort(key=lambda x: x.overall)

    # Allocation: try to spread across 6 strata. Give at least 1 to each non-empty,
    # then round-robin fill.
    chosen: List[int] = []
    non_empty = [key for key, v in strata.items() if v]
    # deterministic key order: incorrect strata first to force coverage
    non_empty.sort(key=lambda t: (t[0], t[1]))  # correct flag then bucket

    # First pass: 1 per stratum (if we have budget)
    for key in non_empty:
        if len(chosen) >= k:
            break
        v = strata[key]
        # pick median element to avoid only extremes in the first pass
        take = v.pop(len(v) // 2)
        chosen.append(take.idx)

    # Remaining: round-robin over strata, alternating extremes for diversity
    keys_cycle = [key for key in non_empty if strata.get(key)]
    cycle_i = 0
    take_from_low = True
    while len(chosen) < k and keys_cycle:
        key = keys_cycle[cycle_i % len(keys_cycle)]
        v = strata[key]
        if not v:
            keys_cycle = [kk for kk in keys_cycle if strata.get(kk)]
            cycle_i += 1
            continue
        take = v.pop(0 if take_from_low else -1)
        chosen.append(take.idx)
        take_from_low = not take_from_low
        cycle_i += 1

    # If still short due to weirdness, backfill from any remaining items
    if len(chosen) < k:
        remaining = [it.idx for key in non_empty for it in strata.get(key, [])]
        for idx in remaining:
            if len(chosen) >= k:
                break
            chosen.append(idx)

    return sorted(set(chosen))[:k]


def main():
    cap_k = 11
    manifest = {
        "version": "postfiltered_v1",
        "cap_per_combo": cap_k,
        "datasets": sorted(FILTERED_DIRS.keys()),
        "combos": {},
    }

    total = 0
    for dataset, dpath in FILTERED_DIRS.items():
        jsonl_files = sorted(dpath.glob("*_filtered_p1_only.jsonl"))
        for jsonl in jsonl_files:
            stem = jsonl.stem
            res_path = dpath / "results" / f"{stem}_results.json"
            if not res_path.exists():
                continue

            items = load_items(res_path)
            if not items:
                continue

            picked = pick_stratified(items, cap_k)
            key = f"{dataset}::{stem}"
            manifest["combos"][key] = {
                "dataset": dataset,
                "jsonl": str(jsonl),
                "results_json": str(res_path),
                "n_available": len(items),
                "picked_indices": picked,
                "n_picked": len(picked),
            }
            total += len(picked)

    manifest["total_picked"] = total

    out_path = OUT_DIR / "validation_manifest_postfiltered.json"
    with out_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved {out_path} with total_picked={total} across combos={len(manifest['combos'])}")


if __name__ == "__main__":
    main()

