#!/usr/bin/env python3
"""
Generate a stratified validation manifest for *post-filtered* responses (filtered-cot-*/).

This manifest is for re-judging with other judges (e.g., GPT-4o, Claude) on the
post-filtered (high-confidence) subsets, without requiring any precomputed mini-judge
scores. We stratify using fields already present in the filtered JSONLs:

Per record (JSONL):
  - idx: original dataset index
  - score: correctness (bool or [bool])
  - answer_confidence: scalar confidence (float)

Sampling strategy (per combo/file):
  - cap K samples per combo (default K=9), but take all if N<K
  - stratify by:
      (A) correctness: correct vs incorrect
      (B) answer_confidence terciles: low/med/high within that combo
  - allocate evenly across available strata when possible, otherwise backfill.

Output:
  analysis/judge_validation/validation_manifest_postfiltered_all.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "analysis" / "judge_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILTERED_DIRS = {
    "gsm8k": REPO / "filtered-cot-gsm8k",
    "math500": REPO / "filtered-cot-math500",
    "svamp": REPO / "filtered-cot-svamp",
    "aqua": REPO / "filtered-cot-aqua",
    "gpqa": REPO / "filtered-cot-gpqa",
    "commonsense_qa": REPO / "filtered-cot-commonsense",
}


@dataclass(frozen=True)
class Item:
    idx: int
    correct: bool
    conf: float


def _to_bool(x) -> bool:
    if isinstance(x, list) and x:
        return bool(x[0])
    return bool(x)


def load_items_from_jsonl(jsonl_path: Path) -> List[Item]:
    items: List[Item] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            idx = r.get("idx", None)
            conf = r.get("answer_confidence", None)
            score = r.get("score", None)
            if idx is None or conf is None or score is None:
                continue
            items.append(Item(idx=int(idx), correct=_to_bool(score), conf=float(conf)))
    return items


def tercile_bucket(values: np.ndarray) -> np.ndarray:
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

    vals = np.array([it.conf for it in items], dtype=float)
    buckets = tercile_bucket(vals)  # 0/1/2

    # strata: correct(0/1) × bucket(0/1/2)
    strata: Dict[Tuple[int, int], List[Item]] = {}
    for it, b in zip(items, buckets):
        key = (1 if it.correct else 0, int(b))
        strata.setdefault(key, []).append(it)

    # sort within stratum by confidence for diversity
    for key in strata:
        strata[key].sort(key=lambda x: x.conf)

    chosen: List[int] = []
    non_empty = [key for key, v in strata.items() if v]
    # deterministic order: incorrect first, then correct; within, low→high conf bucket
    non_empty.sort(key=lambda t: (t[0], t[1]))

    # First pass: take 1 per stratum (median) if possible
    for key in non_empty:
        if len(chosen) >= k:
            break
        v = strata[key]
        take = v.pop(len(v) // 2)
        chosen.append(take.idx)

    # Fill: round-robin, alternate low/high extremes for diversity
    keys_cycle = [key for key in non_empty if strata.get(key)]
    cycle_i = 0
    take_low = True
    while len(chosen) < k and keys_cycle:
        key = keys_cycle[cycle_i % len(keys_cycle)]
        v = strata[key]
        if not v:
            keys_cycle = [kk for kk in keys_cycle if strata.get(kk)]
            cycle_i += 1
            continue
        take = v.pop(0 if take_low else -1)
        chosen.append(take.idx)
        take_low = not take_low
        cycle_i += 1

    # Backfill from any remaining items if still short
    if len(chosen) < k:
        remaining = [it.idx for key in non_empty for it in strata.get(key, [])]
        for idx in remaining:
            if len(chosen) >= k:
                break
            chosen.append(idx)

    # ensure unique, stable length k
    out = []
    seen = set()
    for idx in sorted(chosen):
        if idx in seen:
            continue
        out.append(idx)
        seen.add(idx)
        if len(out) >= k:
            break
    return out


def main():
    cap_k = 9
    manifest = {
        "version": "postfiltered_confidence_v1",
        "cap_per_combo": cap_k,
        "stratify": ["correctness", "answer_confidence_tercile_within_combo"],
        "datasets": sorted(FILTERED_DIRS.keys()),
        "combos": {},
    }

    total = 0
    combos = 0

    for dataset, dpath in FILTERED_DIRS.items():
        if not dpath.exists():
            continue
        for jsonl in sorted(dpath.glob("*_filtered_p1_only.jsonl")):
            items = load_items_from_jsonl(jsonl)
            if not items:
                continue
            picked = pick_stratified(items, cap_k)
            key = f"{dataset}::{jsonl.stem}"
            manifest["combos"][key] = {
                "dataset": dataset,
                "jsonl": str(jsonl),
                "n_available": len(items),
                "picked_indices": picked,
                "n_picked": len(picked),
            }
            total += len(picked)
            combos += 1

    manifest["total_picked"] = total
    manifest["total_combos"] = combos

    out_path = OUT_DIR / "validation_manifest_postfiltered_all.json"
    with out_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved {out_path} with total_picked={total} across combos={combos}")


if __name__ == "__main__":
    main()

