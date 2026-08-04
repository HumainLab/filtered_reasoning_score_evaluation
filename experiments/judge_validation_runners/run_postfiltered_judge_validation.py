#!/usr/bin/env python3
"""
Re-judge *post-filtered* samples with an external judge (GPT-4o / Claude via Portkey).

Inputs:
  - experiments/paper_figures/judge_validation/validation_manifest_postfiltered_all.json
    (generated from results/filtered_cot/*/ JSONLs; includes per-combo picked idx values)
  - results/filtered_cot/*/results/*_results.json
    (must already exist; provides GPT-4o-mini baseline judge scores for those samples)

Output:
  - experiments/paper_figures/judge_validation/judge_validation_postfiltered_<judge_tag>_<timestamp>.json

The script stores, per selected sample:
  - idx
  - mini: {pillar scores}, mini_fused: {pillar scores in [0,1] + overall}
  - judge: {pillar scores}, judge_fused: {pillar scores in [0,1] + overall}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
JUDGE_PKG_DIR = REPO_ROOT / "frs"
ANALYSIS_DIR = REPO_ROOT / "experiments" / "paper_figures" / "judge_validation"

sys.path.insert(0, str(JUDGE_PKG_DIR))

PILLARS = ["faithfulness", "utility", "coherence", "factuality"]
PORTKEY_BASE_URL = "https://api.portkey.ai/v1"


def short_name_from_stem(stem: str) -> str:
    # stem examples:
    #   DeepSeek_R1_Distill_Qwen_1.5B_filtered_p1_only
    #   Qwen2.5_7B_Instruct_filtered_p1_only
    #   Phi_4_reasoning_filtered_p1_only
    m = stem.replace("_filtered_p1_only", "")
    mapping = {
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
    return mapping.get(m, m)


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def load_jsonl_index(path: Path, needed_idxs: List[int]) -> Dict[int, Dict[str, Any]]:
    needed = set(needed_idxs)
    out: Dict[int, Dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            idx = r.get("idx", None)
            if idx is None:
                continue
            idx = int(idx)
            if idx in needed:
                out[idx] = r
                if len(out) == len(needed):
                    break
    return out


def load_mini_results(results_json: Path, needed_idxs: List[int]) -> Dict[int, Dict[str, Any]]:
    with results_json.open() as f:
        data = json.load(f)
    needed = set(needed_idxs)
    out: Dict[int, Dict[str, Any]] = {}
    for r in data.get("results", []):
        idx = r.get("idx", None)
        if idx is None:
            continue
        idx = int(idx)
        if idx in needed:
            out[idx] = r
    return out


def record_to_eval_fields(rec: Dict[str, Any]) -> Tuple[str, str, str]:
    question = rec.get("question", "")
    gold = rec.get("gt", rec.get("answer", ""))
    code_field = rec.get("code", "")
    if isinstance(code_field, list) and code_field:
        cot = code_field[0]
    else:
        cot = str(code_field)
    return question, cot, str(gold)


def judge_one(judge, question: str, cot: str, gold: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from cot_eval_v2.evaluator import PillarsEvaluator

    evaluator = PillarsEvaluator(judge=judge)
    flags, evidence, rule_scores, judge_scores, fused_scores = evaluator.analyze(
        problem=question, cot_text=cot, gold=gold
    )
    return judge_scores, fused_scores


def mk_judge(model: str, portkey_key: Optional[str], portkey_prefix: str):
    from cot_eval_v2.judge import Judge

    if portkey_key:
        return Judge(
            model=f"{portkey_prefix}/{model}",
            mode="ALWAYS",
            diagnostic=False,
            base_url=PORTKEY_BASE_URL,
            api_key=portkey_key,
        )
    # fall back to OpenAI direct
    if not os.environ.get("OPENAI_API_KEY"):
        warnings.warn("OPENAI_API_KEY not set; direct OpenAI calls will likely fail.")
    return Judge(model=model, mode="ALWAYS", diagnostic=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(ANALYSIS_DIR / "validation_manifest_postfiltered_all.json"),
        help="Path to postfiltered manifest JSON.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        required=True,
        help="Validator model name (e.g. gpt-4o or claude-sonnet-4-5).",
    )
    parser.add_argument("--portkey-key", type=str, default=None, help="Portkey API key.")
    parser.add_argument(
        "--portkey-model-prefix",
        type=str,
        default=os.environ.get("PORTKEY_MODEL_PREFIX", ""),
        help="Portkey model prefix (default: $PORTKEY_MODEL_PREFIX)",
    )
    parser.add_argument("--parallel", type=int, default=6, help="Parallel judge calls.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)

    combos = manifest.get("combos", {})
    if not combos:
        raise RuntimeError(f"No combos found in manifest: {manifest_path}")

    judge = mk_judge(args.judge_model, args.portkey_key, args.portkey_model_prefix)

    per_combo_out: Dict[str, Any] = {}
    tasks: List[Tuple[str, int, Dict[str, Any], Dict[str, Any], str, str, str]] = []

    # Build tasks
    for combo_key, c in combos.items():
        jsonl_path = Path(c["jsonl"])
        picked = [int(x) for x in c["picked_indices"]]
        results_json = jsonl_path.parent / "results" / f"{jsonl_path.stem}_results.json"
        if not results_json.exists():
            raise RuntimeError(f"Missing mini results JSON: {results_json}")

        recs = load_jsonl_index(jsonl_path, picked)
        minis = load_mini_results(results_json, picked)

        # sanity
        missing_recs = [i for i in picked if i not in recs]
        missing_mini = [i for i in picked if i not in minis]
        if missing_recs:
            raise RuntimeError(f"{combo_key}: missing {len(missing_recs)} records in JSONL (e.g. {missing_recs[:5]})")
        if missing_mini:
            raise RuntimeError(f"{combo_key}: missing {len(missing_mini)} mini results (e.g. {missing_mini[:5]})")

        for idx in picked:
            rec = recs[idx]
            mini = minis[idx]
            q, cot, gold = record_to_eval_fields(rec)
            tasks.append((combo_key, idx, rec, mini, q, cot, gold))

        per_combo_out[combo_key] = {
            "dataset": c.get("dataset"),
            "jsonl": str(jsonl_path),
            "results_json": str(results_json),
            "short_name": short_name_from_stem(jsonl_path.stem),
            "n_available": c.get("n_available"),
            "picked_indices": picked,
            "sample_pairs": [],
        }

    # Run judging
    t0 = time.time()
    results: Dict[Tuple[str, int], Dict[str, Any]] = {}

    def _run_one(t):
        combo_key, idx, rec, mini, q, cot, gold = t
        js, fused = judge_one(judge, q, cot, gold)
        return combo_key, idx, js, fused, mini

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futs = [pool.submit(_run_one, t) for t in tasks]
        for fut in as_completed(futs):
            combo_key, idx, js, fused, mini = fut.result()
            results[(combo_key, idx)] = {
                "idx": idx,
                "mini": mini.get("judge_scores"),
                "mini_fused": mini.get("fused_scores"),
                "judge": js,
                "judge_fused": fused,
                "original_correct": bool(mini.get("original_correct", False)),
            }

    # Assemble outputs per combo in a stable order
    for combo_key, c in per_combo_out.items():
        for idx in c["picked_indices"]:
            c["sample_pairs"].append(results[(combo_key, idx)])

    out = {
        "type": "postfiltered_judge_validation",
        "manifest": str(manifest_path),
        "judge_model": args.judge_model,
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(tasks),
        "parallel": args.parallel,
        "per_combo": per_combo_out,
        "elapsed_s": round(time.time() - t0, 2),
    }

    tag = args.judge_model.replace("/", "_")
    out_path = ANALYSIS_DIR / f"judge_validation_postfiltered_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved {out_path} ({len(tasks)} samples) in {out['elapsed_s']}s")


if __name__ == "__main__":
    main()

