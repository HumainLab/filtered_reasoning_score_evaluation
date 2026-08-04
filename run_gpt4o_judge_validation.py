#!/usr/bin/env python3
"""
GPT-4o Judge Validation Script

Samples N items per model from existing GPT-4o-mini judged results,
re-judges them with GPT-4o, and computes agreement metrics.

Usage:
  python run_gpt4o_judge_validation.py [--samples-per-model 30] [--dataset gsm8k]
"""

import json
import os
import sys
import time
import random
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR / "backend"
COT_ANALYSIS_DIR = SCRIPT_DIR / "evaluation" / "exports" / "cot_analysis"
OUTPUT_DIR = SCRIPT_DIR / "analysis" / "judge_validation"

sys.path.insert(0, str(BACKEND_DIR))

MODELS = {
    "DeepSeek-R1-Distill-Qwen-1.5B": "DS-R1-1.5B",
    "DeepSeek-R1-Distill-Qwen-7B": "DS-R1-7B",
    "Llama-3.1-8B-Instruct": "LLaMA-3.1-8B",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "Qwen2.5-Math-7B": "Qwen2.5-Math-7B",
    "gemma-7b": "Gemma-7B",
    "phi-4": "Phi-4",
    "Phi-4-reasoning": "Phi-4-Reasoning",
    "Qwen3-4B-Thinking-2507": "Qwen3-4B",
}

PILLARS = ["faithfulness", "utility", "coherence", "factuality"]


def load_api_key() -> str:
    config_path = BACKEND_DIR / "path_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        key = config.get("openai_api_key", "")
        if key:
            return key
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("No OpenAI API key found.")
    return key


PORTKEY_BASE_URL = "https://api.portkey.ai/v1"


def find_latest_result(model_dir: str, dataset: str) -> Optional[Path]:
    d = COT_ANALYSIS_DIR / model_dir / dataset
    if not d.exists():
        return None
    files = sorted(d.glob("cot_analysis_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def sample_from_results(result_path: Path, n: int, seed: int = 42) -> Tuple[List[Dict], Dict]:
    with open(result_path) as f:
        data = json.load(f)

    per_sample = data.get("per_sample", [])
    valid = [s for s in per_sample if s.get("judge_scores") and all(
        s["judge_scores"].get(p) is not None for p in PILLARS
    )]

    rng = random.Random(seed)
    n = min(n, len(valid))
    sampled = rng.sample(valid, n)
    return sampled, data


def load_samples_by_indices(result_path: Path, indices: List[int]) -> Tuple[List[Dict], Dict]:
    """Load specific samples by idx from a result file."""
    with open(result_path) as f:
        data = json.load(f)
    idx_set = set(indices)
    per_sample = data.get("per_sample", [])
    valid = [
        s for s in per_sample
        if s.get("idx") in idx_set
        and s.get("judge_scores")
        and all(s["judge_scores"].get(p) is not None for p in PILLARS)
    ]
    return valid, data


def build_flags_summary(flags_dict: Dict) -> str:
    if not flags_dict:
        return "No issues detected."
    lines = []
    for pillar, flag_list in flags_dict.items():
        if not flag_list:
            continue
        for flag in flag_list:
            issue = flag.get("issue", "unknown")
            step = flag.get("step", "")
            lines.append(f"[{pillar}] {issue} at {step}")
    return "\n".join(lines) if lines else "No issues detected."


def judge_sample_gpt4o(sample: Dict, judge) -> Dict[str, Any]:
    question = sample.get("question", "")
    cot_text = sample.get("cot_text", sample.get("model_output", ""))
    gold = sample.get("ground_truth", "")
    flags = sample.get("flags", {})
    evidence = sample.get("evidence", {})

    flags_summary = build_flags_summary(flags)

    try:
        scores_4o = judge.score(
            problem=question,
            cot=cot_text,
            gold=str(gold),
            flags_summary=flags_summary,
            evidence=evidence,
        )
        return scores_4o
    except Exception as e:
        warnings.warn(f"GPT-4o judge error for idx={sample.get('idx')}: {e}")
        return {p: None for p in PILLARS}


def compute_agreement(mini_scores: List[Dict], gpt4o_scores: List[Dict]) -> Dict:
    from collections import defaultdict
    import math

    per_pillar = {p: {"mini": [], "gpt4o": []} for p in PILLARS}

    for ms, gs in zip(mini_scores, gpt4o_scores):
        for p in PILLARS:
            mv = ms.get(p)
            gv = gs.get(p)
            if mv is not None and gv is not None:
                per_pillar[p]["mini"].append(mv)
                per_pillar[p]["gpt4o"].append(gv)

    results = {}
    for p in PILLARS:
        mini_vals = per_pillar[p]["mini"]
        gpt4o_vals = per_pillar[p]["gpt4o"]
        n = len(mini_vals)
        if n < 2:
            results[p] = {"n": n, "error": "insufficient data"}
            continue

        mean_m = sum(mini_vals) / n
        mean_g = sum(gpt4o_vals) / n

        abs_diffs = [abs(m - g) for m, g in zip(mini_vals, gpt4o_vals)]
        mad = sum(abs_diffs) / n
        exact_match = sum(1 for d in abs_diffs if d == 0) / n
        within_1 = sum(1 for d in abs_diffs if d <= 1) / n

        cov = sum((m - mean_m) * (g - mean_g) for m, g in zip(mini_vals, gpt4o_vals)) / (n - 1)
        std_m = math.sqrt(sum((m - mean_m) ** 2 for m in mini_vals) / (n - 1))
        std_g = math.sqrt(sum((g - mean_g) ** 2 for g in gpt4o_vals) / (n - 1))
        pearson = cov / (std_m * std_g) if std_m > 0 and std_g > 0 else 0.0

        results[p] = {
            "n": n,
            "mean_4o_mini": round(mean_m, 3),
            "mean_4o": round(mean_g, 3),
            "mean_abs_diff": round(mad, 3),
            "exact_match_rate": round(exact_match, 3),
            "within_1_rate": round(within_1, 3),
            "pearson_r": round(pearson, 3),
        }

    all_mini = []
    all_gpt4o = []
    for p in PILLARS:
        all_mini.extend(per_pillar[p]["mini"])
        all_gpt4o.extend(per_pillar[p]["gpt4o"])

    n_all = len(all_mini)
    if n_all >= 2:
        mean_m = sum(all_mini) / n_all
        mean_g = sum(all_gpt4o) / n_all
        abs_diffs = [abs(m - g) for m, g in zip(all_mini, all_gpt4o)]
        cov = sum((m - mean_m) * (g - mean_g) for m, g in zip(all_mini, all_gpt4o)) / (n_all - 1)
        std_m = math.sqrt(sum((m - mean_m) ** 2 for m in all_mini) / (n_all - 1))
        std_g = math.sqrt(sum((g - mean_g) ** 2 for g in all_gpt4o) / (n_all - 1))
        pearson_all = cov / (std_m * std_g) if std_m > 0 and std_g > 0 else 0.0

        results["overall"] = {
            "n": n_all,
            "mean_abs_diff": round(sum(abs_diffs) / n_all, 3),
            "exact_match_rate": round(sum(1 for d in abs_diffs if d == 0) / n_all, 3),
            "within_1_rate": round(sum(1 for d in abs_diffs if d <= 1) / n_all, 3),
            "pearson_r": round(pearson_all, 3),
        }

    return results


def compute_rank_agreement(model_avgs_mini: Dict[str, float], model_avgs_4o: Dict[str, float]) -> Dict:
    models = sorted(model_avgs_mini.keys())
    rank_mini = sorted(models, key=lambda m: model_avgs_mini[m], reverse=True)
    rank_4o = sorted(models, key=lambda m: model_avgs_4o[m], reverse=True)

    pos_mini = {m: i for i, m in enumerate(rank_mini)}
    pos_4o = {m: i for i, m in enumerate(rank_4o)}

    n = len(models)
    d_sq_sum = sum((pos_mini[m] - pos_4o[m]) ** 2 for m in models)
    spearman = 1 - (6 * d_sq_sum) / (n * (n ** 2 - 1)) if n > 1 else 1.0

    return {
        "rank_mini": rank_mini,
        "rank_4o": rank_4o,
        "spearman_rho": round(spearman, 3),
        "rank_changes": {m: pos_mini[m] - pos_4o[m] for m in models},
    }


def main():
    parser = argparse.ArgumentParser(description="GPT-4o judge validation")
    parser.add_argument("--samples-per-model", type=int, default=30)
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Use manifest for all model×dataset combos (from generate_validation_manifest.py)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--portkey-key", type=str, default=None,
                        help="Portkey API key. If set, routes through Portkey gateway.")
    parser.add_argument("--portkey-model-prefix", type=str, default=os.environ.get("PORTKEY_MODEL_PREFIX", ""),
                        help="Portkey model routing prefix (default: $PORTKEY_MODEL_PREFIX)")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Override judge model name (full Portkey path, e.g. @prefix/claude-sonnet-4-5)")
    args = parser.parse_args()

    use_manifest = args.manifest is not None
    manifest_data = None
    if use_manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_absolute():
            manifest_path = SCRIPT_DIR / manifest_path
        with open(manifest_path) as f:
            manifest_data = json.load(f)
        total_in_manifest = manifest_data.get("total_samples", 0)
        n_combos = manifest_data.get("n_combos", 0)

    print("=" * 70)
    print("  GPT-4o JUDGE VALIDATION")
    if use_manifest:
        print(f"  Mode: manifest ({n_combos} combos, {total_in_manifest} samples)")
        print(f"  Manifest: {args.manifest}")
    else:
        print(f"  Dataset: {args.dataset}")
        print(f"  Samples per model: {args.samples_per_model}")
    print(f"  Seed: {args.seed}")
    print(f"  Time: {datetime.now().isoformat()}")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    use_portkey = args.portkey_key is not None
    from app.cot_eval_v2.judge import Judge

    if use_portkey:
        if args.judge_model:
            portkey_model = args.judge_model
        else:
            portkey_model = f"{args.portkey_model_prefix}/gpt-4o"
        judge_4o = Judge(
            model=portkey_model,
            mode="ALWAYS",
            diagnostic=False,
            base_url=PORTKEY_BASE_URL,
            api_key=args.portkey_key,
        )
        print(f"  Portkey API key loaded (ends with ...{args.portkey_key[-6:]})")
        print(f"  Portkey model: {portkey_model}")
    else:
        api_key = load_api_key()
        os.environ["OPENAI_API_KEY"] = api_key
        print(f"  API key loaded (ends with ...{api_key[-6:]})")
        judge_4o = Judge(model="gpt-4o", mode="ALWAYS", diagnostic=False)

    print(f"  GPT-4o judge initialized\n")

    all_model_results = {}
    model_avgs_mini = {}
    model_avgs_4o = {}
    total_api_calls = 0
    lock = threading.Lock()
    global_start = time.time()

    if use_manifest:
        combos_to_run = list(manifest_data["combos"].items())
    else:
        combos_to_run = [
            (f"{model_dir}::{args.dataset}", {
                "model_dir": model_dir,
                "dataset": args.dataset,
                "result_path": str(find_latest_result(model_dir, args.dataset)) if find_latest_result(model_dir, args.dataset) else None,
                "indices": None,
                "n": args.samples_per_model,
            })
            for model_dir in MODELS.keys()
        ]
        combos_to_run = [(k, v) for k, v in combos_to_run if v.get("result_path")]

    for combo_key, combo_info in combos_to_run:
        model_dir = combo_info["model_dir"]
        dataset = combo_info["dataset"]
        result_path = Path(combo_info["result_path"]) if combo_info.get("result_path") else find_latest_result(model_dir, dataset)
        if result_path is None:
            print(f"  SKIP {combo_key}: no result file")
            continue

        short_name = MODELS.get(model_dir, model_dir)
        if use_manifest:
            indices = combo_info.get("indices", [])
            samples, meta = load_samples_by_indices(result_path, indices)
        else:
            samples, meta = sample_from_results(result_path, args.samples_per_model, args.seed)
        print(f"\n{'─'*60}")
        print(f"  {short_name} | {dataset} ({model_dir})")
        print(f"  Source: {result_path.name}")
        print(f"  Samples: {len(samples)} items")
        print(f"{'─'*60}")

        mini_scores = []
        gpt4o_scores = []
        done_count = [0]

        def process_sample(sample):
            idx = sample.get("idx", -1)
            ms = sample.get("judge_scores", {})

            if args.dry_run:
                gs = {p: ms.get(p) for p in PILLARS}
            else:
                gs = judge_sample_gpt4o(sample, judge_4o)

            with lock:
                done_count[0] += 1
                status = "OK" if all(gs.get(p) is not None for p in PILLARS) else "PARTIAL"
                diff_str = " ".join(
                    f"{p[:5]}:{ms.get(p,'?')}->{gs.get(p,'?')}"
                    for p in PILLARS
                )
                print(f"    [{done_count[0]:>3}/{len(samples)}] idx={idx:>4} {status} | {diff_str}")

            return ms, gs

        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(process_sample, s): s for s in samples}
            for future in as_completed(futures):
                ms, gs = future.result()
                mini_scores.append(ms)
                gpt4o_scores.append(gs)

        total_api_calls += len(samples)

        agreement = compute_agreement(mini_scores, gpt4o_scores)
        all_model_results[combo_key] = {
            "model_dir": model_dir,
            "dataset": dataset,
            "short_name": short_name,
            "n_samples": len(samples),
            "agreement": agreement,
            "sample_pairs": [
                {"idx": s.get("idx"), "mini": ms, "gpt4o": gs}
                for s, ms, gs in zip(samples, mini_scores, gpt4o_scores)
            ],
        }

        valid_mini = [s for s in mini_scores if all(s.get(p) is not None for p in PILLARS)]
        valid_4o = [s for s in gpt4o_scores if all(s.get(p) is not None for p in PILLARS)]
        if valid_mini:
            if short_name not in model_avgs_mini:
                model_avgs_mini[short_name] = []
            model_avgs_mini[short_name].extend(
                sum(s[p] for p in PILLARS) / 4 for s in valid_mini
            )
        if valid_4o:
            if short_name not in model_avgs_4o:
                model_avgs_4o[short_name] = []
            model_avgs_4o[short_name].extend(
                sum(s[p] for p in PILLARS) / 4 for s in valid_4o
            )

        ov = agreement.get("overall", {})
        print(f"\n  {short_name} | {dataset} Agreement:")
        print(f"    Pearson r = {ov.get('pearson_r', 'N/A')}")
        print(f"    Exact match = {ov.get('exact_match_rate', 'N/A')}")
        print(f"    Within ±1 = {ov.get('within_1_rate', 'N/A')}")
        print(f"    Mean abs diff = {ov.get('mean_abs_diff', 'N/A')}")

    elapsed = time.time() - global_start

    # Convert per-model lists to averages (for rank agreement)
    model_avgs_mini_scalar = {m: sum(v) / len(v) for m, v in model_avgs_mini.items() if v}
    model_avgs_4o_scalar = {m: sum(v) / len(v) for m, v in model_avgs_4o.items() if v}
    common_models = set(model_avgs_mini_scalar.keys()) & set(model_avgs_4o_scalar.keys())
    rank_agreement = {}
    if len(common_models) >= 3:
        rank_agreement = compute_rank_agreement(
            {m: model_avgs_mini_scalar[m] for m in common_models},
            {m: model_avgs_4o_scalar[m] for m in common_models},
        )

    print("\n" + "=" * 70)
    print("  GLOBAL SUMMARY")
    print("=" * 70)
    print(f"  Total API calls: {total_api_calls}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print()

    print("  Per-pillar agreement (all models pooled):")
    all_mini_pooled = []
    all_4o_pooled = []
    for model_name, mdata in all_model_results.items():
        for pair in mdata["sample_pairs"]:
            all_mini_pooled.append(pair["mini"])
            all_4o_pooled.append(pair["gpt4o"])

    pooled_agreement = compute_agreement(all_mini_pooled, all_4o_pooled)
    for key in PILLARS + ["overall"]:
        pa = pooled_agreement.get(key, {})
        print(f"    {key:15s}: r={pa.get('pearson_r','N/A'):>6}  exact={pa.get('exact_match_rate','N/A'):>6}  ±1={pa.get('within_1_rate','N/A'):>6}  MAD={pa.get('mean_abs_diff','N/A'):>6}")

    if rank_agreement:
        print(f"\n  Model Ranking Agreement (Spearman rho): {rank_agreement['spearman_rho']}")
        print(f"  GPT-4o-mini ranking: {rank_agreement['rank_mini']}")
        print(f"  GPT-4o ranking:      {rank_agreement['rank_4o']}")
        print(f"  Rank changes: {rank_agreement['rank_changes']}")

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "dataset": args.dataset if not use_manifest else "all",
            "manifest": args.manifest,
            "samples_per_model": args.samples_per_model,
            "seed": args.seed,
            "judge_model": portkey_model if use_portkey else "gpt-4o",
            "baseline_model": "gpt-4o-mini",
            "api_gateway": "portkey" if use_portkey else "openai_direct",
        },
        "total_api_calls": total_api_calls,
        "elapsed_s": round(elapsed, 1),
        "pooled_agreement": pooled_agreement,
        "rank_agreement": rank_agreement,
        "per_combo": all_model_results,
        "model_avgs_mini": model_avgs_mini_scalar,
        "model_avgs_4o": model_avgs_4o_scalar,
    }

    suffix = "all" if use_manifest else args.dataset
    model_tag = portkey_model.split("/")[-1] if use_portkey else "gpt4o"
    out_path = OUTPUT_DIR / f"judge_validation_{suffix}_{model_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
