"""
Score pass@16 reasoning traces with Math-Shepherd PRM.

Adapted from analysis/run_prm_baseline.py (Qwen variant). Key differences:
- Loads peiyi9979/math-shepherd-mistral-7b-prm via AutoModelForCausalLM
- Reads canonical pass@16 layout: <jsonl-dir>/Model__Benchmark.jsonl (16 traces/question)
- Token-probability scoring at ки positions, softmax over [+, -] candidate tokens
- Outputs same shard layout (by_pair/<model>___<benchmark>.jsonl + raw_scores.jsonl)
  for compatibility with analysis/aggregate_prm_baseline.py

Each JSONL row includes ``trace_score`` (alias of last-step signal when present, else min,
else fallback) so aggregates that key on ``trace_score`` keep working unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME_DEFAULT = "peiyi9979/math-shepherd-mistral-7b-prm"
STEP_TAG = "ки"
GOOD_TOKEN = "+"
BAD_TOKEN = "-"

logger = logging.getLogger("prm_math_shepherd")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_model_benchmark(filename: str) -> Tuple[str, str]:
    stem = Path(filename).stem
    parts = stem.split("__")
    if len(parts) != 2:
        raise ValueError(f"Cannot parse model__benchmark from {filename}")
    return parts[0], parts[1]


def discover_pairs(
    jsonl_dir: Path,
    models: Optional[List[str]] = None,
    benchmarks: Optional[List[str]] = None,
) -> List[Tuple[Path, str, str]]:
    out: List[Tuple[Path, str, str]] = []
    for f in sorted(jsonl_dir.glob("*__*.jsonl")):
        try:
            m, b = parse_model_benchmark(f.name)
        except ValueError:
            logger.warning("Skipping unparseable filename: %s", f.name)
            continue
        if models and m not in models:
            continue
        if benchmarks and b.lower() not in [x.lower() for x in benchmarks]:
            continue
        out.append((f, m, b))
    return out


def load_done_keys(shard_path: Path) -> Set[Tuple[int, int]]:
    keys: Set[Tuple[int, int]] = set()
    if not shard_path.is_file():
        return keys
    with shard_path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                keys.add((int(obj["question_idx"]), int(obj["trace_idx"])))
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


def concat_shards(shard_dir: Path, output_path: Path) -> int:
    n = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for shard in sorted(shard_dir.glob("*.jsonl")):
            with shard.open(encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
                    n += 1
    return n


def split_into_steps(cot: str) -> List[str]:
    steps = [s.strip() for s in cot.split("\n\n") if s.strip()]
    if len(steps) < 2:
        steps = [s.strip() for s in cot.split("\n") if s.strip()]
    return steps if steps else [cot.strip()]


def format_for_prm(question: str, cot: str) -> str:
    steps = split_into_steps(cot)
    body = " ".join(f"{s} {STEP_TAG}" for s in steps)
    return f"{question} {body}"


def trace_score_for_aggregate(
    min_step: Optional[float],
    last_step: Optional[float],
    *,
    fallback: float = 0.5,
) -> float:
    """Scalar compatible with aggregate_prm_baseline (mean-of-trace_score per pair)."""
    if last_step is not None:
        return float(last_step)
    if min_step is not None:
        return float(min_step)
    return fallback


@torch.no_grad()
def score_one(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    candidate_ids: List[int],
    step_tag_id: int,
    device: torch.device,
    max_length: int = 4096,
) -> Tuple[Optional[float], Optional[float], int]:
    ids = tokenizer.encode(text, return_tensors="pt").to(device)
    if ids.shape[1] > max_length:
        ids = ids[:, :max_length]
    logits = model(ids).logits[:, :, candidate_ids]
    probs = logits.softmax(dim=-1)[0, :, 0]
    step_mask = ids[0] == step_tag_id
    step_scores = probs[step_mask].cpu().tolist()
    if not step_scores:
        return None, None, 0
    return min(step_scores), step_scores[-1], len(step_scores)


def run_pair(
    jsonl_path: Path,
    model_name: str,
    bench_name: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    candidate_ids: List[int],
    step_tag_id: int,
    device: torch.device,
    out_shard: Path,
    questions_per_pair: int,
    traces_per_question: int,
    max_length: int,
    *,
    sanity_check: bool = False,
) -> int:
    done = load_done_keys(out_shard)
    n_new = n_resumed = 0
    t0 = time.time()
    with jsonl_path.open(encoding="utf-8") as fin, out_shard.open("a", encoding="utf-8") as fout:
        for q_count, line in enumerate(fin):
            if questions_per_pair > 0 and q_count >= questions_per_pair:
                break
            row = json.loads(line)
            qid = int(row["idx"])
            question = row["prompt"]
            traces = row["code"][:traces_per_question]
            scores = (row.get("score") or [None] * len(traces))[:traces_per_question]
            preds = (row.get("pred") or [None] * len(traces))[:traces_per_question]
            for ti, cot in enumerate(traces):
                if (qid, ti) in done:
                    n_resumed += 1
                    continue
                try:
                    text = format_for_prm(question, str(cot))
                    mn, lst, nsteps = score_one(
                        model,
                        tokenizer,
                        text,
                        candidate_ids,
                        step_tag_id,
                        device,
                        max_length,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Error on %s/%s q=%s t=%s: %s", model_name, bench_name, qid, ti, e
                    )
                    mn, lst, nsteps = None, None, 0

                ts = trace_score_for_aggregate(mn, lst)
                rec: Dict[str, object] = {
                    "model": model_name,
                    "benchmark": bench_name,
                    "question_idx": qid,
                    "trace_idx": ti,
                    "min_step_score": mn,
                    "last_step_score": lst,
                    "trace_score": ts,
                    "n_steps": nsteps,
                    "correct": scores[ti],
                    "pred": preds[ti],
                }
                fout.write(json.dumps(rec) + "\n")
                n_new += 1
                if sanity_check:
                    if mn is not None and lst is not None:
                        logger.info(
                            "  q=%s t=%s steps=%s min=%.3f last=%.3f correct=%s",
                            qid,
                            ti,
                            nsteps,
                            mn,
                            lst,
                            scores[ti],
                        )
                    else:
                        logger.info(
                            "  q=%s t=%s steps=0 (no step tag detected) correct=%s",
                            qid,
                            ti,
                            scores[ti],
                        )
                if n_new % 50 == 0:
                    rate = n_new / max(time.time() - t0, 1e-3)
                    logger.info(
                        "  %s/%s: %s new, %s resumed, %.2f tr/s",
                        model_name,
                        bench_name,
                        n_new,
                        n_resumed,
                        rate,
                    )
    elapsed = time.time() - t0
    logger.info(
        "  %s/%s: %s new + %s resumed in %.0fs -> %s",
        model_name,
        bench_name,
        n_new,
        n_resumed,
        elapsed,
        out_shard.name,
    )
    return n_new


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prm-model", default=MODEL_NAME_DEFAULT)
    ap.add_argument("--jsonl-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--benchmarks", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument(
        "--questions-per-pair",
        type=int,
        default=0,
        help="0 = all questions in the file",
    )
    ap.add_argument("--traces-per-question", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--no-merge-shards-at-end", action="store_true")
    ap.add_argument(
        "--sanity-check",
        action="store_true",
        help="5 questions on first pair only; verbose per-trace log",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    setup_logging(args.verbose or args.sanity_check)
    jsonl_dir = Path(args.jsonl_dir)
    output_dir = Path(args.output_dir)
    shard_dir = output_dir / "by_pair"
    shard_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s...", args.prm_model)
    tokenizer = AutoTokenizer.from_pretrained(args.prm_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.prm_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    candidate_ids = tokenizer.encode(f"{GOOD_TOKEN} {BAD_TOKEN}")[1:]
    step_tag_id = tokenizer.encode(STEP_TAG)[-1]
    logger.info("good/bad token ids = %s, step tag id = %s", candidate_ids, step_tag_id)

    pairs = discover_pairs(jsonl_dir, models=args.models, benchmarks=args.benchmarks)
    if not pairs:
        logger.error("No pairs found in %s", jsonl_dir)
        sys.exit(1)
    logger.info("Discovered %d (model, benchmark) pairs", len(pairs))

    if args.sanity_check:
        pairs = pairs[:1]
        args.questions_per_pair = 5
        logger.info("[SANITY] %s/%s, 5 questions only", pairs[0][1], pairs[0][2])

    total = 0
    for jsonl_path, m, b in pairs:
        shard_path = shard_dir / f"{m}___{b}.jsonl"
        logger.info("=> %s / %s", m, b)
        total += run_pair(
            jsonl_path,
            m,
            b,
            model,
            tokenizer,
            candidate_ids,
            step_tag_id,
            device,
            shard_path,
            args.questions_per_pair,
            args.traces_per_question,
            args.max_length,
            sanity_check=args.sanity_check,
        )

    logger.info("All pairs done. Total newly scored: %s", total)
    if not args.no_merge_shards_at_end and not args.sanity_check:
        merged = output_dir / "raw_scores.jsonl"
        n = concat_shards(shard_dir, merged)
        logger.info("Merged %s rows -> %s", n, merged)


if __name__ == "__main__":
    main()
