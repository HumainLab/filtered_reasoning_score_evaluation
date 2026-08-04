#!/usr/bin/env python3
"""
Offline Qwen2.5-Math-PRM-7B scoring for filtered reasoning traces.

Pure additive analysis — does not modify evaluation/, backend judge, or FRS pipelines.

Note on multi-trace: filtered `*_filtered_p1_only.jsonl` commonly has ``len(code) == 1``.
``--traces-per-question`` caps scoring at available traces per question; BoN‑16 requires
raw generations with ``n_sampling=16`` populated in ``code``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

LOG = logging.getLogger("prm_baseline")

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_APP = REPO_ROOT / "backend" / "app"

if str(BACKEND_APP) not in sys.path:
    sys.path.insert(0, str(BACKEND_APP))
from cot_eval_v2.parsing import split_steps  # noqa: E402

MODEL_STEMS_DEFAULT = (
    "DeepSeek_R1_Distill_Qwen_1.5B",
    "DeepSeek_R1_Distill_Qwen_7B",
    "Llama_3.1_8B_Instruct",
    "Phi_4_reasoning",
    "Qwen2.5_7B_Instruct",
    "Qwen2.5_Math_7B",
    "Qwen3_4B_Thinking_2507",
    "gemma_7b",
    "phi_4",
)

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

BENCHMARK_DIR_SUFFIX = {
    "gsm8k": "filtered-cot-gsm8k",
    "math500": "filtered-cot-math500",
    "svamp": "filtered-cot-svamp",
    "aqua": "filtered-cot-aqua",
    "gpqa": "filtered-cot-gpqa",
    "commonsense_qa": "filtered-cot-commonsense",
}

SANITY_SAMPLES: List[Tuple[str, str, int, int]] = [
    ("phi_4", "math500", 0, 0),
    ("phi_4", "math500", 1, 0),
    ("DeepSeek_R1_Distill_Qwen_1.5B", "gsm8k", 42, 0),
    ("DeepSeek_R1_Distill_Qwen_1.5B", "gsm8k", 142, 0),
    ("Llama_3.1_8B_Instruct", "gpqa", 3, 0),
    ("Llama_3.1_8B_Instruct", "gpqa", 88, 0),
    ("Llama_3.1_8B_Instruct", "svamp", 100, 0),
    ("Llama_3.1_8B_Instruct", "svamp", 210, 0),
    ("gemma_7b", "gsm8k", 303, 0),
    ("Qwen2.5_7B_Instruct", "gsm8k", 404, 0),
]


def make_step_rewards(logits: torch.Tensor, token_masks: torch.Tensor) -> List[List[float]]:
    probabilities = F.softmax(logits.float(), dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)
    all_scores_res: List[List[float]] = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
        all_scores_res.append(positive_probs.detach().cpu().tolist())
    return all_scores_res


def segment_trace(trace_text: str) -> Tuple[List[str], str]:
    stripped = trace_text.strip()
    if not stripped:
        return [], "fallback"
    parts = [p.strip() for p in stripped.split("\n\n") if p.strip()]
    if len(parts) >= 2:
        return parts, "nn"
    fb = split_steps(stripped)
    cleaned = [x.strip() for x in fb if x.strip()]
    if cleaned:
        return cleaned, "fallback"
    return [stripped], "fallback"


def filtered_jsonl_path(filtered_root: Path, benchmark: str, stem: str) -> Path:
    return filtered_root / BENCHMARK_DIR_SUFFIX[benchmark] / f"{stem}_filtered_p1_only.jsonl"


def load_jsonl_index(path: Path) -> Dict[int, Dict[str, Any]]:
    rows: Dict[int, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("idx") is None:
                continue
            rows[int(obj["idx"])] = obj
    return rows


def parse_question_manifest_csv(path: Path) -> Dict[Tuple[str, str], List[int]]:
    out: Dict[Tuple[str, str], List[int]] = {}
    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames or not {"model_stem", "benchmark", "question_idx"}.issubset(
            set(h.strip() for h in reader.fieldnames if h)
        ):
            raise ValueError("question_sets.csv must have model_stem,benchmark,question_idx")
        for row in reader:
            key = (row["model_stem"].strip(), row["benchmark"].strip())
            out.setdefault(key, []).append(int(row["question_idx"]))
    for key in out:
        out[key] = sorted(set(out[key]))
    return out


def default_question_idxs(rows: Dict[int, Dict[str, Any]], n: int, seed: int) -> List[int]:
    del seed  # reserved
    rng = sorted(rows.keys())
    return rng if len(rng) <= n else rng[:n]


def build_assistant_with_steps(step_texts: List[str]) -> str:
    steps = step_texts if step_texts else ["(empty)"]
    return "<extra_0>".join(steps) + "<extra_0>"


def encode_conversation_messages(
    tokenizer: AutoTokenizer, question_text: str, assistant_body: str
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question_text.strip()},
        {"role": "assistant", "content": assistant_body},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def truncate_for_budget(
    tokenizer: AutoTokenizer,
    question_text: str,
    step_segments: List[str],
    max_length: int,
) -> Tuple[List[str], bool]:
    segs = [s for s in step_segments if s.strip()]
    if not segs:
        segs = ["(empty)"]
    truncated = False
    while True:
        assistant_body = build_assistant_with_steps(segs)
        conv = encode_conversation_messages(tokenizer, question_text, assistant_body)
        ntok = len(tokenizer.encode(conv, add_special_tokens=False))
        if ntok <= max_length:
            return segs, truncated
        if len(segs) > 1:
            segs.pop(0)
            truncated = True
            continue
        blob = segs[0]
        drop = max(int(len(blob) * 0.15), 32)
        segs[0] = blob[drop:].lstrip()
        truncated = True
        if len(segs[0]) < 24:
            return segs, truncated


def infer_logits_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    kwargs: Dict[str, Any] = {"input_ids": input_ids}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    out = model(**kwargs)
    return out.logits if hasattr(out, "logits") else out[0]


def score_single_trace(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    *,
    question: str,
    trace_text: str,
    device: torch.device,
    max_length: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    step_raw, segmentation_method = segment_trace(trace_text)
    segments, truncated = truncate_for_budget(tokenizer, question, step_raw, max_length=max_length)
    if truncated:
        LOG.warning(
            "Truncated reasoning max_length=%s (raw_segments=%s -> kept=%s)",
            max_length,
            len(step_raw),
            len(segments),
        )

    assistant_body = build_assistant_with_steps(segments)
    conv = encode_conversation_messages(tokenizer, question, assistant_body)
    inputs = tokenizer(
        conv,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    input_ids = inputs["input_ids"].to(device)
    attn = inputs.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)

    sep_id = tokenizer.encode("<extra_0>", add_special_tokens=False)[0]
    masks = input_ids.eq(sep_id).float()

    with torch.no_grad():
        logits = infer_logits_forward(model, input_ids, attention_mask=attn)

    prob_lists = make_step_rewards(logits, masks)
    probs = prob_lists[0] if prob_lists else []
    mean_r = float(sum(probs) / len(probs)) if probs else float("nan")

    diag = {
        "segments_after_trunc": len(segments),
        "segments_before_trunc": len(step_raw),
        "conversation_tokens": int(input_ids.shape[1]),
    }
    payload = {
        "step_rewards": probs,
        "trace_score": mean_r,
        "n_steps": len(probs),
        "segmentation_method": segmentation_method,
        "truncated": truncated,
    }
    return payload, diag


def load_done_keys(shard_path: Path) -> Set[Tuple[int, int]]:
    keys: Set[Tuple[int, int]] = set()
    if not shard_path.is_file():
        return keys
    with shard_path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            keys.add((int(obj["question_idx"]), int(obj["trace_idx"])))
    return keys


def run_sanity_samples(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    *,
    filtered_root: Path,
    max_length: int,
) -> None:
    rows_out: List[Dict[str, Any]] = []
    for stem, bench, idx, trace_idx in SANITY_SAMPLES:
        path = filtered_jsonl_path(filtered_root, bench, stem)
        if not path.is_file():
            print(f"[SANITY MISSING FILE] {path}")
            continue
        table = load_jsonl_index(path)
        if idx not in table:
            print(f"[SANITY MISSING IDX] stem={stem} bench={bench} idx={idx} path={path}")
            continue
        rec = table[idx]
        code_field = rec.get("code") or []
        texts: List[str]
        if isinstance(code_field, list):
            texts = [str(x) for x in code_field if isinstance(x, str)]
        elif isinstance(code_field, str):
            texts = [code_field]
        else:
            texts = []
        if trace_idx >= len(texts):
            print(
                f"[SANITY MISSING TRACE_IDX] stem={stem} bench={bench} idx={idx} codes={len(texts)}"
            )
            continue
        trace_body = texts[trace_idx]
        question = rec.get("question") or rec.get("problem") or rec.get("input") or ""

        snippet = trace_body[:500] + ("..." if len(trace_body) > 500 else "")
        raw_steps, meth_raw = segment_trace(trace_body)
        steps_budget, truncated_budget = truncate_for_budget(
            tokenizer, str(question), raw_steps, max_length=max_length
        )

        scored, diag = score_single_trace(
            model,
            tokenizer,
            question=str(question),
            trace_text=str(trace_body),
            device=device,
            max_length=max_length,
        )

        print("\n" + "=" * 80)
        print(
            f"stem={stem} benchmark={bench} idx={idx} trace_idx={trace_idx} segmentation={scored['segmentation_method']} path={path}"
        )
        print(f"trace_snippet (~500 chars):\n{snippet}")
        print(
            f"segment_hints: raw_method={meth_raw} raw_n={len(raw_steps)} post_budget={len(steps_budget)} truncated={truncated_budget}"
        )
        for si, txt in enumerate(steps_budget[: min(25, len(steps_budget))]):
            print(f"  step_preview_{si + 1}: {txt.replace(chr(10), ' ')[:80]}")
        sr = scored["step_rewards"]
        print(f"step_rewards (n={len(sr)}): {sr}")
        print(f"aggregate trace_score (mean): {scored['trace_score']}")

        rows_out.append(
            {
                "stem": stem,
                "benchmark": bench,
                "question_idx": idx,
                "trace_idx": trace_idx,
                "snippet": snippet,
                **scored,
                **diag,
            }
        )

    out_path = REPO_ROOT / "analysis_outputs" / "prm_sanity_stdout.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows_out, indent=2), encoding="utf-8")
    print(f"\n[SANITY wrote JSON to {out_path}]")


def parse_models_arg(arg: Optional[str]) -> Tuple[str, ...]:
    if not arg or arg.strip().lower() in ("", "all"):
        return MODEL_STEMS_DEFAULT
    return tuple(p.strip() for p in arg.split(",") if p.strip())


def run_pair_shard(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    *,
    stem: str,
    benchmark: str,
    filtered_root: Path,
    output_dir: Path,
    manifest: Dict[Tuple[str, str], List[int]],
    questions_per_pair: int,
    traces_cap: int,
    max_length: int,
    seed: int,
) -> None:
    path = filtered_jsonl_path(filtered_root, benchmark, stem)
    if not path.is_file():
        LOG.error("Missing filtered JSONL %s", path)
        return

    shard_dir = output_dir / "by_pair"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"{stem}___{benchmark}.jsonl"
    done = load_done_keys(shard_path)

    rows_idx = load_jsonl_index(path)
    key_manifest = manifest.get((stem, benchmark))

    qidxs = key_manifest if key_manifest else default_question_idxs(rows_idx, questions_per_pair, seed)
    (shard_dir / f"{stem}___{benchmark}_manifest.json").write_text(
        json.dumps({"stem": stem, "benchmark": benchmark, "question_idx": qidxs}, indent=2),
        encoding="utf-8",
    )

    shortage: List[Dict[str, Any]] = []
    for q_idx in qidxs:
        rec = rows_idx.get(q_idx)
        if rec is None:
            continue

        question = (
            rec.get("question") or rec.get("problem") or rec.get("input") or "(no question)"
        )
        code_field = rec.get("code") or []
        if isinstance(code_field, list):
            traces = [str(c) for c in code_field if isinstance(c, str)]
        elif isinstance(code_field, str):
            traces = [code_field]
        else:
            traces = []

        n_avail = len(traces)
        n_score = min(traces_cap, n_avail if n_avail else 0)

        if n_avail < traces_cap:
            shortage.append(
                dict(question_idx=q_idx, n_code_available=n_avail, cap=traces_cap)
            )
        if n_avail == 0:
            continue

        for t_idx in range(n_score):
            if (q_idx, t_idx) in done:
                continue
            scored, _diag = score_single_trace(
                model,
                tokenizer,
                question=str(question),
                trace_text=traces[t_idx],
                device=device,
                max_length=max_length,
            )
            payload = {
                "model": stem,
                "benchmark": benchmark,
                "question_idx": q_idx,
                "trace_idx": t_idx,
                "n_steps": scored["n_steps"],
                "n_code_available": n_avail,
                "step_rewards": scored["step_rewards"],
                "trace_score": scored["trace_score"],
                "segmentation_method": scored["segmentation_method"],
                "truncated": scored["truncated"],
            }
            with shard_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(payload) + "\n")

    if shortage:
        Path(shard_dir / f"{stem}___{benchmark}_trace_shortfall.json").write_text(
            json.dumps(shortage, indent=2), encoding="utf-8"
        )


def concat_shards(output_dir: Path, merged_name: str = "raw_scores.jsonl") -> Path:
    merged = output_dir / merged_name
    with merged.open("w", encoding="utf-8") as dst:
        shard_dir = output_dir / "by_pair"
        for fp in sorted(shard_dir.glob("*.jsonl")):
            with fp.open(encoding="utf-8") as inp:
                for line in inp:
                    if line.strip():
                        dst.write(line)
    LOG.info("Wrote %s", merged)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prm-model", default="Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument(
        "--filtered-root",
        default=os.environ.get("FRS_REPO_ROOT", str(Path(__file__).resolve().parent.parent)),
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "analysis_outputs" / "prm_qwen225_baseline",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["gsm8k", "math500", "svamp", "aqua"],
    )
    parser.add_argument("--models", default="all")
    parser.add_argument("--questions-per-pair", type=int, default=50)
    parser.add_argument("--traces-per-question", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--question-manifest-csv",
        type=Path,
        default=None,
        help=(
            "question_sets.csv: model_stem,benchmark,question_idx. "
            "Default looks under analysis_outputs/trace0_k1_judging/"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Run 10 hard-coded traces incl. gpqa segmentation stress-test; exit.",
    )
    parser.add_argument(
        "--no-merge-shards-at-end",
        action="store_true",
        help="Skip creating raw_scores.jsonl from shards.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    tokenizer = AutoTokenizer.from_pretrained(args.prm_model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.prm_model,
        device_map=os.environ.get("PRM_DEVICE_MAP", "auto"),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()

    manifest: Dict[Tuple[str, str], List[int]] = {}
    csv_default = REPO_ROOT / "analysis_outputs" / "trace0_k1_judging" / "question_sets.csv"
    manifest_src = args.question_manifest_csv or (csv_default if csv_default.is_file() else None)
    if manifest_src:
        manifest = parse_question_manifest_csv(manifest_src)
        LOG.info("Loaded manifest from %s (%d keys)", manifest_src, len(manifest))

    device = next(model.parameters()).device
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.sanity_check:
        run_sanity_samples(
            model, tokenizer, device, filtered_root=args.filtered_root, max_length=args.max_length
        )
        LOG.info("--sanity-check done; exiting before full shards.")
        return

    stems = parse_models_arg(args.models)
    for bm in args.benchmarks:
        if bm not in BENCHMARK_DIR_SUFFIX:
            LOG.error("Unknown benchmark %s", bm)

    for bm in args.benchmarks:
        if bm not in BENCHMARK_DIR_SUFFIX:
            continue
        for stem in stems:
            LOG.info("Scoring %s × %s", stem, bm)
            run_pair_shard(
                model,
                tokenizer,
                device,
                stem=stem,
                benchmark=bm,
                filtered_root=args.filtered_root,
                output_dir=args.output_dir,
                manifest=manifest,
                questions_per_pair=args.questions_per_pair,
                traces_cap=args.traces_per_question,
                max_length=args.max_length,
                seed=args.seed,
            )

    if not args.no_merge_shards_at_end:
        concat_shards(args.output_dir)


if __name__ == "__main__":
    main()
