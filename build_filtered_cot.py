#!/usr/bin/env python3
"""
Build *_filtered_p1_only.jsonl for the FRS judge pipeline.

The in-repo consumer is run_filtered_cot_eval.py (glob *_filtered_p1_only.jsonl).
The producer script was previously offline; this implements the confidence filter
used in rebuttal analyses (global top fraction by bottom-10% mean token prob).

Example (HumanEval):

  python build_filtered_cot.py \\
    --input-jsonl evaluation/outputs/Llama_humaneval.jsonl \\
    --output-dir filtered-cot-humaneval \\
    --model-stem Llama_3.1_8B_Instruct \\
    --top-frac 0.10 \\
    --humaneval-split-reasoning

Then run the judge:

  python run_filtered_cot_eval.py --input-dir filtered-cot-humaneval
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

# Reuse extraction logic when splitting HumanEval reasoning vs code.
try:
    from humaneval_extract_answer import extract_humaneval_answer
except ImportError:
    extract_humaneval_answer = None  # type: ignore


def trace_bottom10_mean_prob(probs: list[float | None]) -> float:
    arr = np.asarray([p for p in probs if p is not None], dtype=float)
    if arr.size == 0:
        return float("nan")
    k = max(1, int(np.floor(0.10 * arr.size)))
    return float(np.sort(arr)[:k].mean())


def _normalize_list_field(val: Any) -> list:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]


def _gold_as_judge_string(record: dict) -> str:
    """Judge expects a string gold reference, not HumanEval's test dict."""
    gt = record.get("gt", "")
    if isinstance(gt, dict):
        entry = gt.get("entry_point", record.get("entry_point", ""))
        prompt = record.get("prompt", record.get("question", ""))
        return f"{prompt}\n\nFunction to implement: {entry}".strip()
    if gt:
        return str(gt)
    return str(record.get("question", record.get("prompt", "")))


def _split_humaneval_reasoning(full_text: str) -> tuple[str, str]:
    """
    Return (reasoning_for_judge, extracted_code_body).
    Falls back to full text as reasoning if extraction yields nothing.
    """
    if not full_text or not full_text.strip():
        return "", ""

    code_body = ""
    if extract_humaneval_answer is not None:
        code_body = extract_humaneval_answer(full_text)

    reasoning = full_text
    # Prefer text before the last fenced python block.
    blocks = list(re.finditer(r"```(?:python)?\s*\n.*?```", full_text, re.DOTALL | re.IGNORECASE))
    if blocks:
        reasoning = full_text[: blocks[-1].start()].strip()
    elif code_body and code_body in full_text:
        reasoning = full_text.replace(code_body, "").strip()

    if not reasoning.strip():
        reasoning = full_text.strip()
    return reasoning, code_body


def expand_traces(record: dict) -> list[dict]:
    """
    Expand one math_eval JSONL record into per-trace dicts with scalar fields.
    """
    code_list = _normalize_list_field(record.get("code"))
    pred_list = _normalize_list_field(record.get("pred"))
    score_list = _normalize_list_field(record.get("score"))
    probs_paths = (record.get("chosen_token_probs_per_path") or {}).get("epoch_0", [])

    n = max(len(code_list), len(pred_list), len(probs_paths), 1)
    traces: list[dict] = []
    for tid in range(n):
        raw_code = code_list[tid] if tid < len(code_list) else (code_list[0] if code_list else "")
        raw_pred = pred_list[tid] if tid < len(pred_list) else (pred_list[0] if pred_list else "")
        score = score_list[tid] if tid < len(score_list) else (score_list[0] if score_list else False)
        probs = probs_paths[tid] if tid < len(probs_paths) else []
        traces.append(
            {
                "trace_idx": tid,
                "raw_code": str(raw_code or ""),
                "raw_pred": str(raw_pred or ""),
                "score": bool(score) if score is not None else False,
                "confidence": trace_bottom10_mean_prob(probs),
            }
        )
    return traces


def build_filtered_records(
    input_jsonl: Path,
    top_frac: float,
    humaneval_split: bool,
) -> tuple[list[dict], dict]:
    """Return rows for filtered JSONL and summary stats."""
    pool: list[dict] = []
    n_questions = 0

    with input_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            n_questions += 1
            qidx = record.get("idx", n_questions - 1)
            judge_gold = _gold_as_judge_string(record)

            for tr in expand_traces(record):
                conf = tr["confidence"]
                if not np.isfinite(conf):
                    continue

                cot_text = tr["raw_code"] or tr["raw_pred"]
                pred_text = tr["raw_pred"] or tr["raw_code"]
                if humaneval_split:
                    merged = cot_text if cot_text else pred_text
                    cot_text, extracted = _split_humaneval_reasoning(merged)
                    if extracted:
                        pred_text = extracted

                row = dict(record)
                row["idx"] = qidx
                row["gt"] = judge_gold
                row["code"] = [cot_text]
                row["pred"] = [pred_text]
                row["score"] = [tr["score"]]
                row["answer_confidence"] = float(conf)
                row["trace_idx"] = tr["trace_idx"]
                pool.append(row)

    if not pool:
        raise RuntimeError(f"No traces with finite confidence in {input_jsonl}")

    pool.sort(key=lambda r: r["answer_confidence"], reverse=True)
    k = max(1, int(np.floor(top_frac * len(pool))))
    kept = pool[:k]

    summary = {
        "input": str(input_jsonl),
        "n_questions": n_questions,
        "n_traces_pooled": len(pool),
        "n_kept": len(kept),
        "top_frac": top_frac,
        "confidence_min_kept": kept[-1]["answer_confidence"] if kept else None,
        "confidence_max_kept": kept[0]["answer_confidence"] if kept else None,
        "humaneval_split_reasoning": humaneval_split,
    }
    return kept, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build filtered CoT JSONL for FRS judge.")
    parser.add_argument("--input-jsonl", type=Path, required=True, help="math_eval output JSONL")
    parser.add_argument("--output-dir", type=Path, required=True, help="e.g. filtered-cot-humaneval")
    parser.add_argument(
        "--model-stem",
        required=True,
        help="Output filename stem, e.g. Llama_3.1_8B_Instruct",
    )
    parser.add_argument(
        "--top-frac",
        type=float,
        default=0.10,
        help="Keep top fraction of all traces by confidence (default 0.10)",
    )
    parser.add_argument(
        "--humaneval-split-reasoning",
        action="store_true",
        help="Split reasoning (code[0]) vs extracted code (pred[0]); stringify gt for judge",
    )
    args = parser.parse_args()

    kept, summary = build_filtered_records(
        args.input_jsonl,
        top_frac=args.top_frac,
        humaneval_split=args.humaneval_split_reasoning,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.model_stem}_filtered_p1_only.jsonl"
    with out_path.open("w", encoding="utf-8") as out:
        for row in kept:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = args.output_dir / f"{args.model_stem}_filtered_build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {len(kept)} rows -> {out_path}")
    print(json.dumps(summary, indent=2))
    print("\nNext: python run_filtered_cot_eval.py --input-dir", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
