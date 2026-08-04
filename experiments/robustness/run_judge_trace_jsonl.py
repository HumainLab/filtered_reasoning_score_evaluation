#!/usr/bin/env python3
"""
Judge every trace in a trace-per-line JSONL file with the FRS GPT-4o-mini rubric.

Input format (one JSON object per line):
  idx, trace_idx, question, gt, code (list or str), score, answer_confidence (optional)

Output: FRS-style checkpoint JSON + per-trace CSV summary.

Usage:
  python analysis/run_judge_trace_jsonl.py \\
    --input-jsonl results/7b/DeepSeek-R1-Distill-Qwen-7B_filtered_p1_only.jsonl \\
    --model-name DS-R1-7B \\
    --output-dir results/7b \\
    --max-workers 6
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from topk_judge_eval import (  # noqa: E402
    DEFAULT_JUDGE_MODEL,
    Judge,
    reasoning_score_from_judge,
)

BIN_LABEL_DEFAULT = "filtered_p1"


def setup_logging(log_file: Path, verbose: bool) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("judge_trace_jsonl")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(threadName)-12s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_checkpoint(path: Path, data: Dict[str, Any]) -> None:
    """Atomic write; caller must not hold locks that workers need during dump."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    snap = copy.deepcopy(data)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def parse_row(row: dict) -> Optional[Dict[str, Any]]:
    try:
        idx = int(row["idx"])
        trace_idx = int(row.get("trace_idx", 0))
    except (KeyError, TypeError, ValueError):
        return None

    code = row.get("code", "")
    if isinstance(code, list):
        cot = code[0] if code else ""
    else:
        cot = str(code)
    if not cot.strip():
        return None

    scores = row.get("score", [])
    if isinstance(scores, list) and scores:
        correct = bool(scores[0])
    else:
        correct = bool(row.get("correct", False))

    conf = row.get("answer_confidence", row.get("confidence", float("nan")))
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = float("nan")

    return {
        "idx": idx,
        "trace_idx": trace_idx,
        "question": str(row.get("question", "")),
        "gt": str(row.get("gt", row.get("answer", ""))),
        "cot": cot,
        "correct": correct,
        "confidence": conf,
        "pred": row.get("pred"),
    }


def judge_trace(
    judge: Judge,
    trace: Dict[str, Any],
    model_name: str,
    benchmark: str,
    bin_label: str,
    max_retries: int = 4,
) -> Dict[str, Any]:
    key = f"{trace['idx']}:{trace['trace_idx']}"
    err = ""
    for attempt in range(max_retries):
        try:
            raw = judge.score(
                problem=trace["question"],
                cot=trace["cot"],
                gold=trace["gt"],
                flags_summary="No automated flags available.",
                evidence={"final_correct": trace["correct"]},
                log_ctx={
                    "eval_model": model_name,
                    "dataset": benchmark,
                    "idx": trace["idx"],
                    "trace_idx": trace["trace_idx"],
                    "bin": bin_label,
                },
            )
            rs = reasoning_score_from_judge(raw)
            ok = rs is not None
            return {
                "idx": trace["idx"],
                "trace_idx": trace["trace_idx"],
                "bin_label": bin_label,
                "confidence": trace["confidence"],
                "correct": trace["correct"],
                "judge_scores": raw,
                "reasoning_score": round(rs, 4) if rs is not None else None,
                "judge_ok": ok,
                "error": "" if ok else "incomplete_judge_scores",
                "_key": key,
            }
        except Exception as e:
            err = str(e)
            time.sleep(min(2**attempt, 30))
    return {
        "idx": trace["idx"],
        "trace_idx": trace["trace_idx"],
        "bin_label": bin_label,
        "confidence": trace["confidence"],
        "correct": trace["correct"],
        "judge_scores": None,
        "reasoning_score": None,
        "judge_ok": False,
        "error": err,
        "_key": key,
    }


def write_csv(path: Path, samples: List[Dict[str, Any]]) -> None:
    import csv

    rows_out = []
    for s in samples:
        js = s.get("judge_scores") or {}
        rows_out.append(
            {
                "idx": s["idx"],
                "trace_idx": s["trace_idx"],
                "confidence": s.get("confidence"),
                "correct": s.get("correct"),
                "faithfulness": js.get("faithfulness"),
                "utility": js.get("utility"),
                "coherence": js.get("coherence"),
                "factuality": js.get("factuality"),
                "reasoning_score": s.get("reasoning_score"),
                "judge_ok": s.get("judge_ok"),
                "error": s.get("error", ""),
            }
        )
    if not rows_out:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-jsonl", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model-name", type=str, required=True)
    ap.add_argument("--benchmark", type=str, default="filtered_p1")
    ap.add_argument("--bin-label", type=str, default=BIN_LABEL_DEFAULT)
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    ap.add_argument(
        "--portkey-key",
        default=os.environ.get("PORTKEY_API_KEY"),
        help="Portkey API key (or PORTKEY_API_KEY env)",
    )
    args = ap.parse_args()

    input_path = args.input_jsonl.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    ck_path = out_dir / f"judged_{stem}.json"
    csv_path = out_dir / f"judged_{stem}.csv"
    log_path = out_dir / f"judged_{stem}.log"

    log = setup_logging(log_path, args.verbose)

    traces: List[Dict[str, Any]] = []
    with open(input_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            t = parse_row(row)
            if t is None:
                log.warning("skip line %d: could not parse trace", line_no)
                continue
            traces.append(t)

    log.info("Loaded %d traces from %s", len(traces), input_path)

    if args.dry_run:
        print(f"Would judge {len(traces)} traces → {ck_path}")
        return 0

    if not args.portkey_key:
        log.error("PORTKEY_API_KEY or --portkey-key required")
        return 1

    ck: Dict[str, Any] = {}
    judged: Dict[str, Dict[str, Any]] = {}
    if ck_path.is_file() and not args.overwrite:
        ck = load_checkpoint(ck_path)
        for s in ck.get("judged_samples", []):
            k = f"{s['idx']}:{s['trace_idx']}"
            judged[k] = s
        log.info("Resume: %d traces already judged", len(judged))

    todo = [t for t in traces if f"{t['idx']}:{t['trace_idx']}" not in judged]
    log.info("Todo: %d / %d traces", len(todo), len(traces))

    if not todo and judged:
        write_csv(csv_path, list(judged.values()))
        log.info("Nothing to do; CSV at %s", csv_path)
        return 0

    judge = Judge(model=args.judge_model, portkey_api_key=args.portkey_key)
    lock = threading.Lock()
    n_ok = 0
    n_fail = 0
    t0 = time.perf_counter()

    def on_done(sample: Dict[str, Any]) -> None:
        nonlocal n_ok, n_fail
        k = sample.pop("_key", f"{sample['idx']}:{sample['trace_idx']}")
        ck_body: Optional[Dict[str, Any]] = None
        with lock:
            judged[k] = sample
            if sample.get("judge_ok"):
                n_ok += 1
            else:
                n_fail += 1
            done = n_ok + n_fail
            if done % 5 == 0 or done == len(todo):
                ck_body = {
                    "metadata": {
                        "model": args.model_name,
                        "benchmark": args.benchmark,
                        "input_file": str(input_path),
                        "bin_label": args.bin_label,
                        "n_traces_total": len(traces),
                        "n_judged_ok": sum(1 for s in judged.values() if s.get("judge_ok")),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    "judged_samples": list(judged.values()),
                }
        if ck_body is not None:
            save_checkpoint(ck_path, ck_body)
            log.info(
                "progress %d/%d ok=%d fail=%d elapsed=%.0fs",
                n_ok + n_fail,
                len(todo),
                n_ok,
                n_fail,
                time.perf_counter() - t0,
            )

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {
            ex.submit(
                judge_trace,
                judge,
                t,
                args.model_name,
                args.benchmark,
                args.bin_label,
            ): t
            for t in todo
        }
        for fut in as_completed(futs):
            on_done(fut.result())

    samples = list(judged.values())
    ck_final = {
        "metadata": {
            "model": args.model_name,
            "benchmark": args.benchmark,
            "input_file": str(input_path),
            "bin_label": args.bin_label,
            "n_traces_total": len(traces),
            "n_judged_ok": sum(1 for s in samples if s.get("judge_ok")),
            "n_judged_fail": sum(1 for s in samples if not s.get("judge_ok")),
            "mean_reasoning_score": float(
                np.nanmean([s["reasoning_score"] for s in samples if s.get("reasoning_score") is not None])
            )
            if samples
            else float("nan"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "judged_samples": samples,
    }
    save_checkpoint(ck_path, ck_final)
    write_csv(csv_path, samples)

    log.info("Done. checkpoint=%s csv=%s", ck_path, csv_path)
    print(f"Judged {len(samples)} traces ({ck_final['metadata']['n_judged_ok']} ok)")
    print(f"Mean reasoning_score: {ck_final['metadata']['mean_reasoning_score']:.4f}")
    print(f"Checkpoint: {ck_path}")
    print(f"CSV: {csv_path}")
    return 0 if ck_final["metadata"]["n_judged_fail"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
