#!/usr/bin/env python3
"""
End-to-end Filtered Reasoning Score (FRS) pipeline in a single file.

Given generation outputs with N (e.g. 16) samples per question per model and
per-token probabilities, this script:

  1. Computes per-trace confidence (bottom-10% mean token probability).
  2. Pools all traces for a model and keeps the global top X% (default 10%)
     by confidence.
  3. Runs the 4-pillar LLM reasoning judge (GPT-4o-mini) on the kept traces,
     routed through Portkey, with up to 100 judge calls in flight at once.
  4. Reports FRS = mean(fused overall) x 100 per model, and writes the
     filtered traces + per-trace judge results to disk.

This file is self-contained: it has its own copy of the judge rubric/parsing
and needs only `openai` + `numpy`. If it is dropped into the repo root (so that
`frs/cot_eval_v2` is importable), it will automatically use the exact
production judge + flag/evidence pipeline instead of the inlined fallback, so
the FRS numbers match the paper pipeline bit-for-bit.

USAGE
-----
  export PORTKEY_API_KEY=sk-...                 # Portkey gateway key
  python frs_pipeline.py \\
      --input "outputs/*.jsonl" \\
      --output-dir results/filtered_cot/humaneval \\
      --top-frac 0.10 \\
      --max-workers 100 \\
      --humaneval

INPUT
-----
  --input may be a single JSONL file, a directory (all *.jsonl inside), or a
  glob. One file == one model. Each line is one question with N samples:

    {
      "idx": 0,
      "question": "...",
      "gt": "243" | {"test": ..., "entry_point": ...},
      "code":  ["<reasoning trace 1>", ...],   # N reasoning strings
      "pred":  ["<final answer 1>", ...],      # N final answers
      "score": [true, false, ...],             # optional correctness per sample
      "chosen_token_probs_per_path": {"epoch_0": [[p,p,...], ...]}  # N prob lists
    }

  The per-token probability field is auto-detected from (in order):
    chosen_token_probs_per_path, probability_log_per_path,
    chosen_token_probs, probability_log
  Each may be {"epoch_0": [[...per token...], ...]} or a plain list of lists.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
# The 4-pillar judge package lives alongside this script (frs/cot_eval_v2).
JUDGE_PKG_DIR = SCRIPT_DIR

DEFAULT_PORTKEY_BASE_URL = "https://api.portkey.ai/v1"
DEFAULT_PORTKEY_MODEL_PREFIX = os.environ.get("PORTKEY_MODEL_PREFIX", "")
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"

PILLARS = ["faithfulness", "utility", "coherence", "factuality"]


# ───────────────────────── confidence ──────────────────────────────────────
def trace_bottom10_mean_prob(probs: List[Optional[float]]) -> float:
    """Bottom-10% mean token probability (matches Appendix-S `confidence`)."""
    arr = np.asarray([p for p in probs if p is not None], dtype=float)
    if arr.size == 0:
        return float("nan")
    k = max(1, int(np.floor(0.10 * arr.size)))
    return float(np.sort(arr)[:k].mean())


def _per_path_probs(record: Dict[str, Any]) -> List[List[float]]:
    """Return list of per-token prob lists, one per trace. Robust to schemas."""
    candidates = [
        "chosen_token_probs_per_path",
        "probability_log_per_path",
        "chosen_token_probs",
        "probability_log",
    ]
    for key in candidates:
        val = record.get(key)
        if val is None:
            continue
        if isinstance(val, dict):
            val = val.get("epoch_0")
        if not isinstance(val, list) or not val:
            continue
        # list of lists -> per-path; flat list of floats -> single trace
        if all(isinstance(x, (int, float)) for x in val):
            if any(v not in (None, []) for v in val):
                return [val]  # single trace, flat probs
            continue
        per_path = [p for p in val if isinstance(p, list) and len(p) > 0]
        if per_path:
            return per_path
    return []


# ───────────────────────── trace expansion ─────────────────────────────────
def _as_list(val: Any) -> list:
    if val is None:
        return []
    return val if isinstance(val, list) else [val]


def _gold_as_judge_string(record: Dict[str, Any]) -> str:
    """Judge wants a string gold reference, not HumanEval's test dict."""
    gt = record.get("gt", "")
    if isinstance(gt, dict):
        entry = gt.get("entry_point", record.get("entry_point", ""))
        prompt = record.get("prompt", record.get("question", ""))
        return f"{prompt}\n\nFunction to implement: {entry}".strip()
    return str(gt) if gt else str(record.get("question", record.get("prompt", "")))


def _split_humaneval_reasoning(full_text: str) -> Tuple[str, str]:
    """Return (reasoning_for_judge, code_after_last_fence)."""
    if not full_text or not full_text.strip():
        return "", ""
    code_body = ""
    blocks = list(re.finditer(r"```(?:python)?\s*\n(.*?)```", full_text, re.DOTALL | re.IGNORECASE))
    if blocks:
        code_body = blocks[-1].group(1).rstrip()
        reasoning = full_text[: blocks[-1].start()].strip()
    else:
        reasoning = full_text.strip()
    if not reasoning.strip():
        reasoning = full_text.strip()
    return reasoning, code_body


def expand_traces(record: Dict[str, Any], humaneval: bool) -> List[Dict[str, Any]]:
    code_list = _as_list(record.get("code"))
    pred_list = _as_list(record.get("pred"))
    score_list = _as_list(record.get("score"))
    probs_paths = _per_path_probs(record)

    n = max(len(code_list), len(pred_list), len(probs_paths), 1)
    qidx = record.get("idx", 0)
    gold = _gold_as_judge_string(record)
    question = record.get("question", record.get("prompt", ""))

    out: List[Dict[str, Any]] = []
    for tid in range(n):
        raw_code = str(code_list[tid] if tid < len(code_list) else (code_list[0] if code_list else ""))
        raw_pred = str(pred_list[tid] if tid < len(pred_list) else (pred_list[0] if pred_list else ""))
        score = score_list[tid] if tid < len(score_list) else (score_list[0] if score_list else False)
        probs = probs_paths[tid] if tid < len(probs_paths) else []

        cot_text = raw_code or raw_pred
        pred_text = raw_pred or raw_code
        if humaneval:
            reasoning, code_body = _split_humaneval_reasoning(cot_text or pred_text)
            cot_text = reasoning
            if code_body:
                pred_text = code_body

        out.append(
            {
                "idx": qidx,
                "trace_idx": tid,
                "question": question,
                "gt": gold,
                "cot_text": cot_text,
                "pred": pred_text,
                "original_correct": bool(score) if score is not None else False,
                "confidence": trace_bottom10_mean_prob(probs),
            }
        )
    return out


def filter_top_traces(traces: List[Dict[str, Any]], top_frac: float) -> List[Dict[str, Any]]:
    pool = [t for t in traces if np.isfinite(t["confidence"])]
    if not pool:
        return []
    pool.sort(key=lambda t: t["confidence"], reverse=True)
    k = max(1, int(np.floor(top_frac * len(pool))))
    return pool[:k]


# ───────────────────────── inlined judge (fallback) ────────────────────────
# Faithful copy of frs/cot_eval_v2/judge.py build_prompt rubric.
_JUDGE_SYSTEM = "You are a careful and consistent evaluator of reasoning quality."

_RUBRIC = """You are an expert evaluator of mathematical and logical reasoning. 
Score the chain-of-thought (CoT) on 4 dimensions using the scoring criteria below.

Each score must be an integer from 1-5 (1 = very poor, 5 = excellent).

## SCORING CRITERIA

### 1. FAITHFULNESS (1-5)
**Definition:** Reasoning is internally consistent, follows logical rules, and stays focused on the problem without hidden shortcuts or leaps.
- 5: Perfect logical consistency, no contradictions, stays completely on-topic
- 4: Minor inconsistencies or slight tangents, but overall coherent
- 3: Some logical gaps or moderate off-topic content
- 2: Significant logical flaws or frequent tangents
- 1: Major contradictions, illogical leaps, or completely off-topic

### 2. UTILITY (1-5)
**Definition:** Each step meaningfully contributes to solving the problem, calculations are correct, and reasoning efficiently leads to the final answer.
- 5: Every step is necessary and correct, efficient path to solution
- 4: Most steps useful, minor inefficiencies or small errors
- 3: Some useful steps mixed with unnecessary ones or calculation errors
- 2: Many unnecessary steps or significant calculation errors
- 1: Mostly useless steps, major calculation errors, or repetitive content

### 3. COHERENCE (1-5)
**Definition:** Steps flow smoothly from one to the next with clear logical progression and smooth transitions.
- 5: Perfect flow, each step naturally follows from the previous
- 4: Good flow with minor awkward transitions
- 3: Some disjointed steps but overall progression
- 2: Choppy flow with unclear connections between steps
- 1: Disjointed, random steps with no clear progression

### 4. FACTUALITY (1-5)
**Definition:** Every step must be factually correct and grounded in the problem context, not hallucinated from surface-level understanding.
- 5: All facts and statements are accurate and grounded in the problem
- 4: Mostly accurate with minor factual errors
- 3: Some factual errors or unsupported claims
- 2: Multiple factual errors or significant hallucinations
- 1: Major factual errors, hallucinations, or completely unsupported claims

## EVALUATION PROCESS
1. Read the problem carefully to understand the context and given information
2. Analyze each step of the CoT reasoning
3. Check each step against the four criteria above
4. Assign scores based on the specific guidelines for each dimension
5. Ensure every step is evaluated for factual accuracy and logical soundness

## Problem
{problem}

## Model Reasoning (CoT)
{cot}

## Gold Answer
{gold}

## Instructions
Carefully evaluate the reasoning against the four criteria above and apply your own judgment.

Do NOT include explanations. Output only the JSON object.

Required JSON schema:
{{
  "faithfulness": <1-5>,
  "utility": <1-5>,
  "coherence": <1-5>,
  "factuality": <1-5>
}}"""


def _extract_json_safely(raw_output: str) -> Dict[str, Optional[int]]:
    none = {p: None for p in PILLARS}
    if not raw_output:
        return none

    def _ok(d: Any) -> bool:
        return isinstance(d, dict) and all(
            isinstance(d.get(k), int) and 1 <= d.get(k) <= 5 for k in PILLARS
        )

    try:
        parsed = json.loads(raw_output)
        if _ok(parsed):
            return parsed
    except Exception:
        pass

    pattern = r'\{[^{}]*"faithfulness"[^{}]*"utility"[^{}]*"coherence"[^{}]*"factuality"[^{}]*\}'
    for m in re.findall(pattern, raw_output, re.IGNORECASE | re.DOTALL):
        try:
            parsed = json.loads(m)
            if _ok(parsed):
                return parsed
        except Exception:
            continue

    for js in reversed(re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw_output)):
        try:
            parsed = json.loads(js)
            if _ok(parsed):
                return parsed
        except Exception:
            continue
    return none


def _fuse_judge_overall(judge_scores: Dict[str, Optional[int]]) -> Dict[str, float]:
    """Mirror backend fuse_with_judge when judge is ALWAYS-on: judge-only."""
    fused: Dict[str, float] = {}
    for p in PILLARS:
        v = judge_scores.get(p)
        if v is None:
            fused[p] = 0.0
        else:
            v = min(5, max(1, int(v)))
            fused[p] = max(0.0, min(1.0, (v - 1) / 4.0))
    fused["overall"] = sum(fused[p] for p in PILLARS) / 4.0
    return fused


class InlineJudge:
    """Standalone judge: Portkey/OpenAI chat completion + robust JSON parse."""

    def __init__(self, model: str, base_url: Optional[str], api_key: Optional[str]):
        from openai import OpenAI

        kwargs: Dict[str, Any] = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        self.client = OpenAI(**kwargs)
        self.model = model

    def score(self, problem: str, cot: str, gold: str) -> Dict[str, Optional[int]]:
        prompt = _RUBRIC.format(problem=problem, cot=cot, gold=gold)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1000,
        )
        return _extract_json_safely(resp.choices[0].message.content or "")


# ───────────────────────── backend judge (preferred) ───────────────────────
def try_load_backend_judge(model: str, base_url: Optional[str], api_key: Optional[str]):
    """Return (judge_obj, evaluator_cls) if the repo backend is importable."""
    if not (JUDGE_PKG_DIR / "cot_eval_v2" / "judge.py").exists():
        return None, None
    try:
        if str(JUDGE_PKG_DIR) not in sys.path:
            sys.path.insert(0, str(JUDGE_PKG_DIR))
        from cot_eval_v2.judge import Judge as BackendJudge
        from cot_eval_v2.evaluator import PillarsEvaluator

        judge = BackendJudge(
            model=model,
            mode="ALWAYS",
            diagnostic=False,
            base_url=base_url,
            api_key=api_key,
        )
        return judge, PillarsEvaluator
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"Backend judge unavailable, using inlined judge: {e}")
        return None, None


# ───────────────────────── progress ────────────────────────────────────────
class Progress:
    def __init__(self, total: int):
        self._lock = threading.Lock()
        self.total = total
        self.done = 0
        self.errors = 0
        self.start = time.time()

    def tick(self, error: bool = False):
        with self._lock:
            self.done += 1
            if error:
                self.errors += 1
            done, total, errs = self.done, self.total, self.errors
        if done % 25 == 0 or done == total:
            el = time.time() - self.start
            rate = done / el if el else 0
            eta = (total - done) / rate if rate else 0
            print(
                f"  judge {done}/{total}  errors={errs}  "
                f"{rate:.1f}/s  eta={eta:.0f}s",
                flush=True,
            )


# ───────────────────────── pipeline ────────────────────────────────────────
def resolve_inputs(input_arg: str) -> List[Path]:
    p = Path(input_arg)
    if p.is_dir():
        return sorted(p.glob("*.jsonl"))
    if any(ch in input_arg for ch in "*?[]"):
        return sorted(Path(x) for x in glob.glob(input_arg))
    return [p]


def build_model_tasks(
    files: List[Path], top_frac: float, humaneval: bool, limit: Optional[int]
) -> Dict[str, List[Dict[str, Any]]]:
    per_model: Dict[str, List[Dict[str, Any]]] = {}
    for fp in files:
        stem = fp.name
        for suffix in ("_filtered_p1_only.jsonl", ".jsonl"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        all_traces: List[Dict[str, Any]] = []
        with fp.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if limit is not None and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                all_traces.extend(expand_traces(rec, humaneval))
        kept = filter_top_traces(all_traces, top_frac)
        for t in kept:
            t["model"] = stem
        per_model[stem] = kept
        print(
            f"[filter] {stem}: pooled {len(all_traces)} traces -> kept {len(kept)} "
            f"(top {top_frac:.0%})",
            flush=True,
        )
    return per_model


def judge_one(task: Dict[str, Any], judge, evaluator_cls) -> Dict[str, Any]:
    problem = task["question"]
    cot = task["cot_text"]
    gold = task["gt"]

    if not cot or not cot.strip():
        return {**_trace_meta(task), "fused_scores": None, "judge_scores": None, "status": "empty_cot"}

    try:
        if evaluator_cls is not None:
            flags, evidence, _rule, judge_scores, fused = evaluator_cls(judge=judge).analyze(
                problem=problem, cot_text=cot, gold=gold
            )
        else:
            judge_scores = judge.score(problem, cot, gold)
            fused = _fuse_judge_overall(judge_scores)
        return {
            **_trace_meta(task),
            "judge_scores": judge_scores,
            "fused_scores": fused,
            "status": "ok",
        }
    except Exception as e:  # noqa: BLE001
        return {
            **_trace_meta(task),
            "judge_scores": None,
            "fused_scores": None,
            "status": "error",
            "error": str(e)[:300],
        }


def _trace_meta(task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "idx": task["idx"],
        "trace_idx": task["trace_idx"],
        "model": task["model"],
        "confidence": task["confidence"],
        "original_correct": task["original_correct"],
        "pred": str(task.get("pred", ""))[:500],
    }


def run(args: argparse.Namespace) -> int:
    api_key = args.portkey_key or os.environ.get("PORTKEY_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: no API key. Set --portkey-key or PORTKEY_API_KEY.", file=sys.stderr)
        return 2

    use_portkey = bool(args.portkey_key or os.environ.get("PORTKEY_API_KEY"))
    base_url = DEFAULT_PORTKEY_BASE_URL if use_portkey else None
    model = f"{args.portkey_model_prefix}/{args.judge_model}" if use_portkey else args.judge_model

    files = resolve_inputs(args.input)
    if not files:
        print(f"ERROR: no input files matched {args.input!r}", file=sys.stderr)
        return 2
    print(f"Inputs ({len(files)}): " + ", ".join(f.name for f in files))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    per_model = build_model_tasks(files, args.top_frac, args.humaneval, args.limit)

    # Persist the filtered traces (FRS judge input) per model.
    for stem, kept in per_model.items():
        fpath = out_dir / f"{stem}_filtered_p1_only.jsonl"
        with fpath.open("w", encoding="utf-8") as out:
            for t in kept:
                out.write(json.dumps({
                    "idx": t["idx"], "trace_idx": t["trace_idx"], "question": t["question"],
                    "gt": t["gt"], "code": [t["cot_text"]], "pred": [t["pred"]],
                    "score": [t["original_correct"]], "answer_confidence": t["confidence"],
                }, ensure_ascii=False) + "\n")

    all_tasks = [t for kept in per_model.values() for t in kept]
    print(f"\nTotal judge calls: {len(all_tasks)} | max_workers={args.max_workers}")

    if args.dry_run:
        for stem, kept in per_model.items():
            print(f"  [dry-run] {stem}: {len(kept)} traces")
        return 0

    judge, evaluator_cls = try_load_backend_judge(model, base_url, api_key)
    if judge is None:
        judge = InlineJudge(model=model, base_url=base_url, api_key=api_key)
        evaluator_cls = None
        print("Judge: inlined standalone judge (judge-only FRS).")
    else:
        print("Judge: backend PillarsEvaluator + Judge (production FRS).")

    progress = Progress(len(all_tasks))
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(judge_one, t, judge, evaluator_cls): t for t in all_tasks}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            progress.tick(error=res.get("status") == "error")

    # Aggregate per model and write results.
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        by_model.setdefault(r["model"], []).append(r)

    summary_rows = []
    print("\n" + "=" * 64)
    print("  FRS RESULTS")
    print("=" * 64)
    for stem in sorted(by_model):
        rows = sorted(by_model[stem], key=lambda r: (r["idx"], r["trace_idx"]))
        overalls = [
            r["fused_scores"]["overall"]
            for r in rows
            if r.get("status") == "ok" and r.get("fused_scores")
        ]
        frs = 100.0 * sum(overalls) / len(overalls) if overalls else float("nan")
        n_err = sum(1 for r in rows if r.get("status") == "error")
        n_empty = sum(1 for r in rows if r.get("status") == "empty_cot")

        result_path = results_dir / f"{stem}_filtered_p1_only_results.json"
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": stem,
                    "frs_pct": frs,
                    "n_judged": len(overalls),
                    "n_total": len(rows),
                    "errors": n_err,
                    "empty_cot": n_empty,
                    "top_frac": args.top_frac,
                    "judge_model": model,
                    "results": rows,
                },
                f,
                indent=2,
                default=str,
            )
        summary_rows.append({"model": stem, "frs_pct": frs, "n": len(overalls), "errors": n_err})
        print(f"  {stem:46s} FRS={frs:6.2f}%  n={len(overalls):4d}  err={n_err}")

    (out_dir / "frs_summary.json").write_text(
        json.dumps({"top_frac": args.top_frac, "judge_model": model, "models": summary_rows}, indent=2),
        encoding="utf-8",
    )
    print("=" * 64)
    print(f"Per-model results: {results_dir}/")
    print(f"Summary: {out_dir / 'frs_summary.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="End-to-end FRS pipeline (filter + LLM judge).")
    ap.add_argument("--input", required=True, help="JSONL file, directory, or glob (one file per model)")
    ap.add_argument("--output-dir", required=True, help="e.g. results/filtered_cot/humaneval")
    ap.add_argument("--top-frac", type=float, default=0.10, help="Keep top fraction by confidence (default 0.10)")
    ap.add_argument("--max-workers", type=int, default=100, help="Concurrent judge calls (default 100)")
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Judge model name (default gpt-4o-mini)")
    ap.add_argument("--portkey-key", default=None, help="Portkey API key (else PORTKEY_API_KEY env)")
    ap.add_argument("--portkey-model-prefix", default=DEFAULT_PORTKEY_MODEL_PREFIX, help="Portkey model prefix")
    ap.add_argument("--humaneval", action="store_true", help="Split reasoning vs code; stringify gt for judge")
    ap.add_argument("--limit", type=int, default=None, help="Max questions per file (debug)")
    ap.add_argument("--dry-run", action="store_true", help="Filter + write inputs, skip judge calls")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
