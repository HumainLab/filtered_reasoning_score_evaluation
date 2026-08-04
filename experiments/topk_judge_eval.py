"""
Top-K Judge Evaluation for FRS Paper
======================================
Self-contained script that runs GPT-4o-mini judge on confidence-selected traces.
For each problem, judges the most confident and least confident trace.

Requires: pip install portkey-ai numpy

Usage (Portkey API key via flag or ``PORTKEY_API_KEY`` env):
    python topk_judge_eval.py single \\
        --input-jsonl /path/to/file.jsonl \\
        --output-dir ./topk_judge_results \\
        --model-name "DS-R1-7B" \\
        --dataset-name "GSM8K" \\
        --portkey-key "$PORTKEY_API_KEY" \\
        --parallel 10

Batch usage for all files:
    python topk_judge_eval.py batch \\
        --data-root "/path/to/threshold" \\
        --output-dir ./topk_judge_results \\
        --datasets GSM8K MATH500 SVAMP \\
        --portkey-key "$PORTKEY_API_KEY" \\
        --parallel 10

Analyze saved results:
    python topk_judge_eval.py analyze --output-dir ./topk_judge_results

Logging (stderr + optional ``--log-file``):
    python topk_judge_eval.py single ... -v
    python topk_judge_eval.py batch ... --log-file ./topk_judge.log

Resume:
    Re-run the same ``single`` or ``batch`` command. Already-finished problem indices
    (``idx``) are skipped. Checkpoints are written atomically (temp file + rename).
    Default ``--checkpoint-every`` is 1 (save after every problem). Use a larger value for less I/O.
"""

import json
import re
import os
import sys
import tempfile
import time
import warnings
import argparse
import glob
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

LOG = logging.getLogger("topk_judge")
LOG_API = logging.getLogger("topk_judge.api")
LOG_RUN = logging.getLogger("topk_judge.run")


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """Console + optional file; thread name in each line for parallel runs."""
    fmt = "%(asctime)s | %(levelname)-8s | %(threadName)-12s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handler_console = logging.StreamHandler(sys.stderr)
    handler_console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    handler_console.setLevel(level)
    handlers: List[logging.Handler] = [handler_console]
    if log_file:
        _d = os.path.dirname(os.path.abspath(log_file))
        if _d:
            os.makedirs(_d, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        fh.setLevel(level)
        handlers.append(fh)
    for lg in (LOG, LOG_API, LOG_RUN):
        lg.handlers.clear()
        for h in handlers:
            lg.addHandler(h)
        lg.setLevel(level)
        lg.propagate = False

# ═══════════════════════════════════════════════════════════════════════════════
# Judge (API client)
# ═══════════════════════════════════════════════════════════════════════════════

# Paper / evaluation: judge is always GPT-4o-mini (OpenAI). Portkey uses a gateway model id.
JUDGE_LLM = "gpt-4o-mini"
DEFAULT_PORTKEY_GATEWAY = os.environ.get("PORTKEY_MODEL_PREFIX", "")
DEFAULT_JUDGE_MODEL = f"{DEFAULT_PORTKEY_GATEWAY}/{JUDGE_LLM}"


class Judge:
    """LLM judge: GPT-4o-mini via Portkey (OpenAI-compatible ``chat.completions`` API)."""

    def __init__(
        self,
        model: str = DEFAULT_JUDGE_MODEL,
        portkey_api_key: Optional[str] = None,
    ):
        # `model` is the Portkey chat model string; default routes to JUDGE_LLM.
        self.model = model
        key = portkey_api_key or os.environ.get("PORTKEY_API_KEY")
        if not key:
            raise ValueError("Portkey API key missing: pass portkey_api_key or set PORTKEY_API_KEY")
        try:
            from portkey_ai import Portkey

            self.client = Portkey(api_key=key)
        except ImportError as e:
            raise ImportError("pip install portkey-ai") from e

    def build_prompt(
        self,
        problem: str,
        cot: str,
        gold: str,
        flags_summary: str,
        evidence: Dict[str, Any],
    ) -> str:
        return f"""You are an expert evaluator of mathematical and logical reasoning. 
Score the chain-of-thought (CoT) on 4 dimensions using the scoring criteria below.

Each score must be an integer from 1-5 (1 = very poor, 5 = excellent).

## SCORING CRITERIA

### 1. FAITHFULNESS (1-5)
**Definition:** Reasoning is internally consistent, follows logical rules, and stays focused on the problem without hidden shortcuts or leaps.

**Scoring Guidelines:**
- 5: Perfect logical consistency, no contradictions, stays completely on-topic
- 4: Minor inconsistencies or slight tangents, but overall coherent
- 3: Some logical gaps or moderate off-topic content
- 2: Significant logical flaws or frequent tangents
- 1: Major contradictions, illogical leaps, or completely off-topic

**Dock Points For:**
- Contradictory statements within the reasoning
- Logical leaps without justification
- Going off-topic or discussing irrelevant matters
- Hidden assumptions not stated explicitly
- Unjustified final answers (not derivable from steps)
- Shortcut reasoning (non-contributing steps)

### 2. UTILITY (1-5)
**Definition:** Each step meaningfully contributes to solving the problem, calculations are correct, and reasoning efficiently leads to the final answer.

**Scoring Guidelines:**
- 5: Every step is necessary and correct, efficient path to solution
- 4: Most steps useful, minor inefficiencies or small errors
- 3: Some useful steps mixed with unnecessary ones or calculation errors
- 2: Many unnecessary steps or significant calculation errors
- 1: Mostly useless steps, major calculation errors, or repetitive content

**Dock Points For:**
- Incorrect calculations or mathematical errors
- Repetitive statements that don't add value
- Unnecessary verbose explanations
- Steps that don't advance toward the solution
- Redundant reasoning or circular logic
- Off-topic steps that don't contribute

### 3. COHERENCE (1-5)
**Definition:** Steps flow smoothly from one to the next with clear logical progression and smooth transitions.

**Scoring Guidelines:**
- 5: Perfect flow, each step naturally follows from the previous
- 4: Good flow with minor awkward transitions
- 3: Some disjointed steps but overall progression
- 2: Choppy flow with unclear connections between steps
- 1: Disjointed, random steps with no clear progression

**Dock Points For:**
- Abrupt transitions between ideas
- Missing connecting logic between steps
- Disjointed or random sequence of reasoning
- Poor organization of thoughts
- Dangling references (use-before-define)
- Disordered reasoning chain

### 4. FACTUALITY (1-5)
**Definition:** Every step must be factually correct and grounded in the problem context, not hallucinated from surface-level understanding.

**Scoring Guidelines:**
- 5: All facts and statements are accurate and grounded in the problem
- 4: Mostly accurate with minor factual errors
- 3: Some factual errors or unsupported claims
- 2: Multiple factual errors or significant hallucinations
- 1: Major factual errors, hallucinations, or completely unsupported claims

**Dock Points For:**
- Hallucinated facts not present in the problem
- Incorrect interpretations of given information
- Making assumptions not supported by the problem context
- Surface-level understanding leading to wrong facts
- Stating things as facts that are actually assumptions
- Claims that contradict the problem evidence

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

## Automated Flag Analysis
{flags_summary}

## Evidence Summary
{json.dumps(evidence, indent=2)}

## Instructions
Based on the automated flag analysis above, carefully evaluate the reasoning. The flags highlight specific issues that should influence your scoring:

- **Faithfulness flags** indicate logical inconsistencies, unjustified conclusions, or shortcut reasoning
- **Utility flags** point to redundant, off-topic, or non-contributing steps
- **Coherence flags** reveal structural problems like dangling references or disordered chains
- **Factuality flags** identify unsupported claims or contradictions with the problem

Use these flags as guidance, but apply your own judgment to determine the final scores. Consider the severity and impact of each flagged issue.

Do NOT include explanations. Output only the JSON object.

Required JSON schema:
{{
  "faithfulness": <1-5>,
  "utility": <1-5>,
  "coherence": <1-5>,
  "factuality": <1-5>
}}"""

    def score(
        self,
        problem: str,
        cot: str,
        gold: str,
        flags_summary: str = "No automated flags available.",
        evidence: Optional[Dict[str, Any]] = None,
        log_ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Optional[int]]:
        if evidence is None:
            evidence = {"final_correct": None, "note": "Standalone evaluation without flag pipeline"}
        ctx = log_ctx or {}
        ctx_s = " ".join(f"{k}={v}" for k, v in sorted(ctx.items()) if v is not None)
        prompt = self.build_prompt(problem, cot, gold, flags_summary, evidence)
        prompt_chars = len(prompt)
        cot_chars = len(cot)
        LOG_API.info(
            "API → request start | model=%s | prompt_chars=%d cot_chars=%d gold=%r | %s",
            self.model,
            prompt_chars,
            cot_chars,
            (gold[:80] + "…") if len(str(gold)) > 80 else gold,
            ctx_s or "(no ctx)",
        )
        if LOG_API.isEnabledFor(logging.DEBUG):
            LOG_API.debug("API → user message head (400 chars): %s", prompt[:400])
        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a careful and consistent evaluator of reasoning quality.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            dt = time.perf_counter() - t0
            raw = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            udict = (
                {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
                if usage is not None
                else {}
            )
            LOG_API.info(
                "API ← response ok | latency_s=%.3f | raw_chars=%d | usage=%s | %s",
                dt,
                len(raw),
                udict,
                ctx_s or "",
            )
            if LOG_API.isEnabledFor(logging.DEBUG):
                LOG_API.debug("API ← raw body: %s", raw[:800] + ("…" if len(raw) > 800 else ""))
            out = self._extract_json(raw)
            if all(out.get(k) is not None for k in ("faithfulness", "utility", "coherence", "factuality")):
                LOG_API.info(
                    "API ← parsed scores | F=%s U=%s C=%s Fa=%s | %s",
                    out.get("faithfulness"),
                    out.get("utility"),
                    out.get("coherence"),
                    out.get("factuality"),
                    ctx_s or "",
                )
            else:
                LOG_API.warning("API ← parse incomplete | out=%s | %s", out, ctx_s or "")
            return out
        except Exception as e:
            dt = time.perf_counter() - t0
            LOG_API.exception("API ✗ failed after %.3fs | %s", dt, ctx_s or "")
            warnings.warn(f"API call failed: {e}")
            return {"faithfulness": None, "utility": None, "coherence": None, "factuality": None}

    def _extract_json(self, raw: str) -> Dict[str, Optional[int]]:
        null = {"faithfulness": None, "utility": None, "coherence": None, "factuality": None}
        try:
            parsed = json.loads(raw)
            if self._valid(parsed):
                return parsed
        except Exception:
            pass
        matches = re.findall(r'\{[^{}]*"faithfulness"[^{}]*\}', raw, re.DOTALL)
        if matches:
            try:
                parsed = json.loads(matches[-1])
                if self._valid(parsed):
                    return parsed
            except Exception:
                pass
        LOG_API.warning("Judge JSON parse failed | raw_head=%r", raw[:300])
        warnings.warn(f"Failed to parse judge output: {raw[:200]}")
        return null

    @staticmethod
    def _valid(d: Any) -> bool:
        if not isinstance(d, dict):
            return False
        for k in ["faithfulness", "utility", "coherence", "factuality"]:
            v = d.get(k)
            if not isinstance(v, int) or v < 1 or v > 5:
                return False
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# Confidence
# ═══════════════════════════════════════════════════════════════════════════════

LOW_PROB_CUTOFF = 0.10


def compute_trace_confidence(token_probs: List[float]) -> float:
    """Mean probability of the lowest 10% of token probs in the trace."""
    if not token_probs:
        return float("nan")
    arr = np.asarray(token_probs, dtype=np.float64)
    n_low = max(1, int(len(arr) * LOW_PROB_CUTOFF))
    kth = min(n_low - 1, len(arr) - 1)
    lowest = np.partition(arr, kth)[:n_low]
    return float(np.mean(lowest))


def reasoning_score_from_judge(scores: Dict[str, Optional[int]]) -> Optional[float]:
    """Normalized [0,1]: (sum - 4) / 16 for four 1–5 scores."""
    vals = [scores.get(k) for k in ["faithfulness", "utility", "coherence", "factuality"]]
    if any(v is None for v in vals):
        return None
    return (sum(vals) - 4) / 16.0


# ═══════════════════════════════════════════════════════════════════════════════
# File discovery
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_MAP = {
    "gsm8k": "GSM8K",
    "math500": "MATH500",
    "svamp": "SVAMP",
    "aqua": "AQuA",
    "gpqa": "GPQA",
    "commonsense_qa": "CommonsenseQA",
    "custom": "MATH500",
}
MODEL_MAP = {
    "DeepSeek_R1_Distill_Qwen_1.5B": "DS-R1-1.5B",
    "DeepSeek_R1_Distill_Qwen_7B": "DS-R1-7B",
    "Llama_3.1_8B_Instruct": "LLaMA-3.1-8B",
    "Qwen2.5_7B_Instruct": "Qwen2.5-7B",
    "Qwen2.5_Math_7B": "Qwen2.5-Math",
    "gemma_7b": "Gemma-7B",
    "phi_4": "Phi-4",
    "Phi_4_reasoning": "Phi-4-Reas.",
    "Qwen3_4B_Thinking_2507": "Qwen3-4B",
}


def extract_model_dataset(filepath: str) -> Tuple[str, str]:
    parts = filepath.replace("\\", "/").split("/")
    model = None
    for i, p in enumerate(parts):
        if p.startswith("source_pass16"):
            if i + 1 < len(parts):
                model = parts[i + 1]
            break
    fname = parts[-1]
    ds_match = re.match(r"^([a-z_0-9]+?)__", fname)
    dataset = ds_match.group(1) if ds_match else "unknown"
    dataset = DATASET_MAP.get(dataset, dataset)
    model = MODEL_MAP.get(model, model)
    return model, dataset


def discover_files(data_root: str, datasets: List[str]) -> Dict[Tuple[str, str], str]:
    pattern = os.path.join(data_root, "data/pass16_sample*", "**", "*.jsonl")
    all_files = glob.glob(pattern, recursive=True)
    file_map: Dict[Tuple[str, str], str] = {}
    for fp in all_files:
        model, dataset = extract_model_dataset(fp)
        if dataset not in datasets:
            continue
        key = (model, dataset)
        if key not in file_map or "_processed" in fp:
            file_map[key] = fp
    return file_map


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def select_traces(row: dict) -> Tuple[Optional[dict], Optional[dict]]:
    scores = row.get("score", [])
    code = row.get("code", [])
    pred = row.get("pred", [])
    probs_all = row.get("chosen_token_probs_per_path", {}).get("epoch_0", [])
    if not isinstance(probs_all, list):
        return None, None
    n = min(len(scores), len(code), len(probs_all))
    if n == 0:
        return None, None

    confidences = []
    for i in range(n):
        c = compute_trace_confidence(probs_all[i] if i < len(probs_all) else [])
        confidences.append((i, c))

    valid = [(i, c) for i, c in confidences if not np.isnan(c)]
    if not valid:
        return None, None

    valid.sort(key=lambda x: x[1])
    least_idx, least_conf = valid[0]
    most_idx, most_conf = valid[-1]

    def build(idx: int, conf: float) -> dict:
        return {
            "trace_idx": idx,
            "confidence": conf,
            "correct": bool(scores[idx]) if idx < len(scores) else None,
            "pred": pred[idx] if idx < len(pred) else None,
            "cot": code[idx] if idx < len(code) else "",
        }

    return build(most_idx, most_conf), build(least_idx, least_conf)


def judge_one_problem(
    judge: Judge,
    row: dict,
    problem_idx: int,
    run_ctx: Optional[Dict[str, Any]] = None,
) -> Optional[dict]:
    question = row.get("question", "")
    gt = row.get("gt", row.get("answer", ""))
    rc = run_ctx or {}
    pidx = row.get("idx", problem_idx)

    most, least = select_traces(row)
    if most is None:
        LOG_RUN.warning("skip problem | no valid traces | idx=%s | %s", pidx, rc)
        return None

    LOG_RUN.info(
        "problem start | idx=%s line_i=%s | most_trace=%s conf=%.4f | least_trace=%s conf=%.4f | %s",
        pidx,
        problem_idx,
        most["trace_idx"],
        most["confidence"],
        least["trace_idx"],
        least["confidence"],
        " ".join(f"{k}={v}" for k, v in sorted(rc.items())),
    )

    result: Dict[str, Any] = {
        "idx": pidx,
        "question": question[:200],
        "gt": gt,
    }

    for label, trace_info in [("most_confident", most), ("least_confident", least)]:
        if trace_info is None:
            result[label] = None
            continue

        log_ctx = {
            "eval_model": rc.get("eval_model"),
            "dataset": rc.get("dataset"),
            "problem_line": problem_idx,
            "idx": pidx,
            "side": label,
            "trace_idx": trace_info["trace_idx"],
            "correct": trace_info["correct"],
        }

        judge_scores = judge.score(
            problem=question,
            cot=trace_info["cot"],
            gold=str(gt),
            flags_summary="No automated flags available.",
            evidence={"final_correct": trace_info["correct"]},
            log_ctx=log_ctx,
        )

        rs = reasoning_score_from_judge(judge_scores)

        result[label] = {
            "trace_idx": trace_info["trace_idx"],
            "confidence": round(trace_info["confidence"], 6),
            "correct": trace_info["correct"],
            "pred": trace_info["pred"],
            "judge_scores": judge_scores,
            "reasoning_score": round(rs, 4) if rs is not None else None,
        }

    LOG_RUN.info("problem done | idx=%s | %s", pidx, " ".join(f"{k}={v}" for k, v in sorted(rc.items())))
    return result


def evaluate_file(
    input_jsonl: str,
    output_dir: str,
    model_name: str,
    dataset_name: str,
    judge: Judge,
    parallel: int = 10,
    max_samples: int = 0,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    checkpoint_every: int = 1,
    progress_log_every: int = 1,
) -> None:
    out_file = os.path.join(output_dir, f"{model_name}_{dataset_name}_topk_judge.json")

    existing_idxs: set = set()
    existing_results: List = []
    if os.path.exists(out_file):
        with open(out_file, encoding="utf-8") as f:
            data = json.load(f)
            existing_results = data.get("outputs/results", [])
            existing_idxs = {r["idx"] for r in existing_results}
        LOG_RUN.info("resume | loaded %d finished idx from %s", len(existing_idxs), out_file)

    rows: List[dict] = []
    with open(input_jsonl, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    if max_samples > 0:
        rows = rows[:max_samples]

    todo = [(i, row) for i, row in enumerate(rows) if row.get("idx", i) not in existing_idxs]

    LOG_RUN.info(
        "batch | %s x %s | input=%s | todo_problems=%d | already_done=%d | rows=%d | parallel=%d | checkpoint_every=%d | progress_log_every=%d",
        model_name,
        dataset_name,
        input_jsonl,
        len(todo),
        len(existing_idxs),
        len(rows),
        parallel,
        checkpoint_every,
        progress_log_every,
    )

    if not todo:
        LOG_RUN.warning("nothing to do | %s x %s", model_name, dataset_name)
        return

    results = list(existing_results)
    done_count = 0
    start_time = time.time()
    run_ctx = {"eval_model": model_name, "dataset": dataset_name}

    def worker(item: Tuple[int, dict]) -> Optional[dict]:
        i, row = item
        return judge_one_problem(judge, row, i, run_ctx)

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {executor.submit(worker, item): item for item in todo}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                LOG_RUN.exception("worker future failed | %s", e)
                warnings.warn(f"Error: {e}")

            done_count += 1
            if checkpoint_every <= 0:
                should_ckpt = done_count == len(todo)
            else:
                should_ckpt = (
                    done_count % checkpoint_every == 0 or done_count == len(todo)
                )
            if should_ckpt:
                _save_output(
                    out_file, model_name, dataset_name, input_jsonl, results, judge_model
                )
                LOG_RUN.info(
                    "checkpoint | wrote %s | n_results=%d | %s x %s",
                    out_file,
                    len(results),
                    model_name,
                    dataset_name,
                )

            elapsed = time.time() - start_time
            rate = done_count / elapsed if elapsed > 0 else 0
            remaining_n = len(todo) - done_count
            eta_s = remaining_n / rate if rate > 0 else 0.0
            if done_count % max(progress_log_every, 1) == 0 or done_count == len(todo):
                LOG_RUN.info(
                    "progress | %s x %s | done=%d/%d | remaining=%d | rate=%.2f prob/s | elapsed=%.1fs | eta_remaining=%.1fs (~%.1f min) | last_ckpt=%s",
                    model_name,
                    dataset_name,
                    done_count,
                    len(todo),
                    remaining_n,
                    rate,
                    elapsed,
                    eta_s,
                    eta_s / 60.0,
                    should_ckpt,
                )

    elapsed = time.time() - start_time
    LOG_RUN.info(
        "evaluate_file finished | %s x %s | total_results=%d | wall_s=%.1f | out=%s",
        model_name,
        dataset_name,
        len(results),
        elapsed,
        out_file,
    )


def _save_output(
    out_file: str,
    model_name: str,
    dataset_name: str,
    input_jsonl: str,
    results: List,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> None:
    """Write JSON atomically (temp file + replace) so a crash mid-write cannot corrupt the checkpoint."""
    sorted_results = sorted(results, key=lambda r: r.get("idx", 0))
    output = {
        "metadata": {
            "model": model_name,
            "dataset": dataset_name,
            "input_file": input_jsonl,
            "gateway": "portkey",
            "judge_llm": JUDGE_LLM,
            "portkey_model_id": judge_model,
            "judge_model": judge_model,
            "n_results": len(sorted_results),
            "timestamp": datetime.now().isoformat(),
        },
        "outputs/results": sorted_results,
    }
    d = os.path.dirname(out_file) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".topk_judge_", suffix=".json.tmp", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        os.replace(tmp_path, out_file)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_results(output_dir: str) -> None:
    json_files = glob.glob(os.path.join(output_dir, "*_topk_judge.json"))
    if not json_files:
        print("No result files found.")
        return

    print("\n" + "=" * 90)
    print("TOP-K JUDGE EVALUATION RESULTS")
    print("=" * 90)
    print(
        f"\n{'Model':<16} {'Dataset':<14} {'N':>5}  "
        f"{'RS_hi':>6} {'RS_lo':>6} {'Gap':>6}  "
        f"{'Acc_hi':>6} {'Acc_lo':>6}"
    )
    print("-" * 90)

    all_rows = []
    for jf in sorted(json_files):
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        meta = data["metadata"]
        results = data["outputs/results"]

        rs_hi, rs_lo, acc_hi, acc_lo = [], [], [], []
        for r in results:
            mc = r.get("most_confident")
            lc = r.get("least_confident")
            if mc and mc.get("reasoning_score") is not None:
                rs_hi.append(mc["reasoning_score"])
                acc_hi.append(1 if mc["correct"] else 0)
            if lc and lc.get("reasoning_score") is not None:
                rs_lo.append(lc["reasoning_score"])
                acc_lo.append(1 if lc["correct"] else 0)

        if rs_hi and rs_lo:
            mean_hi = np.mean(rs_hi) * 100
            mean_lo = np.mean(rs_lo) * 100
            gap = mean_hi - mean_lo
            a_hi = np.mean(acc_hi) * 100
            a_lo = np.mean(acc_lo) * 100

            print(
                f"{meta['model']:<16} {meta['dataset']:<14} {len(results):>5}  "
                f"{mean_hi:>5.1f}% {mean_lo:>5.1f}% {gap:>+5.1f}  "
                f"{a_hi:>5.1f}% {a_lo:>5.1f}%"
            )

            all_rows.append(
                {
                    "model": meta["model"],
                    "dataset": meta["dataset"],
                    "rs_most_confident": round(mean_hi, 2),
                    "rs_least_confident": round(mean_lo, 2),
                    "rs_gap": round(gap, 2),
                    "acc_most_confident": round(a_hi, 2),
                    "acc_least_confident": round(a_lo, 2),
                }
            )

    if all_rows:
        gaps = [r["rs_gap"] for r in all_rows]
        positive = sum(1 for g in gaps if g > 0)
        print(f"\n{'=' * 90}")
        print(
            f"Reasoning score higher for most-confident trace in "
            f"{positive}/{len(gaps)} model-benchmark pairs"
        )
        print(f"Average gap: {np.mean(gaps):+.2f} pp")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def _configure_logging_from_args(args: argparse.Namespace) -> None:
    level = (
        logging.DEBUG
        if getattr(args, "verbose", False)
        else getattr(logging, args.log_level.upper(), logging.INFO)
    )
    setup_logging(level=level, log_file=getattr(args, "log_file", None) or None)


def main() -> None:
    log_parser = argparse.ArgumentParser(add_help=False)
    log_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose DEBUG logging (all API request/response details)",
    )
    log_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default INFO). Use with --log-file for long runs.",
    )
    log_parser.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help="Append the same logs to this file (in addition to stderr)",
    )
    log_parser.add_argument(
        "--progress-log-every",
        type=int,
        default=1,
        metavar="N",
        help="Log progress/ETA line every N completed problems (default 1 = every completion)",
    )

    parser = argparse.ArgumentParser(description="Top-K Judge Evaluation for FRS")
    sub = parser.add_subparsers(dest="command", required=True)

    single = sub.add_parser(
        "single",
        parents=[log_parser],
        help="Evaluate a single JSONL file",
    )
    single.add_argument("--input-jsonl", required=True)
    single.add_argument("--output-dir", default="outputs/topk_judge_results")
    single.add_argument("--model-name", required=True)
    single.add_argument("--dataset-name", required=True)
    single.add_argument("--parallel", type=int, default=10)
    single.add_argument("--max-samples", type=int, default=0)
    single.add_argument(
        "--portkey-key",
        default=os.environ.get("PORTKEY_API_KEY"),
        help="Portkey API key (or set PORTKEY_API_KEY)",
    )
    single.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"Portkey chat model id (default: {DEFAULT_JUDGE_MODEL} → {JUDGE_LLM})",
    )
    single.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save JSON after this many problems complete (and always after the last). "
        "Default 1 = every problem. 0 = only save at end of file.",
    )

    batch = sub.add_parser(
        "batch",
        parents=[log_parser],
        help="Evaluate all files for given datasets",
    )
    batch.add_argument("--data-root", required=True)
    batch.add_argument("--output-dir", default="outputs/topk_judge_results")
    batch.add_argument("--datasets", nargs="+", default=["GSM8K", "MATH500", "SVAMP"])
    batch.add_argument("--parallel", type=int, default=10)
    batch.add_argument("--max-samples", type=int, default=0)
    batch.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Only run these models (default: all)",
    )
    batch.add_argument(
        "--portkey-key",
        default=os.environ.get("PORTKEY_API_KEY"),
        help="Portkey API key (or set PORTKEY_API_KEY)",
    )
    batch.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"Portkey chat model id (default: routes to {JUDGE_LLM})",
    )
    batch.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save JSON after this many problems complete (and always after the last). "
        "Default 1 = every problem. 0 = only save at end of each file.",
    )

    analyze = sub.add_parser(
        "analyze",
        parents=[log_parser],
        help="Analyze existing results",
    )
    analyze.add_argument("--output-dir", default="outputs/topk_judge_results")

    args = parser.parse_args()

    if args.command == "single":
        _configure_logging_from_args(args)
        LOG.info("command=single | cwd=%s", os.getcwd())
        os.makedirs(args.output_dir, exist_ok=True)
        if not args.portkey_key:
            print("Error: provide --portkey-key or set PORTKEY_API_KEY", file=sys.stderr)
            sys.exit(1)
        judge = Judge(model=args.judge_model, portkey_api_key=args.portkey_key)
        evaluate_file(
            args.input_jsonl,
            args.output_dir,
            args.model_name,
            args.dataset_name,
            judge,
            args.parallel,
            args.max_samples,
            judge_model=args.judge_model,
            checkpoint_every=args.checkpoint_every,
            progress_log_every=args.progress_log_every,
        )

    elif args.command == "batch":
        _configure_logging_from_args(args)
        LOG.info("command=batch | cwd=%s", os.getcwd())
        os.makedirs(args.output_dir, exist_ok=True)
        if not args.portkey_key:
            print("Error: provide --portkey-key or set PORTKEY_API_KEY", file=sys.stderr)
            sys.exit(1)
        judge = Judge(model=args.judge_model, portkey_api_key=args.portkey_key)
        file_map = discover_files(args.data_root, args.datasets)
        print(f"Found {len(file_map)} (model, dataset) pairs for {args.datasets}")
        for (model, dataset), filepath in sorted(file_map.items()):
            if args.models and model not in args.models:
                continue
            print(f"\n{'=' * 60}")
            print(f"  {model} x {dataset}")
            print(f"{'=' * 60}")
            evaluate_file(
                filepath,
                args.output_dir,
                model,
                dataset,
                judge,
                args.parallel,
                args.max_samples,
                judge_model=args.judge_model,
                checkpoint_every=args.checkpoint_every,
                progress_log_every=args.progress_log_every,
            )
        analyze_results(args.output_dir)

    elif args.command == "analyze":
        _configure_logging_from_args(args)
        analyze_results(args.output_dir)


if __name__ == "__main__":
    main()
