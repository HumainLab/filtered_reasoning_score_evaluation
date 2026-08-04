#!/usr/bin/env python3
"""
Fewshot CoT Analysis Runner
============================
Runs pillars evaluation on the 4 remaining Qwen3-4B-Thinking fewshot jobs.

Features:
  - Per-sample terminal logging (every sample visible)
  - Checkpoint/resume after every sample
  - 5 jobs concurrently, 5 samples at a time
  - Saves final results to exports directory
  - --test mode to verify on 3 samples first

Usage:
  # Test on 3 samples per job first
  python run_fewshot_analysis.py --test

  # Full run (resumes from checkpoints)
  python run_fewshot_analysis.py

  # Fresh start (ignores checkpoints)
  python run_fewshot_analysis.py --no-resume

  # Custom concurrency
  python run_fewshot_analysis.py --job-concurrency 3 --sample-concurrency 8
"""

import asyncio
import json
import sys
import time
import argparse
import os
import traceback
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "app"))
sys.path.insert(0, str(ROOT / "evaluation"))

CHECKPOINT_DIR = ROOT / "cot_analysis_checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
EXPORTS_DIR = ROOT / "evaluation" / "exports" / "cot_analysis"

# ── Jobs to Process ──────────────────────────────────────────────────────────
FEWSHOT_JOBS = [
    ("12d51b75-a9eb-470a-9413-434d99711053", "commonsense_qa", "commonsense_qa_fewshot"),
    ("97eb9344-36c7-4ffd-9b86-53e80c697632", "gsm8k",          "gsm8k_fewshot"),
    ("360ef525-488c-499c-a4c8-2c3be65ad169", "math500",        "math500_fewshot"),
    ("c98e3f4b-eb76-46e9-8321-0e16104e6e9a", "svamp",          "svamp_fewshot"),
]


# ── Logging ──────────────────────────────────────────────────────────────────
class Logger:
    """Simple logger that writes to both terminal and file."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.fh = open(log_path, "a", buffering=1)  # line-buffered
        self._lock = asyncio.Lock()

    def _ts(self):
        return datetime.now().strftime("%H:%M:%S")

    async def log(self, msg: str, level: str = "INFO"):
        line = f"{self._ts()} | {level:<5} | {msg}"
        async with self._lock:
            print(line, flush=True)
            self.fh.write(line + "\n")
            self.fh.flush()

    async def info(self, msg):  await self.log(msg, "INFO")
    async def debug(self, msg): await self.log(msg, "DEBUG")
    async def error(self, msg): await self.log(msg, "ERROR")
    async def warn(self, msg):  await self.log(msg, "WARN")

    # Sync versions for non-async contexts
    def info_sync(self, msg):  self._write_sync(msg, "INFO")
    def error_sync(self, msg): self._write_sync(msg, "ERROR")

    def _write_sync(self, msg, level):
        line = f"{self._ts()} | {level:<5} | {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


# ── Checkpoint ───────────────────────────────────────────────────────────────
@dataclass
class Checkpoint:
    job_id: str
    model: str
    dataset: str
    prompt_type: str
    total_samples: int
    processed_samples: int
    results: List[Dict]
    status: str  # pending | running | done | error
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    @property
    def path(self) -> Path:
        return CHECKPOINT_DIR / f"{self.job_id}_checkpoint.json"

    def save(self):
        with open(self.path, "w") as f:
            json.dump(asdict(self), f)

    @classmethod
    def load(cls, job_id: str) -> Optional["Checkpoint"]:
        p = CHECKPOINT_DIR / f"{job_id}_checkpoint.json"
        if not p.exists():
            return None
        with open(p) as f:
            return cls(**json.load(f))


# ── Analyzer ─────────────────────────────────────────────────────────────────
class FewshotAnalyzer:
    def __init__(self, api_key: str, sample_concurrency: int, job_concurrency: int, logger: Logger):
        self.api_key = api_key
        self.sample_sem = asyncio.Semaphore(sample_concurrency * job_concurrency)
        self.job_sem = asyncio.Semaphore(job_concurrency)
        self.log = logger
        self.max_samples: Optional[int] = None

        os.environ["OPENAI_API_KEY"] = api_key
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key)

        from app.cot_eval_v2.evaluator import PillarsEvaluator
        from app.cot_eval_v2.scoring import rule_scores, fuse_with_judge
        self.PillarsEvaluator = PillarsEvaluator
        self.rule_scores_fn = rule_scores
        self.fuse_with_judge = fuse_with_judge

    # ── Judge call ───────────────────────────────────────────────────────
    async def _call_judge(self, problem: str, cot: str, gold: str,
                          flags_summary: str, evidence: Dict, tag: str) -> Dict:
        prompt = (
            "You are an expert evaluator of mathematical and logical reasoning.\n"
            "Score the chain-of-thought (CoT) on 4 dimensions. Each score must be an integer from 1-5.\n\n"
            f"## Problem\n{problem}\n\n"
            f"## Model Reasoning (CoT)\n{cot[:3000]}\n\n"
            f"## Gold Answer\n{gold}\n\n"
            f"## Automated Flag Analysis\n{flags_summary}\n\n"
            "## Instructions\n"
            "Score each dimension 1-5 based on:\n"
            "- faithfulness: Internal consistency, no contradictions\n"
            "- utility: Each step contributes, calculations correct\n"
            "- coherence: Smooth flow between steps\n"
            "- factuality: Facts grounded in problem, no hallucinations\n\n"
            'Output ONLY a JSON object:\n'
            '{"faithfulness": <1-5>, "utility": <1-5>, "coherence": <1-5>, "factuality": <1-5>}'
        )

        t0 = time.time()
        try:
            resp = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a careful and consistent evaluator of reasoning quality."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            raw = resp.choices[0].message.content
            elapsed = time.time() - t0
            usage = resp.usage
            tok_info = f" | {usage.prompt_tokens}+{usage.completion_tokens} tok" if usage else ""
            await self.log.debug(f"{tag} Judge API returned in {elapsed:.2f}s{tok_info}")
            return self._parse_judge(raw)
        except Exception as e:
            await self.log.error(f"{tag} Judge API error: {e}")
            return {"faithfulness": None, "utility": None, "coherence": None, "factuality": None}

    def _parse_judge(self, raw: str) -> Dict:
        import re
        for attempt in [raw, (re.search(r"\{[^{}]*\}", raw) or type("_", (), {"group": lambda s: "{}"})()).group()]:
            try:
                parsed = json.loads(attempt)
                return {k: min(5, max(1, int(parsed[k]))) for k in ["faithfulness", "utility", "coherence", "factuality"]}
            except Exception:
                continue
        return {"faithfulness": None, "utility": None, "coherence": None, "factuality": None}

    def _flags_summary(self, flags) -> str:
        parts = []
        for pillar in ["faithfulness", "utility", "coherence", "factuality"]:
            pf = flags.get_flags_by_pillar(pillar)
            if pf:
                parts.append(f"{pillar}: {', '.join(f.issue for f in pf)}")
        return "\n".join(parts) or "No issues detected."

    def _flags_to_dict(self, flags) -> Dict:
        return {p: [f.to_dict() for f in flags.get_flags_by_pillar(p)]
                for p in ["faithfulness", "utility", "coherence", "factuality"]}

    # ── Single sample ────────────────────────────────────────────────────
    async def _analyze_sample(self, sample: Dict, idx: int, tag: str) -> Dict:
        async with self.sample_sem:  # limit total concurrent API calls
            t0 = time.time()
            try:
                # Extract data
                code = sample.get("code", sample.get("output", ""))
                model_output = code[0] if isinstance(code, list) and code else (code if isinstance(code, str) else "")
                question = sample.get("question", "")
                gt = sample.get("gt", "")

                # Parse reasoning
                if "####" in model_output:
                    parts = model_output.split("####")
                    reasoning = parts[0].strip()
                    predicted = parts[1].strip() if len(parts) > 1 else ""
                else:
                    reasoning = model_output.strip()
                    predicted = ""
                full_cot = f"{reasoning}\n#### {predicted}" if predicted else reasoning

                # Rule-based analysis
                evaluator = self.PillarsEvaluator(judge=None)
                flags, evidence, rule_scores_dict, _, _ = evaluator.analyze(
                    problem=question, cot_text=full_cot, gold=gt
                )

                is_correct = sample.get("score", [False])
                if isinstance(is_correct, list):
                    is_correct = is_correct[0] if is_correct else False
                evidence["final_correct"] = bool(is_correct)

                # Judge call
                judge_scores = await self._call_judge(
                    question, full_cot, gt, self._flags_summary(flags), evidence, tag
                )

                # Fuse
                fused = self.fuse_with_judge(rule_scores_dict, judge_scores, evidence)
                overall = sum(fused.get(k, 0) for k in ["faithfulness", "utility", "coherence", "factuality"]) / 4

                elapsed = time.time() - t0
                correct_str = "✓" if is_correct else "✗"
                await self.log.info(
                    f"{tag} {correct_str} F:{fused.get('faithfulness',0):.2f} "
                    f"U:{fused.get('utility',0):.2f} C:{fused.get('coherence',0):.2f} "
                    f"Fa:{fused.get('factuality',0):.2f} | OVR:{overall:.2f} | {elapsed:.1f}s"
                )

                return {
                    "idx": sample.get("idx", idx),
                    "question": question, "ground_truth": gt,
                    "model_output": model_output, "cot_text": full_cot,
                    "faithfulness": fused.get("faithfulness", 0.0),
                    "utility": fused.get("utility", 0.0),
                    "coherence": fused.get("coherence", 0.0),
                    "factuality": fused.get("factuality", 0.0),
                    "overall": overall,
                    "flags": self._flags_to_dict(flags),
                    "judge_scores": judge_scores,
                    "rule_scores": rule_scores_dict,
                    "evidence": evidence,
                }
            except Exception as e:
                elapsed = time.time() - t0
                await self.log.error(f"{tag} FAILED ({elapsed:.1f}s): {e}")
                return {
                    "idx": sample.get("idx", idx),
                    "question": sample.get("question", ""),
                    "ground_truth": sample.get("gt", ""),
                    "model_output": "", "cot_text": "",
                    "faithfulness": 0, "utility": 0, "coherence": 0, "factuality": 0,
                    "overall": 0, "flags": {}, "judge_scores": None,
                    "rule_scores": None, "evidence": None, "error": str(e),
                }

    # ── Single job ───────────────────────────────────────────────────────
    async def _run_job(self, job_id: str, dataset: str, prompt_type: str,
                       job_db: Dict, resume: bool, job_num: int, total_jobs: int):
        async with self.job_sem:
            model = "Qwen3-4B-Thinking-2507"
            prefix = f"[{job_num}/{total_jobs} {dataset}]"

            await self.log.info(f"")
            await self.log.info(f"{'═'*65}")
            await self.log.info(f"{prefix} JOB START: {job_id}")
            await self.log.info(f"{prefix} Model: {model} | Dataset: {dataset} | Prompt: {prompt_type}")
            await self.log.info(f"{'═'*65}")

            # Resume?
            cp = Checkpoint.load(job_id) if resume else None
            if cp and cp.status == "done":
                await self.log.info(f"{prefix} Already done, skipping")
                return cp

            # Load data
            from app.runner import get_job_raw_data
            try:
                raw = get_job_raw_data(job_id).get("data", [])
            except Exception as e:
                await self.log.error(f"{prefix} Failed to load data: {e}")
                return None
            if not raw:
                await self.log.error(f"{prefix} No data found")
                return None

            total = len(raw) if not self.max_samples else min(len(raw), self.max_samples)
            raw = raw[:total]

            # Determine start index
            start_idx = 0
            if cp and cp.status == "running" and cp.processed_samples > 0:
                start_idx = cp.processed_samples
                await self.log.info(f"{prefix} Resuming from sample {start_idx}/{total}")
            else:
                cp = Checkpoint(
                    job_id=job_id, model=model, dataset=dataset,
                    prompt_type=prompt_type, total_samples=total,
                    processed_samples=0, results=[] if not (cp and cp.results) else cp.results,
                    status="running", started_at=time.time(),
                )
                await self.log.info(f"{prefix} Starting fresh, {total} samples")

            cp.status = "running"
            cp.save()

            BATCH_SIZE = 20  # Process 20 samples concurrently
            t0 = time.time()
            i = start_idx
            while i < total:
                batch_end = min(i + BATCH_SIZE, total)
                # Create tasks for the entire batch
                tasks = []
                for j in range(i, batch_end):
                    tag = f"{prefix}[{j+1}/{total}]"
                    tasks.append(self._analyze_sample(raw[j], j, tag))

                # Run batch concurrently
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results and save checkpoint after each batch
                for j, result in enumerate(batch_results):
                    idx = i + j
                    if isinstance(result, Exception):
                        await self.log.error(f"{prefix}[{idx+1}/{total}] Batch exception: {result}")
                        result = {
                            "idx": idx, "question": "", "ground_truth": "",
                            "model_output": "", "cot_text": "",
                            "faithfulness": 0, "utility": 0, "coherence": 0, "factuality": 0,
                            "overall": 0, "flags": {}, "judge_scores": None,
                            "rule_scores": None, "evidence": None, "error": str(result),
                        }
                    cp.results.append(result)

                cp.processed_samples = batch_end
                cp.save()  # save after every batch

                # Progress summary after each batch
                elapsed = time.time() - t0
                done = batch_end - start_idx
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - batch_end) / rate if rate > 0 else 0
                avg_ovr = sum(r["overall"] for r in cp.results) / len(cp.results)
                await self.log.info(
                    f"{prefix} ── BATCH DONE {batch_end}/{total} ({batch_end/total*100:.0f}%) "
                    f"| {elapsed:.0f}s elapsed | ETA {eta:.0f}s | avg_overall={avg_ovr:.3f} ──"
                )

                i = batch_end

            cp.status = "done"
            cp.completed_at = time.time()
            cp.save()

            # Save final JSON
            self._save_results(cp)
            dur = cp.completed_at - (cp.started_at or cp.completed_at)
            avg_ovr = sum(r["overall"] for r in cp.results) / len(cp.results) if cp.results else 0
            await self.log.info(f"{prefix} ✅ DONE in {dur:.0f}s | avg_overall={avg_ovr:.4f}")
            return cp

    def _save_results(self, cp: Checkpoint):
        out_dir = EXPORTS_DIR / cp.model / cp.dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = out_dir / f"cot_analysis_{cp.job_id}_{ts}.json"

        total = len(cp.results)
        summary = {
            "total_samples": total,
            "avg_faithfulness": sum(r["faithfulness"] for r in cp.results) / total if total else 0,
            "avg_utility":     sum(r["utility"] for r in cp.results) / total if total else 0,
            "avg_coherence":   sum(r["coherence"] for r in cp.results) / total if total else 0,
            "avg_factuality":  sum(r["factuality"] for r in cp.results) / total if total else 0,
            "avg_overall":     sum(r["overall"] for r in cp.results) / total if total else 0,
            "analysis_time":   (cp.completed_at or time.time()) - (cp.started_at or time.time()),
        }

        output = {
            "job_id": cp.job_id, "model": cp.model,
            "dataset": cp.dataset, "prompt_type": cp.prompt_type,
            "timestamp": datetime.now().isoformat(),
            "summary": summary, "per_sample": cp.results,
        }
        with open(out_file, "w") as f:
            json.dump(output, f, indent=2)
        self.log.info_sync(f"💾 Saved: {out_file}")

    # ── Run all ──────────────────────────────────────────────────────────
    async def run_all(self, job_db: Dict, resume: bool):
        total = len(FEWSHOT_JOBS)
        await self.log.info(f"")
        await self.log.info(f"{'#'*65}")
        await self.log.info(f"  PARALLEL FEWSHOT COT ANALYSIS")
        await self.log.info(f"  Jobs: {total} | Sample concurrency: {self.sample_sem._value}")
        await self.log.info(f"  Job concurrency: {self.job_sem._value} | Resume: {resume}")
        await self.log.info(f"{'#'*65}")
        await self.log.info(f"")

        t0 = time.time()
        tasks = [
            self._run_job(jid, ds, pt, job_db, resume, i, total)
            for i, (jid, ds, pt) in enumerate(FEWSHOT_JOBS, 1)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.time() - t0
        await self.log.info(f"")
        await self.log.info(f"{'#'*65}")
        await self.log.info(f"  FINAL SUMMARY  ({elapsed:.0f}s / {elapsed/60:.1f} min)")
        await self.log.info(f"{'#'*65}")
        for r in results:
            if isinstance(r, Exception):
                await self.log.error(f"  Exception: {r}")
            elif r is None:
                await self.log.error(f"  Job returned None")
            else:
                icon = "✅" if r.status == "done" else "❌"
                avg = sum(x["overall"] for x in r.results) / len(r.results) if r.results else 0
                await self.log.info(f"  {icon} {r.dataset:<18} {r.processed_samples}/{r.total_samples} samples | avg_overall={avg:.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fewshot CoT Analysis Runner")
    parser.add_argument("--test", action="store_true", help="Test mode: 3 samples per job")
    parser.add_argument("--no-resume", action="store_true", help="Ignore checkpoints, start fresh")
    parser.add_argument("--sample-concurrency", "-s", type=int, default=5, help="Concurrent API calls per job (default: 5)")
    parser.add_argument("--job-concurrency", "-j", type=int, default=5, help="Concurrent jobs (default: 5)")
    args = parser.parse_args()

    # Load config
    config_path = ROOT / "backend" / "path_config.json"
    with open(config_path) as f:
        config = json.load(f)
    api_key = config.get("openai_api_key", "")
    if not api_key:
        print("ERROR: No OpenAI API key in backend/path_config.json")
        sys.exit(1)

    job_db_path = ROOT / "backend" / "job_db.json"
    with open(job_db_path) as f:
        job_db = json.load(f)

    # Logger
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "cot_analysis_logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"fewshot_analysis_{ts}.log"
    logger = Logger(log_path)
    logger.info_sync(f"Log file: {log_path}")

    # Analyzer
    analyzer = FewshotAnalyzer(api_key, args.sample_concurrency, args.job_concurrency, logger)

    if args.test:
        analyzer.max_samples = 3
        logger.info_sync("TEST MODE: 3 samples per job")

    resume = not args.no_resume
    asyncio.run(analyzer.run_all(job_db, resume))
    logger.close()


if __name__ == "__main__":
    main()
