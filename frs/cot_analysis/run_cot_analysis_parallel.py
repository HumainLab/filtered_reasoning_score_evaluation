#!/usr/bin/env python3
"""
Parallel CoT Analysis Script with Checkpoint/Resume Support

Features:
- Runs multiple jobs in parallel with configurable concurrency
- Saves progress after EACH API call for reliable resume
- Uses GPT-4o-mini at temperature 0
- Can resume from checkpoint if interrupted

Usage:
    python run_cot_analysis_parallel.py [--concurrency 5] [--resume] [--job-ids JOB1 JOB2]
"""

import asyncio
import json
import sys
import time
import argparse
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import traceback

# Add paths for imports
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "frs"))
sys.path.insert(0, str(_REPO / "evaluation"))

# Configuration
CHECKPOINT_DIR = Path(__file__).parent / "cot_analysis_checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR = Path(__file__).parent / "cot_analysis_logs"
LOG_DIR.mkdir(exist_ok=True)

# Setup logging
def setup_logging(verbose: bool = False):
    """Setup logging with file and console handlers"""
    log_file = LOG_DIR / f"cot_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create formatter
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler - always detailed
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler - depends on verbose flag
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(detailed_formatter)
    
    # Setup logger
    logger = logging.getLogger('cot_analysis')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"📝 Log file: {log_file}")
    return logger

logger = logging.getLogger('cot_analysis')

@dataclass
class SampleResult:
    """Result for a single sample"""
    idx: int
    question: str
    ground_truth: str
    model_output: str
    cot_text: str
    faithfulness: float
    utility: float
    coherence: float
    factuality: float
    overall: float
    flags: Dict
    judge_scores: Optional[Dict]
    rule_scores: Optional[Dict]
    evidence: Optional[Dict]
    error: Optional[str] = None

@dataclass 
class JobCheckpoint:
    """Checkpoint for a single job"""
    job_id: str
    model: str
    dataset: str
    prompt_type: str
    total_samples: int
    processed_samples: int
    results: List[Dict]
    status: str  # pending, running, done, error
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    
    @property
    def checkpoint_path(self) -> Path:
        return CHECKPOINT_DIR / f"{self.job_id}_checkpoint.json"
    
    def save(self):
        """Save checkpoint to disk"""
        with open(self.checkpoint_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, job_id: str) -> Optional['JobCheckpoint']:
        """Load checkpoint from disk if exists"""
        checkpoint_path = CHECKPOINT_DIR / f"{job_id}_checkpoint.json"
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
                return cls(**data)
        return None


class ParallelCotAnalyzer:
    """Parallel CoT analyzer with checkpoint support"""
    
    def __init__(self, api_key: str, concurrency: int = 5, job_concurrency: int = 1, judge_mode: str = "ALWAYS"):
        self.api_key = api_key
        self.concurrency = concurrency  # API calls per job
        self.job_concurrency = job_concurrency  # Jobs in parallel
        self.judge_mode = judge_mode
        self.semaphore = asyncio.Semaphore(concurrency * job_concurrency)  # Total concurrent API calls
        self.job_semaphore = asyncio.Semaphore(job_concurrency)  # Jobs running at once
        self.max_samples = None  # For testing
        
        # Initialize OpenAI client
        os.environ['OPENAI_API_KEY'] = api_key
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key)
        
        # Import evaluator components
        from cot_eval_v2.evaluator import PillarsEvaluator
        from cot_eval_v2.judge import Judge
        from cot_eval_v2.scoring import rule_scores, fuse_with_judge
        
        self.PillarsEvaluator = PillarsEvaluator
        self.Judge = Judge
        self.rule_scores = rule_scores
        self.fuse_with_judge = fuse_with_judge
        
    async def call_judge_async(self, problem: str, cot: str, gold: str, 
                                flags_summary: str, evidence: Dict, sample_idx: int = 0) -> Dict[str, Optional[int]]:
        """Async wrapper for GPT-4o-mini judge call"""
        prompt = self._build_judge_prompt(problem, cot, gold, flags_summary, evidence)
        
        start_time = time.time()
        logger.debug(f"[Sample {sample_idx}] 🔄 Calling GPT-4o-mini judge...")
        logger.debug(f"[Sample {sample_idx}] Problem (first 100 chars): {problem[:100]}...")
        logger.debug(f"[Sample {sample_idx}] CoT length: {len(cot)} chars")
        logger.debug(f"[Sample {sample_idx}] Gold answer: {gold}")
        logger.debug(f"[Sample {sample_idx}] Flags: {flags_summary[:200] if flags_summary else 'None'}")
        
        try:
            resp = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a careful and consistent evaluator of reasoning quality."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=500
            )
            
            elapsed = time.time() - start_time
            raw_output = resp.choices[0].message.content
            
            # Log token usage if available
            usage = resp.usage
            if usage:
                logger.debug(f"[Sample {sample_idx}] ✅ API response in {elapsed:.2f}s | Tokens: {usage.prompt_tokens} prompt, {usage.completion_tokens} completion")
            else:
                logger.debug(f"[Sample {sample_idx}] ✅ API response in {elapsed:.2f}s")
            
            logger.debug(f"[Sample {sample_idx}] Raw judge output: {raw_output}")
            
            scores = self._parse_judge_response(raw_output)
            logger.debug(f"[Sample {sample_idx}] Parsed scores: {scores}")
            
            return scores
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Sample {sample_idx}] ❌ Judge API error after {elapsed:.2f}s: {e}")
            return {"faithfulness": None, "utility": None, "coherence": None, "factuality": None}
    
    def _build_judge_prompt(self, problem: str, cot: str, gold: str, 
                            flags_summary: str, evidence: Dict) -> str:
        """Build prompt for LLM judge"""
        return f"""You are an expert evaluator of mathematical and logical reasoning.
Score the chain-of-thought (CoT) on 4 dimensions. Each score must be an integer from 1-5.

## Problem
{problem}

## Model Reasoning (CoT)
{cot[:3000]}

## Gold Answer
{gold}

## Automated Flag Analysis
{flags_summary}

## Instructions
Score each dimension 1-5 based on:
- faithfulness: Internal consistency, no contradictions
- utility: Each step contributes, calculations correct  
- coherence: Smooth flow between steps
- factuality: Facts grounded in problem, no hallucinations

Output ONLY a JSON object:
{{"faithfulness": <1-5>, "utility": <1-5>, "coherence": <1-5>, "factuality": <1-5>}}"""

    def _parse_judge_response(self, raw_output: str) -> Dict[str, Optional[int]]:
        """Parse judge response to extract scores"""
        import re
        try:
            # Try direct JSON parse
            parsed = json.loads(raw_output)
            return self._validate_scores(parsed)
        except:
            pass
        
        # Try regex extraction
        try:
            json_match = re.search(r'\{[^{}]*\}', raw_output)
            if json_match:
                parsed = json.loads(json_match.group())
                return self._validate_scores(parsed)
        except:
            pass
        
        return {"faithfulness": None, "utility": None, "coherence": None, "factuality": None}
    
    def _validate_scores(self, parsed: Dict) -> Dict[str, Optional[int]]:
        """Validate and normalize scores"""
        keys = ["faithfulness", "utility", "coherence", "factuality"]
        result = {}
        for k in keys:
            v = parsed.get(k)
            try:
                v = int(v)
                v = min(5, max(1, v))
            except:
                v = None
            result[k] = v
        return result
    
    async def analyze_sample(self, sample: Dict, checkpoint: JobCheckpoint, 
                             sample_idx: int) -> SampleResult:
        """Analyze a single sample with async judge call"""
        async with self.semaphore:
            sample_start = time.time()
            logger.debug(f"[Sample {sample_idx}] ▶️  Starting analysis...")
            
            try:
                # Extract data - use 'code' field for actual model output (not 'answer' which is gold)
                code_field = sample.get("code", sample.get("output", ""))
                # 'code' is typically a list, get first element
                if isinstance(code_field, list):
                    model_output = code_field[0] if code_field else ""
                else:
                    model_output = code_field
                question = sample.get("question", "")
                gt = sample.get("gt", "")
                
                logger.debug(f"[Sample {sample_idx}] Question length: {len(question)} chars")
                logger.debug(f"[Sample {sample_idx}] Model output length: {len(model_output)} chars")
                
                # Parse model output
                if "####" in model_output:
                    parts = model_output.split("####")
                    reasoning = parts[0].strip()
                    predicted = parts[1].strip() if len(parts) > 1 else ""
                else:
                    reasoning = model_output.strip()
                    predicted = ""
                
                full_cot = f"{reasoning}\n#### {predicted}" if predicted else reasoning
                
                # Create evaluator for rule-based analysis (sync)
                logger.debug(f"[Sample {sample_idx}] Running rule-based analysis...")
                rule_start = time.time()
                evaluator = self.PillarsEvaluator(judge=None)
                flags, evidence, rule_scores_dict, _, _ = evaluator.analyze(
                    problem=question,
                    cot_text=full_cot,
                    gold=gt
                )
                logger.debug(f"[Sample {sample_idx}] Rule-based analysis done in {time.time() - rule_start:.2f}s")
                logger.debug(f"[Sample {sample_idx}] Rule scores: {rule_scores_dict}")
                
                # Get correctness from sample
                is_correct = sample.get('score', [False])
                if isinstance(is_correct, list):
                    is_correct = is_correct[0] if is_correct else False
                evidence['final_correct'] = bool(is_correct)
                logger.debug(f"[Sample {sample_idx}] Correct: {is_correct}")
                
                # Build flags summary
                flags_summary = self._build_flags_summary(flags)
                
                # Call judge async
                judge_scores = await self.call_judge_async(
                    question, full_cot, gt, flags_summary, evidence, sample_idx
                )
                
                # Fuse scores
                fused = self.fuse_with_judge(rule_scores_dict, judge_scores, evidence)
                
                overall = (fused.get("faithfulness", 0.0) + fused.get("utility", 0.0) + 
                          fused.get("coherence", 0.0) + fused.get("factuality", 0.0)) / 4.0
                
                elapsed = time.time() - sample_start
                logger.info(f"[Sample {sample_idx}] ✅ Done in {elapsed:.2f}s | "
                           f"F:{fused.get('faithfulness', 0):.2f} U:{fused.get('utility', 0):.2f} "
                           f"C:{fused.get('coherence', 0):.2f} Fa:{fused.get('factuality', 0):.2f} "
                           f"| Overall: {overall:.2f} | Correct: {is_correct}")
                
                return SampleResult(
                    idx=sample.get("idx", sample_idx),
                    question=question,
                    ground_truth=gt,
                    model_output=model_output,
                    cot_text=full_cot,
                    faithfulness=fused.get("faithfulness", 0.0),
                    utility=fused.get("utility", 0.0),
                    coherence=fused.get("coherence", 0.0),
                    factuality=fused.get("factuality", 0.0),
                    overall=overall,
                    flags=self._flags_to_dict(flags),
                    judge_scores=judge_scores,
                    rule_scores=rule_scores_dict,
                    evidence=evidence
                )
                
            except Exception as e:
                elapsed = time.time() - sample_start
                logger.error(f"[Sample {sample_idx}] ❌ Error after {elapsed:.2f}s: {e}")
                logger.debug(f"[Sample {sample_idx}] Traceback: {traceback.format_exc()}")
                # Get model_output safely in case error happened before it was defined
                code_field = sample.get("code", sample.get("output", ""))
                error_model_output = code_field[0] if isinstance(code_field, list) and code_field else (code_field if isinstance(code_field, str) else "")
                return SampleResult(
                    idx=sample.get("idx", sample_idx),
                    question=sample.get("question", ""),
                    ground_truth=sample.get("gt", ""),
                    model_output=error_model_output,
                    cot_text="",
                    faithfulness=0.0, utility=0.0, coherence=0.0, factuality=0.0,
                    overall=0.0, flags={}, judge_scores=None, rule_scores=None,
                    evidence=None, error=str(e)
                )
    
    def _build_flags_summary(self, flags) -> str:
        """Build human-readable flags summary"""
        summaries = []
        for pillar in ["faithfulness", "utility", "coherence", "factuality"]:
            pillar_flags = flags.get_flags_by_pillar(pillar)
            if pillar_flags:
                # Flag has: pillar, step, issue, details
                flag_issues = [f.issue for f in pillar_flags]
                summaries.append(f"{pillar}: {', '.join(flag_issues)}")
        return "\n".join(summaries) if summaries else "No issues detected."
    
    def _flags_to_dict(self, flags) -> Dict:
        """Convert flags to dictionary format"""
        result = {}
        for pillar in ["faithfulness", "utility", "coherence", "factuality"]:
            pillar_flags = flags.get_flags_by_pillar(pillar)
            result[pillar] = [f.to_dict() for f in pillar_flags]
        return result
    
    async def process_job(self, job_id: str, job_data: Dict, resume: bool = True) -> JobCheckpoint:
        """Process a single job with checkpoint support"""
        req = job_data.get('request', {})
        model = req.get('model', 'unknown').split('/')[-1]
        dataset = req.get('dataset', 'unknown')
        prompt_type = req.get('prompt_type', 'unknown')
        
        logger.info(f"{'='*60}")
        logger.info(f"JOB: {job_id}")
        logger.info(f"Model: {model} | Dataset: {dataset} | Prompt: {prompt_type}")
        logger.info(f"{'='*60}")
        
        # Check for existing checkpoint
        checkpoint = None
        if resume:
            checkpoint = JobCheckpoint.load(job_id)
            if checkpoint and checkpoint.status == 'done':
                logger.info(f"✅ Job already completed, skipping")
                return checkpoint
        
        # Load raw data
        logger.info(f"Loading raw data...")
        raw_data = self._load_job_data(job_id, job_data)
        if not raw_data:
            logger.error(f"❌ No data found for job")
            return None
        
        # Limit samples for testing
        if self.max_samples:
            logger.info(f"Limiting to {self.max_samples} samples (testing mode)")
            raw_data = raw_data[:self.max_samples]
        
        total_samples = len(raw_data)
        logger.info(f"Loaded {total_samples} samples")
        
        # Create or resume checkpoint
        if checkpoint and checkpoint.status == 'running':
            logger.info(f"🔄 Resuming from sample {checkpoint.processed_samples}/{total_samples}")
            start_idx = checkpoint.processed_samples
        else:
            checkpoint = JobCheckpoint(
                job_id=job_id,
                model=model,
                dataset=dataset,
                prompt_type=prompt_type,
                total_samples=total_samples,
                processed_samples=0,
                results=[],
                status='running',
                started_at=time.time()
            )
            start_idx = 0
            logger.info(f"Starting fresh from sample 0")
        
        checkpoint.status = 'running'
        checkpoint.save()
        
        logger.info(f"📊 Processing {total_samples - start_idx} remaining samples...")
        
        # Process samples
        job_start_time = time.time()
        for i in range(start_idx, total_samples):
            sample = raw_data[i]
            result = await self.analyze_sample(sample, checkpoint, i)
            
            checkpoint.results.append(asdict(result))
            checkpoint.processed_samples = i + 1
            
            # Save checkpoint after EACH sample
            checkpoint.save()
            
            # Progress update every 10 samples
            if (i + 1) % 10 == 0 or i == total_samples - 1:
                pct = (i + 1) / total_samples * 100
                elapsed = time.time() - job_start_time
                samples_done = i + 1 - start_idx
                if samples_done > 0:
                    avg_time = elapsed / samples_done
                    remaining = (total_samples - i - 1) * avg_time
                    logger.info(f"📈 Progress: {i+1}/{total_samples} ({pct:.1f}%) | "
                               f"Elapsed: {elapsed:.0f}s | ETA: {remaining:.0f}s")
        
        checkpoint.status = 'done'
        checkpoint.completed_at = time.time()
        checkpoint.save()
        
        total_time = checkpoint.completed_at - checkpoint.started_at
        logger.info(f"✅ Job completed in {total_time:.1f}s ({total_time/60:.1f} min)")
        
        # Calculate and log summary stats
        results = checkpoint.results
        avg_overall = sum(r.get('overall', 0) for r in results) / len(results) if results else 0
        avg_faith = sum(r.get('faithfulness', 0) for r in results) / len(results) if results else 0
        avg_util = sum(r.get('utility', 0) for r in results) / len(results) if results else 0
        avg_coh = sum(r.get('coherence', 0) for r in results) / len(results) if results else 0
        avg_fact = sum(r.get('factuality', 0) for r in results) / len(results) if results else 0
        
        logger.info(f"📊 Summary: Overall={avg_overall:.3f} | F={avg_faith:.3f} U={avg_util:.3f} C={avg_coh:.3f} Fa={avg_fact:.3f}")
        
        # Save final results
        self._save_final_results(checkpoint)
        
        return checkpoint
    
    def _load_job_data(self, job_id: str, job_data: Dict) -> List[Dict]:
        """Load raw sample data for a job"""
        from app.runner import get_job_raw_data
        try:
            raw = get_job_raw_data(job_id)
            return raw.get('data', [])
        except Exception as e:
            print(f"    Error loading data: {e}")
            return []
    
    def _save_final_results(self, checkpoint: JobCheckpoint):
        """Save final analysis results to exports directory"""
        exports_dir = _REPO / "evaluation" / "exports" / "cot_analysis"
        output_dir = exports_dir / checkpoint.model / checkpoint.dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"cot_analysis_{checkpoint.job_id}_{timestamp}.json"
        
        # Calculate summary stats
        results = checkpoint.results
        total = len(results)
        
        summary = {
            "total_samples": total,
            "avg_faithfulness": sum(r["faithfulness"] for r in results) / total if total else 0,
            "avg_utility": sum(r["utility"] for r in results) / total if total else 0,
            "avg_coherence": sum(r["coherence"] for r in results) / total if total else 0,
            "avg_factuality": sum(r["factuality"] for r in results) / total if total else 0,
            "avg_overall": sum(r["overall"] for r in results) / total if total else 0,
            "analysis_time": (checkpoint.completed_at or time.time()) - (checkpoint.started_at or time.time())
        }
        
        output = {
            "job_id": checkpoint.job_id,
            "model": checkpoint.model,
            "dataset": checkpoint.dataset,
            "prompt_type": checkpoint.prompt_type,
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "per_sample": results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"    💾 Saved results to {output_file}")
    
    async def process_job_with_semaphore(self, job_id: str, job_data: Dict, resume: bool, job_num: int, total_jobs: int):
        """Wrapper to process job with semaphore for concurrency control"""
        async with self.job_semaphore:
            logger.info(f"")
            logger.info(f"[{job_num}/{total_jobs}] Starting job: {job_id[:8]}...")
            try:
                return await self.process_job(job_id, job_data, resume)
            except Exception as e:
                logger.error(f"❌ Job {job_id[:8]} failed: {e}")
                logger.debug(traceback.format_exc())
                return None

    async def run_all(self, job_ids: List[str], job_db: Dict, resume: bool = True):
        """Run analysis for all jobs with controlled concurrency"""
        logger.info(f"")
        logger.info(f"{'#'*60}")
        logger.info(f"🚀 STARTING PARALLEL COT ANALYSIS")
        logger.info(f"{'#'*60}")
        logger.info(f"Total jobs: {len(job_ids)}")
        logger.info(f"Job concurrency: {self.job_concurrency} (jobs at once)")
        logger.info(f"API concurrency: {self.concurrency} (calls per job)")
        logger.info(f"Total max API calls: {self.concurrency * self.job_concurrency}")
        logger.info(f"Resume: {resume}")
        logger.info(f"")
        
        total_start = time.time()
        
        # Create tasks for all jobs - semaphore controls how many run at once
        tasks = [
            self.process_job_with_semaphore(job_id, job_db.get(job_id, {}), resume, i, len(job_ids))
            for i, job_id in enumerate(job_ids, 1)
        ]
        
        # Run all jobs with controlled concurrency
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and None results
        valid_results = [r for r in results if r is not None and not isinstance(r, Exception)]
        
        total_time = time.time() - total_start
        
        # Print summary
        logger.info(f"")
        logger.info(f"{'#'*60}")
        logger.info(f"📊 FINAL SUMMARY")
        logger.info(f"{'#'*60}")
        
        done = sum(1 for r in valid_results if r and r.status == 'done')
        logger.info(f"Completed: {done}/{len(job_ids)} jobs")
        logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        logger.info(f"")
        
        for r in valid_results:
            if r:
                status_icon = "✅" if r.status == 'done' else "❌"
                logger.info(f"  {status_icon} {r.model}/{r.dataset}: {r.processed_samples}/{r.total_samples} samples")


def load_job_db() -> Dict:
    """Load job database"""
    job_db_path = _REPO / "webui" / "backend" / "job_db.json"
    with open(job_db_path, 'r') as f:
        return json.load(f)


def get_jobs_by_prompt_type(job_db: Dict, prompt_type_filter: str = 'fewshot', 
                           specific_ids: Optional[List[str]] = None,
                           exclude_datasets: Optional[List[str]] = None) -> List[str]:
    """Get all jobs matching prompt type that are DONE or completed"""
    jobs = []
    exclude_datasets = exclude_datasets or []
    
    for job_id, job in job_db.items():
        # Skip cot_analysis entries
        if job_id.startswith('cot_analysis_'):
            continue
        
        # Check if matches specific IDs (support partial matching)
        if specific_ids:
            matches = any(job_id.startswith(sid) or sid.startswith(job_id[:8]) for sid in specific_ids)
            if not matches:
                continue
        
        # Check status (allow both 'DONE' and 'completed')
        status = job.get('status', '').lower()
        if status not in ['done', 'completed']:
            continue
            
        req = job.get('request', {})
        prompt_type = req.get('prompt_type', '')
        dataset = req.get('dataset', '')
        
        # Skip excluded datasets (e.g., 'math' - too large)
        if dataset in exclude_datasets:
            continue
        
        # Match prompt type
        if prompt_type_filter.lower() in prompt_type.lower():
            jobs.append(job_id)
    return jobs


def get_fewshot_jobs(job_db: Dict, specific_ids: Optional[List[str]] = None) -> List[str]:
    """Get all fewshot jobs that are DONE (backward compatibility)"""
    return get_jobs_by_prompt_type(job_db, 'fewshot', specific_ids)


def load_api_key() -> str:
    """Load OpenAI API key from config"""
    config_path = _REPO / "webui" / "backend" / "path_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config.get('openai_api_key', '')


async def main():
    parser = argparse.ArgumentParser(
        description="Run parallel CoT analysis with checkpoint support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all fewshot jobs (default)
  python run_cot_analysis_parallel.py

  # Run all direct prompt jobs
  python run_cot_analysis_parallel.py --prompt-type direct

  # Run with more parallel jobs (faster!)
  python run_cot_analysis_parallel.py -p direct --job-concurrency 10

  # Run 5 jobs at once with 3 API calls each
  python run_cot_analysis_parallel.py -j 5 -c 3

  # Run specific jobs only
  python run_cot_analysis_parallel.py --job-ids 8c91f076 1d02f381

  # Start fresh (ignore existing checkpoints)
  python run_cot_analysis_parallel.py --no-resume

  # List jobs and their status
  python run_cot_analysis_parallel.py --list-jobs -p direct

  # Verbose mode (see detailed API calls)
  python run_cot_analysis_parallel.py --verbose
        """
    )
    parser.add_argument(
        "--concurrency", "-c", type=int, default=5,
        help="Number of parallel API calls per job (default: 5)"
    )
    parser.add_argument(
        "--job-concurrency", "-j", type=int, default=5,
        help="Number of jobs to run in parallel (default: 5)"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Don't resume from checkpoints, start fresh"
    )
    parser.add_argument(
        "--job-ids", nargs="+",
        help="Specific job IDs to analyze (default: all fewshot jobs)"
    )
    parser.add_argument(
        "--list-jobs", action="store_true",
        help="List available fewshot jobs and exit"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max samples per job (for testing)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output (show detailed API calls and scores)"
    )
    parser.add_argument(
        "--prompt-type", "-p", type=str, default="fewshot",
        choices=["fewshot", "direct", "cot"],
        help="Prompt type to analyze (default: fewshot)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    global logger
    logger = setup_logging(verbose=args.verbose)
    
    # Load data
    job_db = load_job_db()
    api_key = load_api_key()
    
    if not api_key:
        logger.error("❌ No OpenAI API key found in backend/path_config.json")
        sys.exit(1)
    
    # Get jobs to process - exclude 'math' dataset (too large, use math500 instead)
    exclude_datasets = ['math']
    jobs_to_analyze = get_jobs_by_prompt_type(
        job_db, 
        prompt_type_filter=args.prompt_type,
        specific_ids=args.job_ids,
        exclude_datasets=exclude_datasets
    )
    
    if args.list_jobs:
        print(f"Found {len(jobs_to_analyze)} {args.prompt_type} jobs:")
        for job_id in jobs_to_analyze:
            job = job_db[job_id]
            req = job.get('request', {})
            model = req.get('model', '').split('/')[-1]
            dataset = req.get('dataset', '')
            checkpoint = JobCheckpoint.load(job_id)
            status = "✅ done" if checkpoint and checkpoint.status == 'done' else "⏳ pending"
            print(f"  {job_id[:8]}... | {model:<35} | {dataset:<15} | {status}")
        return
    
    if not jobs_to_analyze:
        logger.error(f"❌ No {args.prompt_type} jobs found to analyze")
        sys.exit(1)
    
    logger.info(f"🔑 API key loaded (ending in ...{api_key[-4:]})")
    logger.info(f"📁 Checkpoints: {CHECKPOINT_DIR}")
    logger.info(f"📁 Logs: {LOG_DIR}")
    logger.info(f"⚙️  Prompt type: {args.prompt_type}")
    logger.info(f"⚙️  Jobs to analyze: {len(jobs_to_analyze)}")
    logger.info(f"⚙️  API concurrency: {args.concurrency} (per job)")
    logger.info(f"⚙️  Job concurrency: {args.job_concurrency} (parallel jobs)")
    logger.info(f"⚙️  Total max API calls: {args.concurrency * args.job_concurrency}")
    logger.info(f"⚙️  Verbose: {args.verbose}")
    logger.info(f"⚙️  Resume: {not args.no_resume}")
    if args.max_samples:
        logger.info(f"⚙️  Max samples: {args.max_samples} (testing mode)")
    
    # Run analysis
    analyzer = ParallelCotAnalyzer(api_key, concurrency=args.concurrency, job_concurrency=args.job_concurrency)
    analyzer.max_samples = args.max_samples
    await analyzer.run_all(jobs_to_analyze, job_db, resume=not args.no_resume)


if __name__ == "__main__":
    asyncio.run(main())
