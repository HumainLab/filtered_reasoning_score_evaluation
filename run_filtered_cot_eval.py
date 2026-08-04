#!/usr/bin/env python3
"""
Batch CoT Evaluation Runner for filtered-cot JSONL files.

Runs the four-pillar CoT evaluation (Faithfulness, Utility, Coherence, Factuality)
with GPT-4o-mini as the LLM judge on all filtered CoT JSONL files.

Features:
  - Parallel processing: 5 files concurrently, 5 samples per file concurrently
  - Extensive live terminal logging with progress bars and per-sample details
  - Checkpoint/resume support (skips already-evaluated samples)
  - Results saved per-model as JSON in filtered-cot/results/
"""

import json
import os
import sys
import time
import logging
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR / "backend"
DEFAULT_INPUT_DIR = SCRIPT_DIR / "filtered-cot"

# Add backend to sys.path so we can import cot_eval_v2
sys.path.insert(0, str(BACKEND_DIR))

# ── load API key from path_config.json ─────────────────────────────────────────
def load_api_key() -> str:
    """Load OpenAI API key from backend/path_config.json."""
    config_path = BACKEND_DIR / "path_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        key = config.get("openai_api_key", "")
        if key:
            return key
    # Fallback to environment variable
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "No OpenAI API key found. Set it in backend/path_config.json or OPENAI_API_KEY env var."
        )
    return key


# ── thread-safe counters ───────────────────────────────────────────────────────
class ProgressTracker:
    """Thread-safe progress tracking for parallel evaluation."""

    def __init__(self, total_files: int):
        self._lock = threading.Lock()
        self.total_files = total_files
        self.files_done = 0
        # per-file trackers: {filename: {total, done, correct, errors, start_time}}
        self.file_progress: Dict[str, Dict[str, Any]] = {}
        self.global_start = time.time()

    def register_file(self, filename: str, total_samples: int):
        with self._lock:
            self.file_progress[filename] = {
                "total": total_samples,
                "done": 0,
                "correct": 0,
                "errors": 0,
                "start_time": time.time(),
            }

    def record_sample(self, filename: str, correct: bool, error: bool = False):
        with self._lock:
            fp = self.file_progress[filename]
            fp["done"] += 1
            if correct:
                fp["correct"] += 1
            if error:
                fp["errors"] += 1

    def mark_file_done(self, filename: str):
        with self._lock:
            self.files_done += 1

    def get_summary_lines(self) -> List[str]:
        with self._lock:
            elapsed = time.time() - self.global_start
            lines = [
                "",
                f"{'='*80}",
                f"  GLOBAL PROGRESS  |  Files: {self.files_done}/{self.total_files}  |  Elapsed: {elapsed:.0f}s",
                f"{'='*80}",
            ]
            for fname, fp in sorted(self.file_progress.items()):
                pct = fp["done"] / max(fp["total"], 1) * 100
                acc = fp["correct"] / max(fp["done"], 1) * 100 if fp["done"] > 0 else 0
                bar_len = 30
                filled = int(bar_len * fp["done"] / max(fp["total"], 1))
                bar = "#" * filled + "-" * (bar_len - filled)
                status = "DONE" if fp["done"] == fp["total"] else "RUNNING"
                lines.append(
                    f"  [{bar}] {fp['done']:>4}/{fp['total']:<4} ({pct:5.1f}%) "
                    f"acc={acc:5.1f}% err={fp['errors']} {status}  {fname}"
                )
            lines.append(f"{'='*80}")
            lines.append("")
            return lines


# ── logging setup ──────────────────────────────────────────────────────────────
def setup_logging(results_dir: Path, verbose: bool = False) -> logging.Logger:
    """Configure logging to both file and terminal."""
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("cot_eval_batch")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler – everything
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(results_dir / f"eval_log_{ts}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(fh)

    # Console handler – INFO+ (or DEBUG if verbose)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(ch)

    return logger


# ── single-sample evaluation ──────────────────────────────────────────────────
def evaluate_sample(
    record: Dict[str, Any],
    judge,
    logger: logging.Logger,
    filename: str,
) -> Dict[str, Any]:
    """
    Evaluate a single sample using PillarsEvaluator + Judge.

    Returns a result dict with idx, scores, metadata.
    """
    from app.cot_eval_v2.evaluator import PillarsEvaluator

    idx = record.get("idx", -1)
    question = record.get("question", "")
    gt = record.get("gt", record.get("answer", ""))
    cot_text = ""
    code_field = record.get("code", [])
    if isinstance(code_field, list) and code_field:
        cot_text = code_field[0]
    elif isinstance(code_field, str):
        cot_text = code_field

    pred = record.get("pred", [])
    if isinstance(pred, list) and pred:
        pred_str = str(pred[0])
    else:
        pred_str = str(pred)

    score_field = record.get("score", [])
    if isinstance(score_field, list) and score_field:
        original_correct = bool(score_field[0])
    else:
        original_correct = bool(score_field)

    # Handle empty CoT
    if not cot_text or not cot_text.strip():
        logger.warning(f"  [{filename}] idx={idx}: EMPTY CoT – skipping judge")
        return {
            "idx": idx,
            "question": question[:150],
            "gt": gt,
            "pred": pred_str,
            "original_correct": original_correct,
            "cot_length_chars": 0,
            "judge_scores": None,
            "fused_scores": None,
            "flags_summary": "Empty CoT",
            "status": "empty_cot",
        }

    t0 = time.time()

    try:
        evaluator = PillarsEvaluator(judge=judge)
        flags, evidence, rule_scores_dict, judge_scores, fused_scores = evaluator.analyze(
            problem=question, cot_text=cot_text, gold=gt
        )

        elapsed = time.time() - t0

        result = {
            "idx": idx,
            "question": question[:150],
            "gt": gt,
            "pred": pred_str,
            "original_correct": original_correct,
            "cot_length_chars": len(cot_text),
            "judge_scores": judge_scores,
            "rule_scores": rule_scores_dict,
            "fused_scores": fused_scores,
            "flags_summary": flags.summarize_for_prompt()[:500],
            "flags_count": len(flags),
            "evidence": {k: v for k, v in evidence.items() if k != "arith_bad_examples"},
            "status": "ok",
            "eval_time_s": round(elapsed, 2),
        }

        # Detailed log
        fs = fused_scores
        logger.info(
            f"  [{filename}] idx={idx:>4} | "
            f"faith={fs.get('faithfulness',0):.2f} util={fs.get('utility',0):.2f} "
            f"coher={fs.get('coherence',0):.2f} fact={fs.get('factuality',0):.2f} "
            f"overall={fs.get('overall',0):.2f} | "
            f"orig_correct={original_correct} | {elapsed:.1f}s"
        )

        return result

    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"  [{filename}] idx={idx}: ERROR – {e} ({elapsed:.1f}s)")
        logger.debug(traceback.format_exc())
        return {
            "idx": idx,
            "question": question[:150],
            "gt": gt,
            "pred": pred_str,
            "original_correct": original_correct,
            "cot_length_chars": len(cot_text),
            "judge_scores": None,
            "fused_scores": None,
            "flags_summary": f"Error: {str(e)[:200]}",
            "status": "error",
            "eval_time_s": round(elapsed, 2),
        }


# ── per-file evaluation (runs N samples in parallel) ─────────────────────────
def evaluate_file(
    filepath: Path,
    api_key: str,
    samples_parallel: int,
    logger: logging.Logger,
    tracker: ProgressTracker,
    results_dir: Path,
    dry_run: bool = False,
    max_samples: Optional[int] = None,
    portkey: Optional[Tuple[Optional[str], str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate all samples in a single JSONL file.

    Returns summary dict with results and statistics.
    """
    filename = filepath.stem  # e.g. DeepSeek_R1_Distill_Qwen_1.5B_filtered_p1_only

    # Load records
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if max_samples is not None:
        records = records[:max_samples]

    total = len(records)
    tracker.register_file(filename, total)
    logger.info(f"\n{'─'*70}")
    logger.info(f"START: {filename} ({total} samples, {samples_parallel} parallel)")
    logger.info(f"{'─'*70}")

    # Load checkpoint if exists
    result_path = results_dir / f"{filename}_results.json"
    existing_results = {}
    if result_path.exists():
        try:
            with open(result_path) as f:
                prev = json.load(f)
            for r in prev.get("results", []):
                existing_results[r["idx"]] = r
            logger.info(f"  Loaded checkpoint with {len(existing_results)} existing results")
        except Exception:
            pass

    # Create judge (one per file to avoid thread issues with OpenAI client)
    from app.cot_eval_v2.judge import Judge
    portkey_key = None
    portkey_prefix = os.environ.get("PORTKEY_MODEL_PREFIX", "")
    if portkey is not None:
        portkey_key, portkey_prefix = portkey

    if portkey_key:
        # Route via Portkey (OpenAI-compatible)
        judge = Judge(
            model=f"{portkey_prefix}/gpt-4o-mini",
            mode="ALWAYS",
            diagnostic=False,
            base_url="https://api.portkey.ai/v1",
            api_key=portkey_key,
        )
    else:
        os.environ["OPENAI_API_KEY"] = api_key
        judge = Judge(model="gpt-4o-mini", mode="ALWAYS", diagnostic=False)

    all_results = []
    file_start = time.time()

    if dry_run:
        # Dry-run: just count records
        for rec in records:
            idx = rec.get("idx", -1)
            tracker.record_sample(filename, correct=False)
            all_results.append({"idx": idx, "status": "dry_run"})
        tracker.mark_file_done(filename)
        logger.info(f"  DRY RUN complete for {filename}: {total} samples")
        return {
            "filename": filename, "total": total, "evaluated_ok": 0,
            "errors": 0, "empty_cot": 0, "avg_scores": {},
            "eval_time_s": 0, "results": all_results,
        }

    # Evaluate samples in parallel
    def _eval_one(record):
        idx = record.get("idx", -1)
        # Skip if already in checkpoint
        if idx in existing_results:
            res = existing_results[idx]
            tracker.record_sample(
                filename,
                correct=res.get("original_correct", False),
                error=res.get("status") == "error",
            )
            logger.debug(f"  [{filename}] idx={idx}: CACHED (skipping)")
            return res

        result = evaluate_sample(record, judge, logger, filename)
        tracker.record_sample(
            filename,
            correct=result.get("original_correct", False),
            error=result.get("status") == "error",
        )
        return result

    with ThreadPoolExecutor(max_workers=samples_parallel) as pool:
        futures = {pool.submit(_eval_one, rec): rec for rec in records}
        for future in as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                rec = futures[future]
                idx = rec.get("idx", -1)
                logger.error(f"  [{filename}] idx={idx}: FUTURE ERROR – {e}")
                all_results.append({"idx": idx, "status": "error", "error": str(e)})

    # Sort by idx
    all_results.sort(key=lambda r: r.get("idx", -1))

    elapsed = time.time() - file_start

    # Compute summary stats
    ok_results = [r for r in all_results if r.get("status") == "ok"]
    error_count = sum(1 for r in all_results if r.get("status") == "error")
    empty_count = sum(1 for r in all_results if r.get("status") == "empty_cot")

    # Average fused scores
    avg_scores = {}
    if ok_results:
        for pillar in ["faithfulness", "utility", "coherence", "factuality", "overall"]:
            vals = [r["fused_scores"][pillar] for r in ok_results if r.get("fused_scores") and pillar in r["fused_scores"]]
            avg_scores[pillar] = round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "filename": filename,
        "total": total,
        "evaluated_ok": len(ok_results),
        "errors": error_count,
        "empty_cot": empty_count,
        "avg_scores": avg_scores,
        "eval_time_s": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
    }

    # Save results
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"  Saved results to {result_path}")

    tracker.mark_file_done(filename)

    # Print file summary
    logger.info(f"\n{'─'*70}")
    logger.info(f"DONE: {filename}")
    logger.info(f"  Total={total}  OK={len(ok_results)}  Errors={error_count}  Empty={empty_count}  Time={elapsed:.0f}s")
    if avg_scores:
        logger.info(
            f"  Avg Scores: faith={avg_scores.get('faithfulness','N/A')} "
            f"util={avg_scores.get('utility','N/A')} "
            f"coher={avg_scores.get('coherence','N/A')} "
            f"fact={avg_scores.get('factuality','N/A')} "
            f"overall={avg_scores.get('overall','N/A')}"
        )
    logger.info(f"{'─'*70}\n")

    return summary


# ── progress printer (background thread) ─────────────────────────────────────
def progress_printer(tracker: ProgressTracker, logger: logging.Logger, stop_event: threading.Event):
    """Periodically print global progress to the terminal."""
    while not stop_event.is_set():
        stop_event.wait(15)  # Print every 15 seconds
        if stop_event.is_set():
            break
        lines = tracker.get_summary_lines()
        for line in lines:
            logger.info(line)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Batch CoT evaluation for filtered-cot JSONL files")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory containing *_filtered_p1_only.jsonl files (default: filtered-cot)")
    parser.add_argument("--files-parallel", type=int, default=5, help="Number of files to process in parallel (default: 5)")
    parser.add_argument("--samples-parallel", type=int, default=5, help="Number of samples per file to process in parallel (default: 5)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per file (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without API calls")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging (DEBUG level to terminal)")
    parser.add_argument("--files", nargs="*", help="Specific JSONL files to process (default: all)")
    parser.add_argument("--portkey-key", type=str, default=None, help="Portkey API key (routes judge calls via Portkey).")
    parser.add_argument("--portkey-model-prefix", type=str, default=os.environ.get("PORTKEY_MODEL_PREFIX", ""),
                        help="Portkey model prefix (default: $PORTKEY_MODEL_PREFIX)")
    args = parser.parse_args()

    # Resolve paths
    input_dir = Path(args.input_dir) if args.input_dir else DEFAULT_INPUT_DIR
    input_dir = input_dir if input_dir.is_absolute() else SCRIPT_DIR / input_dir
    results_dir = input_dir / "results"

    # Setup
    logger = setup_logging(results_dir, verbose=args.verbose)
    logger.info("=" * 80)
    logger.info("  CoT EVALUATION BATCH RUNNER")
    logger.info(f"  Input dir: {input_dir}")
    logger.info(f"  Time: {datetime.now().isoformat()}")
    logger.info(f"  Files parallel: {args.files_parallel}")
    logger.info(f"  Samples parallel: {args.samples_parallel}")
    logger.info(f"  Max samples: {args.max_samples or 'ALL'}")
    logger.info(f"  Dry run: {args.dry_run}")
    logger.info("=" * 80)

    # Load API key (OpenAI by default; Portkey if provided)
    if not args.dry_run:
        if args.portkey_key:
            api_key = args.portkey_key
            logger.info("  Using Portkey API key for judge calls")
        else:
            api_key = load_api_key()
            logger.info(f"  API key loaded (ends with ...{api_key[-6:]})")
    else:
        api_key = "dry-run-no-key"

    # Find JSONL files
    if args.files:
        jsonl_files = [input_dir / f if not Path(f).is_absolute() else Path(f) for f in args.files]
    else:
        jsonl_files = sorted(input_dir.glob("*_filtered_p1_only.jsonl"))

    if not jsonl_files:
        logger.error(f"No *_filtered_p1_only.jsonl files found in {input_dir}/")
        sys.exit(1)

    logger.info(f"\n  Found {len(jsonl_files)} JSONL files:")
    for fp in jsonl_files:
        with open(fp) as f:
            count = sum(1 for _ in f)
        logger.info(f"    {fp.name}: {count} samples")
    logger.info("")

    # Create progress tracker
    tracker = ProgressTracker(total_files=len(jsonl_files))

    # Start progress printer thread
    stop_event = threading.Event()
    printer_thread = threading.Thread(
        target=progress_printer, args=(tracker, logger, stop_event), daemon=True
    )
    printer_thread.start()

    # Run evaluations in parallel across files
    all_summaries = []
    global_start = time.time()

    with ThreadPoolExecutor(max_workers=args.files_parallel) as pool:
        futures = {
            pool.submit(
                evaluate_file,
                fp,
                api_key,
                args.samples_parallel,
                logger,
                tracker,
                results_dir,
                args.dry_run,
                args.max_samples,
                (args.portkey_key, args.portkey_model_prefix),
            ): fp
            for fp in jsonl_files
        }

        for future in as_completed(futures):
            fp = futures[future]
            try:
                summary = future.result()
                all_summaries.append(summary)
            except Exception as e:
                logger.error(f"FILE-LEVEL ERROR for {fp.name}: {e}")
                logger.debug(traceback.format_exc())

    # Stop progress printer
    stop_event.set()
    printer_thread.join(timeout=2)

    # Print final summary
    total_elapsed = time.time() - global_start
    logger.info("\n" + "=" * 80)
    logger.info("  FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    logger.info(f"  Files processed: {len(all_summaries)}/{len(jsonl_files)}")
    logger.info("")

    for s in sorted(all_summaries, key=lambda x: x["filename"]):
        avg = s.get("avg_scores", {})
        logger.info(
            f"  {s['filename']:50s} "
            f"OK={s['evaluated_ok']:>4} Err={s['errors']:>3} Empty={s['empty_cot']:>3} "
            f"Overall={avg.get('overall', 'N/A'):>6} "
            f"Time={s['eval_time_s']:>5.0f}s"
        )

    logger.info("")
    logger.info(f"  Results saved to: {results_dir}/")
    logger.info("=" * 80)

    # Save global summary
    global_summary_path = results_dir / "global_summary.json"
    global_summary = {
        "timestamp": datetime.now().isoformat(),
        "total_time_s": round(total_elapsed, 1),
        "files_parallel": args.files_parallel,
        "samples_parallel": args.samples_parallel,
        "file_summaries": [
            {k: v for k, v in s.items() if k != "results"}
            for s in all_summaries
        ],
    }
    with open(global_summary_path, "w") as f:
        json.dump(global_summary, f, indent=2, default=str)
    logger.info(f"  Global summary saved to: {global_summary_path}")


if __name__ == "__main__":
    main()
