#!/usr/bin/env python3
"""
CoT Analysis Runner for filtered-cot GPQA (and other filtered-cot dirs).

Runs the four-pillar CoT evaluation (Faithfulness, Utility, Coherence, Factuality)
with GPT-4o-mini as the LLM judge.

Concurrency:
  - 5 jobs (files) at a time
  - 20 samples at a time per file

Saving:
  - Per-file results: results/<stem>_results.json
  - Incremental checkpoint: save after every 20 samples per file (minimize loss on crash)
  - Run manifest: results/run_manifest_<run_id>.json (start + end)
  - Global summary: results/global_summary.json

Logging:
  - Main log: results/logs/eval_<run_id>.log
  - Error-only log: results/logs/eval_errors_<run_id>.log
  - Structured events (JSONL): results/logs/eval_events_<run_id>.jsonl

Usage:
  python run_cot_analysis_gpqa.py
  python run_cot_analysis_gpqa.py --input-dir filtered-cot-gpqa --files-parallel 5 --samples-parallel 20
  python run_cot_analysis_gpqa.py --max-samples 50 --dry-run
"""

import json
import os
import sys
import time
import uuid
import logging
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR / "backend"
DEFAULT_INPUT_DIR = SCRIPT_DIR / "filtered-cot-gpqa"

# Default concurrency: 5 jobs, 20 samples per job
DEFAULT_FILES_PARALLEL = 5
DEFAULT_SAMPLES_PARALLEL = 20
DEFAULT_SAVE_EVERY_N = 20

sys.path.insert(0, str(BACKEND_DIR))


def load_api_key() -> str:
    """Load OpenAI API key from backend/path_config.json or env."""
    config_path = BACKEND_DIR / "path_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        key = config.get("openai_api_key", "")
        if key:
            return key
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "No OpenAI API key. Set backend/path_config.json or OPENAI_API_KEY."
        )
    return key


# ── Progress tracker ─────────────────────────────────────────────────────────────
class ProgressTracker:
    def __init__(self, total_files: int):
        self._lock = threading.Lock()
        self.total_files = total_files
        self.files_done = 0
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
            lines.append(f"{'='*80}\n")
            return lines


# ── Logging and event writer ─────────────────────────────────────────────────────
class RunLogger:
    """Multi-destination logging: main log, error log, structured JSONL events."""

    def __init__(
        self,
        run_id: str,
        results_dir: Path,
        verbose: bool = False,
    ):
        self.run_id = run_id
        self.results_dir = results_dir
        logs_dir = results_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = logs_dir / f"eval_{run_id}.log"
        self.error_log_path = logs_dir / f"eval_errors_{run_id}.log"
        self.events_path = logs_dir / f"eval_events_{run_id}.jsonl"
        self._events_file = open(self.events_path, "a")
        self._lock = threading.Lock()

        # Main logger
        self.logger = logging.getLogger(f"cot_gpqa_{run_id}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        fh = logging.FileHandler(self.log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)

        eh = logging.FileHandler(self.error_log_path)
        eh.setLevel(logging.ERROR)
        eh.setFormatter(fmt)
        self.logger.addHandler(eh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG if verbose else logging.INFO)
        ch.setFormatter(fmt)
        self.logger.addHandler(ch)

    def event(self, event_type: str, **kwargs):
        """Append one JSON line to the events log."""
        with self._lock:
            rec = {"ts": datetime.now().isoformat(), "event": event_type, **kwargs}
            self._events_file.write(json.dumps(rec, default=str) + "\n")
            self._events_file.flush()

    def close(self):
        with self._lock:
            if self._events_file and not self._events_file.closed:
                self._events_file.close()


# ── Single-sample evaluation ─────────────────────────────────────────────────────
def evaluate_sample(
    record: Dict[str, Any],
    judge,
    logger: logging.Logger,
    filename: str,
) -> Dict[str, Any]:
    """Evaluate one sample; returns result dict with idx, scores, status."""
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
    pred_str = str(pred[0]) if isinstance(pred, list) and pred else str(pred)
    score_field = record.get("score", [])
    original_correct = bool(score_field[0]) if isinstance(score_field, list) and score_field else bool(score_field)

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
        fs = fused_scores
        logger.info(
            f"  [{filename}] idx={idx:>4} | "
            f"faith={fs.get('faithfulness',0):.2f} util={fs.get('utility',0):.2f} "
            f"coher={fs.get('coherence',0):.2f} fact={fs.get('factuality',0):.2f} "
            f"overall={fs.get('overall',0):.2f} | orig_correct={original_correct} | {elapsed:.1f}s"
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


# ── Per-file evaluation with incremental save ───────────────────────────────────
def evaluate_file(
    filepath: Path,
    api_key: str,
    samples_parallel: int,
    save_every_n: int,
    logger: logging.Logger,
    run_logger: RunLogger,
    tracker: ProgressTracker,
    results_dir: Path,
    dry_run: bool = False,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate all samples in one JSONL file. Saves every save_every_n samples."""
    filename = filepath.stem

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
    run_logger.event("file_start", file=filename, total=total, samples_parallel=samples_parallel)
    logger.info(f"\n{'─'*70}")
    logger.info(f"START: {filename} ({total} samples, {samples_parallel} parallel, save_every={save_every_n})")
    logger.info(f"{'─'*70}")

    result_path = results_dir / f"{filename}_results.json"
    existing_results = {}
    if result_path.exists():
        try:
            with open(result_path) as f:
                prev = json.load(f)
            for r in prev.get("results", []):
                existing_results[r["idx"]] = r
            logger.info(f"  Loaded checkpoint: {len(existing_results)} existing results")
            run_logger.event("checkpoint_loaded", file=filename, count=len(existing_results))
        except Exception as e:
            logger.warning(f"  Could not load checkpoint: {e}")

    os.environ["OPENAI_API_KEY"] = api_key
    from app.cot_eval_v2.judge import Judge
    judge = Judge(model="gpt-4o-mini", mode="ALWAYS", diagnostic=False)

    all_results: List[Dict[str, Any]] = []
    file_start = time.time()
    completed_since_save = 0

    def _write_checkpoint(current_results: List[Dict[str, Any]], reason: str = "incremental"):
        """Sort by idx and write current results to disk."""
        sorted_results = sorted(current_results, key=lambda r: r.get("idx", -1))
        ok_count = sum(1 for r in sorted_results if r.get("status") == "ok")
        err_count = sum(1 for r in sorted_results if r.get("status") == "error")
        empty_count = sum(1 for r in sorted_results if r.get("status") == "empty_cot")
        avg_scores = {}
        ok_list = [r for r in sorted_results if r.get("status") == "ok"]
        if ok_list:
            for pillar in ["faithfulness", "utility", "coherence", "factuality", "overall"]:
                vals = [r["fused_scores"][pillar] for r in ok_list if r.get("fused_scores") and pillar in r["fused_scores"]]
                avg_scores[pillar] = round(sum(vals) / len(vals), 4) if vals else None
        summary = {
            "filename": filename,
            "total": total,
            "evaluated_ok": ok_count,
            "errors": err_count,
            "empty_cot": empty_count,
            "avg_scores": avg_scores,
            "eval_time_s": round(time.time() - file_start, 1),
            "timestamp": datetime.now().isoformat(),
            "checkpoint_reason": reason,
            "results": sorted_results,
        }
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        run_logger.event("checkpoint_saved", file=filename, count=len(sorted_results), reason=reason)

    if dry_run:
        for rec in records:
            idx = rec.get("idx", -1)
            tracker.record_sample(filename, correct=False)
            all_results.append({"idx": idx, "status": "dry_run"})
        _write_checkpoint(all_results, "dry_run")
        tracker.mark_file_done(filename)
        logger.info(f"  DRY RUN complete for {filename}: {total} samples")
        run_logger.event("file_end", file=filename, total=total, evaluated_ok=0, errors=0, empty_cot=0, eval_time_s=0)
        return {
            "filename": filename, "total": total, "evaluated_ok": 0,
            "errors": 0, "empty_cot": 0, "avg_scores": {},
            "eval_time_s": 0, "results": all_results,
        }

    def _eval_one(record: Dict[str, Any]) -> Dict[str, Any]:
        idx = record.get("idx", -1)
        if idx in existing_results:
            res = existing_results[idx]
            tracker.record_sample(filename, correct=res.get("original_correct", False), error=res.get("status") == "error")
            logger.debug(f"  [{filename}] idx={idx}: CACHED")
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
                completed_since_save += 1
                if completed_since_save >= save_every_n:
                    _write_checkpoint(all_results, "batch")
                    logger.info(f"  [{filename}] Incremental save: {len(all_results)} results")
                    completed_since_save = 0
            except Exception as e:
                rec = futures[future]
                idx = rec.get("idx", -1)
                logger.error(f"  [{filename}] idx={idx}: FUTURE ERROR – {e}")
                all_results.append({"idx": idx, "status": "error", "error": str(e)})
                completed_since_save += 1

    # Final save
    _write_checkpoint(all_results, "final")
    elapsed = time.time() - file_start

    ok_results = [r for r in all_results if r.get("status") == "ok"]
    error_count = sum(1 for r in all_results if r.get("status") == "error")
    empty_count = sum(1 for r in all_results if r.get("status") == "empty_cot")
    avg_scores = {}
    if ok_results:
        for pillar in ["faithfulness", "utility", "coherence", "factuality", "overall"]:
            vals = [r["fused_scores"][pillar] for r in ok_results if r.get("fused_scores") and pillar in r["fused_scores"]]
            avg_scores[pillar] = round(sum(vals) / len(vals), 4) if vals else None

    logger.info(f"  Saved results to {result_path}")
    logger.info(f"\n{'─'*70}")
    logger.info(f"DONE: {filename} | Total={total} OK={len(ok_results)} Errors={error_count} Empty={empty_count} Time={elapsed:.0f}s")
    if avg_scores:
        logger.info(
            f"  Avg: faith={avg_scores.get('faithfulness','N/A')} util={avg_scores.get('utility','N/A')} "
            f"coher={avg_scores.get('coherence','N/A')} fact={avg_scores.get('factuality','N/A')} overall={avg_scores.get('overall','N/A')}"
        )
    logger.info(f"{'─'*70}\n")

    run_logger.event(
        "file_end", file=filename, total=total,
        evaluated_ok=len(ok_results), errors=error_count, empty_cot=empty_count,
        eval_time_s=round(elapsed, 1), avg_overall=avg_scores.get("overall"),
    )
    tracker.mark_file_done(filename)

    return {
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


def progress_printer(tracker: ProgressTracker, logger: logging.Logger, stop_event: threading.Event):
    while not stop_event.is_set():
        stop_event.wait(15)
        if stop_event.is_set():
            break
        for line in tracker.get_summary_lines():
            logger.info(line)


def main():
    parser = argparse.ArgumentParser(
        description="CoT analysis for filtered-cot (GPQA): 5 jobs, 20 samples, extensive saving & logging",
    )
    parser.add_argument("--input-dir", type=str, default=None,
                        help=f"Directory with *_filtered_p1_only.jsonl (default: {DEFAULT_INPUT_DIR.name})")
    parser.add_argument("--files-parallel", type=int, default=DEFAULT_FILES_PARALLEL,
                        help=f"Number of files (jobs) in parallel (default: {DEFAULT_FILES_PARALLEL})")
    parser.add_argument("--samples-parallel", type=int, default=DEFAULT_SAMPLES_PARALLEL,
                        help=f"Samples per file in parallel (default: {DEFAULT_SAMPLES_PARALLEL})")
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY_N,
                        help=f"Save checkpoint every N samples per file (default: {DEFAULT_SAVE_EVERY_N})")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per file (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="No API calls")
    parser.add_argument("--verbose", action="store_true", help="DEBUG to console")
    parser.add_argument("--files", nargs="*", help="Only these JSONL files (default: all)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else DEFAULT_INPUT_DIR
    if not input_dir.is_absolute():
        input_dir = SCRIPT_DIR / input_dir
    results_dir = input_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_logger = RunLogger(run_id, results_dir, verbose=args.verbose)
    logger = run_logger.logger

    run_logger.event(
        "run_start",
        run_id=run_id,
        input_dir=str(input_dir),
        files_parallel=args.files_parallel,
        samples_parallel=args.samples_parallel,
        save_every=args.save_every,
        max_samples=args.max_samples,
        dry_run=args.dry_run,
    )

    manifest_path = results_dir / f"run_manifest_{run_id}.json"
    manifest = {
        "run_id": run_id,
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "config": {
            "input_dir": str(input_dir),
            "files_parallel": args.files_parallel,
            "samples_parallel": args.samples_parallel,
            "save_every_n": args.save_every,
            "max_samples": args.max_samples,
            "dry_run": args.dry_run,
        },
        "paths": {
            "log": str(run_logger.log_path),
            "error_log": str(run_logger.error_log_path),
            "events": str(run_logger.events_path),
        },
    }

    if args.files:
        jsonl_files = [input_dir / f if not Path(f).is_absolute() else Path(f) for f in args.files]
    else:
        jsonl_files = sorted(input_dir.glob("*_filtered_p1_only.jsonl"))

    if not jsonl_files:
        logger.error(f"No *_filtered_p1_only.jsonl files in {input_dir}")
        run_logger.event("run_end", run_id=run_id, status="error", error="no_files")
        manifest["status"] = "error"
        manifest["error"] = "no files"
        manifest["ended_at"] = datetime.now().isoformat()
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        run_logger.close()
        sys.exit(1)

    manifest["files"] = [f.name for f in jsonl_files]
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Run manifest: {manifest_path}")

    logger.info("=" * 80)
    logger.info("  CoT ANALYSIS (GPQA) – 5 jobs, 20 samples, extensive save & log")
    logger.info(f"  Run ID: {run_id}")
    logger.info(f"  Input: {input_dir}")
    logger.info(f"  Files parallel: {args.files_parallel}  Samples parallel: {args.samples_parallel}")
    logger.info(f"  Save every: {args.save_every}  Max samples: {args.max_samples or 'ALL'}")
    logger.info(f"  Logs: {results_dir / 'logs'}")
    logger.info("=" * 80)

    if not args.dry_run:
        api_key = load_api_key()
        logger.info(f"  API key loaded (...{api_key[-6:]})")
    else:
        api_key = "dry-run"

    tracker = ProgressTracker(total_files=len(jsonl_files))
    stop_event = threading.Event()
    printer_thread = threading.Thread(target=progress_printer, args=(tracker, logger, stop_event), daemon=True)
    printer_thread.start()

    all_summaries: List[Dict[str, Any]] = []
    global_start = time.time()

    try:
        with ThreadPoolExecutor(max_workers=args.files_parallel) as pool:
            futures = {
                pool.submit(
                    evaluate_file,
                    fp,
                    api_key,
                    args.samples_parallel,
                    args.save_every,
                    logger,
                    run_logger,
                    tracker,
                    results_dir,
                    args.dry_run,
                    args.max_samples,
                ): fp
                for fp in jsonl_files
            }
            for future in as_completed(futures):
                fp = futures[future]
                try:
                    summary = future.result()
                    all_summaries.append(summary)
                except Exception as e:
                    logger.exception(f"FILE ERROR {fp.name}: {e}")
                    run_logger.event("file_error", file=fp.name, error=str(e))

        total_elapsed = time.time() - global_start

        # Global summary
        global_summary_path = results_dir / "global_summary.json"
        global_summary = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "total_time_s": round(total_elapsed, 1),
            "files_parallel": args.files_parallel,
            "samples_parallel": args.samples_parallel,
            "save_every_n": args.save_every,
            "file_summaries": [{k: v for k, v in s.items() if k != "results"} for s in all_summaries],
        }
        with open(global_summary_path, "w") as f:
            json.dump(global_summary, f, indent=2, default=str)
        logger.info(f"  Global summary: {global_summary_path}")

        logger.info("\n" + "=" * 80)
        logger.info("  FINAL SUMMARY")
        logger.info("=" * 80)
        logger.info(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
        logger.info(f"  Files: {len(all_summaries)}/{len(jsonl_files)}")
        for s in sorted(all_summaries, key=lambda x: x["filename"]):
            avg = s.get("avg_scores", {})
            logger.info(
                f"  {s['filename']:50s} OK={s['evaluated_ok']:>4} Err={s['errors']:>3} Empty={s['empty_cot']:>3} "
                f"Overall={str(avg.get('overall', 'N/A')):>6} Time={s['eval_time_s']:>5.0f}s"
            )
        logger.info(f"  Results: {results_dir}")
        logger.info("=" * 80)

        manifest["status"] = "completed"
        manifest["ended_at"] = datetime.now().isoformat()
        manifest["total_time_s"] = round(total_elapsed, 1)
        manifest["file_count"] = len(all_summaries)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        run_logger.event("run_end", run_id=run_id, status="completed", total_time_s=round(total_elapsed, 1))
    except Exception as e:
        logger.exception(f"Run failed: {e}")
        run_logger.event("run_end", run_id=run_id, status="error", error=str(e))
        manifest["status"] = "error"
        manifest["error"] = str(e)
        manifest["ended_at"] = datetime.now().isoformat()
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
    finally:
        stop_event.set()
        printer_thread.join(timeout=2)
        run_logger.close()


if __name__ == "__main__":
    main()
