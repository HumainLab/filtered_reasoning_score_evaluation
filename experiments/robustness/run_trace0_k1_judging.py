#!/usr/bin/env python3
"""
Trace-0 k=1 reasoning-score baseline (Reviewer kp6q ablation).

For each model×benchmark pair:
  - Sample K distinct questions (default 50) uniformly from questions where trace_idx=0
    has valid token-confidence (same pool eligibility as FRS).
  - Judge **only trace_idx=0** with GPT-4o-mini (same rubric as FRS / unfiltered baseline).
  - Resumable checkpoints + live progress files for monitoring.

Requires: PORTKEY_API_KEY, network access for judge API.
Does **not** run model generation or use GPU.

Usage:
  python analysis/run_trace0_k1_judging.py --repo-root . --sample-questions-per-pair 50

Monitor live progress:
  tail -f logs/trace0_k1_judging_*.log
  watch -n 5 cat analysis_outputs/trace0_k1_judging/progress.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from build_downstream_parquets import discover_jsonl_groups  # noqa: E402
from topk_ablation import compute_trace_confidence  # noqa: E402
from topk_judge_eval import (  # noqa: E402
    DEFAULT_JUDGE_MODEL,
    Judge,
    reasoning_score_from_judge,
    setup_logging as setup_topk_judge_logging,
)

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError:
    pearsonr = None  # type: ignore
    spearmanr = None  # type: ignore

CORE_BENCHMARKS = ("GSM8K", "MATH500", "SVAMP", "AQuA", "GPQA", "CommonsenseQA")
BENCHMARK_TO_CSQA = "CSQA"
DATASET_TO_BENCHMARK = {
    "GSM8K": "GSM8K",
    "MATH500": "MATH500",
    "SVAMP": "SVAMP",
    "AQuA": "AQuA",
    "GPQA": "GPQA",
    "CommonsenseQA": BENCHMARK_TO_CSQA,
}
BIN_LABEL = "trace0_k1_baseline"
TRACE_IDX_FIXED = 0
DEFAULT_FRS_CSV = REPO_ROOT / "outputs/global_pass1_frs_analysis" / "merged_pass1_frs_per_benchmark.csv"


def _safe_name(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_").replace(".", "_").replace(":", "_")


def valid_trace_indices(row: dict) -> List[int]:
    scores = row.get("score", [])
    code = row.get("code", [])
    probs_all = row.get("chosen_token_probs_per_path", {}).get("epoch_0", [])
    if not isinstance(probs_all, list) or not isinstance(code, list):
        return []
    n = min(len(scores), len(code), len(probs_all))
    out: List[int] = []
    for ti in range(n):
        probs = probs_all[ti] if ti < len(probs_all) else []
        if not np.isnan(compute_trace_confidence(probs)):
            out.append(ti)
    return out


def load_jsonl_rows(filepath: str) -> List[dict]:
    rows: List[dict] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def problems_by_idx(rows: List[dict]) -> Dict[int, dict]:
    by_idx: Dict[int, dict] = {}
    for row in rows:
        try:
            by_idx[int(row["idx"])] = row
        except (TypeError, ValueError, KeyError):
            continue
    return by_idx


def pair_rng(model: str, dataset: str, base_seed: int) -> np.random.Generator:
    digest = hashlib.sha256(f"trace0_k1|{base_seed}|{model}|{dataset}".encode()).hexdigest()
    seed_int = int(digest[:16], 16) % (2**32)
    return np.random.default_rng((base_seed + seed_int) % (2**32))


@dataclass
class SampledTrace:
    idx: int
    trace_idx: int
    question: str
    cot: str
    gt: str
    correct: bool
    confidence: float


def sample_trace0_questions(
    by_idx: Dict[int, dict],
    rng: np.random.Generator,
    k_questions: int,
    log: logging.Logger,
) -> Tuple[List[int], List[SampledTrace], Dict[str, Any]]:
    """Uniform over questions with trace 0 having valid confidence."""
    eligible: List[int] = []
    no_trace0_conf: List[int] = []
    for idx, row in sorted(by_idx.items()):
        vti = valid_trace_indices(row)
        if TRACE_IDX_FIXED in vti:
            eligible.append(idx)
        elif vti:
            no_trace0_conf.append(idx)

    n_distinct = len(eligible)
    n_sample = min(k_questions, n_distinct)

    if no_trace0_conf:
        log.debug(
            "Questions with valid traces but trace0 missing/invalid conf: count=%d (not sampled)",
            len(no_trace0_conf),
        )

    if n_distinct < k_questions:
        log.warning(
            "Only %d questions have valid trace_idx=0 (requested %d); using all %d.",
            n_distinct,
            k_questions,
            n_sample,
        )

    if n_sample == 0:
        return [], [], {"eligible_trace0": eligible, "n_distinct": 0, "n_sample": 0}

    chosen = rng.choice(np.array(eligible, dtype=np.int64), size=n_sample, replace=False)
    chosen_list = sorted(int(x) for x in chosen.tolist())

    samples: List[SampledTrace] = []
    for idx in chosen_list:
        row = by_idx[idx]
        ti = TRACE_IDX_FIXED
        code = row.get("code", [])
        scores = row.get("score", [])
        probs_all = row.get("chosen_token_probs_per_path", {}).get("epoch_0", [])
        cot = code[ti] if ti < len(code) and isinstance(code[ti], str) else str(code[ti])
        probs = probs_all[ti] if ti < len(probs_all) else []
        conf = float(compute_trace_confidence(probs))
        samples.append(
            SampledTrace(
                idx=idx,
                trace_idx=ti,
                question=str(row.get("question", "")),
                cot=cot,
                gt=str(row.get("gt", row.get("answer", ""))),
                correct=bool(scores[ti]) if ti < len(scores) else False,
                confidence=conf,
            )
        )

    return chosen_list, samples, {
        "eligible_trace0": eligible,
        "n_distinct_trace0": n_distinct,
        "n_sample": n_sample,
        "sampled_question_ids": chosen_list,
    }


def load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_checkpoint(path: Path, data: Dict[str, Any], lock: Optional[threading.Lock] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if lock is not None:
        with lock:
            js = data.get("judged_samples", {})
            keys = list(js.keys())
            snap = {
                **{k: v for k, v in data.items() if k != "judged_samples"},
                "judged_samples": {k: js[k] for k in keys},
            }
    else:
        snap = copy.deepcopy(data)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


class ProgressTracker:
    """Thread-safe global progress written to progress.json frequently."""

    def __init__(self, path: Path, total_planned: int) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.state: Dict[str, Any] = {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "total_planned_judgments": total_planned,
            "completed_judgments": 0,
            "cache_hits": 0,
            "new_ok": 0,
            "new_fail": 0,
            "pairs_completed": 0,
            "pairs_total": 0,
            "current_pair": None,
            "current_pair_progress": None,
            "elapsed_sec": 0.0,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._t0 = time.perf_counter()

    def _flush_locked(self) -> None:
        """Write progress.json atomically (must hold self.lock)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        snap = copy.deepcopy(self.state)
        tmp = self.path.parent / f".{self.path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snap, f, indent=2)
        for attempt in range(3):
            try:
                tmp.replace(self.path)
                return
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(0.05 * (attempt + 1))

    def update(self, **kwargs: Any) -> None:
        with self.lock:
            self.state.update(kwargs)
            self.state["updated_utc"] = datetime.now(timezone.utc).isoformat()
            self.state["elapsed_sec"] = round(time.perf_counter() - self._t0, 1)
            self._flush_locked()

    def increment(self, cache_hit: bool, ok: bool) -> None:
        with self.lock:
            self.state["completed_judgments"] = int(self.state.get("completed_judgments", 0)) + 1
            if cache_hit:
                self.state["cache_hits"] = int(self.state.get("cache_hits", 0)) + 1
            elif ok:
                self.state["new_ok"] = int(self.state.get("new_ok", 0)) + 1
            else:
                self.state["new_fail"] = int(self.state.get("new_fail", 0)) + 1
            self.state["updated_utc"] = datetime.now(timezone.utc).isoformat()
            self.state["elapsed_sec"] = round(time.perf_counter() - self._t0, 1)
            self._flush_locked()


def setup_logging(log_file: Path, verbose: bool) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("trace0_k1_judging")
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


def judge_one(
    judge: Judge,
    st: SampledTrace,
    model: str,
    dataset: str,
    log: logging.Logger,
) -> Tuple[Optional[dict], Optional[float], str]:
    try:
        raw = judge.score(
            problem=str(st.question),
            cot=str(st.cot),
            gold=str(st.gt),
            flags_summary="No automated flags available.",
            evidence={"final_correct": st.correct},
            log_ctx={
                "eval_model": model,
                "dataset": dataset,
                "idx": st.idx,
                "trace_idx": st.trace_idx,
                "bin": BIN_LABEL,
            },
        )
        rs = reasoning_score_from_judge(raw)
        return raw, float(rs) if rs is not None else None, ""
    except Exception as e:
        log.error("Judge failed idx=%s trace=%s: %s", st.idx, st.trace_idx, e, exc_info=True)
        return None, None, str(e)


def discover_pairs(
    repo_root: Path,
    models: Optional[Set[str]],
    datasets: Optional[Set[str]],
) -> List[Tuple[str, str, str]]:
    groups = discover_jsonl_groups(str(repo_root))
    out: List[Tuple[str, str, str]] = []
    for (model, ds), jpath in sorted(groups.items()):
        if ds not in CORE_BENCHMARKS:
            continue
        if models and model not in models:
            continue
        if datasets and ds not in datasets:
            continue
        out.append((model, ds, jpath))
    return out


def write_comparison_csv(out_dir: Path, pair_results: List[Dict[str, Any]], log: logging.Logger) -> None:
    import pandas as pd

    if not pair_results:
        return
    df = pd.DataFrame(pair_results)
    if DEFAULT_FRS_CSV.is_file():
        frs = pd.read_csv(DEFAULT_FRS_CSV)
        df = df.merge(
            frs[["model", "benchmark", "frs_pct", "pass1_pct"]],
            left_on=["model", "benchmark"],
            right_on=["model", "benchmark"],
            how="left",
        )
    path = out_dir / "trace0_k1_vs_frs_comparison.csv"
    df.to_csv(path, index=False)
    log.info("Wrote comparison table: %s", path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/analysis_outputs" / "trace0_k1_judging",
    )
    ap.add_argument("--logs-dir", type=Path, default=REPO_ROOT / "logs")
    ap.add_argument("--sample-questions-per-pair", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", type=str, default=None)
    ap.add_argument("--datasets", type=str, default=None)
    ap.add_argument("--max-workers", type=int, default=50)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--overwrite-samples", action="store_true")
    ap.add_argument("--overwrite-judging", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--progress-every", type=int, default=5, help="Log + flush progress every N judgments")
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    out_dir = args.output_dir.resolve()
    logs_dir = args.logs_dir.resolve()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"trace0_k1_judging_{ts}.log"
    log = setup_logging(log_path, args.verbose)
    if args.verbose:
        setup_topk_judge_logging(level=logging.DEBUG, log_file=str(log_path))

    models_f = {m.strip() for m in args.models.split(",") if m.strip()} if args.models else None
    datasets_f = {d.strip() for d in args.datasets.split(",") if d.strip()} if args.datasets else None

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_pair_samples").mkdir(exist_ok=True)
    (out_dir / "judging_checkpoints").mkdir(exist_ok=True)

    pairs = discover_pairs(repo, models_f, datasets_f)
    total_planned = len(pairs) * args.sample_questions_per_pair
    progress_path = out_dir / "progress.json"
    tracker = ProgressTracker(progress_path, total_planned)
    tracker.update(pairs_total=len(pairs), pairs_completed=0)

    log.info("=" * 72)
    log.info("TRACE-0 K=1 JUDGING RUN")
    log.info("=" * 72)
    log.info("Pairs: %d | Questions/pair: %d | Planned judgments: %d", len(pairs), args.sample_questions_per_pair, total_planned)
    log.info("Output: %s", out_dir)
    log.info("Log file: %s", log_path)
    log.info("Live progress: %s", progress_path)
    log.info("Resume=%s overwrite_samples=%s overwrite_judging=%s dry_run=%s workers=%d",
             args.resume, args.overwrite_samples, args.overwrite_judging, args.dry_run, args.max_workers)

    run_meta = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "trace_idx_fixed": TRACE_IDX_FIXED,
        "sample_questions_per_pair": args.sample_questions_per_pair,
        "seed": args.seed,
        "judge_model": DEFAULT_JUDGE_MODEL,
        "pairs": [{"model": m, "dataset": d, "jsonl": p} for m, d, p in pairs],
    }
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    if args.dry_run:
        log.info("DRY RUN — sampling only, no API calls")
    elif not os.environ.get("PORTKEY_API_KEY"):
        log.error("PORTKEY_API_KEY not set. Export it or use --dry-run.")
        return 2

    judge: Optional[Judge] = None
    if not args.dry_run:
        judge = Judge(model=DEFAULT_JUDGE_MODEL)
        log.info("Judge ready: %s", DEFAULT_JUDGE_MODEL)

    pair_rows: List[Dict[str, Any]] = []
    global_cache = 0
    global_new = 0
    global_fail = 0

    for pi, (model, dataset, jpath) in enumerate(pairs):
        benchmark = DATASET_TO_BENCHMARK.get(dataset, dataset)
        slug = f"{_safe_name(model)}__{_safe_name(dataset)}"
        sq_path = out_dir / "per_pair_samples" / f"{slug}_sampled_questions.json"
        st_path = out_dir / "per_pair_samples" / f"{slug}_sampled_traces.jsonl"
        ck_path = out_dir / "judging_checkpoints" / f"trace0_judged_{slug}.json"

        log.info("-" * 72)
        log.info("[%d/%d] PAIR START: %s × %s", pi + 1, len(pairs), model, benchmark)
        log.info("JSONL: %s", jpath)
        tracker.update(
            current_pair={"model": model, "benchmark": benchmark, "pair_index": pi + 1},
            current_pair_progress={"done": 0, "total": 0},
        )

        rows = load_jsonl_rows(jpath)
        by_idx = problems_by_idx(rows)
        rng = pair_rng(model, dataset, args.seed)

        samples: List[SampledTrace] = []
        sampled_ids: List[int] = []

        if args.resume and not args.overwrite_samples and sq_path.is_file() and st_path.is_file():
            log.info("Resuming samples from disk")
            with open(sq_path, encoding="utf-8") as f:
                sq_obj = json.load(f)
            sampled_ids = list(sq_obj.get("sampled_question_ids", []))
            with open(st_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        samples.append(
                            SampledTrace(
                                idx=int(d["idx"]),
                                trace_idx=int(d.get("trace_idx", 0)),
                                question=d.get("question", ""),
                                cot=d.get("cot", ""),
                                gt=d.get("gt", ""),
                                correct=bool(d.get("correct", False)),
                                confidence=float(d.get("confidence", float("nan"))),
                            )
                        )
        else:
            sampled_ids, samples, meta = sample_trace0_questions(
                by_idx, rng, args.sample_questions_per_pair, log
            )
            log.info(
                "Sampled %d questions (eligible trace0=%d)",
                len(samples),
                meta.get("n_distinct_trace0"),
            )
            sq_payload = {
                "model": model,
                "dataset": dataset,
                "benchmark": benchmark,
                "seed": args.seed,
                "trace_idx": TRACE_IDX_FIXED,
                "n_distinct_trace0_eligible": meta.get("n_distinct_trace0"),
                "sampled_question_ids": sampled_ids,
                "jsonl_path": jpath,
            }
            with open(sq_path, "w", encoding="utf-8") as f:
                json.dump(sq_payload, f, indent=2)
            with open(st_path, "w", encoding="utf-8") as f:
                for s in samples:
                    f.write(
                        json.dumps(
                            {
                                "idx": s.idx,
                                "trace_idx": s.trace_idx,
                                "confidence": s.confidence,
                                "correct": s.correct,
                                "question": s.question[:200],
                                "cot_chars": len(s.cot),
                            }
                        )
                        + "\n"
                    )

        n_judge = len(samples)
        tracker.update(current_pair_progress={"done": 0, "total": n_judge})

        if args.dry_run or n_judge == 0:
            log.info("[%d/%d] PAIR END (skip judge): n_sample=%d", pi + 1, len(pairs), n_judge)
            tracker.update(pairs_completed=pi + 1)
            continue

        assert judge is not None
        ck_data = load_checkpoint(ck_path) if args.resume and not args.overwrite_judging else {}
        judged: Dict[str, Any] = ck_data.get("judged_samples", {})
        judge_lock = threading.Lock()
        cache_hits = 0
        new_ok = 0
        new_fail = 0
        done = 0

        def process_one(st: SampledTrace) -> None:
            nonlocal cache_hits, new_ok, new_fail, done
            key = f"{st.idx}:{st.trace_idx}"
            with judge_lock:
                prev = judged.get(key)
                if (
                    prev
                    and prev.get("judge_ok")
                    and prev.get("reasoning_score") is not None
                    and not args.overwrite_judging
                ):
                    cache_hits += 1
                    tracker.increment(cache_hit=True, ok=True)
                    done += 1
                    return
            raw, rs, err = judge_one(judge, st, model, dataset, log)
            ok = raw is not None and rs is not None
            with judge_lock:
                judged[key] = {
                    "idx": st.idx,
                    "trace_idx": st.trace_idx,
                    "confidence": st.confidence,
                    "correct": st.correct,
                    "bin_label": BIN_LABEL,
                    "judge_ok": ok,
                    "reasoning_score": rs,
                    "judge_raw": raw,
                    "error": err,
                }
            if ok:
                new_ok += 1
            else:
                new_fail += 1
            tracker.increment(cache_hit=False, ok=ok)
            done += 1
            if done % args.progress_every == 0 or done == n_judge:
                pct = 100.0 * done / n_judge if n_judge else 0
                log.info(
                    "  %s: %d/%d (%.0f%%) pair_cache=%d pair_new=%d pair_fail=%d | GLOBAL ok=%d fail=%d",
                    slug,
                    done,
                    n_judge,
                    pct,
                    cache_hits,
                    new_ok,
                    new_fail,
                    tracker.state.get("new_ok"),
                    tracker.state.get("new_fail"),
                )
                tracker.update(
                    current_pair_progress={"done": done, "total": n_judge},
                    cache_hits=global_cache + cache_hits,
                    new_ok=global_new + new_ok,
                    new_fail=global_fail + new_fail,
                )
                save_checkpoint(
                    ck_path,
                    {
                        "model": model,
                        "dataset": dataset,
                        "benchmark": benchmark,
                        "jsonl_path": jpath,
                        "bin_label": BIN_LABEL,
                        "judged_samples": judged,
                    },
                    lock=judge_lock,
                )

        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
            futs = [ex.submit(process_one, st) for st in samples]
            for fut in as_completed(futs):
                fut.result()

        save_checkpoint(
            ck_path,
            {
                "model": model,
                "dataset": dataset,
                "benchmark": benchmark,
                "jsonl_path": jpath,
                "bin_label": BIN_LABEL,
                "judged_samples": judged,
            },
            lock=judge_lock,
        )

        global_cache += cache_hits
        global_new += new_ok
        global_fail += new_fail

        scores = [
            float(judged[f"{s.idx}:{s.trace_idx}"]["reasoning_score"])
            for s in samples
            if judged.get(f"{s.idx}:{s.trace_idx}", {}).get("judge_ok")
        ]
        arr = np.asarray(scores, dtype=np.float64) if scores else np.array([])
        mean_rs = float(arr.mean()) if len(arr) else float("nan")
        std_rs = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")

        pair_rows.append(
            {
                "model": model,
                "benchmark": benchmark,
                "dataset": dataset,
                "n_sampled": len(samples),
                "n_judged_ok": len(scores),
                "n_judge_fail": new_fail,
                "cache_hits": cache_hits,
                "new_judge_calls": new_ok,
                "mean_reasoning_score_0_1": mean_rs,
                "mean_reasoning_score_pct": mean_rs * 100 if np.isfinite(mean_rs) else mean_rs,
                "std_reasoning_score_0_1": std_rs,
                "jsonl_path": jpath,
                "checkpoint_path": str(ck_path),
            }
        )

        # Append per-pair row to running CSV
        import pandas as pd

        pd.DataFrame([pair_rows[-1]]).to_csv(
            out_dir / "per_pair_scores_running.csv",
            mode="a",
            header=not (out_dir / "per_pair_scores_running.csv").exists(),
            index=False,
        )

        log.info(
            "[%d/%d] PAIR DONE: %s × %s | judged=%d/%d mean_RS=%.3f cache=%d new=%d fail=%d",
            pi + 1,
            len(pairs),
            model,
            benchmark,
            len(scores),
            len(samples),
            mean_rs if np.isfinite(mean_rs) else -1,
            cache_hits,
            new_ok,
            new_fail,
        )
        tracker.update(
            pairs_completed=pi + 1,
            cache_hits=global_cache,
            new_ok=global_new,
            new_fail=global_fail,
            current_pair=None,
            current_pair_progress=None,
        )

    import pandas as pd

    per_pair_df = pd.DataFrame(pair_rows)
    per_pair_df.to_csv(out_dir / "per_pair_scores.csv", index=False)
    write_comparison_csv(out_dir, pair_rows, log)

    run_meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    run_meta["total_cache_hits"] = global_cache
    run_meta["total_new_judge_calls"] = global_new
    run_meta["total_failures"] = global_fail
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    tracker.update(status="completed", pairs_completed=len(pairs))
    log.info("=" * 72)
    log.info("ALL DONE: pairs=%d cache_hits=%d new_calls=%d failures=%d", len(pairs), global_cache, global_new, global_fail)
    log.info("per_pair_scores.csv | progress.json | log=%s", log_path)
    log.info("=" * 72)
    print(f"\nKey outputs:\n  {out_dir / 'progress.json'}\n  {log_path}\n  {out_dir / 'per_pair_scores.csv'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
