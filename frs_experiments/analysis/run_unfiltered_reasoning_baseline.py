#!/usr/bin/env python3
"""
Unfiltered reasoning-score baseline from the same T=0.7 JSONL pool as FRS.

For each model–dataset pair:
  - Load full pass@16 JSONL (same discovery as FRS / build_downstream_parquets).
  - Group by problem id ``idx``; keep only traces with valid confidence (same pool as
    ``reasoning_confidence_bins.traces_to_dataframe``).
  - Sample min(K, n_questions) distinct questions uniformly without replacement, then
    exactly one trace per question uniformly among valid traces for that question.
  - Judge with the same GPT-4o-mini ``Judge`` + ``reasoning_score_from_judge`` as FRS.

Does **not** modify the existing FRS / reasoning_confidence_bins pipeline.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# Repo root = parent of analysis/
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

# Optional SciPy for rank correlation
try:
    from scipy.stats import pearsonr, spearmanr
except ImportError:
    pearsonr = None  # type: ignore
    spearmanr = None  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORE_BENCHMARKS: Tuple[str, ...] = (
    "GSM8K",
    "MATH500",
    "SVAMP",
    "AQuA",
    "GPQA",
    "CommonsenseQA",
)

DEFAULT_FRS_CSV = REPO_ROOT / "global_pass1_frs_analysis" / "paper_frs_by_benchmark.csv"

BIN_LABEL_UNFILTERED = "unfiltered_baseline"


def _safe_name(s: str) -> str:
    return (
        s.replace("/", "_")
        .replace(" ", "_")
        .replace(".", "_")
        .replace(":", "_")
    )


def valid_trace_indices(row: dict) -> List[int]:
    """Trace indices with non-NaN confidence — same eligibility as FRS trace pool."""
    scores = row.get("score", [])
    code = row.get("code", [])
    probs_all = row.get("chosen_token_probs_per_path", {}).get("epoch_0", [])
    if not isinstance(probs_all, list) or not isinstance(code, list):
        return []
    n = min(len(scores), len(code), len(probs_all))
    out: List[int] = []
    for ti in range(n):
        probs = probs_all[ti] if ti < len(probs_all) else []
        conf = compute_trace_confidence(probs)
        if not np.isnan(conf):
            out.append(ti)
    return out


def load_jsonl_rows(filepath: str) -> List[dict]:
    rows: List[dict] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def problems_by_idx(rows: List[dict]) -> Dict[int, dict]:
    """One row per idx; later lines overwrite earlier if duplicate (defensive)."""
    by_idx: Dict[int, dict] = {}
    for row in rows:
        raw = row.get("idx")
        if raw is None:
            continue
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            continue
        by_idx[idx] = row
    return by_idx


def pair_rng(model: str, dataset: str, base_seed: int) -> np.random.Generator:
    """Deterministic RNG per (model, dataset) from global seed (stable across Python runs)."""
    digest = hashlib.sha256(f"{base_seed}|{model}|{dataset}".encode("utf-8")).hexdigest()
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


def sample_traces_for_pair(
    by_idx: Dict[int, dict],
    rng: np.random.Generator,
    k_questions: int,
    log: logging.Logger,
) -> Tuple[List[int], List[SampledTrace], Dict[str, Any]]:
    """
    Uniform: pick distinct questions uniformly, then one trace uniformly per question.
    Only questions with >=1 valid trace are eligible.
    """
    eligible: List[int] = []
    short_traces: List[int] = []
    for idx, row in sorted(by_idx.items()):
        vti = valid_trace_indices(row)
        if not vti:
            continue
        eligible.append(idx)
        scores = row.get("score", [])
        if len(scores) < 16:
            short_traces.append(idx)

    n_distinct = len(eligible)
    n_sample = min(k_questions, n_distinct)

    if short_traces:
        log.warning(
            "Questions with fewer than 16 raw scores (still sampled if valid traces exist): "
            "count=%d examples=%s",
            len(short_traces),
            short_traces[:10],
        )

    if n_distinct < k_questions:
        log.warning(
            "Only %d distinct questions with valid traces (requested %d); using all %d.",
            n_distinct,
            k_questions,
            n_sample,
        )

    if n_sample == 0:
        return [], [], {"eligible_idx": eligible, "n_distinct": n_distinct, "n_sample": 0}

    chosen_idx = rng.choice(np.array(eligible, dtype=np.int64), size=n_sample, replace=False)
    chosen_idx_list = [int(x) for x in sorted(chosen_idx.tolist())]

    samples: List[SampledTrace] = []
    for idx in sorted(chosen_idx_list):
        row = by_idx[idx]
        vti = valid_trace_indices(row)
        ti = int(rng.choice(vti))
        code = row.get("code", [])
        scores = row.get("score", [])
        pred = row.get("pred", [])
        cot = code[ti] if isinstance(code[ti], str) else str(code[ti])
        gt = row.get("gt", row.get("answer", ""))
        correct = bool(scores[ti]) if ti < len(scores) else False
        samples.append(
            SampledTrace(
                idx=idx,
                trace_idx=ti,
                question=str(row.get("question", "")),
                cot=cot,
                gt=str(gt),
                correct=correct,
            )
        )

    meta = {
        "eligible_idx": eligible,
        "n_distinct": n_distinct,
        "n_sample": n_sample,
        "sampled_question_ids": chosen_idx_list,
    }
    return chosen_idx_list, samples, meta


def judge_one_sample(
    judge: Judge,
    row: SampledTrace,
    model_name: str,
    benchmark_name: str,
    log: logging.Logger,
) -> Tuple[Optional[dict], Optional[float], str]:
    """Single trace through the same judge as FRS; returns (raw_judge_dict, reasoning_score, err)."""
    try:
        raw = judge.score(
            problem=str(row.question),
            cot=str(row.cot),
            gold=str(row.gt),
            flags_summary="No automated flags available.",
            evidence={"final_correct": row.correct},
            log_ctx={
                "eval_model": model_name,
                "dataset": benchmark_name,
                "idx": row.idx,
                "trace_idx": row.trace_idx,
                "bin": BIN_LABEL_UNFILTERED,
            },
        )
        rs = reasoning_score_from_judge(raw)
        return raw, float(rs), ""
    except Exception as e:
        log.error(
            "Judge failed idx=%s trace_idx=%s: %s",
            row.idx,
            row.trace_idx,
            e,
            exc_info=True,
        )
        return None, None, str(e)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_checkpoint(
    path: Path,
    data: Dict[str, Any],
    lock: Optional[threading.Lock] = None,
) -> None:
    """Atomically write JSON checkpoint. If ``lock`` is set, snapshot ``judged_samples`` under it
    so ``json.dump`` never iterates a dict mutated concurrently by worker threads.
    """
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


def setup_logging(log_file: Path, *, verbose: bool = False) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("unfiltered_reasoning_baseline")
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


def discover_pairs(
    repo_root: Path,
    models: Optional[Set[str]],
    datasets: Optional[Set[str]],
) -> List[Tuple[str, str, str]]:
    """Returns sorted list of (model, dataset, jsonl_path)."""
    groups = discover_jsonl_groups(str(repo_root))
    out: List[Tuple[str, str, str]] = []
    for (model, ds), jpath in sorted(groups.items()):
        if ds not in CORE_BENCHMARKS:
            continue
        if models is not None and model not in models:
            continue
        if datasets is not None and ds not in datasets:
            continue
        out.append((model, ds, jpath))
    return out


def load_frs_table(path: Path, log: logging.Logger) -> Dict[str, float]:
    """model -> FRS_Avg from paper CSV."""
    if not path.is_file():
        log.warning("FRS CSV not found at %s; ranking comparison will be incomplete.", path)
        return {}
    out: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row.get("model", "").strip()
            frs = row.get("FRS_Avg", "").strip()
            if not m or not frs:
                continue
            try:
                out[m] = float(frs)
            except ValueError:
                continue
    return out


def rank_dict(values: Dict[str, float], higher_is_better: bool = True) -> Dict[str, int]:
    """Dense rank: 1 = best."""
    items = sorted(values.items(), key=lambda kv: kv[1], reverse=higher_is_better)
    return {k: i + 1 for i, (k, _) in enumerate(items)}


def spearman_corr(
    x: Sequence[float],
    y: Sequence[float],
) -> Tuple[Optional[float], Optional[float]]:
    if len(x) < 2:
        return None, None
    if spearmanr is not None:
        sr = spearmanr(x, y)
        sp = float(getattr(sr, "statistic", sr[0]))
        pe_val: Optional[float] = None
        if pearsonr is not None:
            pr = pearsonr(x, y)
            pe_val = float(getattr(pr, "statistic", pr[0]))
        return sp, pe_val
    # NumPy fallback: rank manually
    rx = np.argsort(np.argsort(np.asarray(x)))
    ry = np.argsort(np.argsort(np.asarray(y)))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None, None
    r = float(np.corrcoef(rx, ry)[0, 1])
    return r, r


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    p.add_argument("--output-dir", type=Path, default=None, help="Default: <repo>/analysis_outputs/unfiltered_reasoning")
    p.add_argument("--logs-dir", type=Path, default=None, help="Default: <repo>/logs")
    p.add_argument("--sample-questions-per-pair", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", type=str, default=None, help="Comma-separated filter")
    p.add_argument("--datasets", type=str, default=None, help="Comma-separated filter")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite-samples", action="store_true")
    p.add_argument("--overwrite-judging", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--frs-csv", type=Path, default=DEFAULT_FRS_CSV)
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Console DEBUG (file is always DEBUG); log lines include thread name for parallel runs.",
    )
    return p


@dataclass
class PairResult:
    model: str
    dataset: str
    n_total_traces: int
    n_distinct_questions: int
    n_sampled_questions: int
    n_judged_traces: int
    mean_reasoning_score: Optional[float]
    std_reasoning_score: Optional[float]
    stderr_reasoning_score: Optional[float]
    cache_hits: int
    new_judge_calls: int
    seed: int
    jsonl_path: str
    sampled_questions_path: str
    sampled_traces_path: str
    checkpoint_path: str


def main() -> int:
    args = build_arg_parser().parse_args()
    repo_root: Path = args.repo_root.resolve()
    out_dir = (args.output_dir or (repo_root / "analysis_outputs" / "unfiltered_reasoning")).resolve()
    logs_dir = (args.logs_dir or (repo_root / "logs")).resolve()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"unfiltered_reasoning_baseline_{ts}.log"
    log = setup_logging(log_path, verbose=args.verbose)
    if args.verbose:
        # Mirror Portkey request/response logs (topk_judge.api) into the same log file.
        setup_topk_judge_logging(level=logging.DEBUG, log_file=str(log_path))

    models_f: Optional[Set[str]] = (
        {m.strip() for m in args.models.split(",") if m.strip()} if args.models else None
    )
    datasets_f: Optional[Set[str]] = (
        {d.strip() for d in args.datasets.split(",") if d.strip()} if args.datasets else None
    )

    run_meta = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "output_dir": str(out_dir),
        "log_file": str(log_path),
        "sample_questions_per_pair": args.sample_questions_per_pair,
        "seed": args.seed,
        "models_filter": sorted(models_f) if models_f else None,
        "datasets_filter": sorted(datasets_f) if datasets_f else None,
        "max_workers": args.max_workers,
        "resume": args.resume,
        "overwrite_samples": args.overwrite_samples,
        "overwrite_judging": args.overwrite_judging,
        "dry_run": args.dry_run,
        "verbose": args.verbose,
        "frs_csv": str(args.frs_csv),
        "judge_model": DEFAULT_JUDGE_MODEL,
        "bin_label": BIN_LABEL_UNFILTERED,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_pair_samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "judging_checkpoints").mkdir(parents=True, exist_ok=True)

    log.info("=== Unfiltered reasoning baseline run started ===")
    log.info("Config: %s", json.dumps(run_meta, indent=2))

    pairs = discover_pairs(repo_root, models_f, datasets_f)
    log.info("Discovered %d model–dataset pairs.", len(pairs))

    frs_by_model = load_frs_table(Path(args.frs_csv), log)

    per_pair_dir = out_dir / "per_pair_samples"
    checkpoint_dir = out_dir / "judging_checkpoints"

    pair_results: List[PairResult] = []
    manifest_pairs: List[Dict[str, Any]] = []

    total_cache_hits = 0
    total_new_calls = 0
    total_sampled = 0
    sampled_question_records: List[Dict[str, Any]] = []

    judge: Optional[Judge] = None
    if not args.dry_run:
        if not os.environ.get("PORTKEY_API_KEY"):
            log.error("PORTKEY_API_KEY is not set; cannot run judging. Use --dry-run to sample only.")
            return 2
        judge = Judge(model=DEFAULT_JUDGE_MODEL)
        log.info(
            "Judge initialized; model=%s | parallel workers per pair=%d "
            "(pairs run one after another; up to %d concurrent judge API calls per pair).",
            DEFAULT_JUDGE_MODEL,
            max(1, args.max_workers),
            max(1, args.max_workers),
        )

    for pi, (model, dataset, jpath) in enumerate(pairs):
        slug = f"{_safe_name(model)}__{_safe_name(dataset)}"
        sq_path = per_pair_dir / f"{slug}_sampled_questions.json"
        st_path = per_pair_dir / f"{slug}_sampled_traces.jsonl"
        ck_path = checkpoint_dir / f"unfiltered_judged_{slug}.json"

        log.info(
            "[%d/%d] Pair start: model=%s dataset=%s jsonl=%s",
            pi + 1,
            len(pairs),
            model,
            dataset,
            jpath,
        )

        rows = load_jsonl_rows(jpath)
        n_lines = len(rows)
        by_idx = problems_by_idx(rows)
        # Total traces in pool (valid confidence) for logging
        total_valid_traces = 0
        for row in by_idx.values():
            total_valid_traces += len(valid_trace_indices(row))

        n_distinct_all = len([i for i in by_idx if valid_trace_indices(by_idx[i])])

        rng = pair_rng(model, dataset, args.seed)
        cache_hits = 0
        new_calls = 0
        sampled_ids: List[int] = []
        samples: List[SampledTrace] = []

        load_samples_from_disk = (
            args.resume
            and not args.overwrite_samples
            and sq_path.is_file()
            and st_path.is_file()
        )

        if load_samples_from_disk:
            log.info("Loading existing samples from disk (resume).")
            with open(sq_path, "r", encoding="utf-8") as f:
                sq_obj = json.load(f)
            sampled_ids = list(sq_obj.get("sampled_question_ids", []))
            with open(st_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    samples.append(
                        SampledTrace(
                            idx=int(d["idx"]),
                            trace_idx=int(d["trace_idx"]),
                            question=d.get("question", ""),
                            cot=d.get("cot", ""),
                            gt=d.get("gt", ""),
                            correct=bool(d.get("correct", False)),
                        )
                    )
            log.info(
                "Pair counts (from resumed manifest): distinct_questions_in_manifest=%d sampled_questions=%d",
                sq_obj.get("n_distinct_questions"),
                len(sampled_ids),
            )
            log.info("Resumed sampled question ids (%d): %s", len(sampled_ids), sampled_ids)
        else:
            if args.resume and not args.overwrite_samples and not (
                sq_path.is_file() and st_path.is_file()
            ):
                log.warning(
                    "Resume requested but sample files missing for %s / %s — drawing new sample.",
                    model,
                    dataset,
                )
            if args.overwrite_samples and (sq_path.is_file() or st_path.is_file()):
                log.warning("overwrite_samples: regenerating samples for %s / %s", model, dataset)
            _, samples, meta = sample_traces_for_pair(
                by_idx, rng, args.sample_questions_per_pair, log
            )
            sampled_ids = meta.get("sampled_question_ids", [])
            log.info(
                "Pair counts: total_jsonl_lines=%d total_valid_traces=%d distinct_questions_with_valid_traces=%d "
                "sampled_questions=%d",
                n_lines,
                total_valid_traces,
                meta.get("n_distinct", 0),
                len(samples),
            )
            log.info("Sampled question ids (%d): %s", len(sampled_ids), sampled_ids)

            sq_payload = {
                "model": model,
                "dataset": dataset,
                "seed": args.seed,
                "n_distinct_questions": meta.get("n_distinct"),
                "n_requested": args.sample_questions_per_pair,
                "n_sampled_questions": len(sampled_ids),
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
                                "question": s.question,
                                "cot": s.cot,
                                "gt": s.gt,
                                "correct": s.correct,
                                "bin_label": BIN_LABEL_UNFILTERED,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            log.info("Wrote samples: %s , %s", sq_path, st_path)

        sampled_question_records.append(
            {
                "model": model,
                "dataset": dataset,
                "jsonl_path": jpath,
                "sampled_question_ids": sampled_ids,
                "n_sampled_questions": len(sampled_ids),
            }
        )

        total_sampled += len(samples)

        if not samples and not args.dry_run:
            log.error(
                "No sampled traces for %s / %s — skipping judging (empty eligible pool or zero sample).",
                model,
                dataset,
            )
            pr = PairResult(
                model=model,
                dataset=dataset,
                n_total_traces=total_valid_traces,
                n_distinct_questions=n_distinct_all,
                n_sampled_questions=0,
                n_judged_traces=0,
                mean_reasoning_score=None,
                std_reasoning_score=None,
                stderr_reasoning_score=None,
                cache_hits=0,
                new_judge_calls=0,
                seed=args.seed,
                jsonl_path=jpath,
                sampled_questions_path=str(sq_path.relative_to(repo_root)),
                sampled_traces_path=str(st_path.relative_to(repo_root)),
                checkpoint_path=str(ck_path.relative_to(repo_root)),
            )
            pair_results.append(pr)
            manifest_pairs.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "jsonl_path": jpath,
                    "sampled_questions": str(sq_path),
                    "sampled_traces": str(st_path),
                }
            )
            log.info("[%d/%d] Pair end (empty sample): %s %s", pi + 1, len(pairs), model, dataset)
            continue

        ck_data = load_checkpoint(ck_path) if not args.dry_run else {}
        if args.overwrite_judging:
            ck_data = {}
            log.warning("overwrite_judging: cleared checkpoint for %s", slug)

        judged_samples: Dict[str, Any] = ck_data.get("judged_samples", {})

        if args.dry_run:
            log.info("[dry-run] Skipping judge for %s", slug)
            pr = PairResult(
                model=model,
                dataset=dataset,
                n_total_traces=total_valid_traces,
                n_distinct_questions=n_distinct_all,
                n_sampled_questions=len(samples),
                n_judged_traces=0,
                mean_reasoning_score=None,
                std_reasoning_score=None,
                stderr_reasoning_score=None,
                cache_hits=0,
                new_judge_calls=0,
                seed=args.seed,
                jsonl_path=jpath,
                sampled_questions_path=str(sq_path.relative_to(repo_root)),
                sampled_traces_path=str(st_path.relative_to(repo_root)),
                checkpoint_path=str(ck_path.relative_to(repo_root)),
            )
            pair_results.append(pr)
            manifest_pairs.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "jsonl_path": jpath,
                    "sampled_questions": str(sq_path),
                    "sampled_traces": str(st_path),
                }
            )
            log.info("[%d/%d] Pair end (dry-run): %s %s", pi + 1, len(pairs), model, dataset)
            continue

        assert judge is not None
        n_to_judge = len(samples)
        done = 0
        progress_every = max(1, min(10, n_to_judge // 10 or 1))
        judge_lock = threading.Lock()

        def process_one(st: SampledTrace) -> Tuple[str, str, Optional[float]]:
            """Return (key, status, score). status: cache_hit | new_ok | new_fail."""
            key = f"{st.idx}:{st.trace_idx}"
            with judge_lock:
                prev = judged_samples.get(key)
                if (
                    prev
                    and prev.get("judge_ok")
                    and isinstance(prev.get("reasoning_score"), (int, float))
                    and not args.overwrite_judging
                ):
                    return key, "cache_hit", float(prev["reasoning_score"])
            raw, rs, err = judge_one_sample(judge, st, model, dataset, log)
            ok = raw is not None and rs is not None
            with judge_lock:
                judged_samples[key] = {
                    "idx": st.idx,
                    "trace_idx": st.trace_idx,
                    "bin_label": BIN_LABEL_UNFILTERED,
                    "judge_ok": ok,
                    "reasoning_score": rs,
                    "judge_raw": raw,
                    "error": err,
                }
            if ok and rs is not None:
                return key, "new_ok", float(rs)
            return key, "new_fail", None

        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
            futures = {ex.submit(process_one, st): st for st in samples}
            for fut in as_completed(futures):
                key, status, rs = fut.result()
                done += 1
                if status == "cache_hit":
                    cache_hits += 1
                elif status == "new_ok":
                    new_calls += 1
                elif status == "new_fail":
                    new_calls += 1
                if done % progress_every == 0 or done == n_to_judge:
                    log.info(
                        "Judging progress %s: %d/%d (cache_hits=%d new_calls=%d)",
                        slug,
                        done,
                        n_to_judge,
                        cache_hits,
                        new_calls,
                    )
                if done % progress_every == 0:
                    save_checkpoint(
                        ck_path,
                        {
                            "model": model,
                            "dataset": dataset,
                            "jsonl_path": jpath,
                            "bin_label": BIN_LABEL_UNFILTERED,
                            "judged_samples": judged_samples,
                        },
                        lock=judge_lock,
                    )

        save_checkpoint(
            ck_path,
            {
                "model": model,
                "dataset": dataset,
                "jsonl_path": jpath,
                "bin_label": BIN_LABEL_UNFILTERED,
                "judged_samples": judged_samples,
            },
            lock=judge_lock,
        )

        total_cache_hits += cache_hits
        total_new_calls += new_calls

        scores_list = []
        for st in samples:
            key = f"{st.idx}:{st.trace_idx}"
            ent = judged_samples.get(key)
            if ent and ent.get("judge_ok") and isinstance(ent.get("reasoning_score"), (int, float)):
                scores_list.append(float(ent["reasoning_score"]))

        arr = np.asarray(scores_list, dtype=np.float64)
        mean_v = float(arr.mean()) if arr.size else None
        std_v = float(arr.std(ddof=1)) if arr.size > 1 else (0.0 if arr.size == 1 else None)
        stderr_v = float(std_v / np.sqrt(len(arr))) if arr.size and std_v is not None else None

        log.info(
            "Pair summary %s: n_judged=%d mean=%s std=%s stderr=%s cache_hits=%d new_calls=%d",
            slug,
            len(scores_list),
            f"{mean_v:.6f}" if mean_v is not None else "nan",
            f"{std_v:.6f}" if std_v is not None else "nan",
            f"{stderr_v:.6f}" if stderr_v is not None else "nan",
            cache_hits,
            new_calls,
        )

        pr = PairResult(
            model=model,
            dataset=dataset,
            n_total_traces=total_valid_traces,
            n_distinct_questions=n_distinct_all,
            n_sampled_questions=len(samples),
            n_judged_traces=len(scores_list),
            mean_reasoning_score=mean_v,
            std_reasoning_score=std_v,
            stderr_reasoning_score=stderr_v,
            cache_hits=cache_hits,
            new_judge_calls=new_calls,
            seed=args.seed,
            jsonl_path=jpath,
            sampled_questions_path=str(sq_path.relative_to(repo_root)),
            sampled_traces_path=str(st_path.relative_to(repo_root)),
            checkpoint_path=str(ck_path.relative_to(repo_root)),
        )
        pair_results.append(pr)
        manifest_pairs.append(
            {
                "model": model,
                "dataset": dataset,
                "jsonl_path": jpath,
                "sampled_questions": str(sq_path),
                "sampled_traces": str(st_path),
                "checkpoint": str(ck_path),
            }
        )

        log.info("[%d/%d] Pair end: %s %s", pi + 1, len(pairs), model, dataset)

    # Manifest
    with open(out_dir / "sampled_pairs_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"pairs": manifest_pairs, "n_pairs": len(manifest_pairs)}, f, indent=2)

    with open(out_dir / "sampled_questions.json", "w", encoding="utf-8") as f:
        json.dump({"pairs": sampled_question_records, "n_pairs": len(sampled_question_records)}, f, indent=2)

    # Per-pair CSV / JSON
    pair_rows = [asdict(x) for x in pair_results]
    with open(out_dir / "per_pair_scores.json", "w", encoding="utf-8") as f:
        json.dump(pair_rows, f, indent=2)

    if pair_rows:
        fieldnames = list(pair_rows[0].keys())
        with open(out_dir / "per_pair_scores.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in pair_rows:
                w.writerow(row)

    # Macro averages per model (mean over datasets in this run)
    by_model: Dict[str, List[float]] = {}
    for pr in pair_results:
        if pr.mean_reasoning_score is None:
            continue
        by_model.setdefault(pr.model, []).append(pr.mean_reasoning_score)

    macro_rows: List[Dict[str, Any]] = []
    for m, vals in sorted(by_model.items()):
        a = np.asarray(vals, dtype=np.float64)
        macro_rows.append(
            {
                "model": m,
                "n_benchmarks": len(vals),
                "unfiltered_macro_mean": float(a.mean()),
                "unfiltered_macro_std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            }
        )

    with open(out_dir / "model_macro_averages.json", "w", encoding="utf-8") as f:
        json.dump(macro_rows, f, indent=2)
    if macro_rows:
        with open(out_dir / "model_macro_averages.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(macro_rows[0].keys()))
            w.writeheader()
            for row in macro_rows:
                w.writerow(row)

    # Ranking vs FRS (ranks computed on models that have both macro unfiltered and FRS_Avg)
    unf_vals = {r["model"]: r["unfiltered_macro_mean"] for r in macro_rows}
    models_both = sorted(m for m in unf_vals if m in frs_by_model)
    unf_sub = {m: unf_vals[m] for m in models_both}
    frs_sub = {m: frs_by_model[m] for m in models_both}
    rank_unf = rank_dict(unf_sub) if unf_sub else {}
    rank_frs = rank_dict(frs_sub) if frs_sub else {}

    ranking_rows: List[Dict[str, Any]] = []
    for m in sorted(set(unf_vals.keys()) | set(frs_by_model.keys())):
        u = unf_vals.get(m)
        f = frs_by_model.get(m)
        ru = rank_unf.get(m)
        rf = rank_frs.get(m)
        ranking_rows.append(
            {
                "model": m,
                "unfiltered_reasoning_avg": u,
                "frs_avg": f,
                "unfiltered_rank": ru,
                "frs_rank": rf,
                "delta_rank": (ru - rf) if ru is not None and rf is not None else None,
            }
        )

    with open(out_dir / "ranking_comparison_vs_frs.json", "w", encoding="utf-8") as f:
        json.dump(ranking_rows, f, indent=2)
    if ranking_rows:
        with open(out_dir / "ranking_comparison_vs_frs.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ranking_rows[0].keys()))
            w.writeheader()
            for row in ranking_rows:
                w.writerow(row)

    # Correlation on models with both metrics
    common = [m for m in models_both]
    xs = [unf_vals[m] for m in common]
    ys = [frs_by_model[m] for m in common]
    sp, pe = spearman_corr(xs, ys)

    run_meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    run_meta["n_pairs_processed"] = len(pair_results)
    run_meta["total_sampled_traces"] = total_sampled
    run_meta["total_cache_hits"] = total_cache_hits
    run_meta["total_new_judge_calls"] = total_new_calls
    run_meta["spearman_unfiltered_vs_frs"] = sp
    run_meta["pearson_unfiltered_vs_frs"] = pe
    run_meta["log_file"] = str(log_path)
    run_meta["artifact_paths"] = {
        "sampled_pairs_manifest": str(out_dir / "sampled_pairs_manifest.json"),
        "sampled_questions_index": str(out_dir / "sampled_questions.json"),
        "per_pair_scores_csv": str(out_dir / "per_pair_scores.csv"),
        "per_pair_scores_json": str(out_dir / "per_pair_scores.json"),
        "model_macro_averages_csv": str(out_dir / "model_macro_averages.csv"),
        "model_macro_averages_json": str(out_dir / "model_macro_averages.json"),
        "ranking_comparison_vs_frs_csv": str(out_dir / "ranking_comparison_vs_frs.csv"),
        "judging_checkpoints_dir": str(out_dir / "judging_checkpoints"),
        "per_pair_samples_dir": str(out_dir / "per_pair_samples"),
    }

    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    # Final report
    log.info("=== Run finished ===")
    print("\n" + "=" * 60)
    print("UNFILTERED REASONING BASELINE — FINAL REPORT")
    print("=" * 60)
    print(f"Model–dataset pairs processed: {len(pair_results)}")
    print(f"Total sampled traces (rows):   {total_sampled}")
    print(f"Total cache hits:              {total_cache_hits}")
    print(f"Total new judge calls:         {total_new_calls}")
    print(f"Log file:                      {log_path}")
    print(f"Per-pair scores CSV:           {out_dir / 'per_pair_scores.csv'}")
    print(f"Ranking vs FRS CSV:            {out_dir / 'ranking_comparison_vs_frs.csv'}")
    print(f"Spearman (unfiltered vs FRS):  {sp}")
    if pe is not None:
        print(f"Pearson (unfiltered vs FRS):   {pe}")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
