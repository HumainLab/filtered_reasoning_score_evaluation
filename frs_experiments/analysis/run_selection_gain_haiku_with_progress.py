#!/usr/bin/env python3
"""
Held-out selection-gain replication with Claude Haiku 4.5 + live progress.

Wrapper around ``analysis/run_selection_gain_judging.py`` (does not modify it).
Uses direct Anthropic SDK when ``--judge-model`` is a Claude/Haiku id and
``ANTHROPIC_API_KEY`` is set; otherwise attempts Portkey routing.

Usage (after pre-flight passes):
  python -u analysis/run_selection_gain_haiku_with_progress.py \\
    --judge-model claude-haiku-4-5 \\
    --max-workers 100 \\
    --reuse-worklist \\
    --output-dir analysis_outputs/selection_gain_heldout_haiku/ \\
    --cache-dir analysis/cache/selection_gain_judging_haiku/ \\
    --log-file analysis_outputs/selection_gain_heldout_haiku/run.log \\
    2>&1 | tee analysis_outputs/selection_gain_heldout_haiku/terminal.log

Pre-flight only:
  python analysis/run_selection_gain_haiku_with_progress.py --preflight-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
import traceback
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from topk_judge_eval import (  # noqa: E402
    DEFAULT_JUDGE_MODEL,
    DEFAULT_PORTKEY_GATEWAY,
    Judge,
    reasoning_score_from_judge,
)

from analysis.run_selection_gain_judging import (  # noqa: E402
    BIN_LABEL,
    PROMPT_VERSION,
    aggregate_question_pair,
    build_trace_materialization_store,
    cache_key,
    load_worklist_from_csv,
)

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None  # type: ignore

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

HAIKU_INPUT_USD_PER_MTOK = 1.0
HAIKU_OUTPUT_USD_PER_MTOK = 5.0
PUBLISHED_R = 0.4906
MINI_CACHE_DIR = REPO_ROOT / "analysis" / "cache" / "selection_gain_judging"
MINI_PAIR_LEVEL = REPO_ROOT / "analysis" / "selection_gain_pair_level.csv"
PANEL_PATH = REPO_ROOT / "analysis" / "frs_predictor_panel.csv"
DEFAULT_WORKLIST = REPO_ROOT / "analysis" / "selection_gain_worklist.csv"

ANTHROPIC_PORTKEY_GATEWAY = os.environ.get("ANTHROPIC_PORTKEY_MODEL_PREFIX", "")
DEFAULT_HAIKU_JUDGE_MODEL = f"{ANTHROPIC_PORTKEY_GATEWAY}/claude-haiku-4-5"

HAIKU_MODEL_CANDIDATES = [
    DEFAULT_HAIKU_JUDGE_MODEL,
    f"{ANTHROPIC_PORTKEY_GATEWAY}/claude-haiku-4-5-20251001",
    "claude-haiku-4-5",
    "claude-haiku-4-5-20251001",
    f"{DEFAULT_PORTKEY_GATEWAY}/claude-haiku-4-5",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_dual_logger(log_path: Path, verbose: bool = True) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("selection_gain_haiku")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    fh = RotatingFileHandler(log_path, maxBytes=50_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Anthropic Haiku judge (direct SDK; same rubric as mini Judge)
# ---------------------------------------------------------------------------


class AnthropicHaikuJudge:
    """Claude Haiku via Anthropic Messages API; prompt/parse shared with ``Judge``."""

    def __init__(self, model: str, prompt_helper: Optional[Judge] = None):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set (required for direct Anthropic Haiku routing)")
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError("pip install anthropic") from e
        self.model = model
        self.client = Anthropic(api_key=api_key)
        pk = os.environ.get("PORTKEY_API_KEY")
        self._prompt_helper = prompt_helper or Judge(
            model=DEFAULT_JUDGE_MODEL,
            portkey_api_key=pk or "unused",
        )

    def build_prompt(self, *args: Any, **kwargs: Any) -> str:
        return self._prompt_helper.build_prompt(*args, **kwargs)

    def _extract_json(self, raw: str) -> Dict[str, Optional[int]]:
        return self._prompt_helper._extract_json(raw)

    def score(
        self,
        problem: str,
        cot: str,
        gold: str,
        flags_summary: str = "No automated flags available.",
        evidence: Optional[Dict[str, Any]] = None,
        log_ctx: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Optional[int]], Dict[str, int]]:
        if evidence is None:
            evidence = {"final_correct": None}
        prompt = self.build_prompt(problem, cot, gold, flags_summary, evidence)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            temperature=0.0,
            system="You are a careful and consistent evaluator of reasoning quality.",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text if resp.content else ""
        usage = {
            "input_tokens": int(getattr(resp.usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(resp.usage, "output_tokens", 0) or 0),
        }
        return self._extract_json(raw), usage


def is_portkey_model(model: str) -> bool:
    return model.startswith("@")


def is_claude_model(model: str) -> bool:
    m = model.lower()
    return "claude" in m or "haiku" in m


def make_judge(model: str, log: logging.Logger) -> Any:
    if is_portkey_model(model):
        log.info("Using Portkey Judge for model=%s", model)
        return Judge(model=model)
    if is_claude_model(model) and os.environ.get("ANTHROPIC_API_KEY"):
        log.info("Using direct Anthropic SDK for model=%s", model)
        return AnthropicHaikuJudge(model=model)
    if is_claude_model(model):
        raise ValueError(
            f"Claude model {model!r} requires either a Portkey gateway prefix "
            f"(e.g. {DEFAULT_HAIKU_JUDGE_MODEL}) or ANTHROPIC_API_KEY for direct SDK"
        )
    log.info("Using Portkey Judge for model=%s", model)
    return Judge(model=model)


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


def _load_smoke_trace() -> Tuple[str, str, str, bool, Dict[str, Any]]:
    wl = pd.read_csv(DEFAULT_WORKLIST).iloc[0]
    from analysis.run_selection_gain_judging import load_jsonl_rows, materialize_trace, problems_by_idx

    rows = load_jsonl_rows(str(wl["jsonl_path"]))
    row = problems_by_idx(rows)[int(wl["question_id"])]
    prob, cot, gt, corr = materialize_trace(row, int(wl["trace_id"]))
    meta = {
        "model": wl["model"],
        "benchmark": wl["benchmark"],
        "question_id": int(wl["question_id"]),
        "trace_id": int(wl["trace_id"]),
    }
    return prob, cot, gt, corr, meta


def preflight_smoke(log: logging.Logger, model: Optional[str] = None) -> Optional[str]:
    prob, cot, gt, corr, meta = _load_smoke_trace()
    log.info("Pre-flight smoke trace: %s", meta)

    candidates = [model] if model else list(HAIKU_MODEL_CANDIDATES)
    last_err = ""
    for cand in candidates:
        if not cand:
            continue
        log.info("Trying model string: %s", cand)
        try:
            judge = make_judge(cand, log)
            if isinstance(judge, AnthropicHaikuJudge):
                raw, usage = judge.score(
                    problem=prob,
                    cot=cot,
                    gold=gt,
                    evidence={"final_correct": corr},
                    log_ctx={"preflight": True},
                )
                log.info("=== REQUEST (Anthropic) === model=%s meta=%s", cand, meta)
                log.info("=== RESPONSE === raw=%s usage=%s", raw, usage)
            else:
                raw = judge.score(
                    problem=prob,
                    cot=cot,
                    gold=gt,
                    evidence={"final_correct": corr},
                    log_ctx={"preflight": True},
                )
                usage = {}
                log.info("=== REQUEST (Portkey) === model=%s meta=%s", cand, meta)
                log.info("=== RESPONSE === raw=%s", raw)

            rs = reasoning_score_from_judge(raw)
            ok = raw and all(
                isinstance(raw.get(k), int) and 1 <= raw[k] <= 5
                for k in ("faithfulness", "utility", "coherence", "factuality")
            )
            log.info("Smoke parsed reasoning_score=%s valid=%s", rs, ok)
            if ok:
                log.info("PRE-FLIGHT SMOKE PASS: use --judge-model %s", cand)
                return cand
            last_err = f"incomplete scores from {cand}"
        except Exception as e:
            last_err = str(e)
            log.error("Smoke failed for %s: %s", cand, e)
    log.error("PRE-FLIGHT SMOKE FAIL: %s", last_err)
    return None


def preflight_burst(log: logging.Logger, model: str, n: int = 20) -> Dict[str, Any]:
    prob, cot, gt, corr, _ = _load_smoke_trace()
    judge = make_judge(model, log)
    results: List[Tuple[bool, float, str]] = []
    t0 = time.perf_counter()

    def one_call(i: int) -> Tuple[bool, float, str]:
        t_start = time.perf_counter()
        try:
            if isinstance(judge, AnthropicHaikuJudge):
                raw, _ = judge.score(prob[:400], cot[:1500], gt, evidence={"final_correct": corr})
            else:
                raw = judge.score(prob[:400], cot[:1500], gt, evidence={"final_correct": corr})
            ok = reasoning_score_from_judge(raw) is not None
            err = ""
        except Exception as e:
            ok = False
            err = str(e)
        return ok, time.perf_counter() - t_start, err

    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(one_call, i) for i in range(n)]
        for fut in futs:
            results.append(fut.result())

    elapsed = time.perf_counter() - t0
    ok_n = sum(1 for ok, _, _ in results if ok)
    err_n = n - ok_n
    rate_429 = sum(1 for _, _, e in results if "429" in e or "rate" in e.lower())
    latencies = [lat for ok, lat, _ in results if ok]
    mean_lat = float(np.mean(latencies)) if latencies else float("nan")

    if rate_429 == 0 and ok_n == n:
        tier_guess = "Tier 2+ (no 429s at 20 parallel)"
        safe_workers = 100
    elif rate_429 <= 1:
        tier_guess = "Tier 1–2 (≤5% 429 at 20 parallel)"
        safe_workers = 50
    else:
        tier_guess = "Tier 1 or lower (>5% 429 at 20 parallel)"
        safe_workers = 25

    summary = {
        "n_calls": n,
        "ok": ok_n,
        "errors": err_n,
        "rate_limit_429": rate_429,
        "elapsed_s": elapsed,
        "mean_latency_s": mean_lat,
        "tier_guess": tier_guess,
        "max_safe_workers_estimate": safe_workers,
    }
    log.info("BURST TEST: %s", json.dumps(summary, indent=2))
    return summary


def run_preflight(log: logging.Logger, model: Optional[str] = None) -> int:
    log.info("=" * 72)
    log.info("PRE-FLIGHT: Claude Haiku 4.5 routing")
    log.info("PORTKEY_API_KEY set: %s", bool(os.environ.get("PORTKEY_API_KEY")))
    log.info("ANTHROPIC_API_KEY set: %s", bool(os.environ.get("ANTHROPIC_API_KEY")))
    log.info("=" * 72)

    if not os.environ.get("PORTKEY_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("Neither PORTKEY_API_KEY nor ANTHROPIC_API_KEY is set.")
        return 2

    working = preflight_smoke(log, model=model or DEFAULT_HAIKU_JUDGE_MODEL)
    if not working:
        return 2

    burst = preflight_burst(log, working, n=20)
    log.info(
        "Pre-flight complete. Recommended --judge-model %s | safe workers ~%d",
        working,
        burst["max_safe_workers_estimate"],
    )
    return 0


# ---------------------------------------------------------------------------
# Progress + adaptive concurrency
# ---------------------------------------------------------------------------


@dataclass
class RunStats:
    total: int
    completed: int = 0
    success: int = 0
    cache_hits: int = 0
    rate_limit: int = 0
    timeout: int = 0
    other_errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    active_workers: int = 0
    call_counter: int = 0
    latencies_1m: Deque[Tuple[float, float]] = field(default_factory=deque)
    outcomes_1m: Deque[Tuple[float, str]] = field(default_factory=deque)
    lock: threading.Lock = field(default_factory=threading.Lock)
    failed_traces: List[str] = field(default_factory=list)
    pair_done: Dict[Tuple[str, str], Dict[str, Any]] = field(default_factory=dict)
    throttle_events: List[str] = field(default_factory=list)

    def record_latency(self, latency: float) -> None:
        now = time.time()
        with self.lock:
            self.latencies_1m.append((now, latency))
            cutoff = now - 60
            while self.latencies_1m and self.latencies_1m[0][0] < cutoff:
                self.latencies_1m.popleft()

    def record_outcome(self, kind: str) -> None:
        now = time.time()
        with self.lock:
            self.outcomes_1m.append((now, kind))
            cutoff = now - 60
            while self.outcomes_1m and self.outcomes_1m[0][0] < cutoff:
                self.outcomes_1m.popleft()

    def rolling_rate_per_min(self) -> float:
        with self.lock:
            if len(self.latencies_1m) < 2:
                return 0.0
            span = self.latencies_1m[-1][0] - self.latencies_1m[0][0]
            if span <= 0:
                return float(len(self.latencies_1m))
            return len(self.latencies_1m) / span * 60.0

    def rolling_mean_latency(self) -> float:
        with self.lock:
            if not self.latencies_1m:
                return float("nan")
            return float(np.mean([x[1] for x in self.latencies_1m]))

    def rolling_429_rate(self) -> float:
        with self.lock:
            if not self.outcomes_1m:
                return 0.0
            n429 = sum(1 for _, k in self.outcomes_1m if k == "429")
            return n429 / len(self.outcomes_1m)

    def cost_usd(self) -> float:
        with self.lock:
            return (
                self.input_tokens / 1_000_000 * HAIKU_INPUT_USD_PER_MTOK
                + self.output_tokens / 1_000_000 * HAIKU_OUTPUT_USD_PER_MTOK
            )


def classify_error(exc: BaseException) -> str:
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "rate_limit" in msg:
        return "429"
    if isinstance(exc, TimeoutError) or "timeout" in msg or "timed out" in msg:
        return "timeout"
    if "connection" in msg or "connect" in msg:
        return "timeout"
    return "other"


def judge_with_policy(
    judge: Any,
    problem: str,
    cot: str,
    gold: str,
    correct: bool,
    log_ctx: Dict[str, Any],
    stats: RunStats,
    log: logging.Logger,
    error_log_path: Path,
) -> Tuple[Optional[Dict[str, Any]], Optional[float], str, Dict[str, int]]:
    usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
    call_id = 0
    with stats.lock:
        stats.call_counter += 1
        call_id = stats.call_counter

    rate_attempts = 0
    conn_attempts = 0
    last_err = ""

    while True:
        t0 = time.perf_counter()
        try:
            if isinstance(judge, AnthropicHaikuJudge):
                raw, usage = judge.score(
                    problem=str(problem),
                    cot=str(cot),
                    gold=str(gold),
                    evidence={"final_correct": correct},
                    log_ctx=log_ctx,
                )
            else:
                raw = judge.score(
                    problem=str(problem),
                    cot=str(cot),
                    gold=str(gold),
                    evidence={"final_correct": correct},
                    log_ctx=log_ctx,
                )
            dt = time.perf_counter() - t0
            stats.record_latency(dt)
            rs = reasoning_score_from_judge(raw)
            if rs is not None:
                stats.record_outcome("ok")
                with stats.lock:
                    stats.input_tokens += usage.get("input_tokens", 0)
                    stats.output_tokens += usage.get("output_tokens", 0)
                return raw, float(rs), "", usage
            last_err = "incomplete_judge_scores"
            stats.record_outcome("other")
            with stats.lock:
                stats.other_errors += 1
            return None, None, last_err, usage
        except Exception as e:
            dt = time.perf_counter() - t0
            kind = classify_error(e)
            stats.record_outcome(kind)
            last_err = str(e)
            if kind == "429":
                rate_attempts += 1
                with stats.lock:
                    stats.rate_limit += 1
                if rate_attempts > 5:
                    break
                backoff = min(60, 4 * (2 ** (rate_attempts - 1)))
                log.warning(
                    "429 on call #%d, backing off %.0fs, attempt %d/5",
                    call_id,
                    backoff,
                    rate_attempts,
                )
                time.sleep(backoff)
                continue
            if kind == "timeout":
                conn_attempts += 1
                with stats.lock:
                    stats.timeout += 1
                if conn_attempts > 3:
                    break
                backoff = 2 * (2 ** (conn_attempts - 1))
                log.warning("Timeout/connection on call #%d, backoff %.0fs, attempt %d/3", call_id, backoff, conn_attempts)
                time.sleep(backoff)
                continue
            with stats.lock:
                stats.other_errors += 1
            tb = traceback.format_exc()
            log.error("Non-retryable error on call #%d: %s\n%s", call_id, e, tb)
            with open(error_log_path, "a", encoding="utf-8") as ef:
                ef.write(f"\n[{datetime.now(timezone.utc).isoformat()}] call #{call_id}\n{tb}\n")
            break

    return None, None, last_err, usage


def format_eta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "??:??"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def periodic_stats_loop(stats: RunStats, log: logging.Logger, stop: threading.Event, interval: float = 30.0) -> None:
    while not stop.wait(interval):
        with stats.lock:
            done = stats.completed
            total = stats.total
            succ = stats.success
            ch = stats.cache_hits
            rl = stats.rate_limit
            to = stats.timeout
            ot = stats.other_errors
            aw = stats.active_workers
        rate = stats.rolling_rate_per_min()
        lat = stats.rolling_mean_latency()
        sr = succ / max(1, done - ch) if done > ch else 1.0
        log.info(
            "STATS | %d/%d | success_rate=%.1f%% | 429=%d timeout=%d other=%d | "
            "workers=%d | cache_hits=%d | mean_lat_1m=%.2fs | cost=$%.4f",
            done,
            total,
            100 * sr,
            rl,
            to,
            ot,
            aw,
            ch,
            lat if np.isfinite(lat) else -1,
            stats.cost_usd(),
        )


def run_judging_with_progress(
    worklist: List[Dict[str, Any]],
    trace_store: Dict[Any, Dict[str, Any]],
    judge: Any,
    judge_model: str,
    cache_dir: Path,
    max_workers: int,
    stats: RunStats,
    log: logging.Logger,
    error_log_path: Path,
) -> List[Dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    lock = threading.Lock()

    pair_total: Dict[Tuple[str, str], int] = {}
    pair_ok: Dict[Tuple[str, str], int] = {}
    pair_t0: Dict[Tuple[str, str], float] = {}
    pair_cost: Dict[Tuple[str, str], float] = {}
    pair_errors: Dict[Tuple[str, str], List[str]] = {}
    for w in worklist:
        key = (w["model"], w["benchmark"])
        pair_total[key] = pair_total.get(key, 0) + 1
        pair_ok.setdefault(key, 0)
        pair_t0.setdefault(key, time.perf_counter())
        pair_cost.setdefault(key, 0.0)
        pair_errors.setdefault(key, [])

    concurrency_limit = max_workers
    min_workers = 25
    sem = threading.Semaphore(concurrency_limit)

    def maybe_throttle() -> None:
        nonlocal concurrency_limit
        r429 = stats.rolling_429_rate()
        if r429 <= 0.05:
            return
        if concurrency_limit <= min_workers:
            msg = (
                f"RATE LIMIT CRITICAL: 429 rate {100*r429:.1f}% in 1-min window but already at "
                f"floor {min_workers} workers — consider GPT-4.1 or higher Anthropic tier."
            )
            log.error(msg)
            with stats.lock:
                if msg not in stats.throttle_events:
                    stats.throttle_events.append(msg)
            return
        new_limit = max(min_workers, concurrency_limit // 2)
        if new_limit == concurrency_limit:
            return
        msg = f"THROTTLE: 429 rate {100*r429:.1f}% → reducing max workers {concurrency_limit} → {new_limit}"
        log.warning(msg)
        with stats.lock:
            stats.throttle_events.append(msg)
        concurrency_limit = new_limit
        # Semaphore can't shrink; new acquisitions respect lower effective limit via wrapper below.

    effective_limit = [concurrency_limit]
    limit_lock = threading.Lock()
    in_flight_sem = threading.Semaphore(concurrency_limit)

    def acquire_slot() -> None:
        while True:
            with limit_lock:
                limit = effective_limit[0]
            # Spin-y but simple: wait for slot, respect dynamic limit
            in_flight_sem.acquire()
            with stats.lock:
                if stats.active_workers >= limit:
                    stats.active_workers += 1
                else:
                    stats.active_workers += 1
            with limit_lock:
                if stats.active_workers <= effective_limit[0]:
                    return
            stats.active_workers -= 1
            in_flight_sem.release()
            time.sleep(0.05)

    def release_slot() -> None:
        with stats.lock:
            stats.active_workers = max(0, stats.active_workers - 1)
        in_flight_sem.release()

    def one(w: Dict[str, Any]) -> Dict[str, Any]:
        acquire_slot()
        try:
            maybe_throttle()
            with limit_lock:
                effective_limit[0] = concurrency_limit

            model = w["model"]
            benchmark = w["benchmark"]
            qid = int(w["question_id"])
            tid = int(w["trace_id"])
            st = w["selection_type"]
            ck = cache_dir / f"{cache_key(model, benchmark, qid, tid, st, judge_model)}.json"

            if ck.is_file():
                try:
                    with open(ck, encoding="utf-8") as f:
                        cached = json.load(f)
                    if cached.get("reasoning_score") is not None:
                        with stats.lock:
                            stats.cache_hits += 1
                            stats.completed += 1
                            stats.success += 1
                        pair_ok[(model, benchmark)] += 1
                        return {
                            **w,
                            "reasoning_score": cached["reasoning_score"],
                            "judge_ok": True,
                            "raw_json": json.dumps(cached.get("judge_raw"), default=str),
                            "error": "",
                            "from_cache": True,
                        }
                except (json.JSONDecodeError, OSError):
                    pass

            mat_key = (model, benchmark, qid, tid)
            mat = trace_store.get(mat_key)
            if mat is None:
                with stats.lock:
                    stats.completed += 1
                    stats.other_errors += 1
                pair_errors[(model, benchmark)].append(f"q{qid}t{tid}:trace_not_in_store")
                return {**w, "reasoning_score": None, "judge_ok": False, "error": "trace_not_in_store", "from_cache": False}

            log_ctx = {
                "eval_model": model,
                "dataset": w["dataset"],
                "idx": qid,
                "trace_idx": tid,
                "bin": BIN_LABEL,
                "selection": st,
            }
            raw, rs, err, usage = judge_with_policy(
                judge, mat["problem"], mat["cot"], mat["gold"], mat["correct"], log_ctx, stats, log, error_log_path
            )
            ok = rs is not None
            rec = {
                **w,
                "reasoning_score": rs,
                "judge_ok": ok,
                "raw_json": json.dumps(raw, default=str) if raw else "",
                "error": err,
                "from_cache": False,
            }
            with stats.lock:
                stats.completed += 1
                if ok:
                    stats.success += 1
                    stats.input_tokens += usage.get("input_tokens", 0)
                    stats.output_tokens += usage.get("output_tokens", 0)
                else:
                    stats.failed_traces.append(f"{model}|{benchmark}|q{qid}|t{tid}|{st}|{err}")
            if ok:
                pair_ok[(model, benchmark)] += 1
                pair_cost[(model, benchmark)] += (
                    usage.get("input_tokens", 0) / 1_000_000 * HAIKU_INPUT_USD_PER_MTOK
                    + usage.get("output_tokens", 0) / 1_000_000 * HAIKU_OUTPUT_USD_PER_MTOK
                )
                payload = {
                    "reasoning_score": rs,
                    "judge_raw": raw,
                    "saved_utc": datetime.now(timezone.utc).isoformat(),
                    "usage": usage,
                }
                tmp = ck.with_suffix(".tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
                tmp.replace(ck)
            else:
                pair_errors[(model, benchmark)].append(f"q{qid}t{tid}:{err}")
            return rec
        finally:
            release_slot()

    pbar = tqdm(total=stats.total, desc="Judging", unit="call", dynamic_ncols=True) if tqdm else None
    stop_stats = threading.Event()
    stats_thread = threading.Thread(target=periodic_stats_loop, args=(stats, log, stop_stats), daemon=True)
    stats_thread.start()

    t_run = time.perf_counter()
    pair_reported: Set[Tuple[str, str]] = set()

    def check_pair_complete(fut_pair_key: Tuple[str, str]) -> None:
        m, b = fut_pair_key
        if pair_ok[(m, b)] + len(pair_errors[(m, b)]) < pair_total[(m, b)]:
            return
        if (m, b) in pair_reported:
            return
        pair_reported.add((m, b))
        elapsed = time.perf_counter() - pair_t0[(m, b)]
        em = pair_errors[(m, b)]
        err_suffix = f" | errors: {len(em)}" if em else ""
        log.info(
            "[✓] %s × %s complete (%d/%d, %s, $%.2f)%s",
            m,
            b,
            pair_ok[(m, b)],
            pair_total[(m, b)],
            format_eta(elapsed),
            pair_cost[(m, b)],
            err_suffix,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs: Dict[Future, Dict[str, Any]] = {ex.submit(one, w): w for w in worklist}
        while futs:
            done_set, _ = wait(futs, timeout=0.5, return_when=FIRST_COMPLETED)
            for fut in done_set:
                w = futs.pop(fut)
                rec = fut.result()
                results.append(rec)
                if pbar:
                    rate = stats.rolling_rate_per_min()
                    remaining = stats.total - stats.completed
                    eta_s = remaining / (rate / 60) if rate > 0 else float("inf")
                    pbar.set_postfix_str(f"{rate:.0f} calls/min | ETA {format_eta(eta_s)}", refresh=False)
                    pbar.update(1)
                check_pair_complete((w["model"], w["benchmark"]))

    stop_stats.set()
    stats_thread.join(timeout=1)
    if pbar:
        pbar.close()

    log.info(
        "Judging finished in %s | ok=%d cache=%d fail=%d cost=$%.2f",
        format_eta(time.perf_counter() - t_run),
        stats.success,
        stats.cache_hits,
        len(stats.failed_traces),
        stats.cost_usd(),
    )
    if stats.throttle_events:
        log.warning("Throttle events: %s", stats.throttle_events)
    return results


# ---------------------------------------------------------------------------
# Post-run analysis
# ---------------------------------------------------------------------------


def bootstrap_pearson_ci(x: np.ndarray, y: np.ndarray, n: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n_obs = len(x)
    rs = []
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        if scipy_stats is None:
            break
        pr = scipy_stats.pearsonr(x[idx], y[idx])
        rs.append(float(getattr(pr, "statistic", pr[0])))
    if not rs:
        return float("nan"), float("nan"), float("nan")
    r0 = scipy_stats.pearsonr(x, y)
    r_point = float(getattr(r0, "statistic", r0[0]))
    return r_point, float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def load_mini_scores(worklist: List[Dict[str, Any]], mini_model: str = DEFAULT_JUDGE_MODEL) -> pd.DataFrame:
    rows = []
    for w in worklist:
        ck = MINI_CACHE_DIR / f"{cache_key(w['model'], w['benchmark'], int(w['question_id']), int(w['trace_id']), w['selection_type'], mini_model)}.json"
        rs = np.nan
        if ck.is_file():
            try:
                with open(ck, encoding="utf-8") as f:
                    d = json.load(f)
                rs = d.get("reasoning_score")
            except (json.JSONDecodeError, OSError):
                pass
        rows.append({**w, "mini_reasoning_score": rs})
    return pd.DataFrame(rows)


def post_run_analysis(
    judged: List[Dict[str, Any]],
    worklist: List[Dict[str, Any]],
    judge_model: str,
    out_dir: Path,
    stats: RunStats,
    log: logging.Logger,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _, pair_df = aggregate_question_pair(judged, random_draws=1, logger=log)
    if pair_df.empty:
        log.error("Post-run: empty pair-level aggregation")
        return

    pair_path = out_dir / "selection_gain_pair_level_haiku.csv"
    pair_df = pair_df.rename(columns={"mean_selection_gain": "mean_selection_gain_haiku"})
    pair_df.to_csv(pair_path, index=False)
    log.info("Wrote %s", pair_path)

    mini_pairs = pd.read_csv(MINI_PAIR_LEVEL) if MINI_PAIR_LEVEL.is_file() else pd.DataFrame()
    panel = pd.read_csv(PANEL_PATH) if PANEL_PATH.is_file() else pd.DataFrame()
    merged = pair_df.merge(panel[["model", "benchmark", "frs_pct"]], on=["model", "benchmark"], how="inner")

    r, ci_lo, ci_hi = float("nan"), float("nan"), float("nan")
    rho = float("nan")
    if len(merged) >= 5 and scipy_stats is not None:
        x = merged["frs_pct"].values.astype(float)
        y = merged["mean_selection_gain_haiku"].values.astype(float)
        r, ci_lo, ci_hi = bootstrap_pearson_ci(x, y)
        sp = scipy_stats.spearmanr(x, y)
        rho = float(getattr(sp, "statistic", sp[0]))

    trace_df = pd.DataFrame(judged)
    mini_df = load_mini_scores(worklist)
    agree = trace_df.merge(
        mini_df[["model", "benchmark", "question_id", "trace_id", "selection_type", "mini_reasoning_score"]],
        on=["model", "benchmark", "question_id", "trace_id", "selection_type"],
        how="left",
    )
    agree_ok = agree[agree["judge_ok"] == True]  # noqa: E712
    agree_ok = agree_ok[agree_ok["reasoning_score"].notna() & agree_ok["mini_reasoning_score"].notna()]
    pear_trace = spear_trace = float("nan")
    if len(agree_ok) >= 10 and scipy_stats is not None:
        a = agree_ok["mini_reasoning_score"].astype(float).values
        b = agree_ok["reasoning_score"].astype(float).values
        pear_trace = float(getattr(scipy_stats.pearsonr(a, b), "statistic", scipy_stats.pearsonr(a, b)[0]))
        spear_trace = float(getattr(scipy_stats.spearmanr(a, b), "statistic", scipy_stats.spearmanr(a, b)[0]))

    mean_mini = float(mini_pairs["mean_selection_gain"].mean()) if not mini_pairs.empty else float("nan")
    mean_haiku = float(pair_df["mean_selection_gain_haiku"].mean())

    flag = ""
    if np.isfinite(pear_trace) and pear_trace < 0.3:
        flag = "**FLAG:** Per-trace judge agreement < 0.3 — held-out judge may measure something different.\n"

    summary = f"""# Held-out selection-gain (Claude Haiku 4.5)

## Key numbers

| Metric | Value |
|--------|-------|
| Held-out Pearson r (gain_haiku vs frs_pct) | {r:.4f} |
| Bootstrap 95% CI | [{ci_lo:.4f}, {ci_hi:.4f}] |
| Spearman rho (robustness) | {rho:.4f} |
| Published mini-judge r | {PUBLISHED_R:.4f} |
| Per-trace agreement Pearson (mini vs haiku RS) | {pear_trace:.4f} |
| Per-trace agreement Spearman | {spear_trace:.4f} |
| Mean selection gain (mini) | {mean_mini:.4f} |
| Mean selection gain (haiku) | {mean_haiku:.4f} |
| Judge model | `{judge_model}` |
| Total judge calls | {stats.total} |
| API cost (est.) | ${stats.cost_usd():.2f} |
| Failed traces (after retries) | {len(stats.failed_traces)} |

{flag}
## Rebuttal-ready summary

We replicated the selection-gain predictor analysis using a held-out judge (Claude Haiku 4.5,
`{judge_model}`) on the same frozen 5,400-trace worklist used for the mini-judge run.
Pair-level mean selection gain from Haiku correlates with published FRS at r={r:.3f} (95% CI
[{ci_lo:.3f}, {ci_hi:.3f}]), compared to r={PUBLISHED_R:.4f} for the original mini-judge pipeline.
Per-trace reasoning-score agreement between judges is Pearson r={pear_trace:.3f} (Spearman
{spear_trace:.3f}), indicating {"reasonable" if pear_trace >= 0.5 else "moderate" if pear_trace >= 0.3 else "low"} alignment on the same rubric.
Total API cost was approximately ${stats.cost_usd():.2f}; {len(stats.failed_traces)} traces failed after retries.

## Failed traces

"""
    if stats.failed_traces:
        summary += "\n".join(f"- {t}" for t in stats.failed_traces[:200])
        if len(stats.failed_traces) > 200:
            summary += f"\n- ... and {len(stats.failed_traces) - 200} more"
    else:
        summary += "None."

    key_path = out_dir / "key_numbers.md"
    key_path.write_text(summary, encoding="utf-8")
    log.info("Wrote %s", key_path)

    if plt is not None and len(merged) >= 5:
        fig_path = out_dir / "scatter_gain_vs_frs_haiku.png"
        plt.figure(figsize=(6, 5))
        plt.scatter(merged["frs_pct"], merged["mean_selection_gain_haiku"], alpha=0.7)
        plt.xlabel("FRS (%)")
        plt.ylabel("Mean selection gain (Haiku judge)")
        plt.title(f"Selection gain vs FRS (r={r:.3f})")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        log.info("Wrote %s", fig_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--judge-model", type=str, default=DEFAULT_HAIKU_JUDGE_MODEL)
    ap.add_argument("--max-workers", type=int, default=100)
    ap.add_argument("--reuse-worklist", action="store_true")
    ap.add_argument("--worklist-path", type=Path, default=DEFAULT_WORKLIST)
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "analysis_outputs" / "selection_gain_heldout_haiku")
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "analysis" / "cache" / "selection_gain_judging_haiku")
    ap.add_argument("--log-file", type=Path, default=None)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--limit-calls", type=int, default=0, help="Debug: cap API calls (0=all)")
    args = ap.parse_args()

    out_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    log_path = (args.log_file or out_dir / "run.log").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_dual_logger(log_path)

    if args.preflight_only:
        return run_preflight(log, model=args.judge_model)

    if not os.environ.get("PORTKEY_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("PORTKEY_API_KEY or ANTHROPIC_API_KEY required. Run --preflight-only first.")
        return 2
    if (
        is_claude_model(args.judge_model)
        and not is_portkey_model(args.judge_model)
        and not os.environ.get("ANTHROPIC_API_KEY")
    ):
        log.error(
            "Bare Claude model string requires ANTHROPIC_API_KEY, or use Portkey model "
            "e.g. %s",
            DEFAULT_HAIKU_JUDGE_MODEL,
        )
        return 2

    wl_path = args.worklist_path.resolve()
    if not args.reuse_worklist and not wl_path.is_file():
        log.error("Worklist missing; pass --reuse-worklist")
        return 2

    all_work = load_worklist_from_csv(wl_path, log)
    if args.limit_calls:
        all_work = all_work[: args.limit_calls]

    stats = RunStats(total=len(all_work))
    error_log = out_dir / "errors.log"

    log.info("Materializing traces (OOM-safe)...")
    trace_store = build_trace_materialization_store(all_work, log)

    log.info("Starting Haiku judging: %d calls, max_workers=%d, model=%s", len(all_work), args.max_workers, args.judge_model)
    judge = make_judge(args.judge_model, log)

    judged = run_judging_with_progress(
        all_work,
        trace_store,
        judge,
        args.judge_model,
        cache_dir,
        args.max_workers,
        stats,
        log,
        error_log,
    )

    judged_path = out_dir / "selection_gain_judged_traces_haiku.csv"
    pd.DataFrame(judged).to_csv(judged_path, index=False)
    log.info("Wrote %s", judged_path)

    post_run_analysis(judged, all_work, args.judge_model, out_dir, stats, log)

    log.info("=" * 72)
    log.info(
        "FINAL ERROR SUMMARY | 429=%d timeout=%d other=%d failed_traces=%d",
        stats.rate_limit,
        stats.timeout,
        stats.other_errors,
        len(stats.failed_traces),
    )
    log.info("=" * 72)
    return 0 if len(stats.failed_traces) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
