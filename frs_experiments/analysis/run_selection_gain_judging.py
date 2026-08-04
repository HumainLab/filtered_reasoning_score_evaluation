#!/usr/bin/env python3
"""
Selection-gain experiment: judge top-confidence vs random trace per question, then
predict pair-level mean gain from FRS and baseline metrics.

Reuses: ``topk_ablation.compute_trace_confidence``, ``topk_judge_eval.Judge``,
``reasoning_score_from_judge`` (same rubric as FRS / unfiltered baseline).

Usage:
  python analysis/run_selection_gain_judging.py --repo-root . --dry-run
  python analysis/run_selection_gain_judging.py --repo-root . --questions-per-pair 50 --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore

try:
    import statsmodels.api as sm
except ImportError:
    sm = None  # type: ignore

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None  # type: ignore

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

CORE_BENCHMARKS: Tuple[str, ...] = (
    "GSM8K",
    "MATH500",
    "SVAMP",
    "AQuA",
    "GPQA",
    "CommonsenseQA",
)

DATASET_TO_BENCHMARK = {
    "GSM8K": "GSM8K",
    "MATH500": "MATH500",
    "SVAMP": "SVAMP",
    "AQuA": "AQuA",
    "GPQA": "GPQA",
    "CommonsenseQA": "CSQA",
}

PROMPT_VERSION = "topk_judge_eval_v1"
POLICY_VERSION = "selection_gain_v1"
BIN_LABEL = "selection_gain"

# --- copied from run_unfiltered_reasoning_baseline (avoid import side effects) ---


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
    digest = hashlib.sha256(f"{base_seed}|{model}|{dataset}".encode("utf-8")).hexdigest()
    seed_int = int(digest[:16], 16) % (2**32)
    return np.random.default_rng((base_seed + seed_int) % (2**32))


def confidence_map_for_question(row: dict) -> Dict[int, float]:
    vti = valid_trace_indices(row)
    probs_all = row.get("chosen_token_probs_per_path", {}).get("epoch_0", [])
    out: Dict[int, float] = {}
    for ti in vti:
        probs = probs_all[ti] if ti < len(probs_all) else []
        c = compute_trace_confidence(probs)
        if not np.isnan(c):
            out[ti] = float(c)
    return out


def pick_top_conf_tie_break(conf_map: Dict[int, float]) -> int:
    best = max(conf_map.values())
    cands = [ti for ti, c in conf_map.items() if c == best]
    return int(min(cands))


def materialize_trace(row: dict, trace_idx: int) -> Tuple[str, str, str, bool]:
    code = row.get("code", [])
    scores = row.get("score", [])
    cot = code[trace_idx] if trace_idx < len(code) else ""
    if not isinstance(cot, str):
        cot = str(cot)
    gt = row.get("gt", row.get("answer", ""))
    correct = bool(scores[trace_idx]) if trace_idx < len(scores) else False
    q = str(row.get("question", ""))
    return q, cot, str(gt), correct


def cache_key(
    model: str,
    benchmark: str,
    qid: int,
    trace_idx: int,
    selection_type: str,
    judge_model: str,
) -> str:
    h = hashlib.sha256(
        f"{model}|{benchmark}|{qid}|{trace_idx}|{selection_type}|{judge_model}|{PROMPT_VERSION}".encode(
            "utf-8"
        )
    ).hexdigest()
    return h[:32]


def setup_file_logger(log_path: Path, verbose: bool) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("selection_gain")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(threadName)-12s | %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def judge_with_retry(
    judge: Judge,
    problem: str,
    cot: str,
    gold: str,
    correct: bool,
    log_ctx: Dict[str, Any],
    logger: logging.Logger,
    max_retries: int = 4,
) -> Tuple[Optional[Dict[str, Any]], Optional[float], str]:
    err = ""
    for attempt in range(max_retries):
        try:
            raw = judge.score(
                problem=str(problem),
                cot=str(cot),
                gold=str(gold),
                flags_summary="No automated flags available.",
                evidence={"final_correct": correct},
                log_ctx=log_ctx,
            )
            rs = reasoning_score_from_judge(raw)
            if rs is not None:
                return raw, float(rs), ""
            err = "incomplete_judge_scores"
        except Exception as e:
            err = str(e)
            logger.warning("Judge attempt %d failed: %s", attempt + 1, e)
            time.sleep(min(2**attempt, 30))
    return None, None, err


def build_worklist_for_pair(
    model: str,
    dataset: str,
    benchmark: str,
    jsonl_path: str,
    rng: np.random.Generator,
    questions_per_pair: int,
    min_traces_per_question: int,
    random_draws: int,
    exclude_top_from_random: bool,
    seed: int,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = load_jsonl_rows(jsonl_path)
    by_idx = problems_by_idx(rows)
    eligible: List[int] = []
    short_eligible = 0
    for idx in sorted(by_idx.keys()):
        cm = confidence_map_for_question(by_idx[idx])
        if len(cm) >= min_traces_per_question:
            eligible.append(idx)
        elif len(cm) >= 1:
            short_eligible += 1

    n_eligible = len(eligible)
    n_take = min(questions_per_pair, n_eligible)
    meta: Dict[str, Any] = {
        "n_lines_jsonl": len(rows),
        "n_distinct_questions": len(by_idx),
        "n_eligible_questions_ge_min_traces": n_eligible,
        "n_shortfall_questions": short_eligible,
        "min_traces_per_question": min_traces_per_question,
        "n_requested": questions_per_pair,
        "n_sampled_questions": n_take,
    }
    if n_take == 0:
        logger.warning("No eligible questions for %s %s", model, dataset)
        return [], meta

    chosen = rng.choice(np.array(eligible, dtype=np.int64), size=n_take, replace=False)
    chosen_list = sorted(int(x) for x in chosen.tolist())

    work: List[Dict[str, Any]] = []
    collisions = 0
    tie_events = 0

    for qid in chosen_list:
        row = by_idx[qid]
        cm = confidence_map_for_question(row)
        top_ti = pick_top_conf_tie_break(cm)
        best_c = cm[top_ti]
        n_tie = sum(1 for c in cm.values() if c == best_c)
        if n_tie > 1:
            tie_events += 1

        pool = list(cm.keys())
        if exclude_top_from_random and len(pool) > 1:
            pool_r = [ti for ti in pool if ti != top_ti]
        else:
            pool_r = pool

        if not pool_r:
            logger.warning("Empty random pool qid=%s model=%s — skipping question", qid, model)
            continue

        work.append(
            {
                "model": model,
                "dataset": dataset,
                "benchmark": benchmark,
                "jsonl_path": jsonl_path,
                "question_id": qid,
                "trace_id": top_ti,
                "selection_type": "top_conf",
                "confidence": cm[top_ti],
                "correct": materialize_trace(row, top_ti)[3],
                "random_draw_index": np.nan,
                "exclude_top_from_random": exclude_top_from_random,
                "seed": seed,
                "policy_version": POLICY_VERSION,
            }
        )

        for rd in range(random_draws):
            rti = int(rng.choice(pool_r))
            if rti == top_ti:
                collisions += 1
            work.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "benchmark": benchmark,
                    "jsonl_path": jsonl_path,
                    "question_id": qid,
                    "trace_id": rti,
                    "selection_type": "random",
                    "confidence": cm[rti],
                    "correct": materialize_trace(row, rti)[3],
                    "random_draw_index": rd,
                    "exclude_top_from_random": exclude_top_from_random,
                    "seed": seed,
                    "policy_version": POLICY_VERSION,
                }
            )

    meta["n_tie_breaks_at_top_conf"] = tie_events
    meta["n_top_equals_random_draws"] = collisions
    meta["n_worklist_rows"] = len(work)
    return work, meta


TraceMatKey = Tuple[str, str, int, int]


def build_trace_materialization_store(
    worklist: List[Dict[str, Any]],
    logger: logging.Logger,
) -> Dict[TraceMatKey, Dict[str, Any]]:
    """
    Load one JSONL at a time; extract only (question_id, trace_id) pairs needed for judging.
    Keys: (model, benchmark, question_id, trace_id) -> {problem, cot, gold, correct}.
    """
    by_path: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for w in worklist:
        by_path[str(w["jsonl_path"])].append(w)

    store: Dict[TraceMatKey, Dict[str, Any]] = {}
    n_paths = len(by_path)
    logger.info(
        "Materializing traces: %d unique jsonl paths, %d worklist rows (one file at a time, OOM-safe)",
        n_paths,
        len(worklist),
    )
    t0 = time.perf_counter()
    for jp in tqdm(sorted(by_path.keys()), desc="JSONL→trace store", unit="file", disable=tqdm is None):
        group = by_path[jp]
        model = str(group[0]["model"])
        benchmark = str(group[0]["benchmark"])
        needed = {(int(w["question_id"]), int(w["trace_id"])) for w in group}
        raw_rows = load_jsonl_rows(jp)
        by_idx = problems_by_idx(raw_rows)
        del raw_rows
        for qid, tid in needed:
            key: TraceMatKey = (model, benchmark, qid, tid)
            if key in store:
                continue
            row = by_idx.get(qid)
            if row is None:
                logger.error("Missing question idx=%s in jsonl=%s", qid, jp[-80:])
                continue
            prob, cot, gt, corr = materialize_trace(row, tid)
            store[key] = {"problem": prob, "cot": cot, "gold": gt, "correct": bool(corr)}
        del by_idx

    logger.info(
        "Trace store built: %d unique (model,benchmark,qid,tid) entries in %.2fs",
        len(store),
        time.perf_counter() - t0,
    )
    return store


def load_worklist_from_csv(path: Path, logger: logging.Logger) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Worklist not found: {path}")
    df = pd.read_csv(path)
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        d = {k: r[k] for k in df.columns}
        d["model"] = str(d["model"])
        d["dataset"] = str(d["dataset"])
        d["benchmark"] = str(d["benchmark"])
        d["jsonl_path"] = str(d["jsonl_path"])
        d["question_id"] = int(d["question_id"])
        d["trace_id"] = int(d["trace_id"])
        rd = d.get("random_draw_index")
        if rd is not None and not (isinstance(rd, float) and np.isnan(rd)):
            try:
                d["random_draw_index"] = int(rd)
            except (TypeError, ValueError):
                d["random_draw_index"] = np.nan
        d["confidence"] = float(d["confidence"]) if pd.notna(d.get("confidence")) else d.get("confidence")
        d["correct"] = bool(d["correct"]) if isinstance(d.get("correct"), (bool, np.bool_)) else d.get("correct")
        rows.append(d)
    logger.info("Loaded worklist from %s rows=%d", path, len(rows))
    return rows


def run_judging_phase(
    worklist: List[Dict[str, Any]],
    trace_store: Dict[TraceMatKey, Dict[str, Any]],
    judge: Optional[Judge],
    judge_model: str,
    cache_dir: Path,
    max_workers: int,
    resume: bool,
    reuse_cache: bool,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    lock = threading.Lock()
    stats = {"ok": 0, "fail": 0, "cache_hit": 0}

    def one(w: Dict[str, Any]) -> Dict[str, Any]:
        model = w["model"]
        benchmark = w["benchmark"]
        qid = int(w["question_id"])
        tid = int(w["trace_id"])
        st = w["selection_type"]
        ck = cache_dir / f"{cache_key(model, benchmark, qid, tid, st, judge_model)}.json"

        if reuse_cache and ck.is_file():
            try:
                with open(ck, encoding="utf-8") as f:
                    cached = json.load(f)
                if cached.get("reasoning_score") is not None:
                    with lock:
                        stats["cache_hit"] += 1
                    return {
                        **w,
                        "reasoning_score": cached["reasoning_score"],
                        "judge_ok": True,
                        "raw_json": json.dumps(cached.get("judge_raw"), default=str) if cached.get("judge_raw") else "",
                        "error": "",
                        "from_cache": True,
                    }
            except (json.JSONDecodeError, OSError):
                pass

        if judge is None:
            return {**w, "reasoning_score": None, "judge_ok": False, "error": "no_judge", "from_cache": False}

        mat_key: TraceMatKey = (model, benchmark, qid, tid)
        mat = trace_store.get(mat_key)
        if mat is None:
            return {
                **w,
                "reasoning_score": None,
                "judge_ok": False,
                "error": "trace_not_in_store",
                "from_cache": False,
            }

        prob = mat["problem"]
        cot = mat["cot"]
        gt = mat["gold"]
        corr = mat["correct"]
        log_ctx = {
            "eval_model": model,
            "dataset": w["dataset"],
            "idx": qid,
            "trace_idx": tid,
            "bin": BIN_LABEL,
            "selection": st,
        }
        raw, rs, err = judge_with_retry(
            judge, prob, cot, gt, corr, log_ctx, logger
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
        if ok:
            with lock:
                stats["ok"] += 1
            payload = {
                "reasoning_score": rs,
                "judge_raw": raw,
                "saved_utc": datetime.now(timezone.utc).isoformat(),
            }
            tmp = ck.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
            tmp.replace(ck)
        else:
            with lock:
                stats["fail"] += 1
        return rec

    iterator = worklist
    if tqdm:
        iterator = tqdm(worklist, desc="Judge calls", unit="call")

    if max_workers <= 1:
        for w in iterator:
            results.append(one(w))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(one, w): w for w in worklist}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Judge calls", disable=tqdm is None):
                results.append(fut.result())

    logger.info(
        "Judging stats: cache_hits=%d ok=%d fail=%d (worklist=%d)",
        stats["cache_hit"],
        stats["ok"],
        stats["fail"],
        len(worklist),
    )
    return results


def aggregate_question_pair(
    judged: List[Dict[str, Any]],
    random_draws: int,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(judged)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    n0 = len(df)
    if "judge_ok" in df.columns:
        df = df[df["judge_ok"] == True]
    df = df[df["reasoning_score"].notna()]
    logger.info("Aggregate: judged rows=%d after judge_ok+score=%d dropped=%d", n0, len(df), n0 - len(df))
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    qrows: List[Dict[str, Any]] = []
    for (model, ds, bench, qid), g in df.groupby(["model", "dataset", "benchmark", "question_id"]):
        top = g[g["selection_type"] == "top_conf"]
        rnd = g[g["selection_type"] == "random"]
        if top.empty or rnd.empty:
            continue
        rs_top = top["reasoning_score"].dropna()
        rs_r = rnd["reasoning_score"].dropna()
        if len(rs_top) < 1 or len(rs_r) < 1:
            continue
        m_top = float(rs_top.iloc[0])
        m_rnd = float(rs_r.mean())
        qrows.append(
            {
                "model": model,
                "dataset": ds,
                "benchmark": bench,
                "question_id": int(qid),
                "reasoning_score_top_conf": m_top,
                "mean_reasoning_score_random": m_rnd,
                "selection_gain_question": m_top - m_rnd,
                "n_random_valid": len(rs_r),
                "n_top_valid": len(rs_top),
            }
        )

    qdf = pd.DataFrame(qrows)
    if qdf.empty:
        return qdf, pd.DataFrame()

    pairs: List[Dict[str, Any]] = []
    for (model, bench), g in qdf.groupby(["model", "benchmark"]):
        gains = g["selection_gain_question"].values.astype(float)
        pairs.append(
            {
                "model": model,
                "benchmark": bench,
                "mean_selection_gain": float(np.mean(gains)),
                "std_selection_gain": float(np.std(gains)),
                "stderr_selection_gain": float(np.std(gains) / np.sqrt(len(gains))) if len(gains) else np.nan,
                "n_questions": len(g),
                "mean_top_conf_reasoning": float(g["reasoning_score_top_conf"].mean()),
                "mean_random_reasoning": float(g["mean_reasoning_score_random"].mean()),
            }
        )

    pdf = pd.DataFrame(pairs)
    return qdf, pdf


def loco_selection_gain_loco(
    m: pd.DataFrame,
    out_dir: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    if sm is None or len(m) < 8:
        return pd.DataFrame()
    base = ["pass1_pct", "high_conf_accuracy_pct", "pass16_pct", "snr", "unfiltered_reasoning_mean"]
    rows: List[Dict[str, Any]] = []
    for group_col, label in [("benchmark", "LODO-benchmark"), ("model", "LOMO-model")]:
        groups = sorted(m[group_col].dropna().unique())
        for g in groups:
            train = m[m[group_col] != g].dropna(subset=["mean_selection_gain"] + base + ["frs_pct"])
            test = m[m[group_col] == g].dropna(subset=["mean_selection_gain"] + base + ["frs_pct"])
            if len(train) < len(base) + 3 or len(test) < 2:
                continue
            y_tr = train["mean_selection_gain"].values.astype(float)
            y_te = test["mean_selection_gain"].values.astype(float)
            Xb_tr = train[base].values.astype(float)
            Xb_te = test[base].values.astype(float)
            f_tr = train["frs_pct"].values.astype(float)
            f_te = test["frs_pct"].values.astype(float)
            X4_tr = sm.add_constant(Xb_tr, has_constant="add")
            X5_tr = sm.add_constant(np.column_stack([Xb_tr, f_tr]), has_constant="add")
            X4_te = sm.add_constant(Xb_te, has_constant="add")
            X5_te = sm.add_constant(np.column_stack([Xb_te, f_te]), has_constant="add")
            fit4 = sm.OLS(y_tr, X4_tr).fit()
            fit5 = sm.OLS(y_tr, X5_tr).fit()
            p4 = np.asarray(fit4.params, dtype=float).ravel()
            p5 = np.asarray(fit5.params, dtype=float).ravel()
            pred4 = X4_te @ p4
            pred5 = X5_te @ p5
            ss_tot = np.sum((y_te - np.mean(y_te)) ** 2)
            r2_te4 = 1 - np.sum((y_te - pred4) ** 2) / ss_tot if ss_tot > 0 else float("nan")
            r2_te5 = 1 - np.sum((y_te - pred5) ** 2) / ss_tot if ss_tot > 0 else float("nan")
            rows.append(
                {
                    "fold_type": label,
                    "held_out_group": g,
                    "n_train": len(train),
                    "n_test": len(test),
                    "r2_test_m4": r2_te4,
                    "r2_test_m5": r2_te5,
                    "delta_r2_test_add_frs": r2_te5 - r2_te4,
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        pth = out_dir / "selection_gain_loco_generalization.csv"
        out.to_csv(pth, index=False)
        logger.info("Wrote %s (%d folds)", pth, len(out))
    return out


def predictor_analysis(
    pair_level: pd.DataFrame,
    panel_path: Path,
    out_dir: Path,
    logger: logging.Logger,
) -> None:
    if pair_level.empty or not panel_path.is_file():
        logger.warning("Skipping predictor analysis: empty pair level or missing panel")
        return
    panel = pd.read_csv(panel_path)
    need = [
        "model",
        "benchmark",
        "frs_pct",
        "pass1_pct",
        "pass16_pct",
        "high_conf_accuracy_pct",
        "unfiltered_reasoning_mean",
        "snr",
    ]
    miss = [c for c in need if c not in panel.columns]
    if miss:
        logger.error("Panel missing columns: %s", miss)
        return
    m = pair_level.merge(panel[need], on=["model", "benchmark"], how="inner")
    logger.info("Predictor merge: pair_level=%d merged=%d dropped=%d", len(pair_level), len(m), len(pair_level) - len(m))
    m.to_csv(out_dir / "selection_gain_predictor_panel_merged.csv", index=False)
    loco_selection_gain_loco(m, out_dir, logger)

    y = m["mean_selection_gain"].values.astype(float)
    preds = ["frs_pct", "pass1_pct", "pass16_pct", "high_conf_accuracy_pct", "unfiltered_reasoning_mean", "snr"]
    rows_corr = []
    for p in preds:
        if p not in m.columns:
            continue
        x = m[p].values.astype(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 5 or scipy_stats is None:
            continue
        pr = scipy_stats.pearsonr(y[ok], x[ok])
        sp = scipy_stats.spearmanr(y[ok], x[ok])
        rows_corr.append(
            {
                "predictor": p,
                "pearson_r": float(getattr(pr, "statistic", pr[0])),
                "pearson_p": float(getattr(pr, "pvalue", pr[1])),
                "spearman_rho": float(getattr(sp, "statistic", sp[0])),
                "spearman_p": float(getattr(sp, "pvalue", sp[1])),
                "n": int(ok.sum()),
            }
        )
    pd.DataFrame(rows_corr).to_csv(out_dir / "selection_gain_predictor_results.csv", index=False)

    base = ["pass1_pct", "high_conf_accuracy_pct", "pass16_pct", "snr", "unfiltered_reasoning_mean"]
    if sm is not None:
        mask = np.ones(len(m), dtype=bool)
        for c in base + ["frs_pct", "mean_selection_gain"]:
            if c in m.columns:
                mask &= m[c].notna().values
        m2 = m.loc[mask]
        y2 = m2["mean_selection_gain"].values.astype(float)
        Xb = m2[base].values.astype(float)
        f = m2["frs_pct"].values.astype(float)
        X4 = sm.add_constant(Xb, has_constant="add")
        X5 = sm.add_constant(np.column_stack([Xb, f]), has_constant="add")
        r4 = sm.OLS(y2, X4).fit()
        r5 = sm.OLS(y2, X5).fit()
        reg_rows = [
            {
                "model": "m4_bases",
                "r2": r4.rsquared,
                "r2_adj": r4.rsquared_adj,
                "n": int(r4.nobs),
                "delta_r2_add_frs": float(r5.rsquared - r4.rsquared),
            },
            {
                "model": "m5_plus_frs",
                "r2": r5.rsquared,
                "r2_adj": r5.rsquared_adj,
                "n": int(r5.nobs),
            },
        ]
        pd.DataFrame(reg_rows).to_csv(out_dir / "selection_gain_regression_incremental.csv", index=False)

    if plt is not None and len(m) >= 5 and "frs_pct" in m.columns:
        plt.figure(figsize=(6, 5))
        plt.scatter(m["frs_pct"], m["mean_selection_gain"], alpha=0.7)
        plt.xlabel("FRS (%)")
        plt.ylabel("Mean selection gain")
        plt.title("Selection gain vs FRS")
        plt.tight_layout()
        plt.savefig(out_dir / "figures" / "selection_gain" / "scatter_gain_vs_frs.png", dpi=150)
        plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--questions-per-pair", type=int, default=50)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--min-traces-per-question", type=int, default=2)
    ap.add_argument("--random-draws-per-question", type=int, default=1)
    ap.add_argument("--exclude-top-from-random", action="store_true", help="Random trace never equals top-conf trace (when ≥2 traces).")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reuse-cache", action="store_true", help="Skip API if per-trace cache file exists.")
    ap.add_argument(
        "--reuse-worklist",
        action="store_true",
        help="Load worklist from CSV (default: analysis/selection_gain_worklist.csv); skip STEP 1–2 rebuild.",
    )
    ap.add_argument(
        "--worklist-path",
        type=Path,
        default=None,
        help="With --reuse-worklist, path to CSV (default: <repo>/analysis/selection_gain_worklist.csv).",
    )
    ap.add_argument("--only-worklist", action="store_true", help="Build worklist + summary then exit (no judge).")
    ap.add_argument("--limit-pairs", type=int, default=0)
    ap.add_argument("--limit-questions", type=int, default=0)
    ap.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--models", type=str, default=None)
    ap.add_argument("--datasets", type=str, default=None)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    out_dir = repo / "analysis"
    fig_dir = out_dir / "figures" / "selection_gain"
    cache_dir = out_dir / "cache" / "selection_gain_judging"
    log_dir = out_dir / "logs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"selection_gain_{ts}.log"
    logger = setup_file_logger(log_path, args.verbose)
    if args.verbose:
        setup_topk_judge_logging(level=logging.DEBUG, log_file=str(log_path))

    log_header = lambda t: (logger.info("=" * 72), logger.info(t), logger.info("=" * 72))
    t0 = time.perf_counter()

    all_work: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    pairs: List[Tuple[str, str, str]] = []
    qpp = args.questions_per_pair
    if args.limit_questions:
        qpp = min(qpp, args.limit_questions)

    if args.reuse_worklist:
        wl_path = (args.worklist_path if args.worklist_path is not None else out_dir / "selection_gain_worklist.csv").resolve()
        all_work = load_worklist_from_csv(wl_path, logger)
        work_df = pd.DataFrame(all_work)
        md_path = out_dir / "selection_gain_run_metadata.json"
        if md_path.is_file():
            with open(md_path, encoding="utf-8") as f:
                run_md = json.load(f)
            qpp = int(run_md.get("questions_per_pair", qpp))
        else:
            qpp = int(work_df.groupby(["model", "dataset"])["question_id"].nunique().max())
        pairs_unique = work_df[["model", "dataset", "jsonl_path"]].drop_duplicates()
        pairs = [tuple(row) for row in pairs_unique.values]
        logger.info(
            "Skipped STEP 1–2 (--reuse-worklist): %s | rows=%d | unique_pairs=%d | questions_per_pair=%d",
            wl_path,
            len(all_work),
            len(pairs),
            qpp,
        )
        if args.only_worklist:
            logger.info("Stopping (--only-worklist): worklist loaded from disk; nothing to rebuild.")
            return 0
    else:
        log_header("STEP 1 — Audit + enumeration")

        groups = discover_jsonl_groups(str(repo))
        for (model, ds), jpath in sorted(groups.items()):
            if ds not in CORE_BENCHMARKS:
                continue
            if args.models and model not in {x.strip() for x in args.models.split(",") if x.strip()}:
                continue
            if args.datasets and ds not in {x.strip() for x in args.datasets.split(",") if x.strip()}:
                continue
            pairs.append((model, ds, jpath))

        if args.limit_pairs:
            pairs = pairs[: args.limit_pairs]

        est_work_rows = len(pairs) * qpp * (1 + args.random_draws_per_question)
        est_judge_calls = est_work_rows if not args.dry_run else 0

        logger.info(
            "Discovered pairs=%d | questions_per_pair=%d | random_draws=%d | est_worklist_rows≈%d | est_judge_calls≈%d",
            len(pairs),
            qpp,
            args.random_draws_per_question,
            est_work_rows,
            est_judge_calls,
        )
        logger.info(
            "Confidence: topk_ablation.compute_trace_confidence (mean of lowest 10%% token probs per trace). "
            "Ties on max confidence → smallest trace_idx."
        )
        logger.info("Judge: %s | prompt_version=%s", args.judge_model, PROMPT_VERSION)

        log_header("STEP 2 — Build worklist")

        for model, dataset, jpath in tqdm(pairs, desc="Pairs", disable=tqdm is None):
            bench = DATASET_TO_BENCHMARK.get(dataset, dataset)
            rng = pair_rng(model, dataset, args.random_seed)
            wl, meta = build_worklist_for_pair(
                model,
                dataset,
                bench,
                jpath,
                rng,
                qpp,
                args.min_traces_per_question,
                args.random_draws_per_question,
                args.exclude_top_from_random,
                args.random_seed,
                logger,
            )
            meta.update({"model": model, "dataset": dataset, "benchmark": bench, "jsonl_path": jpath})
            summaries.append(meta)
            for w in wl:
                all_work.append(w)

        work_df = pd.DataFrame(all_work)
        work_path = out_dir / "selection_gain_worklist.csv"
        work_df.to_csv(work_path, index=False)
        logger.info("Wrote %s rows=%d", work_path, len(work_df))

        sum_df = pd.DataFrame(summaries)
        sum_path = out_dir / "selection_gain_worklist_summary.csv"
        sum_df.to_csv(sum_path, index=False)
        logger.info("Wrote %s", sum_path)

        meta = {
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "pairs": len(pairs),
            "questions_per_pair": qpp,
            "random_seed": args.random_seed,
            "random_draws_per_question": args.random_draws_per_question,
            "exclude_top_from_random": args.exclude_top_from_random,
            "min_traces_per_question": args.min_traces_per_question,
            "judge_model": args.judge_model,
            "prompt_version": PROMPT_VERSION,
            "policy_version": POLICY_VERSION,
            "worklist_rows": len(work_df),
            "est_judge_calls": len(work_df),
            "dry_run": bool(args.dry_run or args.only_worklist),
            "log_file": str(log_path),
        }
        with open(out_dir / "selection_gain_run_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info("Wrote selection_gain_run_metadata.json")

    if args.reuse_worklist:
        logger.info(
            "Confidence: topk_ablation.compute_trace_confidence (mean of lowest 10%% token probs per trace). "
            "Ties on max confidence → smallest trace_idx."
        )
        logger.info("Judge: %s | prompt_version=%s", args.judge_model, PROMPT_VERSION)

    if args.only_worklist or args.dry_run:
        stub = out_dir / "selection_gain_followup_report.md"
        with open(stub, "w", encoding="utf-8") as f:
            f.write("# Selection-gain experiment (worklist only)\n\n")
            f.write(f"- Worklist rows: {len(work_df)}\n")
            f.write(f"- Pairs: {len(pairs)}, questions/pair cap: {qpp}\n")
            f.write(f"- See `selection_gain_worklist.csv` and `selection_gain_worklist_summary.csv`.\n")
            f.write("- Run without `--dry-run` to execute judge calls (requires PORTKEY_API_KEY).\n")
        logger.info("Wrote %s", stub)
        logger.info("Dry-run / only-worklist: stopping before judge API.")
        logger.info("Elapsed %.2fs", time.perf_counter() - t0)
        return 0

    if not os.environ.get("PORTKEY_API_KEY"):
        logger.error("PORTKEY_API_KEY not set. Use --dry-run or set key.")
        return 2

    log_header("STEP 5 — Materialize traces (one JSONL at a time, OOM-safe)")
    trace_store = build_trace_materialization_store(all_work, logger)

    log_header("STEP 6 — Judging")
    judge = Judge(model=args.judge_model)
    reuse = args.reuse_cache or args.resume
    judged = run_judging_phase(
        all_work,
        trace_store,
        judge,
        args.judge_model,
        cache_dir,
        max(1, args.max_workers),
        args.resume,
        reuse,
        logger,
    )

    out_j = pd.DataFrame(judged)
    out_j.to_csv(out_dir / "selection_gain_judge_outputs.csv", index=False)
    logger.info("Wrote selection_gain_judge_outputs.csv rows=%d", len(out_j))

    log_header("STEP 7 — Aggregate")
    qdf, pdf = aggregate_question_pair(judged, args.random_draws_per_question, logger)
    qdf.to_csv(out_dir / "selection_gain_question_level.csv", index=False)
    pdf.to_csv(out_dir / "selection_gain_pair_level.csv", index=False)

    log_header("STEP 8 — Predictors")
    panel_p = repo / "analysis" / "frs_predictor_panel.csv"
    predictor_analysis(pdf, panel_p, out_dir, logger)

    log_header("STEP 9–10 — Report")
    report_path = out_dir / "selection_gain_followup_report.md"
    n_judged_ok = int(out_j["judge_ok"].sum()) if "judge_ok" in out_j.columns else len(out_j)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Selection-gain judging experiment\n\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
        f.write(f"Log: `{log_path.relative_to(repo)}`\n\n")
        f.write("## Feasibility (STEP 1)\n\n")
        f.write("- Raw generations: `source_pass16_jsonl_by_model*/**/*.jsonl` via `discover_jsonl_groups`.\n")
        f.write("- Traces per question: up to 16; valid traces = those with non-NaN `compute_trace_confidence` on `chosen_token_probs_per_path.epoch_0`.\n")
        f.write("- Question id: JSONL field `idx`.\n")
        f.write("- Trace id: index into `code` / `score` / token prob lists.\n")
        f.write("- Judge: same `Judge` + rubric as `topk_judge_eval` / unfiltered baseline.\n\n")
        f.write("## Design\n\n")
        f.write(f"- questions_per_pair={qpp}, seed={args.random_seed}, exclude_top_from_random={args.exclude_top_from_random}\n")
        f.write(f"- Worklist rows: {len(work_df)} (= judge calls attempted).\n")
        f.write(f"- Judge calls succeeded (rows with score): {n_judged_ok}\n")
        f.write(f"- Cache directory: `{cache_dir.relative_to(repo)}`\n\n")
        f.write("## Circularity\n\n")
        f.write("- **Lower** than summary-only analyses: outcome is **policy contrast** (top-conf vs random) on **fresh** judge calls.\n")
        f.write("- **Residual coupling:** same judge model as elsewhere; confidence for selection uses the **same** token-prob formula as FRS-related work — FRS may correlate mechanically with selection gain.\n\n")
        if len(pdf):
            f.write("## Selection gain summary\n\n")
            f.write(f"- Pairs with gain: {len(pdf)}\n")
            f.write(f"- Mean gain (macro over pairs): {pdf['mean_selection_gain'].mean():.5f}\n\n")
        pred_p = out_dir / "selection_gain_predictor_results.csv"
        if pred_p.is_file():
            f.write("## Predictor correlations\n\n")
            f.write(f"See `{pred_p.relative_to(repo)}` (import into paper).\n\n")
        f.write("## Paper-facing (STEP 12)\n\n")
        f.write("- **Best headline claim:** Deployment-style **selection gain** (mean judge score of top-confidence trace minus random trace), reported per model×benchmark and correlated with FRS vs baselines.\n")
        f.write("- **This experiment does NOT support:** That FRS is optimal for all routing rules; causal production impact; independence from confidence (policy uses confidence).\n")
        f.write("- **Best figure:** `figures/selection_gain/scatter_gain_vs_frs.png`\n")
        f.write("- **Best table:** `selection_gain_pair_level.csv` + `selection_gain_predictor_results.csv` + `selection_gain_regression_incremental.csv`\n")
        f.write("- **Appendix-only:** `selection_gain_judge_outputs.csv`, full worklist, per-trace cache JSONs.\n")
        f.write("- **Next best experiment:** Second judge on the **same** (question, trace) pairs to quantify judge variance.\n")

    logger.info("Done. Report: %s", report_path)
    logger.info("TOTAL elapsed %.2fs", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
