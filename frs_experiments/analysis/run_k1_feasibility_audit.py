#!/usr/bin/env python3
"""
Audit feasibility of a true k=1 (trace_idx=0) baseline vs FRS using cached data only.

No generation, no judge API, no GPU.

Outputs: analysis_outputs/rebuttal_k1_feasibility/

Usage:
  python analysis/run_k1_feasibility_audit.py --repo-root .
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR_DEFAULT = REPO_ROOT / "analysis_outputs" / "rebuttal_k1_feasibility"

DATASET_TO_BENCHMARK = {
    "GSM8K": "GSM8K",
    "MATH500": "MATH500",
    "SVAMP": "SVAMP",
    "AQuA": "AQuA",
    "GPQA": "GPQA",
    "CommonsenseQA": "CSQA",
}

BENCHMARK_TO_DATASET = {v: k for k, v in DATASET_TO_BENCHMARK.items()}
PASS1_TIE_TOL_PP = 2.0
EXPECTED_PAIRS = 54

# Rough cost assumptions for planning (not charged by this script)
EST_USD_PER_JUDGE_CALL = 0.03
EST_SEC_PER_JUDGE_CALL = 2.5


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("k1_feasibility")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    r, p = spearmanr(x[m], y[m])
    return float(r), float(p), n


def pairwise_tie_median(
    panel: pd.DataFrame, value_col: str, pass1_col: str = "k1_accuracy_pct", tol: float = PASS1_TIE_TOL_PP
) -> float:
    rows: List[float] = []
    sub = panel.dropna(subset=[value_col, pass1_col])
    for _, g in sub.groupby("benchmark"):
        models = g["model"].tolist()
        vals = g[value_col].astype(float).tolist()
        p1s = g[pass1_col].astype(float).tolist()
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                if abs(p1s[i] - p1s[j]) <= tol:
                    rows.append(abs(vals[i] - vals[j]))
    return float(np.median(rows)) if rows else float("nan")


def discover_jsonl_pairs(repo: Path, logger: logging.Logger) -> Dict[Tuple[str, str], Path]:
    sys.path.insert(0, str(repo))
    from topk_ablation import build_file_map  # noqa: E402

    fm = build_file_map(str(repo))
    out = {k: Path(v) for k, v in fm.items()}
    logger.info("JSONL pairs discovered: %d (expected %d)", len(out), EXPECTED_PAIRS)
    missing_bench = set(BENCHMARK_TO_DATASET.keys()) - {b for _, b in out}
    if missing_bench:
        logger.warning("Benchmarks with no JSONL in map: %s", sorted(missing_bench))
    return out


def discover_judge_files(repo: Path) -> Dict[Tuple[str, str], Path]:
    d = repo / "reasoning_confidence_bins_results" / "judging_checkpoints"
    out: Dict[Tuple[str, str]] = {}
    for fp in d.glob("judged_*.json"):
        m = re.match(r"^judged_(.+)__(.+)\.json$", fp.name)
        if m:
            model, dataset = m.group(1), m.group(2)
            bench = DATASET_TO_BENCHMARK.get(dataset, dataset)
            out[(model, bench)] = fp
    return out


def discover_unfiltered_judges(repo: Path) -> Dict[Tuple[str, str], Path]:
    d = repo / "analysis_outputs" / "unfiltered_reasoning" / "judging_checkpoints"
    out: Dict[Tuple[str, str]] = {}
    for fp in d.glob("unfiltered_judged_*.json"):
        m = re.match(r"^unfiltered_judged_(.+)__(.+)\.json$", fp.name)
        if m:
            model = m.group(1).replace("_", "-").replace("--", ".")  # slug restore partial
            # filenames use DS-R1-1_5B style — match via metadata inside file instead
            with open(fp, encoding="utf-8") as f:
                meta = json.load(f)
            model = meta.get("model", m.group(1))
            dataset = meta.get("dataset", m.group(2).replace("_", "."))
            bench = DATASET_TO_BENCHMARK.get(dataset, dataset)
            out[(model, bench)] = fp
    return out


def load_frs_judge_trace0(jpath: Path) -> Dict[int, Dict[str, Any]]:
    """question idx -> trace-0 judge record (FRS checkpoints only)."""
    with open(jpath, encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[int, Dict[str, Any]] = {}
    for s in data.get("judged_samples", []):
        if int(s.get("trace_idx", -1)) != 0:
            continue
        if s.get("judge_ok") is False:
            continue
        out[int(s["idx"])] = s
    return out


def load_all_judged_keys(jpath: Path) -> Set[Tuple[int, int]]:
    with open(jpath, encoding="utf-8") as f:
        data = json.load(f)
    keys: Set[Tuple[int, int]] = set()
    for s in data.get("judged_samples", []):
        if s.get("judge_ok") is False:
            continue
        keys.add((int(s["idx"]), int(s["trace_idx"])))
    return keys


def audit_jsonl_pair(
    model: str,
    benchmark: str,
    path: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    n_q = 0
    n_has_t0 = 0
    n_has_code_t0 = 0
    n_has_probs_t0 = 0
    n_correct_t0 = 0
    n_traces_ne_16 = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_q += 1
            scores = row.get("score", [])
            code = row.get("code", [])
            probs = (row.get("chosen_token_probs_per_path") or {}).get("epoch_0", [])
            if not isinstance(scores, list) or len(scores) == 0:
                continue
            if len(scores) != 16:
                n_traces_ne_16 += 1
            if len(scores) > 0:
                n_has_t0 += 1
                if bool(scores[0]):
                    n_correct_t0 += 1
            if isinstance(code, list) and len(code) > 0 and code[0]:
                n_has_code_t0 += 1
            if isinstance(probs, list) and len(probs) > 0 and isinstance(probs[0], list) and len(probs[0]) > 0:
                n_has_probs_t0 += 1
    return {
        "model": model,
        "benchmark": benchmark,
        "jsonl_path": str(path),
        "n_questions": n_q,
        "n_trace0_slot": n_has_t0,
        "n_trace0_has_code": n_has_code_t0,
        "n_trace0_has_token_probs": n_has_probs_t0,
        "n_trace0_correct": n_correct_t0,
        "k1_accuracy_pct": 100.0 * n_correct_t0 / n_q if n_q else float("nan"),
        "trace0_correct_coverage": n_has_t0 / n_q if n_q else float("nan"),
        "n_lines_trace_count_ne_16": n_traces_ne_16,
    }


def audit_judge_pair(
    model: str,
    benchmark: str,
    frs_jpath: Optional[Path],
    unf_jpath: Optional[Path],
    question_ids: Set[int],
) -> Dict[str, Any]:
    frs_t0: Dict[int, Dict[str, Any]] = load_frs_judge_trace0(frs_jpath) if frs_jpath else {}
    all_frs_keys: Set[Tuple[int, int]] = load_all_judged_keys(frs_jpath) if frs_jpath else set()

    unf_t0: Dict[int, Dict[str, Any]] = {}
    if unf_jpath and unf_jpath.is_file():
        with open(unf_jpath, encoding="utf-8") as f:
            data = json.load(f)
        samples = data.get("judged_samples", {})
        if isinstance(samples, dict):
            for k, s in samples.items():
                if int(s.get("trace_idx", -1)) != 0:
                    continue
                unf_t0[int(s["idx"])] = s
        elif isinstance(samples, list):
            for s in samples:
                if int(s.get("trace_idx", -1)) != 0:
                    continue
                unf_t0[int(s["idx"])] = s

    n_q = len(question_ids)
    judged_t0 = set(frs_t0.keys()) & question_ids
    n_judged_t0 = len(judged_t0)

    bin_labels = [frs_t0[i].get("bin_label") for i in judged_t0 if i in frs_t0]
    confs = [frs_t0[i].get("confidence") for i in judged_t0 if i in frs_t0]

    rs_vals = []
    dim_complete = 0
    for i in judged_t0:
        s = frs_t0[i]
        rs = s.get("reasoning_score")
        if rs is not None:
            v = float(rs)
            rs_vals.append(v * 100 if v <= 1.5 else v)
        js = s.get("judge_scores") or {}
        if all(js.get(d) is not None for d in ("faithfulness", "utility", "coherence", "factuality")):
            dim_complete += 1

    # Of all FRS judged traces, what fraction is trace 0?
    n_all_judged = len(all_frs_keys)
    n_all_t0 = sum(1 for (_, t) in all_frs_keys if t == 0)

    unf_t0_on_q = set(unf_t0.keys()) & question_ids
    unf_t0_is_trace0 = sum(1 for i in unf_t0_on_q if int(unf_t0[i].get("trace_idx", -1)) == 0)

    return {
        "n_questions_jsonl": n_q,
        "n_frs_judged_trace0": n_judged_t0,
        "trace0_judged_coverage": n_judged_t0 / n_q if n_q else float("nan"),
        "n_frs_judged_all_traces": n_all_judged,
        "n_frs_judged_trace0_all": n_all_t0,
        "frs_judge_trace0_fraction_of_judged": n_all_t0 / n_all_judged if n_all_judged else float("nan"),
        "n_trace0_four_dims": dim_complete,
        "k1_rs_mean_pct": float(np.mean(rs_vals)) if rs_vals else float("nan"),
        "k1_rs_n": len(rs_vals),
        "top_bin_0_10_count": sum(1 for b in bin_labels if str(b) == "0-10"),
        "top_bin_other_count": sum(1 for b in bin_labels if str(b) != "0-10"),
        "mean_confidence_trace0_judged": float(np.mean([float(c) for c in confs if c is not None])) if confs else float("nan"),
        "n_unfiltered_judged_total": len(unf_t0_on_q),
        "n_unfiltered_trace0": unf_t0_is_trace0,
        "n_unfiltered_random_not_t0": len(unf_t0_on_q) - unf_t0_is_trace0,
    }


def bootstrap_k1_accuracy(
    path: Path,
    n_draws: int = 200,
    seed: int = 42,
) -> Tuple[float, float]:
    """Simulate k=1 by picking one random trace per question from k=16 pool."""
    rng = np.random.default_rng(seed)
    accs: List[float] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            scores = row.get("score", [])
            if not scores:
                continue
            k = len(scores)
            ti = int(rng.integers(0, k))
            accs.append(1.0 if bool(scores[ti]) else 0.0)
    if not accs:
        return float("nan"), float("nan")
    return 100.0 * float(np.mean(accs)), 100.0 * float(np.std(accs))


def build_missing_worklist(
    coverage: pd.DataFrame,
    jsonl_map: Dict[Tuple[str, str], Path],
    cap_per_pair: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Build worklist; use cap_per_pair to limit rows (full uncapped list is ~42k calls)."""
    rows: List[Dict[str, Any]] = []
    bench_alias = {"CSQA": "CommonsenseQA"}
    for i, r in coverage.iterrows():
        model, bench = r["model"], r["benchmark"]
        if not r.get("n_questions") or int(r["n_questions"]) <= 0:
            continue
        judged: Set[int] = set()
        jpath = r.get("frs_judge_path")
        if pd.notna(jpath) and Path(str(jpath)).is_file():
            judged = set(load_frs_judge_trace0(Path(str(jpath))).keys())
        key = (model, bench)
        if key not in jsonl_map:
            key = (model, bench_alias.get(bench, bench))
        if key not in jsonl_map:
            continue
        n_added = 0
        with open(jsonl_map[key], encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                idx = int(json.loads(line)["idx"])
                if idx in judged:
                    continue
                if cap_per_pair is not None and n_added >= cap_per_pair:
                    break
                rows.append(
                    {
                        "model": model,
                        "benchmark": bench,
                        "question_id": idx,
                        "trace_idx": 0,
                        "jsonl_path": str(jsonl_map[key]),
                        "status": "needs_judge",
                    }
                )
                n_added += 1
        if logger and (i + 1) % 10 == 0:
            logger.info("Worklist progress: %d pairs scanned, %d rows", i + 1, len(rows))
    return pd.DataFrame(rows)


def count_missing_judges(coverage: pd.DataFrame, jsonl_map: Dict[Tuple[str, str], Path]) -> pd.DataFrame:
    """Fast counts only — no per-question rows."""
    bench_alias = {"CSQA": "CommonsenseQA"}
    rows: List[Dict[str, Any]] = []
    for _, r in coverage.iterrows():
        model, bench = r["model"], r["benchmark"]
        n_q = int(r.get("n_questions") or 0)
        if n_q <= 0:
            continue
        n_judged = int(r.get("n_frs_judged_trace0") or 0)
        key = (model, bench)
        if key not in jsonl_map:
            key = (model, bench_alias.get(bench, bench))
        n_missing = n_q - n_judged if key in jsonl_map else n_q
        rows.append(
            {
                "model": model,
                "benchmark": bench,
                "n_questions": n_q,
                "n_judged_trace0": n_judged,
                "n_missing_trace0_judge": max(0, n_missing),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    coverage: pd.DataFrame,
    feasible: pd.DataFrame,
    prelim: pd.DataFrame,
    worklist_50: pd.DataFrame,
    worklist_100: pd.DataFrame,
    worklist_full: pd.DataFrame,
    totals: Dict[str, Any],
    recommendation: str,
) -> None:
    lines = [
        "# k=1 (trace_idx=0) baseline feasibility audit",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}_",
        "",
        "## Critical distinction (read first)",
        "",
        "Our pass@16 JSONL stores **16 stochastic traces per question** at temperature 0.7. "
        "**`trace_idx=0` is the first of those 16 samples**, not a separate k=1 generation run. "
        "A reviewer-requested **true k=1 baseline** would require generating **one** trace per question "
        "(typically its own run). Using trace_idx=0 from the k=16 pool is a **conservative proxy**: "
        "it answers “what if we only looked at the first sample in a multi-sample run?” but **does not** "
        "replicate a dedicated k=1 model call distribution.",
        "",
        "We also have **unfiltered RS** = one **random** trace per question (100 judged/pair), which is "
        "another approximate k=1 judge baseline, usually **not** trace_idx=0.",
        "",
        "## Executive totals",
        "",
        f"- Expected model×benchmark pairs: **{totals['expected_pairs']}** (found JSONL: **{totals['jsonl_pairs']}**)",
        f"- Total questions (trace-0 slots): **{totals['total_questions']:,}**",
        f"- Trace-0 correctness available (JSONL): **{totals['total_t0_correct_slots']:,}** slots, accuracy definable for **{totals['pairs_full_acc']}** pairs",
        f"- FRS judge trace-0 RS available: **{totals['total_t0_judged']:,}** / {totals['total_questions']:,} "
        f"(**{100*totals['global_t0_judge_cov']:.1f}%** global)",
        f"- Pairs with ≥50 trace-0 judged: **{totals['pairs_ge_50']}**",
        f"- Pairs with ≥75 trace-0 judged: **{totals['pairs_ge_75']}**",
        f"- Pairs with ≥100 trace-0 judged: **{totals['pairs_ge_100']}**",
        f"- Pairs with 100% trace-0 judged (all questions): **{totals['pairs_full_judge']}**",
        "",
        "### Per-question availability (JSONL)",
        "",
        f"- Every inspected row has `score[0]` and `code[0]`: **{totals['all_have_code_t0']}** / {totals['jsonl_pairs']} pairs with full code",
        f"- Token probs for trace 0: **{totals['pairs_full_probs']}** pairs with complete prob lists",
        "",
        "### Judge bias check (trace-0 among FRS judged)",
        "",
        f"- Of **{totals['total_frs_judged']}** FRS judged (idx, trace) records, **{totals['total_frs_t0_judged']}** are trace_idx=0 "
        f"(**{100*totals['frs_t0_frac']:.1f}%**).",
        f"- Among trace-0 judged records, in top bin `0-10`: **{totals['t0_in_top_bin']}**; other bins: **{totals['t0_other_bin']}** "
        "(judge subsample is **not** uniform over questions; trace-0 judged set is a **sparse, bin-enriched** subset).",
        "",
        "## Feasible baselines without new API",
        "",
        "| Baseline | Feasible? | Coverage | Notes |",
        "|:---|:---:|:---|:---|",
    ]
    for _, r in feasible.iterrows():
        lines.append(
            f"| {r['baseline_id']} | {r['feasible']} | {r['coverage_note']} | {r['notes']} |"
        )
    lines.extend(
        [
            "",
            "## Preliminary k=1 proxy metrics (trace_idx=0, cached only)",
            "",
        ]
    )
    if len(prelim):
        for _, r in prelim.iterrows():
            lines.append(
                f"- {r['metric']}: **{r['value']}**"
                + (f" (n={r['n_models_with_rs']})" if "n_models" in prelim.columns and pd.notna(r.get("n_models_with_rs")) else "")
            )
    else:
        lines.append("_No preliminary metrics computed._")

    lines.extend(
        [
            "",
            "## Additional judge calls (planning estimates)",
            "",
            "| Scenario | Calls | Est. cost (@ ${:.2f}/call) | Est. wall-clock (@ {:.1f}s/call) |".format(
                EST_USD_PER_JUDGE_CALL, EST_SEC_PER_JUDGE_CALL
            ),
            f"| 50 questions / pair × 54 pairs | {len(worklist_50):,} | ${len(worklist_50)*EST_USD_PER_JUDGE_CALL:,.0f} | {len(worklist_50)*EST_SEC_PER_JUDGE_CALL/3600:.1f} h |",
            f"| 100 questions / pair × 54 pairs | {len(worklist_100):,} | ${len(worklist_100)*EST_USD_PER_JUDGE_CALL:,.0f} | {len(worklist_100)*EST_SEC_PER_JUDGE_CALL/3600:.1f} h |",
            f"| All missing trace-0 (full N per pair) | {len(worklist_full):,} | ${len(worklist_full)*EST_USD_PER_JUDGE_CALL:,.0f} | {len(worklist_full)*EST_SEC_PER_JUDGE_CALL/3600:.1f} h |",
            "",
            "Worklists: `missing_trace0_judge_worklist.csv` (full), capped variants in report metadata.",
            "",
            "## Recommendation",
            "",
            f"**{recommendation}**",
            "",
            "### Rationale",
            "",
        ]
    )

    if recommendation.startswith("1"):
        lines.append(
            "- Trace-0 **accuracy** is complete from JSONL; use it for k=1 correctness ranking vs FRS.\n"
            "- Trace-0 **RS** is too sparse for pair-level means at full N; cite coverage and use unfiltered/random-k1 as approximate judge baselines.\n"
        )
    elif recommendation.startswith("2"):
        lines.append(
            "- A **targeted** judge pass on 50–100 trace-0 traces per pair would yield rebuttal-grade RS with moderate cost.\n"
            "- Full-N judging is likely unnecessary if 100/pair aligns with unfiltered design.\n"
        )
    else:
        lines.append(
            "- Global trace-0 judge coverage is <5–10%; pair-level FRS vs k=1 RS comparisons are not representative without substantial judging.\n"
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `trace0_coverage_by_pair.csv`",
            "- `feasible_k1_baselines.csv`",
            "- `trace0_preliminary_metrics.csv`",
            "- `missing_trace0_judge_worklist.csv`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default=str(REPO_ROOT))
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--bootstrap-draws", type=int, default=200)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging()
    t0 = time.perf_counter()

    jsonl_map = discover_jsonl_pairs(repo, logger)
    frs_judges = discover_judge_files(repo)
    logger.info("FRS judge checkpoints: %d", len(frs_judges))

    # Build expected pair grid
    all_models = sorted({m for m, _ in jsonl_map} | {m for m, _ in frs_judges})
    all_benchmarks = sorted(BENCHMARK_TO_DATASET.keys())
    expected_keys = [(m, b) for m in all_models for b in all_benchmarks]

    coverage_rows: List[Dict[str, Any]] = []
    n_pairs_done = 0
    for key in sorted(jsonl_map.keys()):
        model, benchmark = key
        path = jsonl_map[key]
        logger.info("[%d/%d] JSONL audit %s × %s", n_pairs_done + 1, len(jsonl_map), model, benchmark)
        row = audit_jsonl_pair(model, benchmark, path, logger)
        row["frs_judge_path"] = str(frs_judges.get(key, "")) if key in frs_judges else ""
        row["has_frs_judge"] = key in frs_judges

        # question ids
        q_ids: Set[int] = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    q_ids.add(int(json.loads(line)["idx"]))

        unf_path = repo / "analysis_outputs" / "unfiltered_reasoning" / "judging_checkpoints"
        unf_j = None
        for fp in unf_path.glob("unfiltered_judged_*.json"):
            with open(fp, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("model") == model and DATASET_TO_BENCHMARK.get(meta.get("dataset", ""), meta.get("dataset")) == benchmark:
                unf_j = fp
                break
        jrow = audit_judge_pair(model, benchmark, frs_judges.get(key), unf_j, q_ids)
        row.update(jrow)
        row["frs_judge_path"] = str(frs_judges.get(key, "")) if key in frs_judges else ""

        # bootstrap simulated k=1 accuracy
        boot_mean, boot_std = bootstrap_k1_accuracy(path, n_draws=args.bootstrap_draws)
        row["bootstrap_k1_accuracy_pct_mean"] = boot_mean
        row["bootstrap_k1_accuracy_pct_std"] = boot_std

        coverage_rows.append(row)
        n_pairs_done += 1

    # Pairs missing JSONL entirely
    for key in sorted(set(frs_judges.keys()) - set(jsonl_map.keys())):
        m, b = key
        logger.warning("Judge exists but JSONL missing: %s × %s", m, b)
        coverage_rows.append(
            {
                "model": m,
                "benchmark": b,
                "jsonl_path": "",
                "n_questions": 0,
                "note": "jsonl_missing",
            }
        )

    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(out_dir / "trace0_coverage_by_pair.csv", index=False)
    logger.info("Wrote trace0_coverage_by_pair.csv (%d rows)", len(coverage))

    # Totals
    cov = coverage[coverage["n_questions"] > 0].copy()
    total_q = int(cov["n_questions"].sum())
    total_t0_judged = int(cov["n_frs_judged_trace0"].sum())
    total_frs_judged = int(cov["n_frs_judged_all_traces"].sum())
    total_frs_t0 = int(cov["n_frs_judged_trace0_all"].sum())
    t0_top = int(cov["top_bin_0_10_count"].sum())
    t0_other = int(cov["top_bin_other_count"].sum())

    totals = {
        "expected_pairs": EXPECTED_PAIRS,
        "jsonl_pairs": len(jsonl_map),
        "judge_pairs": len(frs_judges),
        "total_questions": total_q,
        "total_t0_correct_slots": int(cov["n_trace0_slot"].sum()),
        "pairs_full_acc": int((cov["trace0_correct_coverage"] >= 0.999).sum()),
        "all_have_code_t0": int((cov["n_trace0_has_code"] == cov["n_questions"]).sum()),
        "pairs_full_probs": int((cov["n_trace0_has_token_probs"] == cov["n_questions"]).sum()),
        "total_t0_judged": total_t0_judged,
        "global_t0_judge_cov": total_t0_judged / total_q if total_q else 0,
        "pairs_ge_50": int((cov["n_frs_judged_trace0"] >= 50).sum()),
        "pairs_ge_75": int((cov["n_frs_judged_trace0"] >= 75).sum()),
        "pairs_ge_100": int((cov["n_frs_judged_trace0"] >= 100).sum()),
        "pairs_full_judge": int((cov["n_frs_judged_trace0"] >= cov["n_questions"]).sum()),
        "total_frs_judged": total_frs_judged,
        "total_frs_t0_judged": total_frs_t0,
        "frs_t0_frac": total_frs_t0 / total_frs_judged if total_frs_judged else 0,
        "t0_in_top_bin": t0_top,
        "t0_other_bin": t0_other,
    }
    logger.info(
        "Global trace-0 judge coverage: %d/%d (%.2f%%)",
        total_t0_judged,
        total_q,
        100 * totals["global_t0_judge_cov"],
    )

    # Feasible baselines table
    feasible_rows = [
        {
            "baseline_id": "k1_trace0_accuracy_jsonl",
            "feasible": "yes",
            "coverage_note": f"{totals['pairs_full_acc']}/{len(cov)} pairs, {total_q:,} questions",
            "notes": "Full correctness from score[0]; no judge API",
        },
        {
            "baseline_id": "k1_trace0_rs_frs_judge_cache",
            "feasible": "partial",
            "coverage_note": f"{total_t0_judged:,}/{total_q:,} ({100*totals['global_t0_judge_cov']:.1f}%)",
            "notes": "Only questions that happened to be judged at trace_idx=0 in FRS sample",
        },
        {
            "baseline_id": "k1_trace0_rs_new_judge",
            "feasible": "requires_api",
            "coverage_note": f"{total_q - total_t0_judged:,} missing trace-0 judges",
            "notes": "See missing_trace0_judge_worklist.csv",
        },
        {
            "baseline_id": "k1_random_trace_unfiltered",
            "feasible": "yes",
            "coverage_note": "54 pairs × 100 judged (random trace, usually not trace 0)",
            "notes": "Existing unfiltered baseline; approximate k=1 judge story",
        },
        {
            "baseline_id": "k1_bootstrap_simulated_accuracy",
            "feasible": "yes",
            "coverage_note": f"{len(cov)} pairs, {args.bootstrap_draws} draws/pair",
            "notes": "Random one-of-16 trace; not trace_idx=0 specific",
        },
        {
            "baseline_id": "frs_top10_reference",
            "feasible": "yes",
            "coverage_note": "paper_frs / merged table 54 pairs",
            "notes": "From global_pass1_frs_analysis",
        },
    ]
    feasible = pd.DataFrame(feasible_rows)
    feasible.to_csv(out_dir / "feasible_k1_baselines.csv", index=False)

    # Preliminary metrics: merge FRS
    path_merged = repo / "global_pass1_frs_analysis" / "merged_pass1_frs_per_benchmark.csv"
    if path_merged.is_file():
        frs = pd.read_csv(path_merged)[["model", "benchmark", "frs_pct", "pass1_pct"]]
    else:
        frs = pd.DataFrame(columns=["model", "benchmark", "frs_pct", "pass1_pct"])

    prelim = cov[["model", "benchmark", "k1_accuracy_pct", "k1_rs_mean_pct", "n_frs_judged_trace0", "n_questions", "trace0_judged_coverage"]].merge(
        frs, on=["model", "benchmark"], how="left"
    )
    prelim["k1_rs_coverage_pct"] = 100.0 * prelim["trace0_judged_coverage"]

    # Model-rank spearman k1 acc vs FRS, k1 rs vs FRS (pair-level macro)
    macro_acc = prelim.groupby("model")["k1_accuracy_pct"].mean()
    macro_rs = prelim.groupby("model")["k1_rs_mean_pct"].mean()
    macro_frs = prelim.groupby("model")["frs_pct"].mean()
    sp_acc, _, _ = safe_spearman(macro_acc.values, macro_frs.values)
    sp_rs, _, n_rs = safe_spearman(macro_rs.values, macro_frs.values)

    prelim["macro_note"] = ""
    prelim.to_csv(out_dir / "trace0_preliminary_metrics.csv", index=False)

    panel_tie = prelim.dropna(subset=["frs_pct", "pass1_pct"])
    summary = pd.DataFrame(
        [
            {"metric": "spearman_model_rank_k1_accuracy_vs_frs", "value": round(sp_acc, 4)},
            {"metric": "spearman_model_rank_k1_rs_vs_frs", "value": round(sp_rs, 4), "n_models_with_rs": n_rs},
            {
                "metric": "median_abs_delta_frs_at_pass1_tie_2pp",
                "value": round(pairwise_tie_median(panel_tie, "frs_pct", "pass1_pct"), 2),
            },
            {
                "metric": "median_abs_delta_k1_acc_at_pass1_tie_2pp",
                "value": round(pairwise_tie_median(panel_tie, "k1_accuracy_pct", "pass1_pct"), 2),
            },
            {
                "metric": "median_abs_delta_k1_rs_at_pass1_tie_2pp",
                "value": round(
                    pairwise_tie_median(
                        prelim.dropna(subset=["k1_rs_mean_pct", "pass1_pct"]),
                        "k1_rs_mean_pct",
                        "pass1_pct",
                    ),
                    2,
                ),
            },
        ]
    )
    summary.to_csv(out_dir / "trace0_preliminary_summary.csv", index=False)

    # Worklist counts (fast) + capped exemplar rows for 50/100 q/pair
    miss_counts = count_missing_judges(cov, jsonl_map)
    miss_counts.to_csv(out_dir / "missing_trace0_judge_counts.csv", index=False)
    total_missing = int(miss_counts["n_missing_trace0_judge"].sum())
    logger.info("Total missing trace-0 judge calls (estimated): %d", total_missing)

    logger.info("Building capped worklist sample (50 q/pair)...")
    worklist_50 = build_missing_worklist(cov, jsonl_map, cap_per_pair=50, logger=logger)
    worklist_50.to_csv(out_dir / "missing_trace0_judge_worklist_50cap.csv", index=False)
    logger.info("Building capped worklist sample (100 q/pair)...")
    worklist_100 = build_missing_worklist(cov, jsonl_map, cap_per_pair=100, logger=logger)
    worklist_100.to_csv(out_dir / "missing_trace0_judge_worklist_100cap.csv", index=False)
    # Full worklist = counts file only (42k rows); save manifest not every idx
    manifest = miss_counts.copy()
    manifest["est_api_cost_usd_50cap"] = manifest["n_missing_trace0_judge"].clip(upper=50) * EST_USD_PER_JUDGE_CALL
    manifest["est_api_cost_usd_100cap"] = manifest["n_missing_trace0_judge"].clip(upper=100) * EST_USD_PER_JUDGE_CALL
    manifest["est_api_cost_usd_full"] = manifest["n_missing_trace0_judge"] * EST_USD_PER_JUDGE_CALL
    manifest.to_csv(out_dir / "missing_trace0_judge_worklist.csv", index=False)
    worklist_full = manifest

    # Recommendation
    if totals["pairs_ge_50"] >= 40 and totals["global_t0_judge_cov"] >= 0.25:
        rec = "1. **No-cost baseline is enough** for accuracy; RS needs careful caveats or spot-check only."
    elif totals["global_t0_judge_cov"] < 0.05 and totals["pairs_ge_50"] == 0:
        rec = (
            "2. **Small judge-call run needed** for rebuttal-worthy k=1 **reasoning** "
            "(50–100 trace-0 judges per pair). **No-cost k=1 accuracy** from JSONL is already complete."
        )
    elif totals["global_t0_judge_cov"] < 0.02:
        rec = "3. **Substantial judge-call run needed** for k=1 RS at scale (~{:,} calls for full trace-0).".format(
            total_missing
        )
    else:
        rec = "2. **Small judge-call run needed** (50–100 q/pair trace_idx=0)."

    write_report(
        out_dir / "k1_feasibility_report.md",
        coverage,
        feasible,
        summary,
        worklist_50,
        worklist_100,
        worklist_full,
        totals,
        rec,
    )

    logger.info("Missing trace-0 judges: %d calls (full est.)", total_missing)
    logger.info("Recommendation: %s", rec)
    logger.info("Done in %.1fs → %s", time.perf_counter() - t0, out_dir)


if __name__ == "__main__":
    main()
