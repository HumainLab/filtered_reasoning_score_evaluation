#!/usr/bin/env python3
"""
Within-question confidence selection gain (COLM rebuttal yweD / kp6q).

For each question with 16 pass@16 traces, compare reasoning quality of:
  - top-confidence trace (FRS confidence proxy)
  - deterministic random trace (seeded per model+benchmark+idx)
  - trace_idx=0

Reuses existing judge checkpoints; writes worklist for missing scores.
Does NOT call the judge API (use run_selection_gain_judging.py with --reuse-worklist).

Usage:
  python analysis/run_within_question_selection_gain.py --repo-root .
  python analysis/run_within_question_selection_gain.py --math-only
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR_DEFAULT = REPO_ROOT / "outputs/analysis_outputs" / "within_question_selection_gain"

BENCHMARKS_ALL = ("GSM8K", "MATH500", "SVAMP", "AQuA", "GPQA", "CSQA")
MATH_BENCHMARKS = ("GSM8K", "MATH500", "SVAMP", "AQuA")
DATASET_TO_BENCHMARK = {
    "GSM8K": "GSM8K",
    "MATH500": "MATH500",
    "SVAMP": "SVAMP",
    "AQuA": "AQuA",
    "GPQA": "GPQA",
    "CommonsenseQA": "CSQA",
}
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 123
DEFAULT_RANDOM_SEED = 42

JudgeKey = Tuple[str, str, int, int]  # model, benchmark, idx, trace_idx


def setup_logger() -> logging.Logger:
    log = logging.getLogger("within_q_sel_gain")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(h)
    return log


def rs_to_pct(v: float) -> float:
    if v <= 1.5:
        return float(v) * 100.0
    return float(v)


def question_rng(model: str, benchmark: str, idx: int, seed: int) -> np.random.Generator:
    digest = hashlib.sha256(f"within_q_sel|{seed}|{model}|{benchmark}|{idx}".encode()).hexdigest()
    return np.random.default_rng(int(digest[:16], 16) % (2**32))


def parse_judged_checkpoint_name(stem: str) -> Optional[Tuple[str, str]]:
    m = re.match(r"^judged_(.+)__(.+)$", stem)
    if not m:
        return None
    model, dataset = m.group(1), m.group(2)
    benchmark = DATASET_TO_BENCHMARK.get(dataset, dataset)
    return model, benchmark


def iter_judged_samples(data: dict) -> List[dict]:
    raw = data.get("judged_samples", [])
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return list(raw.values())
    return []


def load_judge_index(repo: Path, log: logging.Logger) -> Dict[JudgeKey, Dict[str, Any]]:
    """(model, benchmark, idx, trace_idx) -> {reasoning_score_pct, source}."""
    index: Dict[JudgeKey, Dict[str, Any]] = {}

    def put(key: JudgeKey, rs: float, source: str) -> None:
        if key in index:
            return
        index[key] = {"reasoning_score_pct": rs_to_pct(rs), "judge_source": source}

    # FRS confidence-bin checkpoints
    frs_dir = repo / "outputs/reasoning_confidence_bins_results" / "judging_checkpoints"
    for fp in sorted(frs_dir.glob("judged_*.json")):
        parsed = parse_judged_checkpoint_name(fp.stem)
        if not parsed:
            continue
        model, benchmark = parsed
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        for s in iter_judged_samples(data):
            if s.get("judge_ok") is False:
                continue
            rs = s.get("reasoning_score")
            if rs is None:
                continue
            put((model, benchmark, int(s["idx"]), int(s["trace_idx"])), float(rs), "frs_bins")

    # Trace-0 k1 checkpoints
    t0_dir = repo / "outputs/analysis_outputs" / "trace0_k1_judging" / "judging_checkpoints"
    for fp in sorted(t0_dir.glob("trace0_judged_*.json")):
        m = re.match(r"^trace0_judged_(.+)__(.+)\.json$", fp.name)
        if not m:
            continue
        model = slug_to_model(m.group(1))
        dataset = m.group(2)
        benchmark = DATASET_TO_BENCHMARK.get(dataset, dataset)
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        for s in iter_judged_samples(data):
            if s.get("judge_ok") is False:
                continue
            rs = s.get("reasoning_score")
            if rs is None:
                continue
            put((model, benchmark, int(s["idx"]), int(s["trace_idx"])), float(rs), "trace0_k1")

    # Unfiltered baseline checkpoints
    unf_dir = repo / "outputs/analysis_outputs" / "unfiltered_reasoning" / "judging_checkpoints"
    for fp in sorted(unf_dir.glob("unfiltered_judged_*.json")):
        m = re.match(r"^unfiltered_judged_(.+)__(.+)\.json$", fp.name)
        if not m:
            continue
        model = slug_to_model(m.group(1))
        dataset = m.group(2)
        benchmark = DATASET_TO_BENCHMARK.get(dataset, dataset)
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        for s in iter_judged_samples(data):
            if s.get("judge_ok") is False:
                continue
            rs = s.get("reasoning_score")
            if rs is None:
                continue
            put((model, benchmark, int(s["idx"]), int(s["trace_idx"])), float(rs), "unfiltered")

    # Prior selection-gain CSV outputs
    for rel, src in [
        ("analysis/selection_gain_judge_outputs.csv", "selection_gain"),
        ("outputs/analysis_outputs/selection_gain_heldout_haiku/selection_gain_judged_traces_haiku.csv", "selection_gain_haiku"),
    ]:
        csv_path = repo / rel
        if not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path)
        for _, r in df.iterrows():
            rs = r.get("reasoning_score")
            if pd.isna(rs):
                continue
            if "judge_ok" in df.columns and r.get("judge_ok") not in (True, "True", 1, 1.0):
                continue
            bench = str(r.get("benchmark", ""))
            if bench == "CommonsenseQA":
                bench = "CSQA"
            put(
                (str(r["model"]), bench, int(r["question_id"]), int(r["trace_id"])),
                float(rs),
                src,
            )

    log.info("Judge index loaded: %d unique (model,benchmark,idx,trace_idx) scores", len(index))
    return index


CHECKPOINT_SLUG_TO_MODEL = {
    "DS-R1-1_5B": "DS-R1-1.5B",
    "DS-R1-7B": "DS-R1-7B",
    "LLaMA-3_1-8B": "LLaMA-3.1-8B",
    "Qwen2_5-7B": "Qwen2.5-7B",
    "Qwen2_5-Math": "Qwen2.5-Math",
    "Gemma-7B": "Gemma-7B",
    "Phi-4": "Phi-4",
    "Phi-4-Reas_": "Phi-4-Reas.",
    "Qwen3-4B": "Qwen3-4B",
}


def slug_to_model(slug: str) -> str:
    return CHECKPOINT_SLUG_TO_MODEL.get(slug, slug.replace("_", "."))


def load_pass16_helpers():
    sys.path.insert(0, str(REPO_ROOT))
    from build_downstream_parquets import discover_jsonl_groups  # noqa: E402
    from topk_ablation import compute_trace_confidence  # noqa: E402

    return discover_jsonl_groups, compute_trace_confidence


def confidence_map(row: dict, compute_trace_confidence) -> Dict[int, float]:
    scores = row.get("score", [])
    code = row.get("code", [])
    probs_all = row.get("chosen_token_probs_per_path", {}).get("epoch_0", [])
    if not isinstance(probs_all, list) or not isinstance(code, list):
        return {}
    n = min(len(scores), len(code), len(probs_all))
    out: Dict[int, float] = {}
    for ti in range(n):
        probs = probs_all[ti] if ti < len(probs_all) else []
        c = compute_trace_confidence(probs)
        if not np.isnan(c):
            out[ti] = float(c)
    return out


def pick_top_conf(conf_map: Dict[int, float]) -> int:
    best = max(conf_map.values())
    cands = [ti for ti, c in conf_map.items() if c == best]
    return int(min(cands))


def pick_random_trace(
    conf_map: Dict[int, float], top_ti: int, model: str, benchmark: str, idx: int, seed: int
) -> int:
    pool = [ti for ti in conf_map if ti != top_ti]
    if not pool:
        pool = list(conf_map.keys())
    rng = question_rng(model, benchmark, idx, seed)
    return int(rng.choice(pool))


def materialize(row: dict, trace_idx: int) -> Tuple[str, str, str, bool, str]:
    code = row.get("code", [])
    scores = row.get("score", [])
    pred = row.get("pred", [])
    cot = code[trace_idx] if trace_idx < len(code) else ""
    if not isinstance(cot, str):
        cot = str(cot)
    gt = str(row.get("gt", row.get("answer", "")))
    p = pred[trace_idx] if trace_idx < len(pred) else ""
    correct = bool(scores[trace_idx]) if trace_idx < len(scores) else False
    q = str(row.get("question", row.get("prompt", "")))
    return q, cot, gt, correct, str(p)


def bootstrap_ci(values: np.ndarray, n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> Tuple[float, float, float]:
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    obs = float(np.mean(v))
    rng = np.random.default_rng(seed)
    boots = [float(np.mean(rng.choice(v, size=len(v), replace=True))) for _ in range(n)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, float(lo), float(hi)


def summarize_paired(
    qdf: pd.DataFrame,
    diff_col: str,
    top_col: str,
    other_col: str,
    label: str,
) -> Dict[str, Any]:
    sub = qdf.dropna(subset=[diff_col, top_col, other_col])
    if sub.empty:
        return {"comparison": label, "n_questions": 0}
    diff = sub[diff_col].values.astype(float)
    obs, lo, hi = bootstrap_ci(diff)
    gt = (sub[top_col] > sub[other_col]).sum()
    eq = (sub[top_col] == sub[other_col]).sum()
    lt = (sub[top_col] < sub[other_col]).sum()
    n = len(sub)
    return {
        "comparison": label,
        "n_questions": n,
        "mean_top_conf_rs_pct": float(sub[top_col].mean()),
        f"mean_{label.split('_vs_')[1]}_rs_pct": float(sub[other_col].mean()),
        "mean_paired_diff_pp": obs,
        "bootstrap_ci95_lo": lo,
        "bootstrap_ci95_hi": hi,
        "pct_top_gt_other": gt / n,
        "pct_equal": eq / n,
        "pct_top_lt_other": lt / n,
    }


def write_key_numbers(
    path: Path,
    overall_rows: List[Dict[str, Any]],
    pair_sum: pd.DataFrame,
    coverage: Dict[str, Any],
) -> None:
    tr = next((r for r in overall_rows if r.get("comparison") == "top_conf_vs_random"), {})
    t0 = next((r for r in overall_rows if r.get("comparison") == "top_conf_vs_trace0"), {})
    n_pos = 0
    if len(pair_sum) and "mean_paired_diff_pp" in pair_sum.columns:
        n_pos = int((pair_sum["mean_paired_diff_pp"] > 0).sum())
    n_pairs = len(pair_sum)

    lines = [
        "# Within-question confidence selection gain (yweD / kp6q)",
        "",
        "## Framing",
        "",
        "Holding the **same question** fixed, does selecting the most-confident among 16 traces",
        "yield higher reasoning scores than a random trace or trace-0? This controls for",
        "question difficulty (yweD easy-problem bias) and tests whether confidence filtering",
        "adds signal beyond single-trace sampling (kp6q).",
        "",
        "## Coverage",
        "",
        f"- Questions with 16 traces processed: **{coverage.get('n_questions_total', 0):,}**",
        f"- Selection rows (3 conditions × questions): **{coverage.get('n_selection_rows', 0):,}**",
        f"- Unique traces needing judge: **{coverage.get('n_unique_traces', 0):,}**",
        f"- Judge scores in cache: **{coverage.get('n_judge_hits', 0):,}** ({coverage.get('pct_judge_coverage', 0):.1%})",
        f"- Missing judge calls: **{coverage.get('n_missing_judge', 0):,}**",
        "",
        "## Cache-only results (complete paired questions)",
        "",
    ]
    if tr.get("n_questions", 0):
        lines.extend(
            [
                f"- **Top-conf vs random** (n={tr['n_questions']:,} questions):",
                f"  - Mean paired gain: **{tr['mean_paired_diff_pp']:.2f} pp** "
                f"[{tr['bootstrap_ci95_lo']:.2f}, {tr['bootstrap_ci95_hi']:.2f}]",
                f"  - Top-conf beats random on **{tr['pct_top_gt_other']:.1%}** of paired questions",
                "",
            ]
        )
    else:
        lines.append("- Top-conf vs random: insufficient cached judge overlap (see worklist).\n")

    if t0.get("n_questions", 0):
        lines.extend(
            [
                f"- **Top-conf vs trace-0** (n={t0['n_questions']:,} questions):",
                f"  - Mean paired gain: **{t0['mean_paired_diff_pp']:.2f} pp** "
                f"[{t0['bootstrap_ci95_lo']:.2f}, {t0['bootstrap_ci95_hi']:.2f}]",
                f"  - Top-conf beats trace-0 on **{t0['pct_top_gt_other']:.1%}** of paired questions",
                "",
            ]
        )

    if n_pairs:
        lines.append(
            f"- Positive mean paired gain (top−random) on **{n_pos}/{n_pairs}** model×benchmark pairs."
        )

    lines.extend(
        [
            "",
            "## Rebuttal-ready sentences",
            "",
        ]
    )
    if tr.get("n_questions", 0):
        lines.append(
            f"> Holding the question fixed, the most-confident trace improves reasoning score by "
            f"**{tr['mean_paired_diff_pp']:.1f}** points (0–100 scale) over a random trace from the "
            f"same 16 samples (paired n={tr['n_questions']:,}; top wins on "
            f"{tr['pct_top_gt_other']:.0%} of questions). This directly controls for question difficulty."
        )
    else:
        lines.append(
            "> Cache-only paired analysis pending — run judging on "
            "`missing_judge_worklist.csv` first."
        )

    lines.extend(["", "## Files", "", "- `per_trace_selection_table.csv`", "- `missing_judge_worklist.csv`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_gain_hist(qdf: pd.DataFrame, out_path: Path) -> None:
    if qdf.empty or "paired_diff_top_minus_random_pp" not in qdf.columns:
        return
    sub = qdf["paired_diff_top_minus_random_pp"].dropna()
    if len(sub) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sub, bins=40, edgecolor="0.3", alpha=0.85)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(sub.mean(), color="crimson", ls="--", label=f"mean={sub.mean():.2f} pp")
    ax.set_xlabel("Top-conf RS − random RS (pp, same question)")
    ax.set_ylabel("Questions")
    ax.set_title("Within-question selection gain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--math-only", action="store_true")
    ap.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    ap.add_argument("--min-traces", type=int, default=16, help="Require at least this many valid-confidence traces")
    ap.add_argument("--allow-partial-traces", action="store_true", help="Allow questions with <16 traces")
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logger()

    discover_jsonl_groups, compute_trace_confidence = load_pass16_helpers()
    judge_index = load_judge_index(repo, log)
    file_map = discover_jsonl_groups(str(repo))

    allowed_bench = set(MATH_BENCHMARKS if args.math_only else BENCHMARKS_ALL)
    pairs: List[Tuple[str, str, str, str]] = []
    for (model, dataset), jsonl_path in sorted(file_map.items()):
        benchmark = DATASET_TO_BENCHMARK.get(dataset, dataset)
        if benchmark not in allowed_bench:
            continue
        pairs.append((model, benchmark, dataset, jsonl_path))
    log.info("Processing %d model×benchmark pairs (%s)", len(pairs), "math-only" if args.math_only else "all")

    selection_rows: List[Dict[str, Any]] = []
    question_rows: List[Dict[str, Any]] = []
    missing_work: Dict[JudgeKey, Dict[str, Any]] = {}
    n_questions_total = 0

    for model, benchmark, dataset, jsonl_path in tqdm(pairs, desc="Pairs"):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                idx = int(row["idx"])
                scores = row.get("score", [])
                if not args.allow_partial_traces and (not isinstance(scores, list) or len(scores) != 16):
                    continue
                cm = confidence_map(row, compute_trace_confidence)
                if len(cm) < args.min_traces:
                    continue
                n_questions_total += 1
                top_ti = pick_top_conf(cm)
                rand_ti = pick_random_trace(cm, top_ti, model, benchmark, idx, args.random_seed)
                trace0_ti = 0 if 0 in cm else None

                selections = [
                    ("top_conf", top_ti),
                    ("random", rand_ti),
                ]
                if trace0_ti is not None:
                    selections.append(("trace0", trace0_ti))

                per_cond: Dict[str, Dict[str, Any]] = {}
                for condition, ti in selections:
                    q, cot, gt, correct, pred = materialize(row, ti)
                    jkey: JudgeKey = (model, benchmark, idx, ti)
                    hit = judge_index.get(jkey)
                    rs_pct = hit["reasoning_score_pct"] if hit else np.nan
                    judge_source = hit["judge_source"] if hit else ""

                    rec = {
                        "model": model,
                        "benchmark": benchmark,
                        "dataset": dataset,
                        "jsonl_path": jsonl_path,
                        "idx": idx,
                        "trace_idx": ti,
                        "condition": condition,
                        "confidence": cm.get(ti, np.nan),
                        "correct": correct,
                        "pred": pred,
                        "reasoning_score_pct": rs_pct,
                        "has_judge_score": bool(hit),
                        "judge_source": judge_source,
                    }
                    selection_rows.append(rec)
                    per_cond[condition] = rec

                    if not hit:
                        if jkey not in missing_work:
                            missing_work[jkey] = {
                                "model": model,
                                "benchmark": benchmark,
                                "dataset": dataset,
                                "jsonl_path": jsonl_path,
                                "idx": idx,
                                "trace_idx": ti,
                                "condition": condition,
                                "question": q[:5000],
                                "trace_text": cot[:20000],
                                "pred": pred,
                                "gt": gt,
                                "correct": correct,
                                "confidence": cm.get(ti, np.nan),
                            }

                top_rs = per_cond.get("top_conf", {}).get("reasoning_score_pct", np.nan)
                rnd_rs = per_cond.get("random", {}).get("reasoning_score_pct", np.nan)
                t0_rs = per_cond.get("trace0", {}).get("reasoning_score_pct", np.nan)
                qrec = {
                    "model": model,
                    "benchmark": benchmark,
                    "idx": idx,
                    "top_conf_trace_idx": top_ti,
                    "random_trace_idx": rand_ti,
                    "trace0_trace_idx": trace0_ti if trace0_ti is not None else np.nan,
                    "top_conf_rs_pct": top_rs,
                    "random_rs_pct": rnd_rs,
                    "trace0_rs_pct": t0_rs,
                    "paired_diff_top_minus_random_pp": (
                        float(top_rs) - float(rnd_rs)
                        if np.isfinite(top_rs) and np.isfinite(rnd_rs)
                        else np.nan
                    ),
                    "paired_diff_top_minus_trace0_pp": (
                        float(top_rs) - float(t0_rs)
                        if np.isfinite(top_rs) and np.isfinite(t0_rs)
                        else np.nan
                    ),
                    "has_paired_top_random": np.isfinite(top_rs) and np.isfinite(rnd_rs),
                    "has_paired_top_trace0": np.isfinite(top_rs) and np.isfinite(t0_rs),
                }
                question_rows.append(qrec)

    sel_df = pd.DataFrame(selection_rows)
    qdf = pd.DataFrame(question_rows)
    sel_df.to_csv(out_dir / "per_trace_selection_table.csv", index=False)
    log.info("Wrote per_trace_selection_table.csv rows=%d", len(sel_df))

    # Deduped worklist
    wl_rows = list(missing_work.values())
    wl_df = pd.DataFrame(wl_rows)
    if len(wl_df):
        wl_df = wl_df.drop_duplicates(subset=["model", "benchmark", "idx", "trace_idx"])
    wl_df.to_csv(out_dir / "missing_judge_worklist.csv", index=False)

    n_unique = sel_df.drop_duplicates(subset=["model", "benchmark", "idx", "trace_idx"]).shape[0]
    n_hits = int(sel_df["has_judge_score"].sum())
    n_missing = len(wl_df)

    coverage = {
        "n_questions_total": n_questions_total,
        "n_selection_rows": len(sel_df),
        "n_unique_traces": n_unique,
        "n_judge_hits": n_hits,
        "pct_judge_coverage": n_hits / max(1, len(sel_df)),
        "n_missing_judge": n_missing,
    }

    # Summaries (cache-only where paired complete)
    paired_tr = qdf[qdf["has_paired_top_random"]]
    paired_t0 = qdf[qdf["has_paired_top_trace0"]]

    overall_rows: List[Dict[str, Any]] = []
    overall_rows.append(summarize_paired(qdf, "paired_diff_top_minus_random_pp", "top_conf_rs_pct", "random_rs_pct", "top_conf_vs_random"))
    overall_rows.append(summarize_paired(qdf, "paired_diff_top_minus_trace0_pp", "top_conf_rs_pct", "trace0_rs_pct", "top_conf_vs_trace0"))
    pd.DataFrame(overall_rows).to_csv(out_dir / "overall_summary.csv", index=False)

    bench_rows: List[Dict[str, Any]] = []
    for bench, g in qdf.groupby("benchmark"):
        bench_rows.append({**summarize_paired(g, "paired_diff_top_minus_random_pp", "top_conf_rs_pct", "random_rs_pct", "top_conf_vs_random"), "benchmark": bench})
    pd.DataFrame(bench_rows).to_csv(out_dir / "per_benchmark_summary.csv", index=False)

    pair_rows: List[Dict[str, Any]] = []
    for (model, bench), g in qdf.groupby(["model", "benchmark"]):
        pr = summarize_paired(g, "paired_diff_top_minus_random_pp", "top_conf_rs_pct", "random_rs_pct", "top_conf_vs_random")
        pr.update({"model": model, "benchmark": bench, "n_questions_total": len(g), "n_paired_top_random": int(g["has_paired_top_random"].sum())})
        pair_rows.append(pr)
    pair_sum = pd.DataFrame(pair_rows)
    pair_sum.to_csv(out_dir / "per_pair_summary.csv", index=False)

    write_key_numbers(out_dir / "within_question_selection_gain_key_numbers.md", overall_rows, pair_sum, coverage)
    plot_gain_hist(qdf, out_dir / "confidence_selected_vs_random_gain.png")

    log.info("=== SUMMARY ===")
    log.info("Questions (16-trace): %d", n_questions_total)
    log.info("Selection rows: %d | judge cache hits: %d (%.1f%%)", len(sel_df), n_hits, 100 * n_hits / max(1, len(sel_df)))
    log.info("Missing unique traces (worklist): %d", n_missing)
    log.info("Paired top vs random (cache-complete): %d questions", len(paired_tr))
    log.info("Paired top vs trace0 (cache-complete): %d questions", len(paired_t0))

    tr = overall_rows[0]
    if tr.get("n_questions", 0):
        log.info(
            "CACHE-ONLY top−random: mean=%.2f pp | top wins %.1f%% | CI [%.2f, %.2f]",
            tr["mean_paired_diff_pp"],
            100 * tr["pct_top_gt_other"],
            tr["bootstrap_ci95_lo"],
            tr["bootstrap_ci95_hi"],
        )
        log.info("Meaningful cache-only result: YES (partial — %d/%d questions)", tr["n_questions"], n_questions_total)
    else:
        log.info("Meaningful cache-only result: NO — run judging first")

    log.info("")
    log.info("Next step (judging) — convert worklist and run selection-gain judge runner:")
    log.info("  python analysis/run_within_question_selection_gain.py  # already done")
    log.info("  # Then adapt worklist columns and judge via:")
    log.info("  python analysis/run_selection_gain_judging.py --repo-root . \\")
    log.info("    --reuse-worklist --worklist-path %s \\", out_dir / "missing_judge_worklist_for_selection_gain.csv")
    log.info("    --resume --questions-per-pair 99999")
    log.info("(See missing_judge_worklist.csv; map to selection_gain worklist format if using that runner.)")

    # Emit selection-gain-compatible worklist stub for top missing traces
    if len(wl_df):
        sg_wl = []
        for _, r in wl_df.iterrows():
            sg_wl.append(
                {
                    "model": r["model"],
                    "dataset": r["dataset"],
                    "benchmark": r["benchmark"],
                    "jsonl_path": r["jsonl_path"],
                    "question_id": int(r["idx"]),
                    "trace_id": int(r["trace_idx"]),
                    "selection_type": r["condition"],
                    "confidence": r["confidence"],
                    "correct": r["correct"],
                    "random_draw_index": np.nan,
                    "exclude_top_from_random": True,
                    "seed": args.random_seed,
                    "policy_version": "within_question_selection_gain_v1",
                }
            )
        pd.DataFrame(sg_wl).to_csv(out_dir / "missing_judge_worklist_for_selection_gain.csv", index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
