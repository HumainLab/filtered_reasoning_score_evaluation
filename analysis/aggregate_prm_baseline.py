#!/usr/bin/env python3
"""
Combine PRM raw scores with judged FRS (read-only joins on existing results JSON).

Usage:
    python analysis/aggregate_prm_baseline.py \
      --prm-dir-glob 'analysis_outputs/prm_qwen225_baseline/by_pair/*.jsonl' \
      --filtered-root $FRS_REPO_ROOT
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import math
import statistics
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILTERED = Path(
    os.environ.get("FRS_REPO_ROOT", str(Path(__file__).resolve().parent.parent))
)

BENCHMARK_EXPECTED_ROWS = {
    "gsm8k": 1319,
    "math500": 500,
    "svamp": 1000,
    "aqua": 254,
    "gpqa": 448,
    "commonsense_qa": 1221,
}

BENCHMARK_DIRS = {
    "gsm8k": "filtered-cot-gsm8k",
    "math500": "filtered-cot-math500",
    "svamp": "filtered-cot-svamp",
    "aqua": "filtered-cot-aqua",
    "gpqa": "filtered-cot-gpqa",
    "commonsense_qa": "filtered-cot-commonsense",
}


def filt_path(root: Path, bm: str, stem: str) -> Path:
    return root / BENCHMARK_DIRS[bm] / f"{stem}_filtered_p1_only.jsonl"


def res_json(root: Path, bm: str, stem: str) -> Path:
    return root / BENCHMARK_DIRS[bm] / "results" / f"{stem}_filtered_p1_only_results.json"


def load_manifests(shard_dir: Path) -> Dict[Tuple[str, str], List[int]]:
    m: Dict[Tuple[str, str], List[int]] = {}
    for p in shard_dir.glob("*__*_manifest.json"):
        blob = json.loads(p.read_text(encoding="utf-8"))
        m[(blob["stem"], blob["benchmark"])] = blob.get("question_idx", [])
    return m


def load_prm_rows(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows = []
    for path in paths:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _to_bool(val: Any) -> bool | None:
    if isinstance(val, list) and val:
        return bool(val[0])
    if isinstance(val, bool):
        return val
    if val is None:
        return None
    try:
        return bool(val)
    except Exception:
        return None


def fused_pass_coverage(
    res_p: Path,
    filt_p: Path,
    wanted: Sequence[int],
) -> Tuple[float, float, float]:
    ws = sorted(set(int(x) for x in wanted))
    if not ws:
        return float("nan"), float("nan"), float("nan")

    pass_vals: Dict[int, bool] = {}
    if filt_p.is_file():
        with filt_p.open(encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                rec = json.loads(line)
                idx = rec.get("idx")
                if idx is None:
                    continue
                if int(idx) not in set(ws):
                    continue
                b = _to_bool(rec.get("score"))
                if b is not None:
                    pass_vals[int(idx)] = b

    fused: List[float] = []
    if res_p.is_file():
        blob = json.loads(res_p.read_text(encoding="utf-8"))
        for item in blob.get("results", []):
            if item.get("status") != "ok":
                continue
            idx = item.get("idx")
            if idx is None or int(idx) not in set(ws):
                continue
            ov = (item.get("fused_scores") or {}).get("overall")
            if ov is None:
                continue
            fused.append(float(ov))

    frs_mean = statistics.mean(fused) if fused else float("nan")
    pass_numbers = [
        int(pass_vals[i])
        for i in ws
        if i in pass_vals
    ]
    pass_mean = statistics.mean(pass_numbers) if pass_numbers else float("nan")
    cov = len(fused) / len(ws) if ws else float("nan")
    return frs_mean, pass_mean, cov


def read_manual(csv_path: Path) -> Dict[Tuple[str, str], Tuple[float, float, float | None]]:
    out: Dict[Tuple[str, str], Tuple[float, float, float | None]] = {}
    if not csv_path.is_file():
        return out
    with csv_path.open(encoding="utf-8", newline="") as fp:
        rdr = csv.DictReader(fp)
        for row in rdr:
            key = (row["model"].strip(), row["benchmark"].strip())
            frs = float(row["frs_score"])
            p1 = float(row["pass_at_1"])
            cov_txt = row.get("frs_judge_coverage") or ""
            cov = float(cov_txt) if cov_txt.strip() else None
            out[key] = (frs, p1, cov)
    return out


def rank_disagreements(pr_order: List[str], fr_order: List[str]) -> List[str]:
    pp = {m: i for i, m in enumerate(pr_order)}
    ff = {m: i for i, m in enumerate(fr_order)}
    lines = []
    n = len(pr_order)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = pr_order[i], pr_order[j]
            if pp[a] < pp[b] and ff[a] > ff[b]:
                lines.append(f"{a} ranks above {b} under PRM, but below under FRS")
            elif pp[a] > pp[b] and ff[a] < ff[b]:
                lines.append(f"{b} ranks above {a} under PRM, but below under FRS")
    return lines


def pairwise_score_gaps(rows: Sequence[Dict[str, Any]]) -> List[str]:
    names = [r["model"] for r in rows]
    pr = {r["model"]: float(r["prm_score"]) for r in rows}
    fr = {r["model"]: float(r["frs_score"]) for r in rows if not math.isnan(r["frs_score"])}
    gaps: List[Tuple[float, Tuple[str, str]]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if a not in fr or b not in fr:
                continue
            gaps.append((abs((pr[a] - pr[b]) - (fr[a] - fr[b])), tuple(sorted((a, b)))))
    gaps.sort(reverse=True)
    lines: List[str] = []
    for gval, pair in gaps[:3]:
        lines.append(
            f"{pair}: |(ΔPR pairwise) − (ΔFRS pairwise)| = {gval:.4f}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prm-jsonl", nargs="*", default=[], help="Explicit PRM shard files.")
    parser.add_argument(
        "--prm-dir-glob",
        default=None,
        help='Glob for shard jsonl lines, e.g. "prm_out/by_pair/*.jsonl"',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "analysis_outputs" / "prm_aggregate",
    )
    parser.add_argument("--filtered-root", type=Path, default=DEFAULT_FILTERED)
    parser.add_argument(
        "--manual-frs-table",
        type=Path,
        default=REPO_ROOT
        / "analysis_outputs"
        / "trace0_k1_judging"
        / "per_pair_scores.csv",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = [Path(x) for x in args.prm_jsonl]
    if args.prm_dir_glob:
        from glob import glob

        paths.extend(Path(p) for p in sorted(glob(args.prm_dir_glob)))

    shard_paths = [p for p in paths if p.suffix == ".jsonl" and p.is_file()]
    if not shard_paths:
        raise SystemExit(
            "Provide --prm-jsonl and/or matching --prm-dir-glob (needs *.jsonl files)."
        )

    manifest_dir = shard_paths[0].parent
    manifests = load_manifests(manifest_dir)

    overrides = read_manual(args.manual_frs_table)
    samples = load_prm_rows(shard_paths)
    grp: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for s in samples:
        grp[(s["model"], s["benchmark"])].append(float(s["trace_score"]))

    benchmarks = sorted({key[1] for key in grp})
    models = sorted({key[0] for key in grp})

    out_rows = []
    for bm in benchmarks:
        for stem in models:
            key = (stem, bm)
            if key not in grp:
                continue
            prm_mean = statistics.mean(grp[key])
            idx_list = manifests.get(key, [])
            frs_csv = overrides.get(key)

            filt = filt_path(args.filtered_root, bm, stem)
            respath = res_json(args.filtered_root, bm, stem)

            if idx_list:
                fused, p1, cov = fused_pass_coverage(respath, filt, idx_list)
            else:
                fused, p1, cov = (
                    math.nan if not frs_csv else float(frs_csv[0]),
                    math.nan if not frs_csv else float(frs_csv[1]),
                    math.nan if not frs_csv or frs_csv[2] is None else float(frs_csv[2]),
                )

            if frs_csv:
                fused = float(frs_csv[0])
                p1 = float(frs_csv[1])
                if frs_csv[2] is not None:
                    cov = float(frs_csv[2])

            out_rows.append(
                {
                    "model": stem,
                    "benchmark": bm,
                    "prm_score": prm_mean,
                    "frs_score": fused,
                    "pass_at_1": p1,
                    "frs_judge_coverage": cov,
                    "prm_traces_written": len(grp[key]),
                    "benchmark_expected_rows": BENCHMARK_EXPECTED_ROWS.get(bm),
                }
            )

    by_bm = defaultdict(list)
    for r in out_rows:
        by_bm[r["benchmark"]].append(r)

    spear_all_txt: List[str] = []
    spear_filt_txt: List[str] = []
    disagreement_md: List[str] = []

    caveat = (
        "**phi_4 × math500**: FRS judge coverage is negligible in released artifacts (~9 fused rows).\n"
        "Do **not** read PRM-vs-FRS agreement literally on that row."
    )

    mean_delta_across_bm: List[float] = []

    for bm in sorted(by_bm.keys()):
        block = sorted(by_bm[bm], key=lambda rr: rr["model"])
        pr_ranked = sorted(block, key=lambda rr: (-rr["prm_score"], rr["model"]))
        fr_defined = [
            rr
            for rr in block
            if not math.isnan(rr["frs_score"])
            and not math.isnan(rr["frs_judge_coverage"])
            and rr["frs_judge_coverage"] >= 0.49
        ]
        fr_fallback = sorted(
            [rr for rr in block if not math.isnan(rr["frs_score"])],
            key=lambda rr: (-rr["frs_score"], rr["model"]),
        )

        base_fr = fr_defined if len(fr_defined) >= 3 else fr_fallback

        pr_order = [r["model"] for r in pr_ranked]

        fused_models = sorted(
            [rr for rr in block if not math.isnan(rr["frs_score"])],
            key=lambda rr: (-rr["frs_score"], rr["model"]),
        )
        fr_order_full = [rr["model"] for rr in fused_models]

        ff_map = {m: i for i, m in enumerate(fr_order_full)}
        pr_filtered = [m for m in pr_order if m in ff_map]

        fr_order_aligned = sorted(pr_filtered, key=lambda mm: ff_map[mm])

        for rr in block:
            rr["prm_rank_in_benchmark"] = (
                pr_order.index(rr["model"]) + 1 if rr["model"] in pr_order else ""
            )
            rr["frs_rank_in_benchmark"] = (
                ff_map[rr["model"]] + 1 if rr["model"] in ff_map else ""
            )

        for rr in block:
            pi, fi = rr["prm_rank_in_benchmark"], rr["frs_rank_in_benchmark"]
            if isinstance(pi, int) and isinstance(fi, int):
                rr["rank_delta"] = pi - fi
                mean_delta_across_bm.append(abs(rr["rank_delta"]))
            else:
                rr["rank_delta"] = ""

        xs = [rr["prm_score"] for rr in base_fr]
        ys = [rr["frs_score"] for rr in base_fr]
        rho_all, _ = spearmanr(xs, ys) if len(xs) >= 2 else (float("nan"), float("nan"))

        subset = [rr for rr in base_fr if rr["frs_judge_coverage"] >= 0.5]
        sx = [rr["prm_score"] for rr in subset]
        sy = [rr["frs_score"] for rr in subset]
        rho_f, _ = spearmanr(sx, sy) if len(subset) >= 2 else (float("nan"), float("nan"))

        spear_all_txt.append(
            f"- `{bm}` all usable FRS rows: ρ={rho_all:.3f} (n_models={len(base_fr)})"
        )
        spear_filt_txt.append(
            f"- `{bm}` FRS_cov≥0.5: ρ={rho_f:.3f} (n_models={len(subset)})"
        )

        disagreement_md.append(f"### Benchmark `{bm}`")
        disagreement_md.extend(
            rank_disagreements(pr_filtered, fr_order_aligned)[:18]
        )
        disagreement_md.append("- Top pairwise PRM-vs-FRS score deltas:")
        disagreement_md.extend("  • " + g for g in pairwise_score_gaps(block))

    comp_path = args.output_dir / "prm_vs_frs_comparison.csv"
    fields = [
        "model",
        "benchmark",
        "prm_score",
        "frs_score",
        "pass_at_1",
        "frs_judge_coverage",
        "prm_rank_in_benchmark",
        "frs_rank_in_benchmark",
        "rank_delta",
        "prm_traces_written",
        "benchmark_expected_rows",
    ]
    md_path = args.output_dir / "prm_baseline_key_numbers.md"

    with comp_path.open("w", newline="", encoding="utf-8") as fp:
        wrt = csv.DictWriter(fp, fieldnames=fields)
        wrt.writeheader()
        for row in sorted(
            out_rows, key=lambda r: (r["benchmark"], -r["prm_score"], r["model"])
        ):
            wrt.writerow({k: row.get(k, "") for k in fields})

    avg_abs = (
        statistics.mean(mean_delta_across_bm) if mean_delta_across_bm else float("nan")
    )

    disag_blob = "\n".join(disagreement_md)

    md_path.write_text(
        "## PRM vs FRS — auto summary\n\n"
        "### Spearman correlations (ordering by PRM_score vs fused FRS_mean)\n\n"
        "**Cohort usable for correlation** prefers models with fused FRS and judge coverage≥0.49; "
        "fallback uses any non-NaN FRS if too few survivors.\n"
        + "\n".join(spear_all_txt)
        + "\n\n**Correlation restricted to benchmark rows with judge coverage≥0.5**\n"
        + "\n".join(spear_filt_txt)
        + f"\n\n### Mean |Δrank| averaged across benchmarks: `{avg_abs:.2f}`\n\n"
        "### Structural disagreements & score gaps\n"
        + disag_blob
        + "\n\n### Caveats\n"
        + caveat
        + "\n",
        encoding="utf-8",
    )

    print(comp_path)
    print(md_path)


if __name__ == "__main__":
    main()