#!/usr/bin/env python3
"""
Experiment F: judge API cost accounting from logs + checkpoint counts.

Parses reasoning_confidence_bins_run.log, trace-0 / unfiltered logs, imputes
selection-gain (gpt-4o-mini) from FRS per-call averages. Reports tokens, cost,
wall-clock, per-pair averages at k=16, and k=8 projection (sample-count fidelity).

Usage:
  python analysis/run_exp_f_cost_accounting.py --repo-root .
  python analysis/run_exp_f_cost_accounting.py --reuse-parsed
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "analysis_outputs" / "exp_f_cost_accounting"

INPUT_PRICE_PER_M = 0.15
OUTPUT_PRICE_PER_M = 0.60

# Appendix sample-count ablation: k=8 global Spearman vs k=16 (ranking fidelity)
K8_SPEARMAN_VS_K16 = 0.9905663295639257

CHECKPOINT_COUNTS = {
    "frs_bins": ("reasoning_confidence_bins_results/judging_checkpoints/judged_*.json", "judged_samples"),
    "selection_gain": ("analysis/selection_gain_judge_outputs.csv", None),
    "trace0": ("analysis_outputs/trace0_k1_judging/judging_checkpoints/*.json", "judged_samples"),
    "unfiltered": ("analysis_outputs/unfiltered_reasoning/judging_checkpoints/unfiltered_judged_*.json", "judged_samples"),
    "haiku_sg": ("analysis_outputs/selection_gain_heldout_haiku/selection_gain_judged_traces_haiku.csv", None),
}

LOG_SOURCES = {
    "frs_bins": [REPO_ROOT / "reasoning_confidence_bins_run.log"],
    "trace0": [REPO_ROOT / "logs/trace0_k1_judging_live_stdout.log"],
    "unfiltered": [REPO_ROOT / "logs/unfiltered_full54_console_20260331_050835.log"],
}

USAGE_RE = re.compile(
    r"latency_s=(?P<lat>[0-9.]+).*?usage=(?P<usage>\{[^}]+\})"
)
MODEL_DS_RE = re.compile(r"eval_model=(?P<model>[^\s|]+).*?dataset=(?P<dataset>[^\s|]+)|dataset=(?P<dataset2>[^\s|]+).*?eval_model=(?P<model2>[^\s|]+)")
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def setup_logger(out_dir: Path) -> logging.Logger:
    log = logging.getLogger("exp_f_cost")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    for h in [logging.StreamHandler(sys.stdout), logging.FileHandler(out_dir / "run.log", encoding="utf-8")]:
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def parse_usage_dict(s: str) -> Tuple[int, int]:
    d = ast.literal_eval(s)
    return int(d["prompt_tokens"]), int(d["completion_tokens"])


def parse_log_file(path: Path, run_name: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return pd.DataFrame()
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "usage=" not in line or "prompt_tokens" not in line:
                continue
            m = USAGE_RE.search(line)
            if not m:
                continue
            pin, pout = parse_usage_dict(m.group("usage"))
            ts_m = TS_RE.search(line)
            md = MODEL_DS_RE.search(line)
            model = dataset = None
            if md:
                model = md.group("model") or md.group("model2")
                dataset = md.group("dataset") or md.group("dataset2")
            rows.append(
                {
                    "run": run_name,
                    "prompt_tokens": pin,
                    "completion_tokens": pout,
                    "latency_s": float(m.group("lat")),
                    "model": model,
                    "dataset": dataset,
                    "timestamp": ts_m.group(1) if ts_m else None,
                }
            )
    return pd.DataFrame(rows)


def wall_clock_from_df(df: pd.DataFrame) -> Optional[float]:
    if df.empty or df["timestamp"].isna().all():
        return None
    ts = pd.to_datetime(df["timestamp"].dropna())
    if len(ts) < 2:
        return None
    return float((ts.max() - ts.min()).total_seconds())


def cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return prompt_tokens * INPUT_PRICE_PER_M / 1e6 + completion_tokens * OUTPUT_PRICE_PER_M / 1e6


def count_checkpoints(repo: Path, log: logging.Logger) -> pd.DataFrame:
    rows = []
    for name, (pattern, field) in CHECKPOINT_COUNTS.items():
        path = pattern
        if pattern.endswith(".csv"):
            n = len(pd.read_csv(repo / pattern))
        else:
            files = glob.glob(str(repo / pattern))
            n = 0
            for f in files:
                data = json.load(open(f, encoding="utf-8"))
                if field == "judged_samples":
                    n += len(data.get("judged_samples", []))
                else:
                    n += len(data.get(field, []))
        rows.append({"run": name, "judge_calls": n})
        log.info("Checkpoint count %s: %d", name, n)
    return pd.DataFrame(rows)


def impute_run(df_template: pd.DataFrame, n_calls: int, run_name: str) -> pd.DataFrame:
    """Impute token/latency from FRS per-call means when logs unavailable."""
    if n_calls == 0:
        return pd.DataFrame()
    mean_pin = df_template["prompt_tokens"].mean()
    mean_pout = df_template["completion_tokens"].mean()
    mean_lat = df_template["latency_s"].mean()
    return pd.DataFrame(
        {
            "run": [run_name] * n_calls,
            "prompt_tokens": [mean_pin] * n_calls,
            "completion_tokens": [mean_pout] * n_calls,
            "latency_s": [mean_lat] * n_calls,
            "model": [None] * n_calls,
            "dataset": [None] * n_calls,
            "timestamp": [None] * n_calls,
            "imputed": [True] * n_calls,
        }
    )


def frs_per_pair_from_log(calls: pd.DataFrame, calls_per_pair: int = 250) -> pd.DataFrame:
    sub = calls[(calls["run"] == "frs_bins") & calls["model"].notna()].copy()
    if sub.empty:
        return pd.DataFrame()
    mean_pin = float(sub["prompt_tokens"].mean())
    mean_pout = float(sub["completion_tokens"].mean())
    mean_lat = float(sub["latency_s"].mean())
    pairs = sub.groupby(["model", "dataset"], as_index=False).size().rename(columns={"size": "log_lines"})
    pairs["n_calls"] = calls_per_pair
    pairs["prompt_tokens"] = pairs.apply(lambda _: int(round(mean_pin * calls_per_pair)), axis=1)
    pairs["completion_tokens"] = pairs.apply(lambda _: int(round(mean_pout * calls_per_pair)), axis=1)
    pairs["latency_s"] = mean_lat * calls_per_pair
    pairs["cost_usd"] = pairs.apply(
        lambda r: cost_usd(int(r["prompt_tokens"]), int(r["completion_tokens"])), axis=1
    )
    return pairs.drop(columns=["log_lines"])


def estimate_inference_tokens(repo: Path, log: logging.Logger) -> Dict[str, float]:
    """Sum generated-token proxy from pass16 JSONL (len of token prob lists)."""
    sys.path.insert(0, str(repo))
    from build_downstream_parquets import discover_jsonl_groups

    total_tokens = 0
    total_traces = 0
    t0 = time.perf_counter()
    for _, path in discover_jsonl_groups(str(repo)).items():
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                probs = row.get("chosen_token_probs_per_path", {}).get("epoch_0", [])
                for p in probs:
                    total_traces += 1
                    total_tokens += len(p)
    log.info(
        "Inference tokens (k=16): %d traces, %d tokens (%.1fs)",
        total_traces,
        total_tokens,
        time.perf_counter() - t0,
    )
    return {"n_traces": total_traces, "total_tokens": total_tokens}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--reuse-parsed", action="store_true")
    parser.add_argument("--skip-inference-scan", action="store_true")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    out_dir = repo / "analysis_outputs" / "exp_f_cost_accounting"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logger(out_dir)
    t_wall = time.perf_counter()

    parsed_cache = out_dir / "parsed_judge_calls.csv"
    if args.reuse_parsed and parsed_cache.exists():
        all_calls = pd.read_csv(parsed_cache)
        log.info("Reused %d parsed call rows", len(all_calls))
    else:
        frames = []
        for run_name, paths in LOG_SOURCES.items():
            for p in paths:
                df = parse_log_file(p, run_name)
                if not df.empty:
                    df["imputed"] = False
                    df["source_log"] = str(p.name)
                    frames.append(df)
                    log.info("Parsed %s: %d API lines from %s", run_name, len(df), p.name)
        if not frames:
            raise RuntimeError("No log lines parsed")
        logged = pd.concat(frames, ignore_index=True)
        frs_logged = logged[logged["run"] == "frs_bins"]
        ckpt = count_checkpoints(repo, log)

        # Impute selection_gain (same gpt-4o-mini rubric; no per-call usage in SG logs)
        sg_n = int(ckpt.loc[ckpt["run"] == "selection_gain", "judge_calls"].iloc[0])
        sg_imp = impute_run(frs_logged, sg_n, "selection_gain")
        logged = pd.concat([logged, sg_imp], ignore_index=True)

        all_calls = logged
        all_calls.to_csv(parsed_cache, index=False)

    ckpt = count_checkpoints(repo, log)
    total_judge_calls = int(ckpt["judge_calls"].sum())
    gpt4o_runs = ckpt[~ckpt["run"].isin(["haiku_sg"])]
    gpt4o_calls = int(gpt4o_runs["judge_calls"].sum())

    # Aggregate by run (gpt-4o-mini only for $) — use per-call means × checkpoint counts
    # (logs may duplicate lines ~2× vs actual API calls)
    run_rows = []
    for run_name in ckpt["run"]:
        if run_name == "haiku_sg":
            continue
        n_expected = int(ckpt.loc[ckpt["run"] == run_name, "judge_calls"].iloc[0])
        sub = all_calls[all_calls["run"] == run_name]
        if sub.empty:
            continue
        mean_pin = float(sub["prompt_tokens"].mean())
        mean_pout = float(sub["completion_tokens"].mean())
        mean_lat = float(sub["latency_s"].mean())
        pin = int(round(mean_pin * n_expected))
        pout = int(round(mean_pout * n_expected))
        lat_sum = mean_lat * n_expected
        lat_wall = wall_clock_from_df(sub)
        usd = cost_usd(pin, pout)
        run_rows.append(
            {
                "run": run_name,
                "judge_calls": n_expected,
                "mean_prompt_tokens_per_call": mean_pin,
                "mean_completion_tokens_per_call": mean_pout,
                "prompt_tokens": pin,
                "completion_tokens": pout,
                "total_tokens": pin + pout,
                "cost_usd": usd,
                "mean_latency_s": mean_lat,
                "sum_latency_s": lat_sum,
                "wall_clock_s": lat_wall,
                "log_lines_parsed": len(sub),
                "imputed_calls": int(sub.get("imputed", pd.Series([False] * len(sub))).sum()),
            }
        )

    run_df = pd.DataFrame(run_rows)
    run_df.to_csv(out_dir / "cost_by_run.csv", index=False)

    gpt4o_total_usd = float(run_df["cost_usd"].sum())
    gpt4o_pin = int(run_df["prompt_tokens"].sum())
    gpt4o_pout = int(run_df["completion_tokens"].sum())

    frs_pair = frs_per_pair_from_log(all_calls)
    frs_pair.to_csv(out_dir / "frs_judge_cost_per_pair.csv", index=False)
    avg_frs_per_pair_usd = float(frs_pair["cost_usd"].mean()) if not frs_pair.empty else float("nan")
    avg_frs_per_pair_calls = float(frs_pair["n_calls"].mean()) if not frs_pair.empty else 250.0

    # k=16 full pipeline (gpt-4o-mini judges only)
    per_pair_k16_judge_usd = gpt4o_total_usd / 54.0

    # k=8 projection: half inference tokens; FRS judge protocol unchanged (250/pair)
    inf = {"n_traces": 0, "total_tokens": 0}
    if not args.skip_inference_scan:
        inf = estimate_inference_tokens(repo, log)
    k8_inference_tokens = inf["total_tokens"] * 0.5

    frs_only_usd = float(run_df.loc[run_df["run"] == "frs_bins", "cost_usd"].iloc[0])
    k8_frs_judge_usd = frs_only_usd  # same 250/pair judge budget
    k8_total_usd = k8_frs_judge_usd  # judge-only pipeline; inference is local — report tokens separately

    corr_summary = pd.DataFrame(
        [
            {
                "metric": "total_judge_calls_all_runs",
                "value": total_judge_calls,
                "unit": "calls",
            },
            {
                "metric": "gpt4o_mini_judge_calls",
                "value": gpt4o_calls,
                "unit": "calls",
            },
            {
                "metric": "gpt4o_mini_total_cost_usd",
                "value": gpt4o_total_usd,
                "unit": "usd",
            },
            {
                "metric": "avg_judge_cost_per_pair_k16_usd",
                "value": per_pair_k16_judge_usd,
                "unit": "usd",
            },
            {
                "metric": "avg_frs_judge_cost_per_pair_k16_usd",
                "value": avg_frs_per_pair_usd,
                "unit": "usd",
            },
            {
                "metric": "avg_frs_judge_calls_per_pair",
                "value": avg_frs_per_pair_calls,
                "unit": "calls",
            },
            {
                "metric": "inference_tokens_k16",
                "value": inf["total_tokens"],
                "unit": "tokens",
            },
            {
                "metric": "inference_tokens_k8_projected",
                "value": k8_inference_tokens,
                "unit": "tokens",
            },
            {
                "metric": "frs_judge_cost_k8_projected_usd",
                "value": k8_frs_judge_usd,
                "unit": "usd",
            },
            {
                "metric": "k8_spearman_vs_k16_ranking",
                "value": K8_SPEARMAN_VS_K16,
                "unit": "spearman",
            },
        ]
    )
    corr_summary.to_csv(out_dir / "cost_summary.csv", index=False)

    lines = [
        "# Experiment F: cost accounting",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Judge call inventory (checkpoints)",
        "",
        "| Run | Calls |",
        "|---|---:|",
    ]
    for _, r in ckpt.iterrows():
        lines.append(f"| {r['run']} | {int(r['judge_calls'])} |")
    lines.extend(
        [
            f"| **Total** | **{total_judge_calls}** |",
            "",
            "## gpt-4o-mini judge cost ($0.15/1M input, $0.60/1M output)",
            "",
            f"- Total prompt tokens: **{gpt4o_pin:,}**",
            f"- Total completion tokens: **{gpt4o_pout:,}**",
            f"- Total judge cost (excl. Haiku replication): **${gpt4o_total_usd:.2f}**",
            f"- Avg judge cost per (model, benchmark) pair (all gpt-4o-mini runs): **${per_pair_k16_judge_usd:.3f}**",
            f"- Avg FRS-bin judge cost per pair (250 calls): **${avg_frs_per_pair_usd:.3f}**",
            "",
            "## Wall-clock (from log timestamps)",
            "",
        ]
    )
    for _, r in run_df.iterrows():
        wc = r["wall_clock_s"]
        wc_s = f"{wc:.0f}s" if wc and not np.isnan(wc) else "n/a"
        lines.append(f"- `{r['run']}`: {wc_s} wall-clock | {r['judge_calls']} calls | ${r['cost_usd']:.2f}")

    lines.extend(
        [
            "",
            "## Inference tokens (local generation, k=16 pass@16 JSONL)",
            "",
            f"- Total generated tokens (proxy): **{inf['total_tokens']:,}** across **{inf['n_traces']:,}** traces",
            "",
            "## k=8 projection (sample-count fidelity: global Spearman vs k=16 = "
            f"**{K8_SPEARMAN_VS_K16:.3f}**)",
            "",
            f"- Projected inference tokens at k=8: **{k8_inference_tokens:,.0f}** (×0.5)",
            f"- FRS judge cost at k=8: **${k8_frs_judge_usd:.2f}** (unchanged 250 judged traces/pair)",
            f"- Projected avg FRS judge cost per pair at k=8: **${k8_frs_judge_usd / 54:.3f}**",
            "",
            "## Notes",
            "",
            "- Selection-gain judge tokens **imputed** from FRS per-call averages (no usage in SG logs).",
            "- Haiku replication (5,400 calls) uses Claude Haiku — excluded from gpt-4o-mini $ totals.",
            "- Inference was run locally; only token counts reported (no GPU $ estimate).",
            "",
        ]
    )

    (out_dir / "key_numbers.md").write_text("\n".join(lines), encoding="utf-8")

    elapsed = time.perf_counter() - t_wall
    log.info("Done in %.1fs", elapsed)

    print("\n".join(lines[5:]))
    print(f"\nTotal gpt-4o-mini judge cost: ${gpt4o_total_usd:.2f} ({gpt4o_calls} calls)")
    print(f"Avg cost per pair (k=16, all gpt-4o-mini judges): ${per_pair_k16_judge_usd:.3f}")
    print(f"Avg FRS judge cost per pair (k=16): ${avg_frs_per_pair_usd:.3f}")
    print(f"Projected FRS judge cost at k=8: ${k8_frs_judge_usd:.2f} total (${k8_frs_judge_usd/54:.3f}/pair)")
    print(f"Inference tokens k=16: {inf['total_tokens']:,} | k=8 projected: {k8_inference_tokens:,.0f}")
    print(f"Wall-clock: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
