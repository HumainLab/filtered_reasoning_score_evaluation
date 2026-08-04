#!/usr/bin/env python3
"""Build MASTER_EXPERIMENT_RESULTS.md — Part I (index) + Part II (FRS) + Part III (tables)."""
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _fmt_cell(raw: str) -> str:
    s = raw.strip()
    if s == "":
        return ""
    try:
        x = float(s)
    except ValueError:
        return raw.replace("|", "\\|")
    if abs(x) >= 1e8 or (0 < abs(x) < 1e-6):
        return f"{x:.6g}"
    if abs(x - round(x)) < 1e-9 and abs(x) < 1e12:
        return str(int(round(x)))
    t = f"{x:.6f}".rstrip("0").rstrip(".")
    return t if t else "0"


def csv_to_markdown_table(path: Path) -> str:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return "_Empty file._\n"
    header = rows[0]
    body = rows[1:]
    lines = [
        "| " + " | ".join(h.replace("|", "\\|") for h in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(_fmt_cell(c) for c in padded[: len(header)]) + " |")
    return "\n".join(lines) + "\n"


def _pct_str(val: float | None) -> str:
    if val is None:
        return "—"
    if val != val:  # NaN
        return "—"
    return f"{100.0 * val:.2f}"


def _spread_pp(vals: list[float | None]) -> str:
    """First − last numeric column, in percentage points (like Experiment A spread)."""
    usable: list[float] = []
    for v in vals:
        if v is None or v != v:
            continue
        usable.append(v)
    if len(usable) < 2:
        return "—"
    return f"{100.0 * (usable[0] - usable[-1]):+.2f}"


def disjoint_bins_population_accuracy_wide(path: Path) -> str:
    """Experiment A–style wide tables: one block per dataset, disjoint percentile bands."""
    with path.open(newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    if not rows:
        return "_Empty file._\n"

    # (model, dataset) -> list of (bin_start, bin_end, acc)
    by_pair: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    for r in rows:
        try:
            a = float(r["mean_accuracy_population"])
            bs = float(r["bin_start_pct"])
            be = float(r["bin_end_pct"])
        except (KeyError, ValueError):
            continue
        by_pair[(r["model"], r["dataset"])].append((bs, be, a))

    datasets = sorted({d for _, d in by_pair.keys()})
    lines: list[str] = [
        "*Disjoint bins: each column is **population accuracy** (terminal correctness) within that "
        "confidence percentile band of the ranked pool — **not** cumulative top‑K. "
        "Column labels match the **right edge** of each band (same style as §1 top‑K% columns where they overlap). "
        "This run uses five bands covering the top 50% of the pool (see `bin_label` in the CSV).*\n",
    ]

    for ds in datasets:
        models = sorted({m for (m, d) in by_pair if d == ds})
        # Column order: union of bin end percentiles for this dataset (sorted).
        ends_set: set[float] = set()
        for m in models:
            for t in by_pair.get((m, ds), []):
                ends_set.add(t[1])
        col_ends = sorted(ends_set)
        col_headers = [f"{int(e)}%" if e == int(e) else f"{e:.1f}%" for e in col_ends]

        lines.append(f"\n#### {ds}\n")
        header = "| Model | " + " | ".join(col_headers) + " | Spread (pp) |"
        sep = "| :--- | " + " | ".join("---:" for _ in col_headers) + " | ---: |"
        lines.extend([header, sep])

        for m in models:
            by_end = {t[1]: t[2] for t in by_pair.get((m, ds), [])}
            vals = [by_end.get(e) for e in col_ends]
            cells = [_pct_str(v) for v in vals]
            lines.append(
                "| "
                + m
                + " | "
                + " | ".join(cells)
                + " | "
                + _spread_pp(vals)
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def cumulative_population_accuracy_wide(path: Path) -> str:
    """Wide tables for cumulative_accuracy_population by topk_pct (same column style as §1)."""
    with path.open(newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    if not rows:
        return "_Empty file._\n"

    by_pair: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for r in rows:
        try:
            k = int(float(r["topk_pct"]))
            a = float(r["cumulative_accuracy_population"])
        except (KeyError, ValueError):
            continue
        by_pair[(r["model"], r["dataset"])][k] = a

    ks_global: list[int] = sorted({k for d in by_pair.values() for k in d})
    col_headers = [f"{k}%" for k in ks_global]

    lines = [
        "*Cumulative top‑K% (population accuracy): column **K%** is mean terminal correctness over **all traces "
        "in the top K%** of the confidence-ranked pool (same cumulative semantics as the accuracy ablation’s "
        "top‑K% slices, modulo implementation details in each script).*\n",
    ]

    datasets = sorted({d for _, d in by_pair.keys()})
    for ds in datasets:
        models = sorted({m for (m, d) in by_pair if d == ds})
        lines.append(f"\n#### {ds}\n")
        header = "| Model | " + " | ".join(col_headers) + " | Spread (pp) |"
        sep = "| :--- | " + " | ".join("---:" for _ in col_headers) + " | ---: |"
        lines.extend([header, sep])
        for m in models:
            mp = by_pair.get((m, ds), {})
            vals: list[float | None] = [mp.get(k) for k in ks_global]
            cells = [_pct_str(v) if v is not None else "—" for v in vals]
            lines.append(
                "| "
                + m
                + " | "
                + " | ".join(cells)
                + " | "
                + _spread_pp(vals)
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


PART_I = """# Master experiment results — FRS / confidence / accuracy

> **Generated:** {generated}
> **Workspace:** `threshold/` (paths below are relative to this folder unless noted)

This document indexes **accuracy** ablations, **median‑split** analyses, **confidence‑bin** population accuracy (disjoint and cumulative), sample‑count ablations, and judge‑free top‑K% trace accuracy.  
**Part II** reproduces the numerical tables from `FRS_experiment_results.md`.  
**Part III** embeds CSV data in **wide tables** where noted (regenerate after re-running experiments).

---

## Quick index — where everything lives

| Topic | Script(s) | Main outputs |
|:---|:---|:---|
| **Top‑K% accuracy** (pooled traces, confidence filter) | `topk_ablation.py` | `topk_ablation_results/topk_ablation_results.csv`, `topk_ablation_*.png`, `topk_ablation_all_datasets_accuracy.png` |
| **Median split** (high vs low confidence half) | `correctness_conditioned.py` | `correctness_conditioned_results/correctness_conditioned.csv`, plots |
| **Population accuracy by confidence bins** | `reasoning_confidence_bins.py` | `reasoning_by_confidence_bin.csv`, `reasoning_cumulative_topk.csv`, PNGs |
| **Judge‑free top‑K% trace accuracy** | `reasoning_confidence_bins.py topk-accuracy` | `reasoning_confidence_bins_results/topk_trace_accuracy.csv` (if generated) |
| **Sample‑count ablation** (k = 4, 8, 12, 16 traces) | `sample_count_ablation.py` | `sample_count_ablation_results/ablation_*.csv`, `checkpoints/chunks/`, PDFs |

---

## Shared experimental setup

### Models & benchmarks

- **9 models** and **6 benchmarks** (AQuA, CommonsenseQA, GPQA, GSM8K, MATH500, SVAMP) — see Part II §1.1–1.2 for the full table.
- **Data:** pass@**16** JSONL per problem: 16 stochastic traces (temperature **0.7**), each with `score[]`, CoT `code[]`, and token probabilities for confidence.

### Confidence definition (used almost everywhere)

**Per‑trace confidence** = mean probability of the **lowest 10%** of token probabilities in that trace (mean over the bottom decile of token probs).  
Rationale: weakest tokens carry more information about uncertainty than near‑deterministic tokens.

---

## 1. Top‑K% accuracy ablation (`topk_ablation.py`)

**Question:** If we pool all traces, rank by confidence, and take **only the top K%** of traces (by a **confidence threshold** on the pooled distribution), how does **accuracy** change?

**Cutoffs:** `K ∈ {{10, 20, 30, 50, 70, 100}}` (see `PERCENTILES` in `topk_ablation.py`).

**Outputs:** `topk_ablation_results/topk_ablation_results.csv` (accuracy %, n traces, mean confidence per row).  
**Figures:** per‑dataset line charts, `topk_ablation_spread_heatmap.png`, combined grid `topk_ablation_all_datasets_accuracy.png`.

**Note:** The implementation uses a **percentile threshold** on the pooled `confidence` column (not always exactly `⌊n·K/100⌋` traces). For **count‑based** top‑K% aligned with the reasoning‑bins pipeline, use `python reasoning_confidence_bins.py topk-accuracy` (see §4).

---

## 2. Correctness‑conditioned / median split (`correctness_conditioned.py`)

**Question:** Split traces at the **median confidence** (per model×dataset). Is accuracy higher in the **high‑confidence half** than the **low‑confidence half**?

**Outputs:** `correctness_conditioned_results/correctness_conditioned.csv`, plots such as `median_split_gap_by_dataset.png`.

**Full tables:** Part II §3 and Part III (CSV dump).

---

## 3. Population accuracy by confidence bins (`reasoning_confidence_bins.py`)

**Question:** After ranking traces by confidence, split the pool into **equal‑count percentile bins** and report **mean terminal correctness** over **all traces in each bin** (`mean_accuracy_population`) and over **cumulative top‑K% slices** (`cumulative_accuracy_population` in `reasoning_cumulative_topk.csv`).

**Wide tables (Experiment A–style columns):** Part III formats **disjoint** bin accuracy and **cumulative** top‑K% accuracy with columns **10% | 20% | …** aligned with the accuracy ablation where the run supports it. Disjoint bands in the current CSV cover the **top 50%** of the pool in five steps; use `--n-bins 10` and `--top-pool-frac 1.0` to obtain ten disjoint deciles through **100%**.

**Key CSVs:**

| File | Contents |
|:---|:---|
| `reasoning_by_confidence_bin.csv` | Per bin: `mean_accuracy_population` (plus optional judge metadata from the same run) |
| `reasoning_cumulative_topk.csv` | Cumulative: `cumulative_accuracy_population` at each `topk_pct` |
| `reasoning_sampling_metadata.json` | `top_pool_fraction`, `n_bins`, bin definitions |

**Plots:** `reasoning_bins_*.png`, `reasoning_cumulative_*.png`.

**Methods detail:** `reasoning_confidence_bins_methods.md`.

---

## 4. Judge‑free top‑K% trace accuracy (`topk-accuracy` subcommand)

**Command:**

```bash
python reasoning_confidence_bins.py topk-accuracy --data-root . --output-dir ./reasoning_confidence_bins_results \\
  --top-pool-frac 0.5 --k-pcts 10,20,30,40,50
```

**What it does:** After optional **pool slice** (`--top-pool-frac`), sort remaining traces by confidence, take the **first ⌊n·K/100⌋** traces for each K, report **mean(correct)** — no API calls.

**Output:** `reasoning_confidence_bins_results/topk_trace_accuracy.csv` (embedded in Part III when present).

---

## 5. Sample‑count ablation — why k ∈ {{4, 8, 12, 16}}? (`sample_count_ablation.py`)

### Motivation

- The production dataset is **pass@16**: up to **16** traces per problem.
- Many analyses (median split, FRS‑style “gap”) need **enough** traces to split into high/low confidence halves **with** variance across bootstrap seeds.
- **k = 16** is the **full** sample: **deterministic** median split (no subsampling noise for that setting) — used as the **reference** when comparing Spearman correlation of rankings vs k=16.
- **k ∈ {{4, 8, 12}}** are **strict subsets**: for each problem we **subsample k traces without replacement** (with multiple RNG seeds), compute median split on that subset, and measure **FRS‑style** metrics (high‑confidence minus low‑confidence accuracy, etc.).

### What the script reports

- Per `(model, dataset, k_sub)`: mean/std of the FRS metric across bootstrap seeds.
- **Spearman vs k=16:** how much **ranking of models** (by dataset) agrees with the full k=16 reference when only k traces are available.

### Numerical outputs (all embedded in Part III)

| File | Description |
|:---|:---|
| `ablation_frs_variance_table.csv` | Mean bootstrap std of FRS (across models) by dataset × `k_sub` |
| `ablation_rankings.csv` | Per model×dataset×`k_sub`: mean/std FRS, Spearman vs k=16 |
| `ablation_rankings_global_spearman.csv` | Global Spearman vs k=16 (single row per `k_sub`) |
| `ablation_regime_separation.csv` | Per `k_sub`, model, dataset: mean FRS gap vs reference gap at k=16, sign agreement |
| `ablation_per_bootstrap_detail.csv` | **Full log:** one row per bootstrap × model × dataset × `k_sub` (`frs_accuracy_proxy_pct`, `acc_unconfident_half_pct`, `gap_pp`) |

The shard files `checkpoints/chunks/<Model>__<Dataset>.csv` are **splits** of the same per‑bootstrap rows as `ablation_per_bootstrap_detail.csv` (one shard per model×dataset); Part III embeds **only** the consolidated file so nothing is duplicated.

### Figures

- `ablation_ranking_stability.pdf`, `ablation_regime_heatmap.pdf`, `ablation_frs_variance.pdf`

### Why not only {{4, 2, 8}}?

The code uses **`K_VALUES = [4, 8, 12, 16]`** (see `sample_count_ablation.py`): **12** bridges small subsamples and full pass@16; **16** is the **no‑subsampling** baseline (variance 0 in `ablation_frs_variance_table` for k=16).

---

## 6. Figures (examples)

| Pattern | Example path |
|:---|:---|
| All‑datasets top‑K accuracy grid | `topk_ablation_results/topk_ablation_all_datasets_accuracy.png` |
| Sample ablation | `sample_count_ablation_results/ablation_*.pdf` |
| Bins / cumulative | `reasoning_confidence_bins_results/reasoning_bins_*.png`, `reasoning_cumulative_*.png` |

---

## How to regenerate this markdown

```bash
cd threshold
python build_master_results_doc.py
```

## How to regenerate experiments

```bash
# Top-K ablation (from JSONL)
python topk_ablation.py --data_root . --output_dir ./topk_ablation_results

# Correctness conditioned
python correctness_conditioned.py --data_root . --output_dir ./correctness_conditioned_results

# Reasoning bins (needs PORTKEY_API_KEY if using judge path inside the script)
python reasoning_confidence_bins.py run --data-root . --output-dir ./reasoning_confidence_bins_results

# Judge-free top-K accuracy only
python reasoning_confidence_bins.py topk-accuracy --data-root . --output-dir ./reasoning_confidence_bins_results

# Sample count ablation
python sample_count_ablation.py --data_root . --output_dir ./sample_count_ablation_results
```

---

# Part II — Full numerical tables (from `FRS_experiment_results.md`)

The following section is the **complete** prior document (Experiments A–B, summary stats, findings, limitations).

"""


PART_III_HEAD = """

---

# Part III — Tables (CSV-backed)

Sections below are **generated** from CSVs under `threshold/`. If a file is missing, that subsection is omitted.

"""


def main() -> None:
    generated = date.today().isoformat()
    frs_path = ROOT / "FRS_experiment_results.md"
    if not frs_path.is_file():
        raise SystemExit(f"Missing {frs_path}")
    frs = frs_path.read_text(encoding="utf-8")

    chunks: list[str] = [PART_I.format(generated=generated), frs, PART_III_HEAD]

    def add_section(title: str, body: str) -> None:
        chunks.append(f"\n### {title}\n\n")
        chunks.append(body)
        if not body.endswith("\n"):
            chunks.append("\n")

    add_section(
        "Top‑K ablation — `topk_ablation_results/topk_ablation_results.csv`",
        csv_to_markdown_table(ROOT / "topk_ablation_results/topk_ablation_results.csv")
        if (ROOT / "topk_ablation_results/topk_ablation_results.csv").is_file()
        else "_File not found._\n",
    )

    add_section(
        "Correctness conditioned — `correctness_conditioned_results/correctness_conditioned.csv`",
        csv_to_markdown_table(ROOT / "correctness_conditioned_results/correctness_conditioned.csv")
        if (ROOT / "correctness_conditioned_results/correctness_conditioned.csv").is_file()
        else "_File not found._\n",
    )

    rb = ROOT / "reasoning_confidence_bins_results/reasoning_by_confidence_bin.csv"
    if rb.is_file():
        add_section(
            "Disjoint bins — population accuracy (wide, Experiment A–style columns)",
            disjoint_bins_population_accuracy_wide(rb),
        )

    rc = ROOT / "reasoning_confidence_bins_results/reasoning_cumulative_topk.csv"
    if rc.is_file():
        add_section(
            "Cumulative top‑K% — population accuracy (wide, same column style as §1 where overlapping)",
            cumulative_population_accuracy_wide(rc),
        )

    chunks.append(
        "\n### §5 Sample‑count ablation — all numerical outputs\n\n"
        "These tables match **§5** in Part I. "
        "Shard files under `sample_count_ablation_results/checkpoints/chunks/` are not repeated here "
        "(they partition `ablation_per_bootstrap_detail.csv` by model×dataset).\n\n"
    )

    for rel, title in [
        ("sample_count_ablation_results/ablation_frs_variance_table.csv", "FRS variance by dataset"),
        ("sample_count_ablation_results/ablation_rankings.csv", "Rankings vs k=16 (mean/std FRS, Spearman)"),
        ("sample_count_ablation_results/ablation_rankings_global_spearman.csv", "Global Spearman vs k=16"),
        ("sample_count_ablation_results/ablation_regime_separation.csv", "Regime separation"),
    ]:
        p = ROOT / rel
        add_section(f"{title} — `{rel}`", csv_to_markdown_table(p) if p.is_file() else f"_File not found: `{rel}`_\n")

    detail = ROOT / "sample_count_ablation_results/ablation_per_bootstrap_detail.csv"
    add_section(
        "Per-bootstrap detail (full log) — `sample_count_ablation_results/ablation_per_bootstrap_detail.csv`",
        csv_to_markdown_table(detail) if detail.is_file() else "_File not found._\n",
    )

    opt = ROOT / "reasoning_confidence_bins_results/topk_trace_accuracy.csv"
    if opt.is_file():
        add_section(
            "Judge‑free top‑K trace accuracy — `reasoning_confidence_bins_results/topk_trace_accuracy.csv`",
            csv_to_markdown_table(opt),
        )

    out = ROOT / "MASTER_EXPERIMENT_RESULTS.md"
    out.write_text("".join(chunks), encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
