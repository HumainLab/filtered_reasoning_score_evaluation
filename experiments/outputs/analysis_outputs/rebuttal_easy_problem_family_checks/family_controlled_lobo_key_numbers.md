# Family-controlled LOBO — key numbers

_Generated: 2026-05-22T16:18:00.964208+00:00_

## Method

Leave-one-**benchmark**-out: aggregate train FRS (or pass@1) over 5 benchmarks per model, Spearman vs held-out benchmark. Variants drop DS-R1-7B, DS-R1-1.5B, or both (7 models).

## Headline

- **All 9 models:** mean LOBO Spearman(FRS→FRS) = **0.7117** (6/6 folds positive).
- **Exclude both DS-R1 (7 models):** mean = **0.495** (6/6 positive).
- **exclude_DS-R1-7B:** mean = **0.588** (6/6 positive).
- **exclude_DS-R1-1.5B:** mean = **0.6634** (6/6 positive).

## Does LOBO survive removing both DS-R1?

**Yes, directionally.** Mean correlation remains **positive** with 7 models, though magnitude may be **weaker** than the 9-model panel. Report both numbers; do not claim identical strength.

## What we can safely claim

- Cross-benchmark FRS ranking alignment is **not solely** driven by including both DS-R1 variants.
- With **n=7**, LOBO is **illustrative**; prefer reporting mean/median ρ and fold counts, not p-values alone.

## What NOT to overclaim (n=7 after removing DS-R1)

- Independent replication or tight confidence intervals.
- That every held-out benchmark fold stays significant.
- Causal transfer of FRS across domains.

See `family_controlled_lobo_pivot.csv` for per-fold Spearman values.
