# Experiment F: cost accounting

Generated: 2026-05-23T06:12:06.660693+00:00

## Judge call inventory (checkpoints)

| Run | Calls |
|---|---:|
| frs_bins | 13500 |
| selection_gain | 5400 |
| trace0 | 2700 |
| unfiltered | 5400 |
| haiku_sg | 5400 |
| **Total** | **32400** |

## gpt-4o-mini judge cost ($0.15/1M input, $0.60/1M output)

- Total prompt tokens: **56,794,228**
- Total completion tokens: **918,000**
- Total judge cost (excl. Haiku replication): **$9.07**
- Avg judge cost per (model, benchmark) pair (all gpt-4o-mini runs): **$0.168**
- Avg FRS-bin judge cost per pair (250 calls): **$0.088**

## Wall-clock (from log timestamps)

- `frs_bins`: 879s wall-clock | 13500 calls | $4.75
- `selection_gain`: n/a wall-clock | 5400 calls | $1.90
- `trace0`: 694s wall-clock | 2700 calls | $0.62
- `unfiltered`: 373s wall-clock | 5400 calls | $1.81

## Inference tokens (local generation, k=16 pass@16 JSONL)

- Total generated tokens (proxy): **660,703,384** across **682,848** traces

## k=8 projection (sample-count fidelity: global Spearman vs k=16 = **0.991**)

- Projected inference tokens at k=8: **330,351,692** (×0.5)
- FRS judge cost at k=8: **$4.75** (unchanged 250 judged traces/pair)
- Projected avg FRS judge cost per pair at k=8: **$0.088**

## Notes

- Selection-gain judge tokens **imputed** from FRS per-call averages (no usage in SG logs).
- Haiku replication (5,400 calls) uses Claude Haiku — excluded from gpt-4o-mini $ totals.
- Inference was run locally; only token counts reported (no GPU $ estimate).
