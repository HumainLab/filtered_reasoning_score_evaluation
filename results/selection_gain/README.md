# Appendix S — Selection-gain materials

## Layout

| Path | Description |
|------|--------------|
| `selection_gain_appendix_s.zip` | Original archive from the paper bundle (tiny; keep as provenance). |
| `appendix_s/` | Unzipped Appendix S payloads (canonical extraction target). |
| `appendix_s/sorted/` | Same CSVs as `appendix_s/`, deterministically sorted for diff-friendly review. |

### Unzipped files (`appendix_s/`)

1. **`selection_gain_judge_outputs.csv`** (~5,400 trace rows + header)  
   - Per-trace GPT judge outputs for the selection-gain experiment: `model`, `benchmark`, `question_id`, `trace_id`, `selection_type` (e.g. top-confidence vs random), `reasoning_score`, `confidence`, `raw_json`, etc.

2. **`selection_gain_pair_level.csv`** (54 pairs + header)  
   - One row per model×benchmark: `mean_selection_gain`, means of top vs random reasoning, `n_questions` (= 50).

3. **`selection_gain_predictor_results.csv`**  
   - Precomputed correlations vs mean selection gain across 54 pairs (includes `frs_pct` Pearson **r ≈ 0.491**, matching Appendix S).

### Sorted copies (`appendix_s/sorted/`)

- **`selection_gain_judge_outputs.csv`** — sorted by `(model, benchmark, question_id, trace_id, selection_type)`.
- **`selection_gain_pair_level.csv`** — sorted by `(benchmark, model)`.
- **`selection_gain_predictor_results.csv`** — sorted by `predictor` name.

To re-extract from the zip:

```bash
cd reasoning-models-eval/selection-gain
unzip -o selection_gain_appendix_s.zip -d appendix_s
```
