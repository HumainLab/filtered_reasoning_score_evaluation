# Within-question confidence selection gain (yweD / kp6q)

## Framing

Holding the **same question** fixed, does selecting the most-confident among 16 traces
yield higher reasoning scores than a random trace or trace-0? This controls for
question difficulty (yweD easy-problem bias) and tests whether confidence filtering
adds signal beyond single-trace sampling (kp6q).

## Coverage

- Questions with 16 traces processed: **42,678**
- Selection rows (3 conditions × questions): **128,034**
- Unique traces needing judge: **122,517**
- Judge scores in cache: **10,167** (7.9%)
- Missing judge calls: **113,045**

## Cache-only results (complete paired questions)

- **Top-conf vs random** (n=349 questions):
  - Mean paired gain: **0.27 pp** [-2.81, 3.22]
  - Top-conf beats random on **32.1%** of paired questions

- **Top-conf vs trace-0** (n=1,071 questions):
  - Mean paired gain: **0.25 pp** [-1.25, 1.61]
  - Top-conf beats trace-0 on **21.7%** of paired questions

- Positive mean paired gain (top−random) on **27/54** model×benchmark pairs.

## Rebuttal-ready sentences

> Holding the question fixed, the most-confident trace improves reasoning score by **0.3** points (0–100 scale) over a random trace from the same 16 samples (paired n=349; top wins on 32% of questions). This directly controls for question difficulty.

## Files

- `per_trace_selection_table.csv`
- `missing_judge_worklist.csv`
