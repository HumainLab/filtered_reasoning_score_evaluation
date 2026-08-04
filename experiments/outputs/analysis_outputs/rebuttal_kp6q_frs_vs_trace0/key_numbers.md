# kp6q: FRS vs trace-0 vs unfiltered — key numbers

## Data coverage

- Panel pairs: **54/54**
- Unfiltered RS: **available (54/54)**
- Macro mean FRS: 68.9% | trace-0: 65.0% | unfiltered: 68.5%
- Spearman(trace-0, FRS) over 54 pairs: **0.658**

## 1. Near-equal accuracy (|Δpass@1| ≤ 2 pp)

| Metric | Value |
|--------|-------|
| Eligible pairs | 15 |
| Mean |Δpass@1| | 0.93 pp |
| Mean |ΔFRS| | **12.18 pp** |
| Mean |Δtrace-0| | 9.64 pp |
| Mean |Δunfiltered| | 6.42 pp |
| Frac pairs FRS gap > trace-0 gap | **66.7%** |
| Frac pairs FRS gap > unfiltered gap | 66.7% |
| Amplification FRS | **13.14×** |
| Amplification trace-0 | 10.40× |
| Amplification unfiltered | 6.93× |

At |Δpass@1| ≤ 3 pp: 27 pairs, FRS amp 7.35× vs trace-0 5.46×.

## 2. LOBO transfer (mean Spearman ρ, 6 folds)

| Train → Test | Mean ρ |
|--------------|--------|
| FRS → FRS | **0.712** |
| trace-0 → FRS | 0.316 |
| trace-0 → trace-0 | 0.654 |
| unfiltered → FRS | 0.466 |
| unfiltered → unfiltered | 0.708 |
| pass@1 → FRS | 0.277 |

## 3. Rank reversals (≥2 pp both sides)

- FRS vs trace-0: **62/181** (34%)
- pass@1 agrees trace-0 winner: 61% of reversals
- LOBO FRS agrees FRS winner: **73%**
- FRS vs unfiltered: 49/179 (27%)

## 4. Bootstrap 95% CIs

- frac_frs_gap_gt_trace0_pass1_le_2pp: 0.667 [0.400, 0.867] (n=15)
- amplification_frs_pass1_le_2pp: 19.006 [8.512, 32.530] (n=14)
- amplification_trace0_pass1_le_2pp: 10.836 [6.885, 15.902] (n=14)
- frac_frs_gap_gt_trace0_pass1_le_3pp: 0.630 [0.444, 0.815] (n=27)
- amplification_frs_pass1_le_3pp: 12.466 [6.125, 20.099] (n=26)
- amplification_trace0_pass1_le_3pp: 7.332 [4.810, 10.460] (n=26)
- frac_frs_gap_gt_trace0_pass1_le_5pp: 0.676 [0.529, 0.824] (n=34)
- amplification_frs_pass1_le_5pp: 10.672 [5.571, 17.478] (n=33)
- amplification_trace0_pass1_le_5pp: 6.125 [3.905, 8.676] (n=33)
- lobo_frs_to_frs: 0.712 [0.594, 0.826] (n=6)
- lobo_trace0_to_frs: 0.316 [0.174, 0.458] (n=6)
- lobo_unfiltered_to_frs: 0.466 [0.322, 0.587] (n=6)
- frac_frs_reverses_trace0: 0.343 [0.276, 0.414] (n=181)

## Files

See README.md and CSV summaries in this directory.
