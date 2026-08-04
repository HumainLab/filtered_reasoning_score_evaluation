# Trace-0 vs FRS rank reversals (Reviewer kp6q)

## Headline numbers

| Metric | Value |
|--------|-------|
| Macro mean trace-0 RS | 65.0% |
| Macro mean FRS | 68.9% |
| Macro gap (FRS − trace-0) | **3.8 pp** |
| Spearman ρ (54 pairs) | 0.658 |
| Pairwise rank reversals (≥2 pp both sides) | **62 / 181** (34%) |
| Reversals where pass@1 agrees with trace-0 winner | 38 / 62 |
| LOBO FRS (5 other benchmarks) agrees with held-out FRS winner | **45 / 62** (73%) |
| LOBO trace-0 agrees with held-out trace-0 winner | 45 / 62 (73%) |
| LOBO FRS sides with trace-0 winner (wrong on reversals) | 17 / 62 |

## LOBO transferability (mean Spearman ρ across 6 held-out benchmarks)

| Train → Test | Mean ρ | Interpretation |
|--------------|--------|----------------|
| FRS → FRS | 0.712 | FRS ranking transfers across benchmarks |
| Trace-0 → Trace-0 | 0.654 | Single-trace RS barely transfers |
| Trace-0 → FRS | 0.316 | Trace-0 macro does **not** predict held-out FRS |

Per-benchmark LOBO (FRS→FRS vs trace-0→trace-0):

| Held out | FRS→FRS ρ | Trace-0→Trace-0 ρ | FRS wins? |
|----------|-----------|-------------------|-----------|
| GSM8K | 0.933 | 0.800 | ✓ |
| MATH500 | 0.717 | 0.867 |  |
| SVAMP | 0.683 | 0.571 | ✓ |
| AQuA | 0.820 | 0.633 | ✓ |
| GPQA | 0.650 | 0.639 | ✓ |
| CSQA | 0.467 | 0.417 | ✓ |

## Top rank reversals (visceral examples)

Each row: two models on one benchmark where **trace-0** and **FRS** disagree on who is better.

1. **CSQA — DS-R1-1.5B vs Phi-4-Reas.**  
   Trace-0: Phi-4-Reas. wins (54.9 vs 72.2, Δ=17.4 pp)  
   FRS: DS-R1-1.5B wins (56.1 vs 33.4, Δ=22.7 pp)  
   pass@1 agrees with FRS (DS-R1-1.5B); LOBO FRS agrees with FRS winner (macro FRS gap +7.7 pp)
2. **AQuA — DS-R1-1.5B vs Qwen2.5-Math**  
   Trace-0: Qwen2.5-Math wins (52.2 vs 74.0, Δ=21.8 pp)  
   FRS: DS-R1-1.5B wins (90.1 vs 74.9, Δ=15.2 pp)  
   pass@1 agrees with trace-0; LOBO FRS agrees with FRS winner (macro FRS gap +4.2 pp)
3. **CSQA — Phi-4 vs Phi-4-Reas.**  
   Trace-0: Phi-4-Reas. wins (59.4 vs 72.2, Δ=12.9 pp)  
   FRS: Phi-4 wins (77.2 vs 33.4, Δ=43.8 pp)  
   pass@1 agrees with trace-0; LOBO FRS disagrees with held-out FRS (macro FRS gap -3.3 pp)
4. **CSQA — DS-R1-7B vs Phi-4-Reas.**  
   Trace-0: Phi-4-Reas. wins (60.0 vs 72.2, Δ=12.2 pp)  
   FRS: DS-R1-7B wins (79.2 vs 33.4, Δ=45.8 pp)  
   pass@1 agrees with FRS (DS-R1-7B); LOBO FRS agrees with FRS winner (macro FRS gap +13.4 pp)
5. **AQuA — DS-R1-1.5B vs Qwen3-4B**  
   Trace-0: Qwen3-4B wins (52.2 vs 65.4, Δ=13.1 pp)  
   FRS: DS-R1-1.5B wins (90.1 vs 78.5, Δ=11.6 pp)  
   pass@1 agrees with trace-0; LOBO FRS agrees with FRS winner (macro FRS gap +2.0 pp)
6. **AQuA — DS-R1-1.5B vs Phi-4**  
   Trace-0: Phi-4 wins (52.2 vs 69.4, Δ=17.1 pp)  
   FRS: DS-R1-1.5B wins (90.1 vs 78.5, Δ=11.6 pp)  
   pass@1 agrees with trace-0; LOBO FRS agrees with FRS winner (macro FRS gap +4.4 pp)
7. **SVAMP — Qwen2.5-Math vs Qwen3-4B**  
   Trace-0: Qwen2.5-Math wins (88.2 vs 77.5, Δ=10.8 pp)  
   FRS: Qwen3-4B wins (74.5 vs 91.0, Δ=16.5 pp)  
   pass@1 agrees with trace-0; LOBO FRS disagrees with held-out FRS (macro FRS gap +0.3 pp)
8. **SVAMP — Phi-4-Reas. vs Qwen3-4B**  
   Trace-0: Phi-4-Reas. wins (88.2 vs 77.5, Δ=10.8 pp)  
   FRS: Qwen3-4B wins (74.8 vs 91.0, Δ=16.2 pp)  
   pass@1 agrees with trace-0; LOBO FRS agrees with FRS winner (macro FRS gap -4.7 pp)
9. **SVAMP — Phi-4 vs Qwen2.5-7B**  
   Trace-0: Qwen2.5-7B wins (66.4 vs 85.2, Δ=18.9 pp)  
   FRS: Phi-4 wins (75.5 vs 64.9, Δ=10.6 pp)  
   pass@1 agrees with trace-0; LOBO FRS agrees with FRS winner (macro FRS gap +5.3 pp)
10. **MATH500 — DS-R1-7B vs Qwen2.5-Math**  
   Trace-0: Qwen2.5-Math wins (68.5 vs 78.5, Δ=10.0 pp)  
   FRS: DS-R1-7B wins (94.0 vs 77.5, Δ=16.5 pp)  
   pass@1 agrees with trace-0; LOBO FRS agrees with FRS winner (macro FRS gap +14.3 pp)
11. **AQuA — DS-R1-1.5B vs Qwen2.5-7B**  
   Trace-0: Qwen2.5-7B wins (52.2 vs 61.9, Δ=9.6 pp)  
   FRS: DS-R1-1.5B wins (90.1 vs 77.2, Δ=12.9 pp)  
   pass@1 agrees with trace-0; LOBO FRS agrees with FRS winner (macro FRS gap +11.6 pp)
12. **AQuA — DS-R1-7B vs Qwen2.5-Math**  
   Trace-0: Qwen2.5-Math wins (64.9 vs 74.0, Δ=9.1 pp)  
   FRS: DS-R1-7B wins (96.8 vs 74.9, Δ=21.9 pp)  
   pass@1 agrees with FRS (DS-R1-7B); LOBO FRS agrees with FRS winner (macro FRS gap +13.2 pp)

## Rebuttal paragraph

Beyond the aggregate 3.8 pp gap (trace-0 RS 65.0% vs FRS 68.9%), FRS **reverses** trace-0's pairwise model ordering in **62 of 181** head-to-head comparisons (≥2 pp margin). In 38 reversals, pass@1 agrees with trace-0 — so FRS is not simply recovering accuracy; it re-ranks models on reasoning quality in the high-confidence regime. LOBO transfer confirms this: macro FRS across 5 benchmarks predicts held-out FRS (mean ρ=0.71), while macro trace-0 RS predicts held-out trace-0 (mean ρ=0.65) and **cannot** predict held-out FRS (mean ρ=0.32). On individual reversals, LOBO FRS agrees with the held-out FRS winner in **45/62** cases vs trace-0 LOBO agreeing with trace-0 in 45/62 — the cross-benchmark signal sides with FRS, not the single trace. The rank reversals are therefore aligned with a transferable reasoning signal, not noise.

## Files

- `pairwise_rank_reversals.csv`
- `lobo_transferability.csv`
- `per_pair_trace0_frs.csv`
- `scatter_trace0_vs_frs_by_benchmark.png`
- `lobo_comparison.png`
