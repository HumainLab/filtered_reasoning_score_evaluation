# Rebuttal paragraph (Reviewer kp6q)

## Draft

If sampling and evaluating one trace reveals a similar trend, why bother using FRS?

Trace-0 reasoning score and FRS are correlated (Spearman ρ≈0.66 over 54 pairs), and trace-0 LOBO transfer to held-out trace-0 (mean ρ=0.65) is comparable to FRS→FRS (ρ=0.71) on some benchmarks — we do not claim FRS replaces all single-trace information. However, FRS adds **informativeness where accuracy is tied** and **cross-benchmark reasoning signal** that trace-0 does not carry.

Among **15** model pairs with |Δpass@1|≤2 pp, FRS separates models **13.1×** more than pass@1 itself (mean |ΔFRS|=12.2 pp vs mean |Δpass@1|=0.9 pp), versus **10.4×** for trace-0 RS and **6.9×** for unfiltered one-trace-per-question RS. FRS wins the head-to-head gap comparison in **67%** of these near-tie pairs (67% vs unfiltered).

For transferability, macro FRS predicts held-out FRS (LOBO ρ=0.71), but macro trace-0 **does not** predict held-out FRS (ρ=0.32) — the aggregate 3.8 pp macro gap reflects re-ranking that a single trace macro cannot recover. We observe **62** concrete pairwise rank reversals (≥2 pp) where FRS and trace-0 disagree; in **73%** of these, LOBO FRS from other benchmarks sides with the FRS winner. Unfiltered RS is closer to FRS on LOBO→FRS (ρ=0.47) but still under-amplifies near-equal pass@1 (6.9×) and diverges in rank (ρ≈0.45 vs FRS in prior analysis).

**Bottom line:** one trace gives a correlated snapshot; FRS aggregates confidence-filtered reasoning quality to (i) discriminate models at similar pass@1 and (ii) produce a benchmark-transferable ranking that trace-0 macro does not predict.
