# Held-out selection-gain (Claude Haiku 4.5)

## Key numbers

| Metric | Value |
|--------|-------|
| Held-out Pearson r (gain_haiku vs frs_pct) | 0.3674 |
| Bootstrap 95% CI | [0.1286, 0.5556] |
| Spearman rho (robustness) | 0.3514 |
| Published mini-judge r | 0.4906 |
| Per-trace agreement Pearson (mini vs haiku RS) | 0.8254 |
| Per-trace agreement Spearman | 0.8274 |
| Mean selection gain (mini) | -0.0260 |
| Mean selection gain (haiku) | -0.0300 |
| Judge model | `@irom-ll37364-an-069b10/claude-haiku-4-5` |
| Total judge calls | 5400 |
| API cost (est.) | $0.00 |
| Failed traces (after retries) | 0 |


## Rebuttal-ready summary

We replicated the selection-gain predictor analysis using a held-out judge (Claude Haiku 4.5,
`@irom-ll37364-an-069b10/claude-haiku-4-5`) on the same frozen 5,400-trace worklist used for the mini-judge run.
Pair-level mean selection gain from Haiku correlates with published FRS at r=0.367 (95% CI
[0.129, 0.556]), compared to r=0.4906 for the original mini-judge pipeline.
Per-trace reasoning-score agreement between judges is Pearson r=0.825 (Spearman
0.827), indicating reasonable alignment on the same rubric.
Total API cost was approximately $0.00; 0 traces failed after retries.

## Failed traces

None.