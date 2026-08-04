# Experiment A: style/length partial-correlation

Generated: 2026-05-23T05:48:39.581381+00:00

## Headline

- Raw FRS-proxy (mean top-conf reasoning) ↔ selection gain: **r = 0.514, p = 0.0001**
- Partial (control length + TTR + max 4-gram rep jointly): **r = 0.308, p = 0.0234**
- N = 54 (model, benchmark) pairs

## Single-feature partials (control one covariate at a time)

| Control removed | Partial r | p | Δ from raw |
|---|---:|---:|---:|
| length | 0.469 | 0.0003 | -0.045 |
| TTR (lexical diversity) | 0.313 | 0.0211 | -0.201 |
| max 4-gram repetition | 0.477 | 0.0003 | -0.037 |

**Read:** large |Δ| ⇒ that feature explains more of the raw FRS↔gain correlation.

## Sensitivity: exclude `Phi-4-Reas.` (N = 48 pairs)

- Raw r = **0.543** (p = 0.0001)
- TTR-only partial r = **0.388** (p = 0.0065, Δ from raw = -0.155)
