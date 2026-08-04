# `filtered-cot/` — earlier MATH500 judge run

This directory has no benchmark suffix for historical reasons. It contains an
**earlier judge run over the same MATH500 filtered traces** as
`filtered-cot-math500/`, not a separate benchmark.

Evidence: identical model file names, identical retained-trace counts (e.g. 148
for DS-R1-7B), and the source JSONLs match `filtered-cot-math500/` byte for byte.
The two runs differ only in judging date and slightly in scores:

| | `filtered-cot/` | `filtered-cot-math500/` |
| --- | --- | --- |
| Judged | 2026-02-09 | 2026-03-10 |
| DS-R1-7B faithfulness | 0.8277 | 0.8294 |

The small differences are judge non-determinism across API calls at
`temperature=0`, which is worth knowing when comparing runs.

**Which to use:** prefer `filtered-cot-math500/` as the newer run. This
directory is kept because it has complete results for `phi_4`, where the newer
run is partial.

Paper numbers come from
`frs_experiments/reasoning_confidence_bins_results/judging_checkpoints/`, not
from either of these directories.
