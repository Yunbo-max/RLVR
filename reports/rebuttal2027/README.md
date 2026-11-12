# Historical original-EED rebuttal evidence

All JSON files in this directory are copied by unchanged Git blob identity
from RLVR commit `b8974183142a99574cca0b74daf57cca2035eebf`.
They are retained to preserve positive, null and adverse findings while new
development moves to the Rich64 foundation.

| File | Scope |
|---|---|
| `codearc__qwen25_7b__all_summary.json` | All 17 arms, 500 distinct problems, seed 1701 |
| `runbugrun__qwen25_7b__all_summary.json` | All 17 arms, 500 distinct problems, seed 1701 |
| `codearc__gemma3_4b__all_summary.json` | All 17 arms, 500 distinct problems, seed 1701 |
| `codearc__qwen25_7b__p0_summary.json` | Earlier subset summary; use all-summary for the full matrix |
| `runbugrun__qwen25_7b__p0_summary.json` | Earlier subset summary; use all-summary for the full matrix |
| `archived_prediction_audit.json` | Reanalysis of original prediction archives; not replicated LoRA training seeds |

The corresponding C00-C16/D01-D04 runners, configs, tests, prompt identities,
original PBPF dependency and amended runtime records remain in the
[complete source snapshot](https://github.com/Yunbo-max/RLVR/tree/b8974183142a99574cca0b74daf57cca2035eebf).
Read its [results](https://github.com/Yunbo-max/RLVR/blob/b8974183142a99574cca0b74daf57cca2035eebf/docs/research/2026-11-09-eesd-rebuttal-p0-results.md)
and [handoff](https://github.com/Yunbo-max/RLVR/blob/b8974183142a99574cca0b74daf57cca2035eebf/docs/research/2026-11-10-eesd-rebuttal-handoff.md).

These reports do not measure behavioral Shapley or the proposed Joint/Mix
method. Their paired intervals are descriptive, conditional on the one
training seed, and unadjusted across the named contrasts. This consolidation
does not independently rescore raw outcomes or certify missing runs.
