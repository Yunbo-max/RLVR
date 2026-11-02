# Rich64 Data Protocol — Train/Eval Eligibility Split

**Date:** 2026-11-02. **Branch:** `research/eesd-t64-budget-matched-20261101`.

This document fixes the data-protocol ambiguity that previously coupled the
final evaluation pool to the training-side F64+C8+A8 (>=80 unique-input)
requirement. It supersedes the earlier "validation+test union, >=80" freeze.

## 1. The two eligibility thresholds are now separate

| Pool | Source | Eligibility | Why |
|---|---|---|---|
| **Training** | official `train` split | >=80 unique input tests (F64 + C8 + A8), no conflicting oracles | grouped Shapley needs 64 public + 8 calibration + 8 reserved inputs |
| **Meta** | reserved from `train` split | same as training (>=80) | independent intervention problems, chosen first |
| **Final evaluation** | official `test` split only (test-only) | >=10 unique input tests, no conflicting oracles | evaluation does NOT run grouped Shapley; it only needs a reasonably reliable official test set |

The 64-feedback requirement is a *training* property. It is **not** imposed on
final evaluation, because evaluation never builds F64+C8+A8 Shapley groups.

## 2. Final evaluation uses the full official test bank

Each evaluation problem carries its **entire official test bank** (every unique
active input), not a fixed 80-test slice. Test counts therefore vary per problem
(frozen cohort: min 15, median 101, max 123 unique inputs). Pass@1 means
"all official tests of that problem pass".

This avoids treating "problems that happen to offer >=80 test inputs" as
representative of RunBugRun as a whole.

## 3. Frozen cohort (seed 1701, frozen 2026-11-02)

- train bug instances: 8000 (across 1033 distinct train problems, <=11/problem)
- meta problems: 64 (reserved from train)
- final evaluation problems: **199** (official test split, >=10 unique inputs)
- `train ∩ eval = ∅`, `train ∩ meta = ∅`, `meta ∩ eval = ∅` (problem_id level)

## 4. Statistical unit is problem_id, not bug instance

RunBugRun derives from IBM CodeNet (~4053 distinct problems). Multiple bug
instances can share one `problem_id`, so bug instances are **not** independent
statistical samples. All paired/bootstrap statistics must cluster by
`problem_id`. The frozen cohort's training side reuses problem_ids within the
train split by design (cap 16 instances/problem), and the evaluation side is one
problem_id per row.

## 5. Feasibility audit (official RunBugRun SQLite, Python language=5)

Distinct Python problems with active tests, by split:

| split | problems with tests | >=80 unique inputs | bug instances >=80 |
|---|---|---|---|
| train | 1583 | 1098 | 129,621 |
| validation | 249 | 161 | 5,645 |
| test | 247 | 162 | 5,678 |
| test (>=10 unique inputs) | — | 200 | — |

problem_id overlap train∩test = 0, train∩validation = 0 (validation/test overlap
heavily, which is why the old union only yielded 179 and was abandoned for a
test-only pool).

8K training is feasible: 1098 train problems x 16 instances/problem cap = 11,755
>= 8000. The 8K target uses bug **instances**, not 8000 independent problems.

## 6. Provenance

- `gold_fixed_code_never_queried`: true (the fixed_code column is never read)
- final evaluation = fixed official test bank per problem, not all RunBugRun
  official tests of the whole corpus
- frozen manifest records `db_sha256`, per-file `file_sha256`, and
  `final_evaluation_protocol`
