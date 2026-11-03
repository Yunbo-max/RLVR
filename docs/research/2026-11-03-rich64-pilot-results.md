# Rich64 64-Bug Pilot Results — 7-Arm Comparison

**Date:** 2026-11-03. **Branch:** `research/eesd-t64-budget-matched-20261101`.

First full seven-arm rich64 pilot run under the corrected execution protocol
(child-process test isolation, microbatch=16 coalition scoring, official
test-only evaluation). Seed 1701, 64 train bugs, G=4, T=64, K=4 groups, 5 rounds.

## 1. Execution protocol (post P0 fixes)

- **Test isolation**: each test runs in its own `python -I -S -B -c` child inside
  one bwrap sandbox; `start_new_session` + `killpg` reaps forked grandchildren on
  timeout. batch vs sequential consistency verified (9 tests).
- **Coalition scoring**: 16 group-coalitions scored in a single batched teacher
  forward (microbatch=16); microbatch 1/4/16 numerically identical, batch equals
  sequential per-coalition LogP (max abs diff ~5e-7).
- **Evaluation**: official test split only, full official test bank per problem
  (min 15, median 101, max 123 unique inputs), 199 independent problems.

## 2. Final R5 held-out performance

| arm | pass@1 | mean test-pass fraction | optimizer steps |
|---|---|---|---|
| shapley | **0.4844** | **0.6154** | 23 |
| random_subset | 0.4844 | 0.6075 | — |
| matched_random_updates | 0.4688 | 0.6033 | — |
| direct_max | 0.4688 | 0.6017 | 23 |
| sdpo (full feedback) | 0.4688 | 0.5612 | — |
| no_update (frozen) | 0.4531 | 0.5821 | 0 |
| audit_gate | 0.4375 | 0.5490 | — |

## 3. Paired statistics (R5, clustered by problem_id)

- shapley vs frozen: delta=+0.03125, fixes=3, regressions=1, **p=0.625 (n.s.)**
- shapley vs direct_max: delta=+0.015625, fixes=1, regressions=0, **p=1.0 (n.s.)**
- cost: shapley 23 steps ≡ direct_max 23 steps; both 20,432 teacher subset calls.

## 4. Conclusion (negative but clean)

1. **Shapley has no measurable independent value** at 64-bug scale: +0.031 over
   frozen is not significant (p=0.625), +0.016 over direct max is p=1.0.
2. **Shapley ≡ random_subset** (identical R5 pass@1 0.4844): sparse selective
   updates, not attribution quality, drive the small gain.
3. audit_gate degenerates (0.4375 < frozen 0.4531); sdpo full-feedback is
   high-variance.
4. Consistent with the earlier T4 lc-pilot, now confirmed under T64 + child-
   process isolation + official test-only evaluation.

## 5. Limitations

64 bugs, single seed 1701, low statistical power (all paired p >= 0.5). The
definitive question requires the 512-bug x 3-seed mechanism study (Shapley vs
Direct Max with R0/R2/R4 single-step Delta(theta) meta interventions).

## 6. Compute

shapley arm wallclock ~3.26h (vs >100h before the two batched optimizations);
7 arms run in two parallel 1xH100 jobs (matrixA ~8.5h, matrixB ~2.2h).
