# EESD2 Rebuttal Completion Implementation Plan

> **For agentic workers:** Use systematic debugging and targeted regression tests for each task; preserve the independent-case interface and review the integrated branch before publishing.

**Goal:** Finish the authorized EESD2 continuation: independent C00–C16 cases, usable original model/dataset coverage, and reviewer-specific evidence collection.

**Architecture:** Preserve the submitted four-state lexical Dirichlet EED and the existing exact-source PBPF bridge. Add a separately named prospective matched protocol for portable evaluation across the twelve registered cells; its results must never be called exact replications of the historical mixed decoding/execution protocols. Reuse case receipts, frozen inputs and per-case output directories.

**Tech stack:** Python, pytest, PyYAML, PyTorch/Transformers/PEFT on the user's GPU host; the pinned PBPF source for task semantics and isolated code execution.

**Spec:** `docs/research/2026-11-07-iclr2027-final-reviewer-case-matrix.md`, `docs/research/2026-11-07-dirichlet-token-credit-rebuttal.md`, and the user's request to complete EESD2 with separate Python cases and full model/dataset coverage.

## Global constraints

- C00–C16 remain separately executable files with one immutable output directory per cell/case/seed.
- Original EED uses four transitions, alpha 0.1, lambda 0.5, kappa 16 and common forward KL beta 0.03. Token-aware C10–C12 are an extension.
- All comparison arms share source rows, initial adapter bytes, token schedule and heldout source/problem IDs.
- Original source pin: `0200283849274b49e303638b91f142acd8bba984` in `yuhanlydia/PBPF`.
- No invented source pairs, missing future labels, historical optimizer counts or GPU results.
- Original Gemma prompt helper is absent from the pinned repository; historical FP4/sampled decoding and direct Replay execution differ from the prospective matched protocol. Preserve these distinctions.
- No generated program executes outside an isolated sandbox. Infrastructure failures must not become model test failures.
- The user explicitly instructed that the failed CI workflow must not be modified or repaired; do not edit Actions/workflow configuration or manually rerun it.

## Review focus

- The prepared model/cell/revision and runtime adapter must remain bound through training, generation, evaluation and collection.
- P0 must be independently collectable without waiting for C10–C12; C00 must be included in fresh-vs-frozen comparisons.
- Duplicate, missing, cross-domain, leaked or changed source/problem mappings must fail before a scientific report is emitted.
- Gemma's model class and chat template must be handled explicitly; no unavailable historical prompt file may be silently substituted.
- Planned code coverage, CPU verification, GPU execution and measured scientific outcomes must be separate statuses.

## Task 1: Close training and shared-initialization gaps

**Files:** `scripts/run_eesd_rebuttal_case.py`, `scripts/run_eesd_weighted_sft.py`, new `scripts/create_eesd_rebuttal_init.py`, targeted tests.

- [x] Audit current independent launchers and training behavior; reproduce substantive defects with targeted tests.
- [x] Fix only defects affecting model identity, actual training, frozen provenance or independent outputs.
- [x] Add a model-config-driven shared initializer for all three pinned backbones, with a reusable manifest and no overwrite.
- [x] Verify the affected CPU tests and document GPU prerequisites.

## Additional recovered work: D03 historical predictor audit

- [x] Recover source/problem/test identities from twelve original cache/NPZ archives.
- [x] Recompute the effective-arm development selection grid and the matched fixed/effective primary predictions using pinned original source.
- [x] Verify 36,000 archived labels/probabilities and publish per-seed NLL/Brier/argmax metrics with 10,000 problem-cluster bootstrap draws.
- [x] Distinguish five execution-outcome classes from the four correction transitions, and candidate-generation seeds from downstream fitting seeds.
- [x] Preserve sign flips and the limit that unavailable external execution banks have not been independently revalidated.

## Task 2: Add explicit prospective matched evaluation

**Files:** new `src/eesd/rebuttal_eval.py`, new `scripts/run_eesd_rebuttal_matched_evaluation.py`, related tests, coverage registry/documentation.

**Interface:** `--case-dir`, `--pbpf-root`, `--public-root`, `--evaluator-root`, `--eval-ids`, `--action prepare|run|both`; per-case `evaluation/plan.json`, candidate bank, heldout report and completion receipt.

- [x] Reproduce currently unsupported Gemma/Replay cases using CPU fixture manifests.
- [x] Validate original materialization schemas, public/private bindings, exact source population and frozen problem map without deriving training information from heldout outcomes.
- [x] Implement public-only deterministic prompts and one greedy candidate per source for all three backbones under a declared common NF4/BF16 protocol.
- [x] Execute original task semantics in the pinned sandbox, with an explicit readiness probe and fatal infrastructure errors.
- [x] Seal independent reports and mark this as a new matched protocol, never exact historical replication.

## Task 3: Collect P0 and all requested ablations independently

**Files:** `scripts/collect_eesd_rebuttal_case_evals.py`, new case-summary script as needed, tests.

- [x] Add selectable P0, token-factorial and all-case suites while retaining the existing six-arm output contract.
- [x] Bind trained/frozen adapters, original prepared ID maps, model/data/protocol, measured training budget and schedules before collection.
- [x] Report paired problem-cluster Pass@1 differences, fixes/regressions and relevant contrasts without calling source bootstrap a fit-seed interval.
- [x] Reject partial/mixed/edited receipts and preserve every individual result.

## Task 4: Finish handoff, verification and GitHub update

**Files:** top-level README, `scripts/rebuttal_cases/README.md`, reviewer case matrix, completion report.

- [x] Make the rebuttal entrypoint discoverable from the README and give exact separate-case commands and the original/new evaluation choice.
- [x] Refresh the 3 x 4 matrix to distinguish prospective implementation coverage from historical reproduction and unavailable inputs.
- [x] Run all repository CPU tests, inspect results, and get an independent integrated code review.
- [x] Publish the reviewed update to the existing authorized rebuttal branch; verify remote SHA before the final handoff.
- [x] Explain completed software, reviewer mapping, run order and the remaining real GPU/data inputs without claiming measured results.

## Execution notes

- Continuation starts from branch `research/eesd-rebuttal-per-case-20261107`, commit `ee1a3de`.
- The user has already authorized continuation and GitHub updates. No new design-approval round is needed.
- A clean separate checkout is used for edits; the pinned PBPF checkout remains unchanged and read-only.

## Final local verification

- Full suite after the final collector fix: 403 passed, 10 skipped, 4 failed in 52.71 seconds. The four failures are the pre-existing recursive-execution tests blocked by this host's bubblewrap namespace error.
- Collector, summary and real-writer pipeline: 47 passed. Independent evaluator/CLI/pipeline review: 73 passed; all five final source-map regressions also passed.
- No outstanding critical or important issue remains in the bounded integrated review. Native GPU inference/training and successful real sandbox execution are not claimed.
- Completion report: `docs/research/2026-11-07-eesd2-completion.md`. The user-excluded CI configuration remains unchanged.
