# Dual Recursive Verifier–Solver OPSD: second-pass code audit

**Branch:** `research/dual-recursive-verifier-opsd-20261114`

**Audit scope:** inspect the new dual-role implementation against its stated scientific protocol and run local, isolated CPU tests. The patched files passed `26` targeted CPU tests, Python `compileall`, shell `bash -n`, and YAML parsing. These tests do **not** instantiate the pinned 7B model, load actual RunBugRun data, run bubblewrap, or certify an end-to-end GPU experiment.

## High-priority issues found and remediated

1. **Silent oracle fallback confounded the new method.** A generated verifier rejected by the training-side gate formerly caused the Solver teacher to consume a **different information regime**: direct trusted execution feedback containing correct expected stdout. This could improve Solver scores without a successful generated verifier. All generated-verifier arms now default to `--missing-verifier skip` and record each skipped rollout. The previous fallback is opt-in with `--missing-verifier fixed` and MUST be labeled as a separate ablation.
2. **Missing information-matched oracle baseline.** `solver_fixed` exposes expected output and is not directly comparable to a generated verifier's PASS/FAIL-only feedback. New `solver_oracle_verdict` provides perfect local truth labels in the **same input/actual/status/verdict fields**, without expected stdout or a source-identity hint in the teacher prompt. Provenance is recorded separately. It remains an upper-bound baseline relative to the local, not yet qualified official RunBugRun, comparator.
3. **Execution failures could be misclassified by generated verifiers.** A compile error, timeout or runtime exception was previously forwarded as if its empty/partial stdout were a normal answer. Now known execution failure is directly `FAIL`, while only normally executed program outputs go to the generated checker.
4. **Malformed verifier proposals produced no learning signal.** The first version required `static_valid` before running Verifier OPSD. Now invalid source receives a small static-error signal (e.g., `syntax`), allowing its sampled code continuation to be distilled. Crucially, the privileged teacher does **not** see the original source code as an upfront copy; that would leak future response tokens.
5. **Nested disallowed imports bypassed AST preflight.** The static checker now rejects unsupported imports inside functions as well as at module level. Mandatory bubblewrap remains the actual security boundary; AST checks are not a sandbox.
6. **Resume receipts did not seal adapter bytes.** They now store both role-adapter directory SHA-256 digests, checked before resuming. Resume refuses absent or changed adapter directories.
7. **Verifier progress reporting suffered selection bias.** Metrics previously described only gate-accepted generated verifiers. New metrics record **all** proposals' source validity, held-out balanced accuracy, false acceptance/rejection, and coverage in addition to accepted-only quality and the Solver's skip/fallback rate. Otherwise selection could make a stagnant generator look like an improving generator.

## Critical scientific constraints still open

- The generated API `verify(input_text, candidate_output) -> bool` is a general semantic output checker. Implementing a robust checker for a heterogeneous CodeNet/RunBugRun problem can be as hard as solving the underlying programming task. With a short generated code budget, most verifiers may fail AST validity or finite gate acceptance. **Before allocating a large training budget, measure actual source-validity, acceptance, all-proposal hidden accuracy and proportion of Solver updates that truly use a generated checker.** A near-zero rate means the research hypothesis has not been exercised.
- The trusted positive/negative examples are defined against the existing pinned stdout matcher and occasional synthetic output perturbations. Equivalent alternative correct outputs may be erroneously labeled wrong. The benchmark MUST be checked against official RunBugRun semantics before claims of verified rewards or official performance. Finite gate accuracy is not formal correctness.
- Teacher and student use two separate LoRA adapters per role. Verifier code selected in round `r` comes from the **beginning-of-round frozen Verifier**, while its parameter update affects new Verifier proposals only in round `r+1`. This intentional one-round lag avoids same-round target drift; do not describe it as immediate within-round Solver use of the newly trained Verifier.
- Sampling for each round uses the frozen role policy at the round start, followed by within-round minibatch steps. This is a *round-frozen on-policy collection*, not exact per-optimizer-step freshly sampled on-policy data.
- The `verifier_only` arm must be judged on **all-proposal** held-out verifier metrics, not only the selected passing programs. The `dual` vs `solver_regenerated_verifier` comparison isolates Verifier parameter updates better than `dual` vs `solver_fixed`; both arms still have diverging on-policy Solver distributions after the initial round.
- No GPU training or real bubblewrap qualification was completed in this audit environment. Use a 1-round pilot with a handful of distinct train problems and reserve final test cases strictly; then decide whether it is scientifically appropriate to proceed to five rounds.

## Local checks

```bash
python -m pytest -q tests/test_dual_verifier*.py
python -m compileall -q src scripts tests
bash -n scripts/run_dual_verifier_matrix.sh
```

The original `main`, Rich64 and Joint/Mix code, and `.github/workflows/ci.yml` were not changed.
