# EESD2 continuation: implementation and evidence handoff

**2026-11-07 — submitted lexical Dirichlet EED preserved.**

This continuation completes the independent-case implementation, adds a common
prospective evaluation path for all twelve registered model/dataset cells,
and executes a recoverable historical D03 predictor analysis. It does not
claim new LoRA training or new C00–C16 benchmark results.

The work continues `research/eesd-rebuttal-per-case-20261107` from
`ee1a3de9f14c70a1e27140eb2c9070e6b1c9e96e`. The original PBPF source remains
unchanged at `0200283849274b49e303638b91f142acd8bba984`.
The user explicitly excluded the failed CI workflow from this task: no
Actions/workflow files were changed and no workflow was manually rerun.
`pyproject.toml` and `uv.lock` are also unchanged.

## Completed software

| Area | Delivered behavior | Evidence boundary |
|---|---|---|
| Independent experiments | C00–C16 retain separate Python entrypoints, locked case IDs and immutable case/seed output directories; D01–D04 remain separate audits. | Entrypoints are implemented, not seventeen completed GPU experiments. |
| Shared initialization | New initializer handles the three pinned backbones, verifies zero LoRA-B initialization, freezes adapter bytes and writes a receipt. Gemma targets its language layers. | Native quantized initialization requires the user's cached models and CUDA host. |
| Training provenance | Preparation and execution bind model revision, adapter structure, source code, correction data, command and actual scheduled response tokens. Reports record optimizer, LoRA configuration and dependency versions. | CPU contract tests do not establish successful GPU training. |
| Prospective evaluation | New greedy NF4/BF16 protocol covers all 3 × 4 cells in code, with public-only prompts, exact saved/loaded adapter checks, frozen populations and isolated original task semantics. | This is a declared new protocol, not exact historical Gemma/Replay replication. |
| Historical bridge | Existing pinned PBPF generation/evaluation bridge retains its four supported cells, with stronger model/cell/configuration bindings. | Missing historical Gemma environment and Replay direct-execution differences remain explicit. |
| Collection | P0, the original six token arms, all cases or an explicit subset can be collected independently. Frozen/trained cases have the appropriate checks. | Different inputs, schedules, evaluator implementations, execution policies or protocols are rejected. |
| Statistical summary | Source-weighted descriptive results, equal-problem estimands, paired problem-cluster intervals, fixes/regressions and case contrasts are reported. | Intervals condition on fitted models; they do not measure variation over fitting seeds. Missing controls are not estimable. |

Exact commands and the full case table are in
[the independent-case guide](../../scripts/rebuttal_cases/README.md).
The [reviewer matrix](2026-11-07-iclr2027-final-reviewer-case-matrix.md)
maps each concern to its cases and interpretation rules.

The original four correction transitions, lexical relevance, alpha 0.1,
lambda 0.5, kappa 16 and common KL beta 0.03 remain the submitted EED setup.
C10–C12 are explicitly an extension. The separate Shapley/Rich64 study is
not substituted for the rebuttal method.

### Evaluation and collection contracts

The prospective evaluator freezes the model configuration, exact revision,
adapter contents, correction provenance, public/private materializations,
source-to-problem mapping, local evaluator code and pinned PBPF source.
Generation validates only public inputs; private test files are opened by
the evaluation phase. A sandbox readiness failure aborts before model
loading. Infrastructure errors never trigger a direct-execution fallback.

Completed results contain a sealed candidate bank, raw test outcomes,
runtime information and completion receipts. The collector compares the
saved `local_source_hashes` and `pbpf_source_hashes` across matched cases;
sharing a protocol name alone cannot make different implementations
comparable. It compares historical saved bindings without requiring the
collector's current checkout to equal the archived implementation.

The real C00 writer now retains correction-file provenance required by
evaluation. The collection audit accepts the real writer's empty error
dictionary and the legacy empty list, while rejecting nonempty or malformed
errors. RBR/CodeARC materializations can retain repeated input strings with
distinct original test IDs; Replay preserves its original uniqueness rule.

## Executed D03 historical analysis

The recovery restored **36,000 future-outcome observations** from twelve
original archives: two backbones × two datasets × three candidate-generation
seeds, with 500 primary candidates and six remaining outcomes per archive.
These are repeated observations, not 36,000 independent tasks.

It checked archive seals and original source/configuration identities,
recomputed the complete effective-arm development selection grid, and
reproduced the saved fixed/adaptive probability arrays using the pinned
original predictor. The matched arms share selected alpha, lexical strength,
visible outcomes and future input query; only total evidence mass changes.
The audit uses 10,000 paired problem-cluster bootstrap draws, seed 314159.

The findings constrain the rebuttal claim:

- DeepSeek/RunBugRun fixed-minus-adaptive NLL differences are **+0.006092105**,
  **−0.003665547** and **+0.000668732** for seeds 1701/1702/1703. The direction
  changes across candidate banks.
- CodeARC has no positive matched mass effect in these archived pairs.
- Every pair has **100% argmax agreement**. With a symmetric prior, changing
  a positive scalar evidence mass preserves class ordering and changes
  confidence. The observed NLL/Brier changes therefore do not demonstrate
  improved top-class accuracy or downstream correction learning.
- These archives predict five execution-outcome classes. They are distinct
  from the four correction transitions used in EED training.

The original external candidate banks and private materializations were
unavailable, so their execution provenance was not independently rerun. The
verified relation is cached outcomes → original predictor recomputation →
archived arrays and metrics. No Gemma/Replay predictor records or new GPU
outcomes were created.

See the [full D03 report](2026-11-07-eesd2-archived-prediction-audit.md) and
[machine-readable audit](../../reports/rebuttal2027/archived_prediction_audit.json)
for every seed, interval, selection, concentration bin and checksum.

## Verification performed

The final repository-wide local command was:

```bash
PYTHONPATH=src python -m pytest -q
```

Environment: Python 3.12, pytest 9.2.0, with PyYAML, NumPy and pyarrow 24.0.0
available in the local test environment. The native torch/transformers/peft
GPU stack was not available. Dependency manifests were not changed for this
local validation.

**Final result: 403 passed, 10 skipped, 4 failed in 52.71 seconds.**
The four failures are unchanged existing host-dependent tests:

- `tests/test_recursive_execution.py::test_batch_matches_sequential_execution`
- `tests/test_recursive_execution.py::test_batch_isolates_builtins_mutation_across_tests`
- `tests/test_recursive_execution.py::test_batch_isolates_environment_mutation_across_tests`
- `tests/test_recursive_execution.py::test_batch_times_out_forked_grandchild`

Their subprocess output reports `bwrap: open /proc/.../ns/ns failed: No such
file or directory`. The same four failures were present before these changes.
They remain reported as failures; no sandbox policy or test was weakened to
make this host pass. This local result is not a claim about GitHub Actions.

The final collector/summary/pipeline selection passed **47 tests**. The final
source-map regressions were first observed failing, then passed after the
comparison was added. Independent review also ran **73 evaluator/CLI/pipeline
tests** and the five final source-map regressions. No critical or important
finding remains in that bounded review.

The pipeline integration uses the real case writer, evaluator preparation,
bank/report writers, collector and summarizer. Generation and execution are
explicit test doubles marked as non-scientific evidence. Native quantized
loading, actual adapter inference and successful isolated benchmark execution
are still required on the experiment host.

## Continue with the actual experiment

The next downstream run is **P0: C00/C01/C02/C03**, independently prepared,
trained where applicable, and evaluated with one consistent protocol.
The available registry is
[model_dataset_matrix.json](../../configs/rebuttal/model_dataset_matrix.json).
Its coverage command reports four historical-bridge cells, twelve prospective
implementation cells and **zero new scientific GPU results**.

1. Supply the real correction JSONL, frozen source/problem IDs and original
   public/private materializations for the chosen cell. Run D01 and preserve
   actual metadata; do not invent identities to pass validation.
2. Use cached pinned model weights, a consistent ML stack, BF16-capable CUDA
   and an operational original sandbox. Create one shared initial adapter
   per model/seed and reuse its exact bytes across cases.
3. Prepare and run each P0 launcher in a new output directory. The documented
   524,288 response-token budget is a prospective setting, not a recovered
   historical optimizer receipt.
4. Prepare and run each evaluation separately. Choose either the supported
   historical bridge or the common prospective protocol consistently for the
   whole comparison.
5. Collect with `--suite p0`, then summarize. The EED-specific comparisons are
   **C03−C02** and **C03−C01**; C03−C00 measures the whole policy update.
6. Add the reviewer controls and token extension only with their matched
   cases. Token contrasts are C11−C03, C10−C02 and C11−C12; the factorial
   interaction is `(C11−C10)−(C03−C02)`. Preserve negative results and report
   absent controls as not estimable.

Existing completed outputs must not be overwritten to adopt this revision.
New preparations freeze this source version and belong in new immutable
case/evaluation directories. The remaining input and runtime requirements
are concrete experiment prerequisites, not completed scientific results.
