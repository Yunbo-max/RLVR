# Individually runnable ICLR 2027 Dirichlet-EESD rebuttal cases

**2026-11-07. Code and tests only; not 17 completed experiments.**
All case launchers are separate, executable Python files. Each launcher
accepts only **its own locked case ID**, calls the exact same scientific
trainer, and writes to its own immutable output folder. This avoids
unfair differences caused by copying the optimizer code into 17 files.

## C00 through C16 (17 files)

| Python launcher | Locked case | Scientific control |
|---|---|---|
| c00_frozen.py | C00_FROZEN | frozen initial policy, no update |
| c01_equal.py | C01_EQUAL | correction CE with weight=1 + common KL |
| c02_fixed.py | C02_FIXED | fixed-mass original four-state Dirichlet + KL |
| c03_eed.py | C03_EED | submitted adaptive four-state EED + KL |
| c04_constant.py | C04_CONSTANT | train-mean weight shrinkage + KL |
| c05_shuffled.py | C05_SHUFFLED | shuffled correction-specific weights + KL |
| c06_no_uncertainty.py | C06_NO_UNCERTAINTY | lambda=0 EED posterior mean + KL |
| c07_final_filter.py | C07_FINAL_FILTER | only public all-tests-pass corrections |
| c08_confidence.py | C08_CONFIDENCE | public post-correction pass fraction |
| c09_terminal_eed.py | C09_TERMINAL_EED | EED times public final pass fraction |
| c10_fixed_edit.py | C10_FIXED_EDIT | fixed mass + normalized edit-token CE |
| c11_eed_edit.py | C11_EED_EDIT | reviewer-motivated EED + edit-token CE |
| c12_eed_random_token.py | C12_EED_RANDOM_TOKEN | EED + random same-count token emphasis |
| c13_kappa8.py | C13_KAPPA8 | temperature kappa=8, EED |
| c14_kappa32.py | C14_KAPPA32 | temperature kappa=32, EED |
| c15_uniform_relevance.py | C15_UNIFORM_RELEVANCE | original EED with uniform test relevance |
| c16_permuted_relevance.py | C16_PERMUTED_RELEVANCE | same relevance mass permuted across tests |

Extra separate CPU/data audits: d01_source_audit.py,
d02_relevance_diagnostic.py, d03_prediction_protocol.py,
d04_descriptor_and_repair_audit.py.

## Original paper model/dataset coverage (3 x 4 = 12 cells)

The full experiment registry is
configs/rebuttal/model_dataset_matrix.json.
Each scientific CASE is configured using a **cell**, separate from its
case ID, so C03_EED can run independently on every model/dataset pair
for which original correction data is supplied.

| Dataset | Qwen2.5-Coder-7B | DeepSeek-Coder-6.7B | Gemma-3-4B |
|---|---|---|---|
| RunBugRun | training implemented; PBPF sandbox eval bridge implemented | training implemented; PBPF sandbox eval bridge implemented | training implemented; ORIGINAL evaluator needs nonportable Gemma environment |
| CodeARC | training implemented; PBPF sandbox eval bridge implemented | training implemented; PBPF sandbox eval bridge implemented | training implemented; ORIGINAL evaluator needs nonportable Gemma environment |
| APPS Replay | training implemented; legacy Replay direct-eval does NOT meet secure new sandbox requirement | training implemented; same blocker | training implemented; original Gemma + Replay runtime missing |
| CodeContests Replay | training implemented; legacy Replay direct-eval does NOT meet secure new sandbox requirement | training implemented; same blocker | training implemented; original Gemma + Replay runtime missing |

**IMPORTANT**: "training implemented" means the pinned NF4/BF16 LoRA
training code accepts the named model and a compatible historical raw
correction JSONL. It does NOT certify the model is cached, sources exist,
fine-tuning has succeeded on an H100, or heldout test IDs are disjoint.
None of the twelve cells has NEW measured C00-C16 scientific results here.
The four implemented PBPF evaluator combinations require local pinned
PBPF checkout, source materializations, and an operational sandbox.
The eight blocked evaluator combinations must not be silently scored
under a different protocol and compared with the old paper results.

### Check the coverage matrix first

    python scripts/check_eesd_rebuttal_coverage.py

Optionally place inputs in
  <INPUT_ROOT>/<cell>/corrections.jsonl
  <INPUT_ROOT>/<cell>/eval_source_problem_ids.jsonl
  <INPUT_ROOT>/<cell>/shared_initial_adapter/
and run:

    python scripts/check_eesd_rebuttal_coverage.py \
      --inputs-root /sealed/eesd-cells \
      --output runs/rebuttal2027/input_coverage.json

This prints 12 per-cell code/data/evaluator statuses, makes NO GPU calls
and explicitly distinguishes missing data from blocked evaluators.

For a single existing raw correction bank, verify source model/dataset:

    python scripts/check_eesd_rebuttal_coverage.py \
      --cell codearc__deepseek_6p7b \
      --corrections /sealed/codearc-deepseek/corrections.jsonl

Use --strict-record-identity if every correction record really contains
original model_id, model_revision and domain/dataset metadata. Many old
records have no dataset field; they will be marked
"incomplete_record_identity_metadata" and require a separate sealed
original-source manifest. Do not invent dataset labels or source IDs
just to satisfy the audit.

## Run ONE independent case: DeepSeek / CodeARC

    python scripts/rebuttal_cases/c03_eed.py \
      --cell codearc__deepseek_6p7b \
      --model-config configs/rebuttal/deepseek_coder_6p7b.yaml \
      --corrections /sealed/deepseek-codearc/corrections.jsonl \
      --init-adapter /sealed/deepseek-codearc/shared_initial_adapter \
      --eval-ids /sealed/deepseek-codearc/eval_source_problem_ids.jsonl \
      --require-problem-disjoint \
      --seed 1701 --budget 524288 \
      --out-root runs/rebuttal2027 --action prepare

    python scripts/rebuttal_cases/c03_eed.py \
      --cell codearc__deepseek_6p7b \
      --seed 1701 --out-root runs/rebuttal2027 --action run

For another case, run a **different Python file**, e.g.
scripts/rebuttal_cases/c02_fixed.py, with exactly the same
--cell, --model-config, --corrections, --init-adapter, --eval-ids,
--seed and --budget. Independent case outputs are:

    runs/rebuttal2027/codearc__deepseek_6p7b/C02_FIXED/seed1701/
    runs/rebuttal2027/codearc__deepseek_6p7b/C03_EED/seed1701/

Do not treat 524288 as a recovered historical budget; the value is the
prospective config default and requires original optimizer receipts to
claim exact numerical replication.

### Qwen/RunBugRun example

    python scripts/rebuttal_cases/c11_eed_edit.py \
      --cell runbugrun__qwen25_7b \
      --model-config configs/rebuttal/qwen25_coder_7b.yaml \
      --corrections /sealed/qwen-runbugrun/corrections.jsonl \
      --init-adapter /sealed/qwen-runbugrun/shared_initial_adapter \
      --eval-ids /sealed/qwen-runbugrun/eval_source_problem_ids.jsonl \
      --require-problem-disjoint \
      --seed 1701 --budget 524288 \
      --out-root runs/rebuttal2027 --action both

The Token-aware cases (C10–C12) **fail closed** if real original program
a0 is absent, because a synthetic task without a before-program has no
well-defined edit-token mask.

## Frozen independent evaluation (currently supported on 4 cells)

After completing a separate C03 case, use the pinned original PBPF code:

    python scripts/run_eesd_rebuttal_case_evaluation.py \
      --case-dir runs/rebuttal2027/codearc__deepseek_6p7b/C03_EED/seed1701 \
      --pbpf-root /checked-out/pinned-PBPF \
      --public-root /sealed/codearc-public \
      --evaluator-root /sealed/codearc-evaluator \
      --eval-ids /sealed/deepseek-codearc/eval_source_problem_ids.jsonl \
      --domain codearc --action prepare

    python scripts/run_eesd_rebuttal_case_evaluation.py \
      --case-dir runs/rebuttal2027/codearc__deepseek_6p7b/C03_EED/seed1701 \
      --action run

C00 is the frozen control: prepare/run c00_frozen.py independently, then
evaluate in its own directory. The evaluator requires unchanged original
source SHA, source IDs, model/adaptor identity and full-program sandbox.
To collect six arms, use
scripts/collect_eesd_rebuttal_case_evals.py and then
scripts/summarize_submitted_eed_token_factorial.py.
The original PBPF evaluator records may not contain independent
problem_id, so the collector REQUIRES an externally frozen and verified
source->problem map for valid problem-cluster inference.

## Review and priority

**P0**: C00,C01,C02,C03 + D01,D02,D04 (D03 if raw prediction labels
are recovered). P0 directly answers the two detailed soundness reviews.

**P1**: C10,C11,C12 for Token-aware vs random; C04,C05,C09 and C15,C16
for mean-shrinkage, final-quality and test-relevance hypotheses.

**P2**: C06–C08 if not run in P1, C13/C14 training temperature
sensitivity, and independent 1702/1703 fitting seeds.

For reviewer-to-case detail read
docs/research/2026-11-07-iclr2027-final-reviewer-case-matrix.md.

## What this repository does not claim

- C00–C16 are separate **available entrypoint files**, not 17 completed runs.
- The 12 original paper model/domain cells are **registered**, not all
  scientifically runnable with the data and evaluators presently available.
- CPU CI, YAML/CLI coverage and mathematical tests cannot replace GPU runs.
- DeepSeek CodeARC +5.4pp was the previous full update vs frozen, not
  proof that adaptive mass beats corrected SFT.
- A successful edit-token extension does not retroactively show that
  submitted adaptive Dirichlet weights independently improve learning.
