# D03: recovered historical predictor records

**Verified on 2026-11-07. This report is an archive reanalysis, not a new GPU experiment.**

The pinned PBPF archive contains 12 candidate-bank records: RunBugRun and
CodeARC, Qwen2.5-Coder-7B and DeepSeek-Coder-6.7B, and seeds 1701/1702/1703.
Each record has 500 primary candidates, four visible executions per candidate,
and six remaining test outcomes: **3,000 observations per archive, 36,000 total**.
The source/problem/test identities and labels were recovered from the original
cache; they were not reconstructed from aggregate paper tables or invented.

## What was verified

The restoration command checks the clean original checkout at
`0200283849274b49e303638b91f142acd8bba984`, exact predictor source hashes,
cache/report/NPZ completion checksums, pinned experiment configuration, and
candidate model/revision/seed. It checks 200 development and 500 primary
records per archive and their source/problem disjointness. It recomputes the
complete effective-arm development selection grid and chosen parameters from
the development cache, then recomputes primary predictions using the original
pure predictor code. The reconstructed labels, source ordering, probability
arrays and effective masses agree with the archived values (rtol 1e-12,
atol 1e-14). Saved values are retained for the analysis after this check.

The scientific contrast is **fixed_at_effective_params minus effective**:
both arms share the effective arm's development-selected alpha and lexical
strength, the same four visible outcomes and future input query; only total
Dirichlet evidence mass changes. Separately tuned fixed vs effective scores
would confound mass with parameter selection and are not this contrast.

All detailed metrics, parameter selections, checksums and concentration bins
are in [archived_prediction_audit.json](../../reports/rebuttal2027/archived_prediction_audit.json).
The source archives are in
[PBPF at the pinned commit](https://github.com/yuhanlydia/PBPF/tree/0200283849274b49e303638b91f142acd8bba984/artifacts/eesd-eval-20260921/mechanism).

## Per-seed result

Positive NLL difference favors adaptive mass. Intervals use 10,000 paired
problem-cluster bootstrap draws, seed 314159, with 500 problems per row.
They condition on that candidate bank and selected predictor; they are not
intervals over downstream training seeds. The three archive seeds identify
candidate generation, followed by development-set predictor selection.

| Cell | Candidate seed | Fixed NLL | Adaptive NLL | Fixed − adaptive NLL | 95% problem-cluster CI |
|---|---:|---:|---:|---:|---|
| `codearc__deepseek_6p7b` | 1701 | 0.414856343 | 0.414856343 | +0.000000000 | [+0.000000000, +0.000000000] |
| `codearc__deepseek_6p7b` | 1702 | 0.446670686 | 0.446686212 | -0.000015526 | [-0.000033211, +0.000002771] |
| `codearc__deepseek_6p7b` | 1703 | 0.415952491 | 0.415952491 | +0.000000000 | [+0.000000000, +0.000000000] |
| `codearc__qwen25_7b` | 1701 | 0.472475404 | 0.472716815 | -0.000241411 | [-0.000422513, -0.000044375] |
| `codearc__qwen25_7b` | 1702 | 0.483710286 | 0.483931944 | -0.000221657 | [-0.000408042, -0.000014377] |
| `codearc__qwen25_7b` | 1703 | 0.512444405 | 0.512572672 | -0.000128267 | [-0.000326796, +0.000101228] |
| `runbugrun__deepseek_6p7b` | 1701 | 0.275430836 | 0.269338731 | +0.006092105 | [+0.002114581, +0.010746735] |
| `runbugrun__deepseek_6p7b` | 1702 | 0.318359404 | 0.322024951 | -0.003665547 | [-0.004467681, -0.002870075] |
| `runbugrun__deepseek_6p7b` | 1703 | 0.307583680 | 0.306914948 | +0.000668732 | [+0.000227019, +0.001180127] |
| `runbugrun__qwen25_7b` | 1701 | 0.270372198 | 0.270357759 | +0.000014440 | [-0.000007548, +0.000039475] |
| `runbugrun__qwen25_7b` | 1702 | 0.248975312 | 0.248755737 | +0.000219575 | [-0.000091452, +0.000568122] |
| `runbugrun__qwen25_7b` | 1703 | 0.256772902 | 0.256060885 | +0.000712017 | [+0.000108020, +0.001415313] |

**DeepSeek / RunBugRun changes sign across seeds**: positive at 1701,
negative at 1702, positive at 1703. CodeARC does not show a positive matched
mass effect for either backbone in these archives. Qwen / RunBugRun has
small positive point estimates; two of its three intervals include zero.
The intervals are descriptive per contrast and are not corrected for multiple
comparisons. These records do not support a broad, stable mechanism claim.

## Why argmax agreement is exactly one

Every paired archive has 100% argmax agreement and identical top-class
accuracy between fixed and adaptive mass. For the matched predictor with a
symmetric Dirichlet prior, this is expected. Let `s_c` be the relevance-weighted
fraction of visible outcomes assigned to class c, and m the total mass:

    q_c(m) = (alpha + m * s_c) / (K * alpha + m)
    q_c(m) - q_d(m) = m * (s_c - s_d) / (K * alpha + m)

Changing a positive m preserves class ordering. It changes predictive
confidence and scoring rules such as NLL/Brier; it cannot by itself establish
better top-class classification, correction learning, or downstream Pass@1.
The latter still requires new C03 versus C02/C01 matched training.

## Outcome and visibility semantics

These prediction archives use five **execution outcomes** in this exact order:
`PASS`, `WRONG_OUTPUT`, `RUNTIME_EXCEPTION`, `TIMEOUT`, `COMPILE_ERROR`.
They are distinct from the four **correction transitions** used by submitted
EED training: FIX, REGRESS, PRESERVE and UNRESOLVED.

The frozen candidate generator had access to four public tests. The analytic
predictor receives their observed execution outcomes and the input of each
remaining test as a query. The remaining test outcome is the prediction label,
not a feature. Therefore "future" means a heldout execution of the same
candidate, not a newly generated candidate or a heldout task. The history-size
sweep changes both the observed history and which test indices remain as
targets; its entire NLL decrease cannot be attributed to adaptive mass, or to
history size alone while claiming a fixed target population.

## Boundaries of the recovery

- The original external generated banks and source materializations were not
  available here. Their full provenance has **not** been independently
  revalidated. The historical execution receipt is `direct-no-sandbox`.
- The verified relation is original cached outcomes → recomputed predictor
  probabilities → archived arrays and metrics. This is narrower than rerunning
  the original programs against the original private evaluators.
- This recovery covers four cells. It creates no Gemma or Replay predictor
  records, no new LoRA fit, and no new C00–C16 downstream evaluation.
- Development/primary disjointness in this cache does not establish the
  disjointness of a separate correction-training dataset; D01 remains required.

## Reproduce without GPU

```bash
PYTHONPATH=src python scripts/restore_eesd_archived_predictions.py \
  --pbpf-root /checked-out/pinned-PBPF \
  --output runs/rebuttal2027/archive-recovery \
  --bootstrap 10000
```

The output directory must be new. The command writes four observation-level
JSONL files and a complete `summary.json`; it never generates new labels,
executes candidate programs or modifies the original checkout. A recovered
JSONL file can also be passed to `audit_eesd_future_outcome_pairs.py` or the
separate D03 case launcher.
