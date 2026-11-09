# EESD rebuttal — first prospective P0 results (2026-11-09)

Scope: seed 1701, 500 primary sources per cell, matched prospective greedy
NF4/BF16 sandbox protocol. All 17 arms (C00–C16) trained and evaluated for two
cells; summary machine-readable files under `reports/rebuttal2027/`. The other
four cells are in flight (see open items).

## codearc__qwen25_7b (CodeARC / Qwen2.5-Coder-7B)

Problem-mean all-tests-pass (95% CI), N=500:

| Arm | Estimate | CI |
|---|---|---|
| C00 frozen baseline | 20.2% | [16.6, 23.8] |
| C01 equal weight | 20.2% | [16.8, 23.8] |
| C02 fixed-mass Dirichlet | 19.0% | [15.6, 22.4] |
| C03 adaptive EED (submitted) | 19.4% | [16.0, 23.0] |
| C04 constant mass | 19.4% | [15.8, 22.8] |
| C05 shuffled weights | 20.2% | [16.6, 23.8] |
| C06 lambda=0 | 19.8% | [16.4, 23.4] |
| C07 final-filter only | 20.8% | [17.2, 24.4] |
| C08 confidence | 19.6% | [16.2, 23.2] |
| C09 terminal EED | 20.2% | [16.6, 23.8] |
| C10 fixed + edit tokens | 20.0% | [16.4, 23.6] |
| C11 EED + edit tokens | 19.6% | [16.2, 23.2] |
| C12 EED + random tokens | 19.2% | [15.8, 22.8] |
| C13 kappa=8 | 20.4% | [16.8, 24.0] |
| C14 kappa=32 | 19.8% | [16.4, 23.4] |
| C15 uniform relevance | 20.4% | [16.8, 24.0] |
| C16 permuted relevance | 19.4% | [16.0, 23.0] |

Key contrasts: C03−C02 = +0.4pp [−0.4, +1.2]; C03−C01 = −0.8pp [−2.8, +1.2];
**C03−C07 = −1.4pp [−2.6, −0.4] (only CI excluding zero; direction against
the submitted method)**; C10−C02 = +1.0pp [+0.0, +2.0] (borderline).

## runbugrun__qwen25_7b (RunBugRun / Qwen2.5-Coder-7B)

| Arm | Estimate | CI |
|---|---|---|
| C00 frozen baseline | 55.4% | [51.0, 59.8] |
| C01 equal weight | 54.8% | [50.4, 59.2] |
| C02 fixed-mass Dirichlet | 57.4% | [53.0, 61.8] |
| C03 adaptive EED (submitted) | 57.0% | [52.6, 61.4] |
| C04 constant mass | 57.4% | [53.0, 61.8] |
| C05 shuffled weights | 56.4% | [52.0, 60.6] |
| C06 lambda=0 | 57.8% | [53.4, 62.2] |
| C07 final-filter only | 56.0% | [51.6, 60.4] |
| C08 confidence | 56.4% | [52.0, 60.8] |
| C09 terminal EED | 55.8% | [51.4, 60.2] |
| C10 fixed + edit tokens | 56.2% | [51.8, 60.4] |
| C11 EED + edit tokens | 56.0% | [51.6, 60.2] |
| C12 EED + random tokens | 56.4% | [52.0, 60.8] |
| C13 kappa=8 | 56.4% | [52.0, 60.8] |
| C14 kappa=32 | 57.0% | [52.6, 61.4] |
| C15 uniform relevance | 55.4% | [51.0, 59.8] |
| C16 permuted relevance | 57.2% | [52.8, 61.4] |

Key contrasts: **C03−C15 = +1.6pp [+0.4, +3.0] (significant; learned
relevance beats uniform relevance)**; C03−C01 = +2.2pp [−0.4, +4.8];
C03−C00 = +1.6pp [−0.6, +4.0]; C03−C02 = −0.4pp [−1.6, +0.6] (null).
All other contrasts straddle zero.

## Honest interpretation

1. Evidence weighting is real on RunBugRun: adaptive EED significantly
   outperforms the uniform-relevance control (C15).
2. Adaptive mass does not beat fixed mass anywhere: C03−C02 is null on both
   cells (±0.4pp).
3. CodeARC shows no method advantage; the single CI-excluding-zero contrast
   (C03−C07) favors the final-filter control.
4. Consistent with the archived D03 re-analysis: effects are weak and flip
   direction across candidate banks; confidence changes do not imply Pass@1
   gains. No claim of stable, universal improvement is made.
5. Seeds 1702/1703 confirmation rounds are not yet run (gated on the 1701
   results above).

## Environment amendments (replacement machine; PBPF checkout unmodified)

1. `--greedy --candidates 1` (transformers 5.x rejects greedy with
   num_return_sequences>1).
2. New direct-execution lock `direct-execution-lock-20261108.json`
   (`f935cb53…`); 20260921 lock preserved as history.
3. `python3` interpreter pinned for direct stages (lock binds the binary).
4. GPFS lacks renameat2(RENAME_NOREPLACE): wrappers substitute `os.rename`.
5. NEL-safe JSONL reads (U+0085/U+2028/U+2029 escaped at read time).
6. Gemma PEFT 0.14 canonical LoRA target names accepted in adapter validation.
7. Sealed per-cell identity manifest for the dataset_identity gap
   (prepare + run re-validation).
8. bwrap shim: `--proc/--dev` → read-only binds, `--tmpfs` → `--dir`;
   LD_PRELOAD shim skips the root MS_SLAVE marking only when seccomp returns
   EPERM. Node kernel heterogeneity required re-running six runbugrun arms so
   every arm's sandbox identity matches (collector requirement).

## Open items

- codearc__gemma3_4b: trained, evaluations submitted.
- codearc__deepseek_6p7b, runbugrun__deepseek_6p7b, runbugrun__gemma3_4b:
  matrices submitted, in flight.
- Replay cells: blocked (train-reserve re-materialization is a separate
  upstream pipeline).
- Seeds 1702/1703: after the 1701 gates.
