# EESD: Shapley-Guided Recursive Self-Distillation — Full Run Results (2026-10-28)

**Status: completed (negative result).** One model, one benchmark, one principal
baseline. The recursive self-distillation pipeline ran end-to-end on a single H100,
but none of the three distillation arms improved over the frozen `no_update` baseline
on unseen RunBugRun problems. Per gate #6 of the experiment design, this does not
support a claim that exact per-test token Shapley attribution beats full-feedback
SDPO-style distillation.

## Setup

- Model: `Qwen/Qwen2.5-Coder-7B-Instruct` @ `c03e6d358207e414f1eca0bb1891e29f1db0e242`,
  LoRA r=16 / alpha=32, NF4 double-quant, single H100 80GB.
- Benchmark: RunBugRun Python, official SQLite (restored locally; 716,944 bugs,
  178,800 Python, 163,163 train-split / 7,839 test-split).
- Frozen data: `runs/rbr-frozen-1701` — 64 train sources + 64 unseen eval sources,
  disjoint `problem_id`, 4 public train tests, `fixed_code` never read.
- Protocol: 8 rollouts/source, 4 public tests, 3 recursive rounds, seed 1701.
- Arms: `no_update`, `sdpo` (matched full-feedback reverse-KL), `loo` (5 coalitions),
  `shapley` (16 coalitions).

## Infrastructure deviations from the preregistered gate

1. **bubblewrap is unusable** on the shared H100 hosts: `/usr/bin/bwrap` exists but
   fails `Can't mount proc on /newroot/proc: Operation not permitted` (unprivileged
   mount/user namespaces blocked). `recursive_execution.py` now falls back to a
   resource-limited soft sandbox (RLIMIT CPU/AS/FSIZE/CORE + `python -I -B`) and
   records `sandbox_mode=soft` in each `plan.json`. No filesystem/network isolation.
2. **Generated code was markdown-fenced** by the Instruct model at temperature 0.7,
   turning every rollout into `COMPILE_ERROR`. Added `_extract_python` (fence
   stripping) in `run_recursive_shapley_sdpo.py`; this restored the no-update baseline
   from `0.0` to `0.484` pass@1.
3. **Prompt budget** raised from 2048 to 4096 tokens (one train + one eval source
   exceed 2048; max observed 3561).

## Pilot (4 train / 8 eval / G=4) — pipeline verification only

| arm | round0 pass@1 | round3 pass@1 | mean_pass r0→r3 |
|---|---|---|---|
| no_update | 0.625 | 0.625 | 0.738 → 0.738 |
| sdpo | 0.625 | 0.625 | 0.738 → 0.738 |
| loo | 0.625 | 0.625 | 0.738 → 0.776 |
| shapley | 0.625 | 0.625 | 0.738 → 0.776 |

## Full run (64 train / 64 eval / G=8) — primary result

Unseen-problem all-tests-pass@1 (greedy), round 3:

| arm | pass@1 | Δ vs sdpo | fixes vs r0 | regressions vs r0 | 95% CI vs sdpo |
|---|---|---|---|---|---|
| **no_update** | **0.4844** (31/64) | +0.0156 | — | — | [-0.109, +0.141] |
| sdpo | 0.4688 (30/64) | 0.0 (ref) | 8 | 9 | — |
| shapley | 0.4531 (29/64) | -0.0156 | 7 | 9 | [-0.047, 0.0] |
| loo | 0.4375 (28/64) | -0.0312 | 6 | 9 | [-0.078, 0.0] |

Per-round mean test-pass fraction (secondary metric):

| arm | r0 | r1 | r2 | r3 |
|---|---|---|---|---|
| no_update | 0.6233 | 0.6233 | 0.6233 | 0.6233 |
| sdpo | 0.6233 | 0.5604 | 0.5173 | 0.5621 |
| loo | 0.6233 | 0.5817 | 0.5075 | 0.5680 |
| shapley | 0.6233 | 0.5660 | 0.5337 | 0.5756 |

## Train-side metrics (do not even improve on train tasks)

Train public-test pass fraction (sampled rollouts) stays flat across rounds:

| arm | r0 | r1 | r2 |
|---|---|---|---|
| sdpo | 0.492 | 0.477 | 0.487 |
| loo | 0.492 | 0.488 | 0.488 |
| shapley | 0.492 | 0.489 | 0.478 |

Shapley arm optimization trace (confirms training actually runs):

| round | grad_norm (mean) | distillation_kl | anchor_kl |
|---|---|---|---|
| 0 | 0.9305 | 0.0709 | 0.0026 |
| 1 | 0.8078 | 0.0524 | 0.0030 |
| 2 | 0.7501 | 0.0535 | 0.0042 |

## Conclusion

1. The recursive self-distillation objective (`reverse-KL(student || frozen
   full-feedback teacher) + 0.02·KL(student || old)`) is optimized — gradients are
   nonzero, KL decreases, and the updated adapter is correctly propagated each round —
   but it does **not** translate into more correct programs: token-level distribution
   alignment to the feedback-conditioned teacher does not improve greedy repair.
2. On unseen problems all three arms **regress** below the frozen baseline, and
   Shapley does **not** beat the cheaper LOO control (nor SDPO). CI intervals include
   0; differences are not statistically significant.
3. This is a genuine negative result, not a code/infra bug (verified: soft sandbox
   positive control, fence-stripped execution, non-zero credit weights with
   `zero_credit=0.0`, correct teacher-forcing token shift, 64 optimizer steps/round).

**Recommended next step (gate #7):** before adding datasets/models or claiming any
method advantage, reproduce a faithful official SDPO (EMA/JSD/importance sampling) to
confirm whether the matched baseline is too weak, or revise the mechanism (e.g. a
successful peer teacher rather than the frozen feedback-conditioned self).
