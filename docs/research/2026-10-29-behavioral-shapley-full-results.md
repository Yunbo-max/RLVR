# EESD v2: Behavioral Shapley Full Run Results (2026-10-29)

**Status: completed (positive but small effect).** The behavior-aligned Shapley
feedback selection (`shapley_select`) beats the matched full-feedback SDPO baseline
and the calibration-only `audit_gate` control on unseen RunBugRun problems, with
zero regressions against either. The effect over `audit_gate` and the frozen
baseline is small (+2 problems), so this is a directional positive signal, not a
large-margin result. The independent reserved-audit check is positive but weak.

## Setup

- Model: `Qwen/Qwen2.5-Coder-7B-Instruct` @ `c03e6d...`, LoRA r=16/alpha=32, NF4,
  single H100 80GB, G=8 rollouts, 3 recursive rounds, seed 1701.
- Data: RunBugRun Python official SQLite; 64 train + 64 unseen eval (disjoint
  `problem_id`), 4+4+4 separation (feedback / train-calibration / reserved-audit),
  `fixed_code` never read.
- Arms: `no_update`, `sdpo` (matched full-feedback reverse-KL), `audit_gate`
  (calibration-only behavioral gate, no Shapley), `shapley_select` (exact signed
  test Shapley + feedback subset selection).
- Primary metric: fresh unseen-problem all-tests-pass@1 (greedy).

## Infrastructure changes required to run on the shared H100

1. **bwrap was missing** from the job image and procfs mounts were blocked. Built a
   self-contained `bin/bwrap` (binary + libcap/libselinux/libpcre2) and dropped
   `--proc /proc` from the sandbox. Full user/mount/pid/net/ipc/uts namespace
   isolation is preserved (verified `WORKSPACE_ISOLATED`, `NETWORK_ISOLATED`).
2. **Prompt budget** raised 4096 → 8192 (one eval source has a 7233-token prompt).

## Full run results (64 unseen problems, round 3)

| arm | pass@1 | Δ vs sdpo | 95% CI vs sdpo | fixes vs base | regr vs base |
|---|---|---|---|---|---|
| no_update (base) | 0.4688 (30/64) | +0.109 | [0.031, 0.203] | — | — |
| sdpo | 0.3594 (23/64) | 0.0 (ref) | — | 1 | 8 |
| audit_gate | 0.4688 (30/64) | +0.109 | [0.047, 0.188] | 4 | 4 |
| **shapley_select** | **0.5000 (32/64)** | **+0.141** | **[0.063, 0.234]** | 4 | 2 |

### Paired comparisons (per-problem, round 3)

- shapley_select vs sdpo: **9 fixes, 0 regressions** (strictly better).
- shapley_select vs audit_gate: **2 fixes, 0 regressions** (strictly better, small).
- shapley_select vs no_update: 4 fixes, 2 regressions (net +2).

The SDPO arm regresses (1 fix / 8 regressions), so full-feedback distillation
hurts on this setup; behavior-calibrated selection avoids those regressions and
adds a small gain.

## Independent reserved-audit check

`reserved_audit.json` re-evaluates the generated candidates on the 4 held-out
train-side audit tests (never used in selection/training):

- `mean_selected_minus_full_on_heldout`: **+0.00158** (selected feedback subset
  slightly beats full feedback on held-out data; positive but tiny).
- `fraction_accepted_with_positive_heldout_alignment`: **0.933** (93.3% of accepted
  selections align positively with held-out outcomes).
- `accept_fraction`: 0.078 (only 7.8% of source-rounds selected feedback; the rest
  abstained, consistent with the weak-evidence-abstention design).

## Conclusion

1. The core gate is met: `shapley_select` beats **both** `sdpo` (+9, CI excludes 0)
   and `audit_gate` (+2), so the gain is attributable to Shapley feedback selection,
   not merely to the extra calibration information.
2. The margin over `audit_gate` and the frozen base is small (+2 problems = 3.1%),
   and the reserved-audit advantage is positive but weak (+0.0016). This is a
   **directional positive result**, not a large-margin claim.
3. `sdpo` (full-feedback reverse-KL) regresses sharply (23/64), confirming the
   earlier finding that token-weight-only distillation without behavior alignment
   harms unseen-problem repair.

**Caveats / next steps:** confirmation seeds 1702/1703 are reserved and not yet run;
the +2 over `audit_gate` should be reproduced before strong claims. The comparison
`sdpo` is a matched baseline, not the full official SDPO (EMA/JSD/importance
sampling). This does not supersede the 2026-10-28 negative result; it tests a new
mechanism.
