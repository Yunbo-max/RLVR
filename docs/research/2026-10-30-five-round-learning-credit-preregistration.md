# EESD — Five-Round Learning-Credit Experiment (2026-10-30)

**Research status: prospective.** Code and CPU/CI checks are not equivalent to new H100 results. The preceding three-round single-seed Behavioral Shapley result was 32/64 unseen programs, compared with 30/64 for frozen Qwen and audit gate and 23/64 for a matched (not official full) SDPO-style baseline. The +2 over audit gate was not significant, and only ~7.8% of training source-rounds generated updates. This follow-up asks WHY, not merely whether, the method can win.

## Core falsifiable claim

The central hypothesis is that attribution-based screening of individual public execution feedback messages provides more transferable learning credit than direct subset score maximization or generic conservative training. A matched direct maximization arm is essential.

**Two independent sizes:**

- T=4 public feedback tests per candidate => exactly 2^4=16 coalitions.
- G=8 candidate programs per train task in the primary five-round study.
- G=16 and G=32 are secondary *rollout-count* experiments at the same T=4, not test-count increases.
- T=16 and T=32 would require 65,536 and 4,294,967,296 exact coalitions per candidate respectively: this implementation deliberately does not support them. Use sampled permutations in a separately budgeted future study.

**One model / one dataset:** Qwen2.5-Coder-7B-Instruct revision c03e6d358207e414f1eca0bb1891e29f1db0e242; RunBugRun Python official SQLite; H100 with NF4/BF16 LoRA rank16.

## Data boundary

The frozen dataset extractor, scripts/prepare_learning_credit_data.py, writes separate immutable SHA256-checked files:

- Train: 64 original buggy code tasks (official train split), each with 4 public feedback tests F, 4 additional *training calibration* tests C, 4 *reserved* audit tests A. All three sets have disjoint test IDs. The model receives only F in feedback prompts. C guides training evidence choice but never enters teacher context. A is held out until post-training.
- Meta intervention: 24 different train-split problem IDs never used for LoRA training or feedback selection. Used to evaluate TRUE one-step counterfactual parameter updates.
- Final: 128 entirely different test-split problem IDs, held out from all selection, tuning and training. Measure one newly generated greedy repair per task at checkpoints R0,R1,R2,R3,R4,R5.

No corrected gold/reference code is queried or used. However trusted finite input/output labels are used, so this is a *partial verification* setting, NOT a fully label-free test-generator experiment.

The runner fails closed without a proper bubblewrap user/mount/network isolation sandbox. Never remove isolation just to launch a run on a shared H100.

## Mathematical objects

For each round-k frozen model pi_k, task x and G newly sampled code implementations y_i, each candidate i receives its own public execution feedback F_i. For every coalition S of the 4 public feedback tests, teacher-force the exact same code into the frozen model and compute its average selected-token log-probability shift:

    Delta_i(S) = mean_t log pi_k(y_it | x,F_i,S,y_i,<t)
                 - mean_t log pi_k(y_it | x,y_i,<t)

Let c_ih be pass/fail on independently administered TRAINING CALIBRATION test h, and z_ih=c_ih-mean_r c_rh. Define the behavior-alignment game:

    V_h(S) = mean_i [z_ih * Delta_i(S)]
    V(S) = mean_h V_h(S)

A positive V means feedback shifts model token preference in the same direction as independently measured partial code quality. This is a proxy, not true general correctness and not normalized probability mass across programs. Compute signed exact test Shapley:

    phi_j,h = sum over S not containing j:
      |S|!(T-|S|-1)! / T! * [V_h(S+{j})-V_h(S)].

For each coalition define the conservative training-side selection score:

    J(S) = mean_h V_h(S) - 0.5*SE_h[V_h(S)] - 0.001*|S|.

Shapley selects only from subsets of tests j with mean_h phi_j,h > 0.5*SE_h[phi_j,h], choosing their argmax J(S) when greater than 0.001, else abstaining.

**Direct_max** selects argmax J(S) over ALL 15 nonempty subsets, WITHOUT Shapley. It has the SAME 16 coalition scoring budget. On any identical candidate pool with a nonempty Shapley choice, in-sample J(S_direct) >= J(S_shapley). Consequently, a Shapley advantage is only defensible if observed on INDEPENDENT heldout behavior or after REAL parameter updates, not in-sample J.

For a selected subset S*, the token gate uses signed agreement between candidate-centered calibration quality a_i and token probability shift d_it:

    w_it = |a_i| * clip( [sign(a_i)*d_it]_+ / 0.15,0,1 ).

Student without feedback learns frozen feedback-conditioned full-vocabulary distributions:

    L = mean_{i,t} w_it * [
      KL(pi_student(.|x,y_i,<t) || stopgrad(pi_k(.|x,F_i,S*,y_i,<t)))
      + .02 KL(pi_student(.|x,y_i,<t) || stopgrad(pi_k(.|x,y_i,<t)))
    ].

No per-rollout unit-mean normalization: weak evidence yields weak gradients. Negative Shapley is not automatically a harmful test; the *behaviorally conditioned effect* determines admission.

## Seven five-round primary arms

| Arm | Feedback rule | What it falsifies |
|---|---|---|
| no_update | Frozen Qwen | No improvement over initial model |
| sdpo | All four feedback tests, equal token KL | Whole-feedback learning versus selective learning |
| audit_gate | All tests only if positive calibration gate | Maybe abstention alone is enough |
| direct_max | Exact all-subset maximization of J(S), NO Shapley | Maybe direct optimization beats Shapley |
| shapley | Eligible individual Shapley tests, then maximize J(S) | Proposed attribution-specific contribution |
| random_subset | Same Shapley-accepted training source and same number of feedback tests, but randomized subset | Maybe arbitrary feedback of the same cardinality works |
| matched_random_updates | Random train source IDs, full-feedback KL, EXACT same actual optimizer-step count per round as Shapley | Maybe fewer optimizer updates alone explains success |

Random schedules are frozen from Shapley's TRAINING logs only; do not use final-evaluation performance. The random-update-count arm uses deterministic backup train source IDs if an initially scheduled source cannot actually produce a gradient; if counts still differ, abort the run. Random-subset may have fewer realized updates due to zero directional token credit: report this explicitly and do not call it exact step matched.

All arms start from the same byte-identical initial LoRA checkpoint, then independently resample and update THEIR OWN model at each round. A single full matrix is 5 updates per arm, 6 evaluations per arm. In a full-scoring Shapley/direct arm at T=4,G=8,64 sources,5 rounds the planned worst-case teacher subcoalition count is 40,960, excluding other forward passes. GPU-time, token counts, sandbox executions and real optimizer steps must all be reported.

## Real learning-credit intervention (not just teacher preference)

The separate scripts/probe_learning_credit_delta_theta.py evaluates the causal chain:

    Feedback S -> one REAL controlled LoRA update Delta(theta) -> new meta-program correctness.

Select fixed training anchors solely from Shapley TRAINING acceptance receipts for round k. Reuse exact same code candidates, F/C executions, fixed frozen pi_k checkpoint and 16-coalition token scores. Each intervention arm (Shapley subset, direct-max subset, same-cardinality random subset, all feedback) starts a SEPARATE student clone from the IDENTICAL pre-update LoRA checkpoint, performs one fresh AdamW step or abstains, then generates repairs for the SAME untouched meta programming problems. Compare post-minus-pre Pass@1, mean test-pass fraction, fixes, regressions and actual gradient norm. The meta set is NEVER used to select feedback within this experiment.

This shows a local causal effect for training-accepted anchor sources, not a general statistical guarantee for all training problems. With few anchors/meta tasks, the estimate may be noisy. Do not use the reported meta effects to tune the same hyperparameters; a future fresh split is needed for confirmation.

After training, a separate reserved 4-test audit checks whether selected coalition behavior-alignment transfers beyond calibration tests.

## Actual commands (from repository root)

Prerequisites: pinned Qwen weights in local HF cache, official RunBugRun SQLite available, functional bwrap namespace isolation, PyTorch/PEFT deps and one H100.

    uv sync --extra ml --extra test
    export PYTHONPATH="$PWD/src"
    uv run pytest -q tests/test_learning_credit_*.py

    python scripts/prepare_learning_credit_data.py \
      --db /data/runbugrun.sqlite --output runs/lc-frozen-1701 \
      --train-sources 64 --meta-sources 24 --eval-sources 128 --seed 1701

    # PILOT: 8 train sources, G=4, 16 final eval tasks; FIVE updates.
    bash scripts/run_learning_credit_matrix.sh \
      runs/lc-frozen-1701 runs/lc-pilot-1701 pilot 1701

    # PRIMARY: 64 train sources, G=8, 128 final eval tasks; FIVE updates.
    bash scripts/run_learning_credit_matrix.sh \
      runs/lc-frozen-1701 runs/lc-full-1701 full 1701

    # Direct parameter-update intervention on disjoint meta task examples.
    python scripts/probe_learning_credit_delta_theta.py \
      --data-dir runs/lc-frozen-1701 \
      --shapley-run runs/lc-full-1701/shapley \
      --output runs/lc-intervention-r0-1701 \
      --round 0 --anchors 4 --meta-limit 12

    # OPTIONAL G ablation on 16 train / 32 final tasks, fixed T=4, five rounds.
    bash scripts/run_learning_credit_rollout_scales.sh \
      runs/lc-frozen-1701 runs/lc-g-scales-1701 \
      runs/lc-full-1701/shared_initial_adapter

Source-paired five-round output: paired_summary.json under each matrix root, alongside per-arm reserved_audit.json and optional intervention results. No output path may be silently overwritten.

**Confirmation seeds:** on the same frozen data manifest, run full 1702 and full 1703 with distinct output roots and new initial adapters; each arm WITHIN one seed still shares one adapter. Then:

    python scripts/aggregate_learning_credit_seeds.py \
      --roots runs/lc-full-1701 runs/lc-full-1702 runs/lc-full-1703 \
      --output runs/lc-confirmation-summary.json

A single-seed task-paired bootstrap does NOT quantify training-seed variance.

## Explicit decision gates

1. Abort when isolation, dataset integrity, LoRA init identity, or round counts differ.
2. If Shapley and direct_max choose identical subsets in almost every source, the claimed attribution selection mechanism is NOT identifiable.
3. If direct_max has equal/better heldout feedback alignment, meta Delta(theta) and final unseen repair, Shapley is not necessary despite being mathematically valid.
4. If matched_random_updates does as well, conservative low update frequency, not feedback credit, may explain the previous +2/64.
5. If random_subset does as well at similar actual update count, selecting specific evidence messages is not shown useful.
6. Positive teacher LogP shift alone is not learning credit. Require actual LoRA intervention evidence and unseen-task improvement, plus matched compute costs.
7. Before a strong paper claim, reproduce across >=3 training seeds and compare to the FULL official SDPO baseline; previous matched SDPO-style is an ablation, not the authors' complete algorithm.

**Current five-round status:** code and CPU validation are separate from H100 execution. The GitHub three-round result must NOT be rewritten as a new five-round result.
