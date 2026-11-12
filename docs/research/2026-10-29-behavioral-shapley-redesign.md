# EESD v2: Behavioral Shapley for Selective Recursive Self-Distillation

Research date: 2026-10-29. Status: IMPLEMENTED, CPU-TESTED, NOT YET RUN ON THE FULL QWEN MODEL.

**Do not replace the negative result.** The previous H100 experiment is preserved at [2026-10-28 Shapley recursive full results](2026-10-28-shapley-recursive-sdpo-full-results.md). On 64 unseen RunBugRun programs, three rounds of token-weight-only Shapley were 29/64 versus SDPO-style full-feedback 30/64 versus frozen Qwen 31/64. This protocol tests a *new mechanism*, not a retroactive success claim.

## 1. Exact scientific question

The old Shapley system asked which test changed the teacher's confidence in sampled code. Its weight depended on the magnitude and agreement of these changes, but all tests were still used to form the teacher, and even bad code was distilled.

The revised question is: can a test's influence on sampled-token probabilities predict a **useful redistribution of confidence between different code behaviors**, and can we avoid imprinting unsupported feedback during recursive learning?

This is not correctness proof. RunBugRun has trusted finite input/output probes, but no fixed-code reference is read by our training code. Calibration probes are extra training-time partial verification; this is NOT the stronger fully unlabelled setting.

## 2. Four-way data contract: no hidden-label contamination

Model: Qwen2.5-Coder-7B-Instruct, pinned revision c03e6d358207e414f1eca0bb1891e29f1db0e242. Exactly one benchmark: RunBugRun Python, official SQLite.

Train sources (64): each must contain at least twelve independent test cases:

- F = 4 public feedback tests: actual/expected output and errors are put in self-teacher prompts.
- C = 4 separate **training-calibration** tests: provide pass/fail **only to the training-side behavioral selector**, NEVER to the teacher prompt.
- A = 4 **reserved audit** tests: held out entirely from training and from selector; file is only opened to check test ID disjointness.
- E = completely unseen problem IDs from the official RunBugRun test split (64 problems). All official E tests are used only for final independent evaluation.

The new extractor is scripts/prepare_behavioral_runbugrun.py; it does not query fixed_code. Do NOT use the earlier frozen 4+4 dataset and silently relabel its train_audit_do_not_train file as trainable: doing so violates the earlier protocol.

## 3. Mathematical definition

At recursive round k, freeze Qwen model pi_k and sample G=8 distinct or potentially repeated complete Python programs y_i under the same task x. Each program is executed against the four public tests to produce its *own* rich feedback F_ij (input, expected, actual, exception, PASS/FAIL). Score the same fixed **canonical executed code** in the frozen model under all 16 subsets of public feedback. Each candidate is always scored against its own feedback; therefore the following is a behavioral preference *surrogate*, not a normalized probabilistic distribution over the candidate pool.

For candidate i, position t and test subset S:

    s_it(S) = log pi_k(y_it | x, F_i,S, y_i,<t)
    delta_i(S) = (1/L_i) sum_t [s_it(S) - s_it(empty)].

On training-calibration probe h, let c_ih be pass=1 / fail=0 of candidate i. Center over candidates on each probe:

    z_ih = c_ih - (1/G) sum_r c_rh.

Define the alignment game for each calibration probe:

    V_h(S) = (1/G) sum_i z_ih * delta_i(S)
    V(S)   = (1/H) sum_h V_h(S).

A positive V means test feedback increases the frozen model's probability of programs doing better on the **independent calibration probes**, and/or decreases the probability of programs that do worse. It is zero when all code rollouts are behaviorally identical on every calibration probe. No claim of global correctness follows.

The signed behavioral Shapley of public test j is

    phi_j = sum_(S subset [T]\{j})
            (|S|! (T-|S|-1)! / T!) [V(S union {j}) - V(S)].

Because the game is linear in the log-probability shifts, the same quantity can also be derived by combining the per-test per-token Shapley with calibration-centered program quality. We compute all 16 coalitions to preserve interactions.

**Important:** For every j, compute one phi_j,h on each calibration probe h, rather than only one aggregate phi_j. Let mean_j and SE_j denote its average and standard error across H=4 probes. Mark test j eligible only if

    mean_j > kappa * SE_j      (kappa=0.5, exploratory).

For each eligible-only joint subset S, score:

    J(S) = mean_h V_h(S) - kappa * SE_h[V_h(S)] - gamma * |S|

with gamma=0.001; select the maximizing S if J(S)>epsilon=0.001; otherwise abstain, select no evidence, perform no update. The joint coalition score handles interference between individual positive attributions. Unlike simply summing Shapley values (which collapses to V(all)-V(empty)), this selection genuinely uses individual causal-style attribution and coalition interactions.

The SE penalty is a small-sample heuristic, **not** a valid statistical certificate of positive generalization with four calibration probes. Tune only on training/development sources. Never inspect E for test selection.

## 4. Dense token-level supervision with behavioral sign

For selected feedback S*, compute token preference shift:

    d_it = s_it(S*) - s_it(empty).

Define a_i = mean_h c_ih - mean_(r,h)c_rh, the candidate's relative calibration success. Construct confidence-PRESERVING weights

    w_it = |a_i| * clip( max(0,sign(a_i) * d_it)/0.15, 0,1 ).

Interpretation: a good candidate + stronger probability of that token => promote it; a poor candidate + reduced probability of that token => learn to avoid it; any wrong-direction token or all-equal calibration results => zero weight. The target is still the actual frozen teacher *distribution* conditioned on selected feedback (not a gold program).

Student p_theta gets no feedback and is optimized via:

    L = mean_(i,t) [ w_it * (
           KL(p_theta(.|x,y_i,<t) || stopgrad(q_k(.|x,F_i,S*,y_i,<t)))
        + 0.02 KL(p_theta(.|x,y_i,<t) || stopgrad(p_k(.|x,y_i,<t)))
        ) ].

Implementation averages per candidate sequence then divides by the count of generated candidates; all arms use the same reduction and tokenizer. Crucially there is **NO** division by sum_t w_it and **NO** per-rollout unit-mean normalization. A weak/zero evidence signal must lead to a weak/zero gradient, not a full-strength update.

Each round generates NEW candidates with updated theta_k, freezes this theta_k as teacher, distils into theta_(k+1), and repeats for three rounds. Existing Markdown-fenced answers are canonicalized to the exact executable Python text *before* token-level scoring/training, removing the earlier mismatched target of self-distilling markup instead of executed code.

## 5. Exactly matched comparison

One model + one dataset throughout:

| Arm | Teacher feedback | Train-time calibration probes | Token update |
|---|---|---|---|
| no_update | none | none | frozen base |
| sdpo | all 4 public feedback tests | executed but NOT consulted | equal token-weight feedback-conditioned KL |
| audit_gate | all 4 public tests | yes, same 4 | behavior-aligned candidate/token gate, no Shapley |
| shapley_select | Shapley-chosen subset of public tests | yes, same 4 | identical behavioral token gate, **plus per-test coalition selection** |

The sdpo control is a matched SDPO-style full-feedback reverse-KL baseline, NOT the complete published SDPO implementation with EMA/JSD/importance sampling. A submission-grade comparison also needs the original authors' implementation. Shapley adds heavy teacher scoring; compute and cost must be reported.

Primary metric: *one newly generated, greedy repair* per previously unseen RunBugRun test problem, all official tests pass. Secondary: mean test pass fraction, fixes and regressions against frozen base, round-1/2/3 trajectories, source-paired 95% bootstrap intervals, training abstention, number of selected feedback tests, GPU compute, sandbox failures. Identical initial program hashes across arms are required. Do not interpret pilot tasks or positive training-only movement as evidence of generalization.

## 6. Preregistered procedure and exact commands

Prerequisites: official RunBugRun SQLite database, pinned model cached on H100, Python/torch/peft dependencies, a working isolated execution backend. The previous shared H100 allowed only a **soft** sandbox due to blocked bubblewrap mount/user namespaces. That is NOT sufficient for untrusted generated Python. The NEW runner deliberately fails closed; provision an isolated VM/container with functioning bubblewrap isolation before executing it. Do not work around this by disabling isolation.

From repo root (fresh output paths are mandatory):

    export PYTHONPATH="$PWD/src"
    python scripts/prepare_behavioral_runbugrun.py \
      --db /data/runbugrun.sqlite \
      --output runs/behavioral-data-1701 \
      --train-sources 64 --eval-sources 64 --seed 1701

    python -m pytest -q tests/test_behavioral_shapley.py \
      tests/test_behavioral_runner.py tests/test_behavioral_prepare.py \
      tests/test_behavioral_summary.py

    python scripts/run_behavioral_shapley.py \
      --data-dir runs/behavioral-data-1701 --arm shapley_select \
      --output runs/prospective-dry-run --dry-run

    bash scripts/run_behavioral_matrix.sh \
      runs/behavioral-data-1701 runs/behavioral-pilot-1701 pilot

Only after the pilot validates isolation, nonzero gradients, selected feedback, nonempty audit contrast, matched initial checkpoint and no contamination:

    bash scripts/run_behavioral_matrix.sh \
      runs/behavioral-data-1701 runs/behavioral-full-1701 full

The runner creates immutable per-round evaluation, training and LoRA output folders. Summary appears at runs/behavioral-full-1701/paired_results.json. The matrix runs all four arms with a fixed random seed and refuses to overwrite results. Repeated model initialization is checked by comparing initial greedy generated code hashes before interpreting outcomes.

## 7. Expected failure modes and stop gate

- If all candidate programs pass/fail identical calibration tests, no useful *relative* behavioral credit exists: abstain rather than hallucinate a reward.
- If sign-aligned Shapley fails to predict holdout benefit, reject this value function; Shapley is not a correctness verifier.
- If shapley_select beats sdpo but not audit_gate, the gain comes from calibration/candidate gating, **not** from Shapley; do not claim Shapley innovation.
- If Shapley is slower than a simpler gate with equal accuracy, consider cheaper LOO only after the mechanism is shown useful.
- If E performance does not beat original frozen Qwen, the central recursive improvement claim remains unsupported.
- Do not switch models/datasets or announce success before this primary RunBugRun pilot/full comparison.
- A trial restricted to H=4 calibration tests risks selection overfit; a strict follow-up should test calibration size and preserve a separate audit stream. Confirmation seeds 1702/1703 are reserved, not yet run.

Code files: src/eesd/behavioral_shapley.py, src/eesd/strict_execution.py, scripts/run_behavioral_shapley.py, scripts/prepare_behavioral_runbugrun.py, scripts/run_behavioral_matrix.sh, scripts/summarize_behavioral_shapley.py.

**State of evidence as of this commit:** local unit tests pass; no new 7B/H100 outputs are reported, and the October 28 negative results remain the latest actual benchmark result.


## 8. Independent post-training feedback audit (automated in matrix)

The four reserved train-side audit tests are **not used** when selecting feedback or computing gradient weights. After all three recursive training rounds are finished, each Shapley training record contains locally stored canonical candidate programs and per-coalition mean log-probability shifts. The matrix invokes:

    python scripts/audit_behavioral_shapley.py \
      --data-dir runs/behavioral-data-1701 \
      --experiment-dir runs/behavioral-full-1701/shapley_select \
      --output runs/behavioral-full-1701/reserved_audit.json

It **only evaluates** already generated programs in a proper sandbox; there are no GPU weight updates. It recomputes the behavioral alignment game using previously unseen reserved audit outcomes, comparing the selected subset with full feedback. Key fields are mean_selected_minus_full_on_heldout and fraction_accepted_with_positive_heldout_alignment. These audit values must never be used to adapt selection hyperparameters on the same run; if they are inspected during development, create an entirely new disjoint experiment manifest for final confirmation.

The extra runner metadata (candidate_sources and candidate_shift_means) resides solely in gitignored local runs/ output, not in GitHub. It is sufficient to reconstruct the independent audit without loading the Qwen model again.

The local pure-math/mocked runner test suite for this revision passes 19 tests; a live 7B+Bubblewrap end-to-end run has not yet been observed.
