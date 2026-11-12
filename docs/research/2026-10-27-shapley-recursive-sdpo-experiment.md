# EESD: Shapley-Guided Recursive Self-Distillation (2026-10-27)

**Research status: prospective.** Code is uploaded; the 20 local CPU tests pass. No full 7B/H100 run or learning gain has been measured. This branch preserves the old lexical EED, old Shapley-Dirichlet and the separate Gate-0/1 research history. All results in the branch are design/software verification unless accompanied by a real-model evaluation receipt.

## One model, one benchmark, one principal baseline

- Model: **Qwen/Qwen2.5-Coder-7B-Instruct**, immutable revision **c03e6d358207e414f1eca0bb1891e29f1db0e242**, LoRA rank 16, NF4/BF16, one H100.
- Benchmark: **RunBugRun Python**, official SQLite dataset from https://github.com/giganticode/run_bug_run. The raw database includes buggy code, problem text and executable I/O tests. The small buggy/fixed Hugging Face Parquet alone lacks the execution harness/tests, and is not sufficient.
- Main baseline: **SDPO-style full-feedback self-distillation** (ICML 2026; https://arxiv.org/abs/2601.20802). It uses identical rollouts, feedback, frozen round teacher and reverse-KL objective, without Shapley credit. **This is an ablated, matched SDPO-style baseline, NOT a full reproduction of the official SDPO package** (which has EMA/JSD/importance sampling/success-peer options). For a submission-grade comparison, run the official SDPO implementation too.
- Necessary ablations: **no_update**, **LOO-token attribution**, **exact 4-test Shapley-token attribution**.
- Task budget: 64 training source programs, 64 unseen evaluation source programs, eight sampled repairs per train source, four public tests each, three *recursive* training rounds. Pilot before full: 4 train sources, 8 eval sources, G=4. Exactly one greedy new candidate per unseen evaluation source each round.

**No gold fixed code is read or trained on.** Existing public I/O expected outputs are trusted behavioral supervision; the model does NOT know the complete correct program, and passing finite tests is NOT a correctness proof. Test- and problem-level leakage is prohibited. The data extractor uses SQL fields b.id, b.problem_id, b.buggy_code, p.text and tests.{id,input,output}, but never b.fixed_code. It requires distinct train/eval problem IDs (dataset split numbers 0 and 2). Train-public, train-audit and evaluation test files are emitted separately, checksummed and immutable; training reads only the train-public file.

## Precise novel mechanism under test

At round k, freeze the Qwen checkpoint at theta_k and generate eight code rollouts y_i. Execute each on the four public tests. Preserve rich textual feedback: input, expected output, actual output, exception and outcome.

For each actual sampled code token y_it, teacher-force the **same frozen Qwen** under every subset of the four execution feedback records:

    v_it(S) = log pi_{theta_k}(y_it | problem, buggy code, feedback_S, y_i,<t)

The signed Shapley effect of test j on token t is

    phi_ijt = sum_{S not containing j} |S|!(4-|S|-1)!/4!
              * [v_it(S union {j}) - v_it(S)].

This is **counterfactual token preference attribution**, NOT ground-truth test reliability. The negative sign is retained: a negative attribution may help by reducing probability of wrong code. Do not interpret sign alone as good or bad.

Instead of old EESD's single correction-level Dirichlet weight, use token-local evidence breadth and directional agreement. Let

    a_it = sum_j abs(phi_ijt)
    E_it = a_it**2 / sum_j phi_ijt**2
    C_it = abs(sum_j phi_ijt) / (a_it + epsilon)
    raw_w_it = (E_it/4) * C_it * min(1, a_it/0.15)

and raw_w=0 when attribution is effectively zero; normalize nonzero weights to unit mean per rollout. This is a **proposed experimental heuristic** and not a proof of correct behavior. Critically, a zero-Shapley feedback event cannot fall back to maximal evidence confidence. Signs control cancellation/coherence, while the teacher's full-vocabulary distribution preserves the actual increase/decrease direction.

The self-teacher target is the frozen **full feedback** distribution, NOT code similarity and NOT a gold solution:

    q_it^F = pi_{theta_k}(. | x, feedback_all, y_i,<t)

The student gets **no feedback** when being trained. Both SDPO-style and Shapley arms optimize the SAME reverse-KL objective at actual generated token positions:

    loss = sum_t w_it KL(p_theta(.|x,y_<t) || stopgrad(q_it^F)) / sum_t w_it
           + 0.02 mean_t KL(p_theta(.|x,y_<t) || stopgrad(p_{theta_k}(.|x,y_<t))).

For SDPO set all w=1. For LOO, compute test marginal influence only from the full coalition and four leave-one-out coalitions, then use the same weight function. For Shapley enumerate all 16 coalitions. As a result, success or failure of Shapley relative to LOO can be isolated from the student loss.

## Truly recursive update, without stale sampling policy

For each arm separately:

    theta_0 --fresh code/test feedback--> frozen teacher theta_0
        --dense self-distillation--> theta_1
    theta_1 --fresh code/test feedback--> frozen teacher theta_1
        --dense self-distillation--> theta_2
    theta_2 --fresh code/test feedback--> frozen teacher theta_2
        --dense self-distillation--> theta_3

At the start of every round, the **sampling model** and **feedback self-teacher** are an identical frozen snapshot theta_k. Within the round, the student starts from theta_k and takes one optimizer step per train source. At next round it becomes theta_{k+1}; all rollouts are regenerated. This avoids the earlier rlvr/train_grpo.py stale sampler. It is *round-synchronous self-distillation*, not strict minibatch on-policy GRPO. No successful peer or external teacher is needed.

## Data and run instructions

First download and restore the **official** RunBugRun SQL database as described in giganticode/run_bug_run README. The database is not stored in GitHub and must be supplied under your own local path. The execution worker **requires bubblewrap and functional unprivileged mount/PID/user namespaces**; it never falls back to unrestricted subprocess execution. Verify its positive control on the H100 host. Do not use generated code directly on the shared host.

From the repo root:

    uv sync --extra ml --extra test
    export PYTHONPATH="$PWD/src"
    python scripts/prepare_recursive_runbugrun.py \
      --db /data/runbugrun.db --output runs/rbr-frozen-1701 \
      --train-sources 64 --eval-sources 64 --seed 1701

Check the plans (no GPU/model required):

    python scripts/run_recursive_shapley_sdpo.py \
      --data-dir runs/rbr-frozen-1701 --arm shapley \
      --output runs/dry-run-unused --dry-run

CPU software tests:

    python -m pytest -q tests/test_recursive_credit.py \
      tests/test_recursive_execution.py tests/test_recursive_runbugrun.py \
      tests/test_recursive_runner.py tests/test_recursive_summary.py

Run the **four-arm pilot**, then inspect every report before executing the full preset:

    bash scripts/run_recursive_sdpo_matrix.sh runs/rbr-frozen-1701 runs/pilot-1701 pilot
    bash scripts/run_recursive_sdpo_matrix.sh runs/rbr-frozen-1701 runs/full-1701 full

This script executes the frozen initial no-update arm, the SDPO-style baseline, LOO and Shapley sequentially. Each output is create-once and should not be overwritten. For strict across-arm checkpoint identity use --init-adapter pointing to the SAME independently created adapter directory. With no explicit adapter, identical seed 1701 and initialization code are used and round-0 greedy outputs must match or the summary refuses interpretation.

The final paired report is under output_root/paired_results.json. It compares unseen-problem all-tests-pass@1 on the same IDs across rounds 0/1/2/3, source-cluster bootstrap CI versus SDPO, fixes, regressions, sampled-code diversity, zero-credit fraction and token-credit statistics. Preflight if necessary uses 4 training tasks, 8 evaluation tasks, 4 rollouts. **Preflight estimates are not publication-grade benchmarks.**

## Minimum scientific decision gates

1. Validate no gold corrected code, no train/eval problem overlap, exact public test count, pinned model snapshot and dataset-file hashes.
2. Verify the sandbox executes an isolated positive control; crash if bwrap cannot start (never turn sandbox failure into all-zero scores).
3. Confirm actual nonzero student LoRA gradients, no teacher updates, correct autoregressive token shifts, no NaNs, and that next-round rollouts come from the updated model.
4. Confirm matched round-0 metrics and compute budgets across arms. Record all four tests per rollout, all 16 Shapley subsets, walltime and GPU peak memory; SDPO needs 2 teacher queries per rollout including no-feedback anchor, LOO needs 6, Shapley needs 16.
5. Judge **unseen programming problems**; not merely fresh tests of training problems. Report uncertainty and per-round negative transfers, do not cherry-pick positive tasks.
6. If Shapley improves training-only tasks but not held-out task pass@1, or is not better than the cheaper LOO control at comparable test budgets, do NOT declare success or add new datasets/models before revising the mechanism.
7. Only after this one-model, one-benchmark gate succeeds, run confirmation seeds 1702/1703 and a **faithful official SDPO** external comparison.

### Important implementation limits

- Built-in result files represent a research implementation. The main training code has NOT been exercised with real 7B weights or a real restored RunBugRun SQLite snapshot here, so H100 compatibility, end-to-end packaging and method gains remain unverified.
- A shared training set with trusted input/output tests is a partial verifier; it does **not** cover the harder zero-label setting where all test expectations are generated autonomously.
- LoRA is parameter-efficient optimization of both arms, not the definition of self-distillation. Full-model fine-tuning is not required to test the attribution hypothesis.
- Exact 16-coalition teacher evaluations cost much more than full-feedback SDPO. Compare not only final accuracy but also accuracy per evaluation/test budget and per GPU-hour.
