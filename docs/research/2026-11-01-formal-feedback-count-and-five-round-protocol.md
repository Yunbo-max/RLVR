# EESD formal experiment: is 4 execution feedback tests enough?

Date: 2026-11-01. **Prospective. Not a new H100 result.**

## Decision

**Four public feedback tests are not excessive.** They are the smallest useful setting for exact individual-feedback Shapley attribution with enough possible interactions (2^4=16 coalitions). But four are likely too narrow to support a general claim about learning from *rich* execution environments, especially if they trigger identical outcomes on all G sampled programs. The prior 3-round study accepted only ~7.8% of training source-rounds; this is a reason to test evidence sparsity, not proof that T=4 caused the low acceptance. Four independent calibration probes were also noisy.

The scientific plan has TWO levels rather than casually increasing tests while changing the dataset.

**Mechanism proof:** T=4 public feedback, C=8 independent calibration, A=8 reserved audit, G=8 candidates, exact 16-coalition Shapley vs exact direct-J subset maximization and five control arms. Five recursive updates and six untouched validation snapshots.

**Feedback richness sensitivity:** T=8 public feedback, C=8, A=8, G=8, same model, same eligible problem IDs, same initial checkpoint and *nested* frozen feedback test bank. Exact 2^8=256 coalition scoring is 16x the T4 cost, so the T8 GPU runner MUST be refactored to use a budgeted permutation Shapley estimator (e.g., paired forward/reverse permutations) and a teacher-query-matched direct greedy/beam subset-search comparator. Do not compare exact T4 with 8-test approximate results without reporting the changed scoring budget and approximation uncertainty. T8 implementation is NOT included in the existing T4 runner. Do NOT run T8 by changing a constant from 4 to 8.

T=16/32 public tests are further stress experiments only, not planned for full exact enumeration.

## Why T=4 might fail

There are three distinct deficiencies, none established merely by counting tests:

1. Lack of *behavioral discrimination*: all eight code rollouts may have the same outcomes on every calibration probe. Then the centered behavioral game V(S)=0 and the selector should abstain.
2. Redundant or misleading feedback: distinct test IDs may reuse an identical input or give different expected outputs; four superficially different records may contribute little new behavior. A rich environment often has diverse exceptions, wrong outputs, edge cases and passing controls. Record diversity directly from executable PASS/FAIL vectors; do NOT assume diversity from test IDs alone.
3. Small calibration variance estimate: with only four independent calibration probes, a purportedly positive phi_j may simply be noisy. Increase C to eight separately from increasing public feedback T.

Record the fraction of train tasks with at least two distinct rollout outcome vectors, mean test-outcome entropy, fraction with nonzero calibration contrast, selector abstention, heldout-A alignment, average selected-mask size and cost per accepted update. Evaluate both T4 and T8 on the SAME eligible source population; re-sample three frozen public-test subsets to measure sensitivity to which tests were chosen.

## Mandatory data feasibility audit (run on actual restored SQLite)

Official RunBugRun includes short executable programs and the tests table, but its README does NOT establish the complete distribution of distinct tests per Python problem. We cannot assert that 512 independently eligible train problems or 256 independently eligible final problems with >=24 unique inputs are available until measuring them.

    python scripts/audit_runbugrun_feedback_feasibility.py \
      --db /data/runbugrun.db \
      --output runs/rbr-feasibility-20261101.json

Inspect split_stats.train/test eligible_without_conflicting_inputs["24"]:
- distinct_problem_ids is the maximum sample size when selecting one buggy program per problem.
- total_bug_instances counts alternative bugs from the SAME problems and must NOT be treated as independent problem samples.
- Inconsistent expected outputs for an identical input exclude the problem from the strict cohort.
- Split 0 (train) and split 2 (test) may share problem IDs in the raw database; the frozen selected cohorts must not.

**If fewer eligible problems exist than requested, do not replace independent problems with near-duplicate submissions and pretend statistical power increased.** Instead register a smaller number of independent problem clusters and optionally evaluate additional bugs as a secondary clustered unit. Report coverage/exclusion rates explicitly.

### T=4 formal frozen data

The executable preparer, scripts/prepare_formal_runbugrun_t4.py, requires >=24 distinct input tests per selected problem, picks at most one bug per problem ID and eliminates inconsistent expected outputs. Its nested public bank is:

- master public: 8 candidates (T=4 uses the first four, four are unused)
- calibration C: 8 distinct inputs never placed into the feedback prompts
- reserved audit A: 8 distinct inputs not used until the completed run
- meta intervention: separate problem IDs from train-split, not used for training/selection
- final evaluation: distinct official test-split problem IDs

This prevents an unfair T4-vs-T8 comparison with different problem difficulty, subject to the future T8 trainer using the same frozen master pool. The record SHA256 and source IDs are immutable.

## Main experiment, 5 recursive rounds, 1 model, 1 dataset

Preferred target, CONDITIONAL on the real feasibility audit:

| Element | Formal target |
|---|---|
| Model | Qwen2.5-Coder-7B-Instruct @ pinned commit, NF4+LoRA r16 |
| Data | RunBugRun Python, one bug per distinct problem_id for primary |
| Train | 512 distinct problem IDs (fallback only if census says insufficient) |
| Independent meta | 64 distinct problem IDs (for single-step Delta(theta) diagnosis) |
| Heldout official test | 256 distinct problem IDs (or all eligible if fewer) |
| F / C / A | 4 public + 8 calibration + 8 reserved audit, from >=24 unique-input bank |
| Generated programs | G=8 per train source per round |
| Recursive updates | 5, with fresh rollouts each round |
| Seeds | 1701, 1702, 1703 |
| Controls | No Update, SDPO-style, Audit Gate, DirectMax, Shapley, Random Subset, Exactly Matched Random Updates |
| Main comparison | Shapley vs DirectMax exact 16-coalition, AFTER five rounds |
| Other endpoints | R0–R5 pass@1 / test pass fraction; regressions/fixes; no-feedback policy KL; heldout-A evidence alignment; actual steps; GPU cost |

Prefer a confirmatory final evaluation including as many *independent official test problems* as feasible; 256 unique problems is a practical first complete formal target but has limited power to detect a 2-3 percentage point difference. If 1,000 bugs are added from the same test problems, bootstrap/aggregate by problem_id, not by buggy submission, or the confidence intervals will be too optimistic.

**Exact T4 scoring cost per heavyweight arm**, with N_train=512, G=8 and five updates:

    512 * 8 * 5 * 16 = 327,680 teacher subset forward evaluations

Both Shapley and DirectMax require this; three seeds need nearly two million subset evaluations for the two main arms, before generative inference, additional teacher distributions, code executions and other baselines. Thus estimate GPU hours by measuring a small real H100 pilot. Do NOT invent wallclock estimates.

### Seven-arm fair comparison

The most important comparison is Shapley vs DirectMax with the SAME F/C/A bank, sixteen coalition values, optimizer and training starting point. Because DirectMax optimizes a superset of Shapley-eligible coalitions, DirectMax has at least as good the training-side J(S). Shapley can only demonstrate value through better INDEPENDENT audit/parameter-update/generalization behavior.

Include a random feedback subset of the same size on the Shapley-accepted training sources and a random-training-source arm matching the actual number of optimizer steps per round. Also measure real parameter/policy KL movement: matching counts does NOT match update strength. Full official SDPO, with its actual training tricks, should be a separate publication baseline; the current branch's SDPO-style is an ablation, not equivalent.

### True learning-credit intervention

At frozen checkpoints R0, R2 and R4, pick accepted training-source anchors by *TRAIN-ONLY* criteria and compare Shapley, DirectMax, random same-cardinality subset and Full Feedback. Each is reset to the identical pre-update LoRA state, given identical generated candidate code and the same tests, performs one REAL LoRA update or abstains, then generates code on independent meta problems. Record:

    Delta_meta = Pass@1_meta(after step) - Pass@1_meta(before step)

Also report continuous per-test pass-rate changes, gradient norm, update/no-update, realized policy KL and improvement/regression events. Four anchors/twelve meta tasks (the old micro probe) is too small for formal inference; use all eligible training anchors up to a preregistered cap and >=64 disjoint meta tasks if feasible. Do not select evidence with meta performance.

### Statistics and preregistration

- Primary: paired Shapley vs DirectMax on R5, grouped by problem_id; one fixed primary contrast and Holm correction for auxiliary planned contrasts.
- Report task-cluster 95% confidence intervals and per-training-seed results; the independent training seeds capture optimizer/sampling variability that a single-source bootstrap misses.
- R0..R5 curves are mandatory; **do not choose the best round using final evaluation**.
- The independent reserved A probes assess whether chosen feedback has higher heldout alignment than full feedback or direct J(S).
- Measure public-feedback outcome diversity and acceptance by F count, not only end-to-end pass@1.
- Log completion/exception/timeout categories, full input-token budgets and whether rich feedback was truncated.
- Record GPU-hours, peak memory and teacher subset calls per corrected program; a more expensive Shapley with no better heldout behavior is not justified.
- Source code and exact selection manifests should be frozen before reading the final test scores; save anonymized per-task pass/fail and experiment receipts without publishing executable untrusted raw code if unsafe.
- If even this larger experiment gives <~3pp gain on a limited number of independent problem clusters, say the result is inconclusive rather than claim robust superiority.

## Exact executable T4 workflow

    export PYTHONPATH="$PWD/src"
    uv sync --extra ml --extra test
    uv run pytest -q tests/test_formal_feedback_feasibility.py tests/test_formal_runbugrun_t4.py

    # 1. Census: prove eligibility. Do this before freezing any final test list.
    python scripts/audit_runbugrun_feedback_feasibility.py \
      --db /data/runbugrun.db \
      --output runs/feasibility.json

    # 2. If eligible, freeze ONE immutable cohort for ALL seeds/controls.
    python scripts/prepare_formal_runbugrun_t4.py \
      --db /data/runbugrun.db --output runs/formal-t4-frozen \
      --train-sources 512 --meta-sources 64 --eval-sources 256 --seed 1701

    # 3. Pilot, then exact-T4 mechanism, then full 5-round three-seed matrix.
    bash scripts/run_formal_t4_matrix.sh \
      runs/formal-t4-frozen runs/formal-t4-pilot-1701 pilot 1701

    bash scripts/run_formal_t4_matrix.sh \
      runs/formal-t4-frozen runs/formal-t4-mechanism-1701 mechanism 1701

    bash scripts/run_formal_t4_matrix.sh \
      runs/formal-t4-frozen runs/formal-t4-full-1701 full 1701
    bash scripts/run_formal_t4_matrix.sh \
      runs/formal-t4-frozen runs/formal-t4-full-1702 full 1702
    bash scripts/run_formal_t4_matrix.sh \
      runs/formal-t4-frozen runs/formal-t4-full-1703 full 1703

    # 4. Summarize independent training-seed effects:
    python scripts/aggregate_learning_credit_seeds.py \
      --roots runs/formal-t4-full-1701 runs/formal-t4-full-1702 \
              runs/formal-t4-full-1703 \
      --output runs/formal-t4-confirmation.json

The branch contains the runnable T4 exact-coalition workflow and the feasibility audit. **The T8 approximate-coalition GPU experiment remains a separately needed implementation, explicitly marked prospective in configs/experiments/eesd_formal_feedback_4v8.yaml.** Do not claim that the T8 runner or full H100 results already exist.

### Two explicit stop gates

1. If the 24-unique-input cohort cannot provide 512/64/256 distinct problem IDs, change the target only after recording the eligibility census. Stop rather than silently bias the benchmark.
2. If heldout-A evidence alignment and actual Delta(theta) do not show Shapley outperforming equal-budget DirectMax or match-updated controls, adding more feedback does not establish the alleged advantage. Report this as a negative finding.
