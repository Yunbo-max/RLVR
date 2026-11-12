# EESD-Rich64: Budget-Matched Execution Feedback Learning Credit

**Preregistration date:** 2026-11-01. **Status:** experimental code and unit/CI tests only. This branch does NOT contain new H100/5-round model-accuracy results. Do not reuse the older positive 3-round 32/64 figure as if it were a new 64-feedback finding.

## 1. Scientific motivation and exact research question

Earlier EESD used T=4 executed public tests for each of G=8 code rollouts and 64 distinct training problems. Its more recent 3-round Behavioral Shapley experiment reported 32/64 heldout programs passing versus 30/64 frozen and audit-only and 23/64 matched SDPO-style. The +2 relative to audit-only was too small to establish a robust independent Shapley benefit, and ~7.8% of source-rounds accepted updates.

The NEW hypothesis is not simply "more tests = better." It is:

> When many distinct execution records are available, can a fixed-budget, behavior-grounded Shapley decomposition avoid harmful feedback-induced self-updates and improve unseen programming correctness relative to direct value maximization, randomized feedback groups and equally sparse parameter updates?

We want to scale from T4 to **up to 64 actually executed tests** without increasing the teacher **subset-coalition score count per candidate** beyond the earlier K4 exact16 budget. The experiment MUST distinguish test execution cost, teacher subset evaluations, prompt length, sample generation cost and actual parameter KL.

## 2. Five genuinely different quantities

| Symbol | Primary value | Meaning |
|---|---|---|
| N | up to 8,000 BUG instances | Sampled buggy submissions, NOT 8K independent problems |
| G | 8, with G4/G16/G32 ablations | Candidate repaired programs sampled per bug |
| T | 4,8,16,32,64 | Public tests whose outcomes the feedback selector is allowed to see |
| K | 4 | **Groups of execution evidence**, Shapley *players* |
| B | 2^K=16 | Frozen teacher-forcing coalition scores **per candidate** for group Shapley and matched exhaustive group DirectMax |
| R | 5 | Recursive rounds; evaluate six checkpoints R0..R5 |

Each evaluated candidate actually runs against the **same 64 public tests** in each T arm. For T4, results from public tests 5..64 are computed for matched execution cost but NEVER supplied to group formation, teacher prompts or learning credit. For T64, all 64 public outcomes enter group formation. The independent calibration C8 and reserved audit A8 are never public feedback and remain identical across arms.

T is a number of test *inputs*, NOT four/five error classes. K=4 is a number of compressed feedback groups. Exact Shapley requires only 16 groups-subsets, not 2^64 test-subsets. The groupwise φ_g cannot honestly be described as φ_j for each of 64 individual tests.

**Data feasibility warning:** IBM CodeNet, RunBugRun's source, has about 4,053 distinct programming problems overall. Thus N=8K can only mean BUG instances with repeated problem_id. The 8K target requires ≥500 eligible distinct training problems at a cap of 16 bug instances/problem, PLUS 64 disjoint meta problems and 256 disjoint official test problems, all with ≥80 unique valid input tests. Run scripts/audit_runbugrun_feedback_feasibility.py on the ACTUAL SQLite file first. We do NOT know that such an eligible population exists. If it does not, report the shortfall and adjust a *new* frozen protocol, not fabricate test inputs or treat repeated problems as independent.

## 3. Why group four pieces of evidence when 64 tests exist?

For each training bug x:

1. Frozen pi_k samples G candidate programs y_i.
2. Execute each y_i on 64 fixed public test inputs, obtaining actual output, expected output, exceptions, and PASS/WRONG_OUTPUT/RUNTIME_EXCEPTION/COMPILE_ERROR/TIMEOUT statuses.
3. For a given T, consider ONLY the first T public tests. For each test j construct an outcome signature across G candidates, z_j=(status_1j,...,status_Gj). Cluster these T test inputs deterministically into K=4 *behavioral groups* based on similarities between signatures. No calibration/audit outcomes participate in this grouping.
4. For candidate i, form four bounded-size JSON feedback summaries, one per group. Each group reports total executed count, outcome category counts and a few representative failing/passing input-output examples. Maximum 440 characters/group. Both group-Shapley and group-DirectMax use the SAME summaries and teacher prompts.
5. Enumerate all 16 GROUP subsets S ⊆ {g1,g2,g3,g4}; teacher-force the same canonical code output against each selected grouped-feedback prompt. Candidate i's log-prob difference Δ_i(S) is compared with the independent C8 pass/fail quality to construct V_h(S).
6. Use exact signed group-Shapley eligibility or directly maximize the same conservative J(S). Both see the SAME full V array and EXACTLY 16 groups-coalitions per candidate; cost/memory logs report actual token lengths and wallclock.

Mathematically:

    Delta_i(S) = mean_t log pi_k(y_it | x, summary(F_{i,S}), y_i,<t)
               - mean_t log pi_k(y_it | x, y_i,<t)

    V_h(S) = (1/G) sum_i (c_ih - mean_r c_rh) * Delta_i(S)

    phi_g = sum_{S not containing group g}
            |S|!(K-|S|-1)!/K! [mean_h V_h(S+g) - mean_h V_h(S)]

The actual code also computes per-calibration-test φ_g,h and uses the conservative group admissibility rule, then J(S) = mean_h V_h(S) - 0.5*SE_h V_h(S) - 0.001|S|. No feedback -> no gradient. The frozen teacher's selected-feedback full-vocabulary distribution is distilled into the student without feedback using the same reverse-KL family as the older baseline, with signed behavior/Token gate.

**Critical limitation:** groups compress observations and hide most raw I/O examples. This trades away individual-test interpretability and possibly useful rare failures. It is the explicit price of maintaining a near-constant teacher-scoring budget. We must test whether that compression loses information; a score improvement cannot be claimed as the superiority of individual-test Shapley.

## 4. Rigorous, compute-conscious ablation matrix

### A — Feedback richness, matched teacher subset budget

T ∈ {4,8,16,32,64}; keep K=4, B=16, G=8, same source IDs/test bank, same initial LoRA. Compare Shapley-group vs exhaustive group-DirectMax at each T. Both have 16 conditional teacher subset scores/candidate; every scored program executes 64 public tests (+8 calibration tests). The exact number of *tokens* processed differs with the feedback summaries, so log actual wallclock and teacher input token lengths; identical query counts do not imply identical FLOPs.

For each T record both R0..R5 pass@1 and information saturation curves computed FROM THE SAME 64 public executions:

    unique_code_outcome_patterns(T)
    distinct_test_outcome_signatures(T)
    mean_test_outcome_entropy_bits(T)
    calibration_behavior_contrast
    group Shapley vs DirectMax subset disagreement rate
    selected group count, abstention and actual optimizer steps
    training cost per fixed programming problem success

If T16 and T64 give the same distinct outcome patterns, further tests may be redundant.

### B — Shapley necessity (T64)

All seven registered arms, on the SAME frozen data:

- no_update, frozen Qwen;
- sdpo, simplified grouped-full-feedback reverse KL (NOT the full original SDPO publication);
- audit_gate, full-group feedback if C8 behavior-aligned;
- direct_max, exact argmax J(S) across all 15 nonempty group subsets, NO Shapley;
- shapley, groupwise exact signed Shapley admissibility + argmax J(S);
- random_subset, same source accepted by Shapley and same number of groups, but randomized group identity;
- matched_random_updates, same REAL optimizer steps/round as Shapley, randomly selected training bugs, full group feedback.

DirectMax is mathematically guaranteed to have >= Shapley *training* J(S) on the identical candidate pool when both choose a subset. If Shapley wins, it must be on independent heldout behavior/parameter updates, as a potential regularization effect, NOT superior in-sample optimization.

The random-update baseline exactly matches realized step count but not necessarily update magnitude; log actual gradient norm, policy KL and LoRA parameter movement where possible. A random-group selection may skip a step if it has no aligned token evidence, so report its actual steps, not simply the scheduled count.

### C — Is group Shapley sufficient, or is individual-test credit important?

64 individual tests would need 2^64 exact coalitions. The code implements a budgeted unbiased *single-test* Shapley marginal estimator:

    phi_j = E_{k uniform 0..T-1, S uniform subsets of size k excluding j}
              [ V(S union {j}) - V(S) ].

With T64, B=128/256/512 teacher value queries allow only **1/2/4 marginal pairs/test**, respectively. One pair per test has no meaningful within-test variance estimate; 2-4 remain noisy. This is a second, potentially more expensive mechanism diagnostic, NOT the main group-based full trainer yet.

The matching non-Shapley comparator searches random/greedy feedback subsets under the SAME B value-query cap; do not compare a 512-query individual estimator to a 16-query grouped DirectMax as if compute were matched.

Before any five-round individual-test training, verify estimator variance on T4 and T8 problems against exact Shapley and measure the heldout feedback attribution stability. A candidate method needing large B for reliable test-level attribution must justify its GPU-hour cost.

### D — Data, rollout, and horizon scaling

- N = 64 pilot, N=512 mechanism, N=8,000 BUG-instance scale (conditional on eligibility).
- G = 4 pilot, 8 main, 16/32 on a smaller fixed source population.
- R=0..5, with no best-round cherry-picking.
- Three independent training seeds 1701, 1702, 1703 for the main T64 Shapley-vs-DirectMax and its most relevant controls.
- R0/R2/R4 single-step Delta(theta) interventions on disjoint meta programming problems: same candidate pool, same pre-update adapter, same public/calibration tests, choose groups by Shapley/DirectMax/random/full, execute ONE real LoRA step, measure meta repair correctness.

### E — Number of tests used for calibrated credit (separate from F64)

The C8 tests are not included in teacher prompts. Increasing public T alone will not fix unreliable calibration if all G sampled candidates perform identically on C8. C=4 vs8 vs16 should be reported as a separate *calibration quantity* ablation with a correspondingly large frozen input-test bank. The current T64 frozen corpus requires C8+A8; adding C16 requires preparing a new immutable dataset manifest, not borrowing audit labels.

## 5. True compute budget (do NOT promise single-H100 training time)

Main grouped T64 scored candidate: 64 sandbox public test executions +8 calibration executions + 16 group coalition teacher-forcing passes + selected distribution KL calls + model generation.

**Mechanism N=512, G=8, R=5 with repeated same tasks**, per heavy arm, if all candidates valid:

    public executions = 512*8*5*64 = 1,310,720
    calibration executions = 512*8*5*8 = 163,840
    teacher subset scores = 512*8*5*16 = 327,680

**Large N=8,000, G=8, R=5 in ROTATING panels (1,600 distinct bug instances per round)**, per heavy arm:

    public executions = 8,000*8*64 = 4,096,000
    calibration executions = 8,000*8*8 = 512,000
    teacher subset scores = 8,000*8*16 = 1,024,000

These are *planned upper budgets for scored, valid candidates*, not measured receipts. Total wallclock could be much larger than the prior experiment; one H100 may not be enough for all seven arms and three seeds economically. Run the real N64 H100 pilot and estimate GPU-hours per arm before full expansion.

The ROTATING 8K regime does not train the same 8K bugs in all five rounds, but the MODEL pi_k is recursively updated. The repeated N512 panel is needed to separately study reuse of the same coding problems. Do not blur these designs.

## 6. Frozen data and how to run the code

Official RunBugRun: https://github.com/giganticode/run_bug_run ; source CodeNet: https://github.com/IBM/Project_CodeNet

Prerequisites: restored official SQLite available locally, pinned Qwen2.5-Coder-7B-Instruct weights cached, NF4/LoRA packages installed, and WORKING bubblewrap namespace sandbox. No raw model-generated Python may run without sandbox isolation.

First audit how many Python problem IDs have >=80 unique nonconflicting inputs:

    export PYTHONPATH="$PWD/src"
    python scripts/audit_runbugrun_feedback_feasibility.py \
      --db /data/runbugrun.db \
      --output runs/rbr-T64-census.json

Inspect split_stats.train and split_stats.test, especially eligible_without_conflicting_inputs["80"] and the 16-bugs-per-problem theoretical upper bound. Do NOT run 8K if these are insufficient.

Only if the census supports it, freeze EXACTLY ONE dataset for all methods/T/seeds:

    python scripts/prepare_t64_runbugrun_cohort.py \
      --db /data/runbugrun.db \
      --output runs/rbr-rich64-frozen \
      --train-bugs 8000 --meta-problems 64 \
      --eval-problems 256 --max-per-problem 16

All T values take the same fixed first T public test rows; 64 public outcomes are executed but hidden from smaller-T selectors, C8 and A8 are not used in those prompts. The final test metric is **all-tests-pass on a fixed 80-case evaluation suite**, not necessarily the entirety of every problem's official RunBugRun tests. Report this definition precisely.

1. Before H100: run CPU unit tests for feedback grouping, unbiased individual estimator, cohort extraction and dry-run source/prompt isolation.
2. Pilot: use 64 training Bug instances and 64 independent final problems, G4, T64, all seven arms; evaluate real sandbox, rich-feedback prompt budget and GPU/time.
3. Mechanism: 512 training instances and 256 independent eval problems, G8, all seven arms at T64, repeated for five rounds. T4..T64 richness sweep compares Shapley/DirectMax under a shared initial adapter.
4. Scale: 8K train instances across five rotating rounds, G8, T64, Shapley vs DirectMax vs Audit Gate vs frozen. Use seed1701 first and then 1702/1703 only after computational feasibility/positive scientific signal.
5. Real parameter intervention: evaluate R0, R2 and R4 accepted anchors on 64 distinct untouched Meta problems; never choose the feedback using meta outcomes.

Commands:

    uv sync --extra ml --extra test
    uv run pytest -q tests/test_feedback_group_credit.py \
      tests/test_t64_runbugrun_cohort.py tests/test_t64_grouped_five_round.py

    bash scripts/run_t64_group_matrix.sh \
      runs/rbr-rich64-frozen runs/rich64-pilot 64 pilot 1701

    bash scripts/run_t64_feedback_sweep.sh \
      runs/rbr-rich64-frozen runs/rich64-Tsweep mechanism 1701

    bash scripts/run_t64_group_matrix.sh \
      runs/rbr-rich64-frozen runs/rich64-8k-1701 64 scale 1701

    python scripts/probe_t64_group_delta_theta.py \
      --data-dir runs/rbr-rich64-frozen \
      --shapley-run runs/rich64-8k-1701/shapley \
      --output runs/rich64-delta-theta-r0 \
      --round 0 --anchors 16 --meta-limit 64

Output: per-arm plan.json and R0..R5 evaluation.json, R0..R4 training.jsonl/summary.json, paired_summary.json, withheld-A reserved_audit.json, and nested feedback_richness_sweep.json. Frozen source hashes must match across arms and T.

## 7. Publishing / falsification gates

1. Prove F64+C8+A8 real input eligibility and report sampling selection bias; zero gold corrections.
2. The same G, source bank, 64 executed public test inputs, four group players, sixteen teacher group coalition scores and fixed group summary limit for every T ablation; record actual token/FLOP/GPU differences.
3. If T64 gives no higher unique code-outcome patterns, test entropy or heldout correction than T16/T32, explicitly say 64 tests were mostly redundant.
4. If Shapley cannot beat DirectMax on disjoint heldout repair and single-step Meta Delta(theta), do not claim its attribution is necessary.
5. If randomized group subsets or equal-update-count controls do as well, evidence selection may be unnecessary or sparsity alone may explain gains.
6. Under richer feedback, groups may discard rare test-specific signal; compare budgeted individual-test attribution and group randomization at matched query budgets before asserting hierarchical superiority.
7. Five-round R0..R5 curves and independent training seeds are mandatory; do not choose best checkpoint on the final holdout.
8. Publication baseline must eventually include the FULL SDPO implementation, not only the matched SDPO-style loss in this branch.
9. If 8K Bug/80-input constraints are infeasible, publish the actual eligible size and preserve disjoint problem_ids; do NOT manufacture more independent tasks.

**Scope honesty:** The current branch includes an executable K4 group-Shapley five-round GPU runner, strict cohort extractor, T sweep, unit tests and a budgeted 64-individual-test Shapley estimator. The latter is a reusable pure value-game component, not yet integrated as a full individual-Shapley GPU five-round arm. No new 64-feedback H100 results have been produced in this branch.
