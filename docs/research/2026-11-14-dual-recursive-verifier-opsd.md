# Dual Recursive OPSD — executable-verifier self-evolution for code

**Implementation stage: code/CPU-test ready; real CUDA/bubblewrap inference, optimizer training, official scoring equivalence, and any scientific gain remain UNVERIFIED.**

Branch: `research/dual-recursive-verifier-opsd-20261114`; base: `main` at `0a118fd`.
The old Joint/Mix source, Rich64, SDPO external bridge, and CI workflow are retained **unchanged**. This is a separate prospective study, not an EESD rebuttal result.

## 1. Core hypothesis and precise novelty target

There are two **separate** token-distribution updates:

1. **Verifier generation policy** `g_phi`: write an executable Python `verify(input_text, candidate_output) -> bool` program. Given independently evaluated mistakes of its *own sampled verifier source*, its frozen self-teacher sees those mistakes; the student does not see the new mistakes. Reverse-KL on the **verifier program's sampled token prefixes** trains the generator of verification programs. This is not iterative in-context editing alone.
2. **Solver policy** `pi_theta`: generate runnable stdin/stdout code using the current Solver LoRA. A frozen current Solver self-teacher reads feedback from a **gated** Verifier program, and an otherwise identical student receives no such feedback, with reverse-KL over the solver's on-policy tokens. The two roles use separate adapters of the same pinned Qwen base.

At round r:

`v_ki^r ~ g_phi_r(v|x)`; `diagnostic(v_ki^r) = errors(verify(v_ki^r), trusted_train_cases)`.

`L_V = E_{v~g_phi_r} [ mean_t KL(g_phi(v_t|x,v_<t) || stopgrad(g_phi_r(v_t|x,error(v),v_<t))) + beta KL(g_phi||g_phi_r(v_t|x,v_<t)) ]`.

`y_ij^r ~ pi_theta_r(y|x)`; `F_ij^r = selected_verifier_r(x, execute(y_ij^r))`, but only from candidates accepted by a disjoint **TRAIN-SIDE gate**. Any source failing the gate triggers a transparent fixed-oracle fallback.

`L_S = E_{y~pi_theta_r} [ mean_t KL(pi_theta(y_t|x,y_<t) || stopgrad(pi_theta_r(y_t|x,F_r,y_<t))) + beta KL(pi_theta||pi_theta_r(y_t|x,y_<t)) ]`.

No Shapley, no online 2^T subset search, no claimed correctness of an LLM-synthesized reward merely because its code executes. Negative examples are **verifier mistakes** rather than proof that the task solution is wrong.

## 2. Why five papers matter (verified scopes)

| Work | Training and code | Benchmarks reported | Distinction to test |
|---|---|---|---|
| SPADE (`2608.19197`, repo `spade-rl/spade`) | Shared policy GRPO; designs and runs complete executable Gym-style environments with hint-based regret and environment memory | 8 held-out math/science/code/reasoning, BFCL-v4, ACEBench-Agent | No mistake-conditioned token-level OPSD of an executable grader source |
| STV (`2605.30290`, repo `AR-FORUM/stv`) | Reference-conditioned self-verifier teacher; alpha divergence JS/OPD **plus verdict RL**; Generator uses trained verifier **frozen** for multi-turn RL (not Solver OPSD) | DAPO Hard/Hardest, SciKnowEval, Qwen3-8B | Main inspiration for privileged information asymmetry; must beat its verifier training baseline |
| CoVera (`2026.findings-emnlp.1634`) | One model co-trained to propose code and reason about correctness using execution-derived labels | Code-generation and correctness verification benchmarks; see original tables before quoting per-dataset scores | Verifier is reasoning/verdict output rather than a generated executable function |
| Who Verifies the Verifier? (`2610.11464`) | Inspector composition over mostly deterministic detectors, gated by independently labeled reference anchor and consensus | MBPP+, text-to-SQL and report generation; always-pass failure when anchor removed | Proof of necessity of trustworthy independent gate, not OPD on verifier-source tokens |
| CoVer (`2026.emnlp-main.2463`) | Single-policy GRPO on coder/test author; anchored information-gain reward and test diversity pruning | LiveBench, MBPP, LiveCodeBench, CodeContests, CodeForces | Test-generation/self-play is already known; dual OPSD must bring additional gains |

Additional independent eval recommendation: VerifierBench (`2610.28679`, 222 tasks / 4,993 hidden samples). Its setup does **not** plug into the current RunBugRun function signature; a separate adapter is future work.

**Novelty caveat:** OPD of the verifier itself is not new after STV. Generating executable verifiers is not new after SPADE/Who Verifies/VerifierBench. The hypothesis needing verification is **mistake-conditioned token-level OPSD over generated executable verifier source and simultaneous solver OPSD over its validated feedback**, with controlled recursive improvement on unseen code tasks. Do not claim "nobody has done this" as a theorem.

## 3. Trusted data boundary and verifier API

Reuses an already frozen RunBugRun cohort built by `scripts/prepare_t64_runbugrun_cohort.py`. Source unit for analysis is one distinct `problem_id`. The historical cohort has public task/test pairs, independent training calibration, reserved audit, and official independent test-split evaluation. We do NOT read gold `fixed_code` nor copy held-out evaluation answers into verifier prompts.

For each training task and random seed, its public inputs are **pre-partitioned by input identity** into:

- **Verifier training anchors:** `verifier_train_tests` input/output pairs; derive positive `output=expected` and negative genuine incorrect solver outputs or deterministic perturbations. Only training-anchor misjudgments enter the verifier self-teacher and next-round reflection memory.
- **Verifier selection gate:** `verifier_gate_tests` disjoint inputs, never shown in prompts. Require both positive and negative classes, a bounded false-accept rate, false-reject rate, and full decision coverage. Gate scores control acceptance but do not teach the verifier.
- **Solver-feedback tests:** `solver_feedback_tests` disjoint inputs. If a gated generated verifier is available, the solver self-teacher sees its PASS/FAIL verdict, actual output, and non-oracle execution status (EXECUTED/exception). It **does not see reference expected outputs or correctness status** in this arm.
- **Reserved verifier audit:** disjoint frozen inputs from `train_audit_do_not_train.jsonl`; output only into diagnostic files, never gate/adapt/teacher input. Do not use these audit outcomes to tune the method on the same cohort.
- **Final test:** official disjoint test-split problems with the full frozen official test bank, no verifier evolution signal or manual best-round selection.

`verify(input_text: str, candidate_output: str) -> bool` is an **input/output checker**, not a complete proof of program correctness. The script checks AST safety heuristics and executes generated code ONLY through existing mandatory bubblewrap. It does not `exec()` generated verifier programs in the trainer process. A verifier producing anything other than a single boolean verdict abstains. Silent accept-all/reject-all fails the gate. The local judge currently compares to one canonical expected stdout and can mislabel a different yet semantically valid output; real scientific runs **must audit this semantic issue against official RunBugRun** and include valid alternative outputs. Synthetic wrong outputs are provisional, not guaranteed specification-invalid for all tasks.

No LLM writes or changes the trusted reference labels. The validator itself is not self-validated by self-reported Solver progress.

## 4. Training protocol & what actually changes each round

At R0, use the same immutable LoRA initialization for both roles. Each round:

1. Freeze current Solver; sample **new** Solver on-policy codes; evaluate untouched final problems at this fixed checkpoint. Original sampled token IDs are preserved for reverse KL, while executable code is extracted separately.
2. Freeze current Verifier policy; generate `K_v` executable verification programs **per task**. Execute each on balanced training anchor cases and disjoint gate cases. Derive true verifier false-positive/false-negative mistakes from trusted anchors. Use independently reserved inputs for report-only audit.
3. If arm is `dual` or `verifier_only`, train the **Verifier LoRA** on its **own current-round sampled verifier tokens**: teacher conditions on training-anchor mistakes, student does not. Keep new mistakes as bounded next-round reflection history; freeze the completed adapter until the next round.
4. Gate generated verifier programs **using gate only**; if no program passes, record fallback to the fixed trusted execution feedback. Solver `dual/static/regenerated` self-teacher uses gate-accepted verdicts. The fixed arm uses the same direct reference feedback as the historic RunBugRun program repair pipeline, not a generated verifier.
5. If arm updates Solver, train separate Solver LoRA on its own current-round generated program tokens using standard reverse-KL against the frozen feedback-conditioned self-teacher. No Shapley, token weighting, BO, 2^T query, external reward model, or GRPO.
6. Save each adapter and `round_receipt.json` *before* entering the next round. Re-sample both roles; repeat. R5 is the fixed final endpoint, not the best intermediate checkpoint.

### Ablations

- `solver_fixed`: fixed trusted program execution feedback; Solver OPSD only.
- `solver_static_verifier`: Verifier program proposed/selected once at round 0, reused for all Solver rounds; no Verifier weight updates.
- `solver_regenerated_verifier`: new Verifier programs each round from a **frozen** Verifier model; no Verifier weight updates.
- `verifier_only`: Verifier OPSD runs, Solver remains frozen; measures verifier improvement without solver co-evolution.
- `dual`: new verifier proposals each round with explicit mistake-conditioned Verifier OPSD; current solver learns from the selected verifier feedback via OPSD.

Treat `solver_fixed` as **information-different** (it sees oracle expected outputs), not as an equal-information comparison. Add compute-matched token-budget baselines and offline SFT/verdict RL if full paper experiments support the hypothesis.

### Prospective metrics

Verifier: gate and reserved audit balanced accuracy, FPR, FNR, abstention/coverage, no-verifier fallback rate, valid alternative rejection rate, new failure-family coverage, and source-code validity. Solver: R0–R5 pass@1 on independent full official cases, mean test-pass fraction, pass@k where separately sampled, action/token cost and GPU wall time. Confidence intervals bootstrap distinct `problem_id` clusters; use multiple training seeds. Count candidate generations, full-vocab teacher forwards, sandbox CPU executions and input lengths separately.

Do not claim a mathematical convergence guarantee: both policy distributions change, reward coverage may drift, and the accepted verifier's finite gate can overfit. Repeated failure modes or a zero-valid-verifier rate are valid negative results.

## 5. Exact setup and commands

### Prepare real frozen dataset **on source system**

```bash
python scripts/prepare_t64_runbugrun_cohort.py \
  --db /ABS/RUNBUGRUN.sqlite \
  --output /ABS/dual-verifier-cohort \
  --train-bugs 8000 --meta-problems 64 --eval-problems 199 --seed 1701
```

This preparation requires a real authoritative RunBugRun SQLite dataset and one-time manual scorer qualification. If an existing locked cohort is available, reuse it rather than silently regenerating it. The data prepare script is unchanged on this branch.

### Create identical initial LoRA (one per seed, reused by all arms)

```bash
python scripts/create_learning_credit_init.py \
  --output /ABS/initial-lora-seed1701 --seed 1701
```

### Check every arm **without** running model or untrusted code

```bash
DATA_DIR=/ABS/dual-verifier-cohort \
INIT_ADAPTER=/ABS/initial-lora-seed1701 \
OUTPUT_ROOT=/ABS/dual-opsd-runs \
SEED=1701 DRY_RUN=1 bash scripts/run_dual_verifier_matrix.sh
```

### Small actual smoke (real cached model, CUDA BF16, Linux mandatory bubblewrap)

```bash
python scripts/run_dual_verifier_opsd.py \
  --data-dir /ABS/dual-verifier-cohort \
  --init-adapter /ABS/initial-lora-seed1701 \
  --output /ABS/dual-opsd-runs/dual-smoke-1701 \
  --arm dual --seed 1701 --rounds 1 \
  --train-limit 4 --eval-limit 4 --rollouts 2 --verifier-proposals 2
```

### Pilot matrix, then expand after real smoke qualification

```bash
DATA_DIR=/ABS/dual-verifier-cohort \
INIT_ADAPTER=/ABS/initial-lora-seed1701 \
OUTPUT_ROOT=/ABS/dual-opsd-runs \
SEED=1701 TRAIN_LIMIT=32 EVAL_LIMIT=32 ROUNDS=5 \
SOLVER_ROLLOUTS=4 VERIFIER_PROPOSALS=2 DRY_RUN=0 \
  bash scripts/run_dual_verifier_matrix.sh

python scripts/summarize_dual_verifier_opsd.py \
 --runs /ABS/dual-opsd-runs/dual-seed1701-N32-R5 \
        /ABS/dual-opsd-runs/solver_fixed-seed1701-N32-R5 \
        /ABS/dual-opsd-runs/solver_static_verifier-seed1701-N32-R5 \
        /ABS/dual-opsd-runs/solver_regenerated_verifier-seed1701-N32-R5 \
        /ABS/dual-opsd-runs/verifier_only-seed1701-N32-R5 \
 --output /ABS/dual-opsd-runs/pilot-summary.json
```

Subsequent confirmatory experiments may use `train-limit 128/512`, `eval-limit 199`, seeds `1701/1702/1703`, but only if actual eligible distinct problem counts, scorer agreement, budget and GPU memory permit it. Run each seed with its own immutable initial LoRA; the scorer/evaluation population must remain appropriately disjoint. Do not report nominal sample counts in place of successful, real runs.

`--resume` accepts only *completed round receipts* and refuses partial rounds or changed plan identities. A partially written round must be diagnosed manually, not silently overwritten. Every arm writes JSONL proposals, verifier audit metrics, solver updates, round checkpoints, R0–R5 fresh evaluation, and a final completion receipt.

## 6. Acceptance and stop rules

1. Python compilation; pure CPU tests for gate, collapsed verifier, bwrap API, prompt/label isolation, parser and paired bootstrap.
2. Actual Linux mandatory-bwrap smoke that demonstrates the execution stdout, timeout and official-match semantics. Synthetic negatives must be checked for validity.
3. Actual CUDA BF16 model-loading/memory smoke; verify both role adapters update independently and checkpoint reload preserves outputs.
4. Independent heldout verifier audit must not be used for admission, hyperparameter tuning or the self-teacher; safeguard against over-restrictive verifiers.
5. The system must not silently switch from generated to trusted reference feedback; the source is logged per Solver sample.
6. If no generated verifier passes the gate, stop claiming verifier-evolution training benefit. If verifier audit improves but Solver remains flat, report that as a negative dual-recursive result.
7. The old GitHub CI workflow is left byte-identical, per owner instruction.

**This branch supplies a runnable research implementation under these prerequisites; it does not imply a completed GPU result or a proven improvement over STV, CoVera, CoVer, SPADE, or any other benchmark.**
