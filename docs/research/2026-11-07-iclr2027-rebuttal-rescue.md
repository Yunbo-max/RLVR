# ICLR 2027 Submission 46512 — Reviewer-directed rebuttal rescue (2026-11-07)

**Scientific status:** Rebuttal planning and a four-transition CPU weight generator are implemented. NO new H100 results, no claim that matched correction training has already run.

**ICLR author rules:** https://iclr.cc/Conferences/2027/AuthorGuidelines . Deadline is November 18, 2026 23:59 AoE. Maximum 10 main text pages with additional appendix. Author response may revise title, abstract and text; AC and reviewers may ignore changes that are substantially different from original submission.

## 1. Main policy on proposing a "new method"

Preserve the **submitted Dirichlet-EESD** as the primary algorithm. New execution-feedback variants may be marked "additional reviewer-motivated ablation/extension" but MUST NOT replace the original method or inherit its submitted results. Group Shapley / three-state axiomatic EESD is a distinct later design, not a clean update to this submission.

## 2. Code-versus-submission mismatch: urgent

The actual 2026-09 paper paper/main.tex specifies:
- four transition categories FIX, REGRESS, PRESERVE, UNRESOLVED;
- lexical signed-hash kernel relevance s_i=exp(16*cos(diff,execution_descriptor));
- Dirichlet posterior alpha_post[k]=0.1 + MASS*r[k];
- utility v=(1,-1,0,0);
- weight w=max(0,mean_utility-0.5*SD_utility);
- loss w*correction_CE + 0.03*forward_KL(previous_policy || student_policy);
- zero-weight rows should STILL incur the KL anchor.

Current src/eesd/distillation.py uses **three** states and posterior mean benefit with NO uncertainty penalty; its scorer ignores the old uncertainty argument.
Current scripts/run_eesd_weighted_sft.py drops zero-weight correction rows and sets beta=0 for equal_weight. Therefore it cannot be run unmodified to produce a submission-faithful exact matching of the old method and its requested baselines. This mismatch must be rectified BEFORE reporting any training comparison to GRLz/JToX.

New scripts/prepare_submitted_dirichlet_rebuttal_weights.py reconstructs the submitted **four-state** posterior and correct weight, using already existing numerical evidence helpers. It provides CPU-only weight receipts, including reviewer-specific sensitivity diagnostics. This is NOT a trained model or a downstream result.

## 3. Priority experiment P0: direct correction-learning attribution

**Hold fixed**: original cached corrections, training prompt and target programs, before/after public tests, original lexical relevance scores, source IDs, LoRA adapter initialization, model revision, forward-KL coefficient (0.03), full row schedule including zero-weight rows, learning rate, response-token budget and optimizer steps.

Run on principal diagnostic DeepSeek-Coder-6.7B / CodeARC (+5.4pp original), plus the negative Gemma-3-4B / APPS Replay (-6.8pp original) if feasible. Label both as post-hoc diagnostic cells because original outcomes were previously examined. Report all original 12 cells, including negative ones. Repeat three train seeds for principal model if feasible.

Core matched training arms:
1. equal_weight + exact SAME KL anchor;
2. fixed_mass Dirichlet + KL;
3. adaptive EED + KL (the submitted algorithm);
4. train-mean constant weight + KL (tests whether lower supervised strength alone matters);
5. shuffled EED weights + KL, retaining distribution/zero fraction but dissociating weight and correction;
6. EED minus uncertainty penalty + KL (lambda=0).

Run arms (1), (2), (3) FIRST; they directly answer the two technical 2-point reviewers. Others are valuable mechanism controls if compute remains.

Training-vs-testing source overlap must be audited (including exact bug IDs and independent problem IDs) and disclosed; absence of proof must be marked "unknown", not quietly asserted disjoint. Record total optimizer steps, response tokens, KL, LoRA trainable modules, gradient norms, and per-example correction weights.

**Decision rule:** if adaptive EED does not beat both matched equal and fixed weighting on an unseen source-disjoint test population, do not claim EED mass improves correction learning. Reframe as a probability-calibration mechanism with mixed learning outcomes.

## 4. Reviewer JToX: method limitation and small extension

The submitted utility gives identical weight to:
- one FIX + three UNRESOLVED, leaving only 1/4 after-correction tests passing;
- one FIX + three PRESERVE, leaving all 4/4 after-correction tests passing.

This is a REAL limitation. A clear optional **reviewer-motivated extension** is terminal-public-quality conditioning:
    w_quality = w_EED * mean_i(1[after_i==PASS]).

It is not automatically a new main method and has no reported training benefit yet. Use it as an ablation ONLY after experimental verification; it uses no additional hidden test labels. More complex "new Shapley" changes are not appropriate in this rebuttal.

## 5. Reviewer GRLz: missing protocol and mass controls

The pseudo-count prediction NLL improvements must not be conflated with downstream code correctness. In the paper list exact train/dev/eval source IDs and hashes, how "future outcome" is defined, the public/future observation split, predictor hyperparameters and how the three initializations differ. Distinguish source bootstrap from fitting-seed variance. Report the correction token budget as a number (not merely "response token accounting"). No made-up data overlap assertions.

## 6. Reviewer kZZR: readability is a soundness issue

Use one concrete example with the input program, a corrected program, four public outcomes, masses, weights, and resulting CE coefficient. Translate the method into one sentence: "We attach a conservative weight to each self-generated correction, determined from the concentration of its executed supporting evidence, while anchoring the update to the previous model." Do not add more theory. The method is a correction training-weighting rule, not a demonstrated autonomous RSI agent.

## 7. Temperature sensitivity, cheap CPU diagnostics

Given original scores s_i=exp(kappa*cos_i) with kappa=16, new weight generator evaluates kappa in {8,16,32} by raising original positive relevance scores to powers {0.5,1,2}. Report E(p), zero weight fraction, weight distribution, and alignment to final public pass fraction. Reweighting alone does not prove causal downstream benefit; train an extra arm only if core P0 is resolved.

## 8. Datasets and model architectures

The submission ALREADY contains three backbones and four domains (12 model/domain cells). No reviewer made adding more architectures the primary demand. Adding another model or dataset before matched fixed/equal ablations is low return. If a fresh source-disjoint test is accessible, it is more valuable than one extra architecture. Do not call CodeARC Replay an official leaderboard reproduction.

## 9. Command for numerical weight preparation (not a GPU result)

    export PYTHONPATH="$PWD/src"
    python scripts/prepare_submitted_dirichlet_rebuttal_weights.py \
      --input /path/to/original_corrections.jsonl \
      --output runs/rebuttal-submitted-four-state \
      --alpha 0.1 --lambda-penalty 0.5 --seed 1701

Only after implementing a dedicated faithful CE+KL trainer and validating identical schedules, run H100 correction-learning P0. The weight generator outputs weights.jsonl and manifest.json with file hashes. Never reuse new three-state EESD as if it were the old submitted four-state method.

## 10. Exact communication principles

- The original downstream matrix contains 4 positive, 1 unchanged, 7 negative point estimates. Do not hide these.
- Existing DeepSeek/CodeARC +5.4pp is FULL update vs frozen, not proof of a particular EED weight.
- Reviewers/ACs can ignore significant manuscript changes: put genuine original-method controls into one additional main page; append all details and raw diagnostics.
- No new scientific result can be stated until the model is trained and evaluated.
- Maintain anonymity in uploaded manuscript and OpenReview replies, including code URLs.
