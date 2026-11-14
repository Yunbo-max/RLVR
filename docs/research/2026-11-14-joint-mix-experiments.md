# Joint-feedback distillation: implementation and falsification protocol

**Status:** Implemented in two separate Python entry points; mathematical and CPU tests pass. **No real GPU or program-execution study has been run as part of this update.** Historical Shapley / Rich64 findings are not results for this method.

## Research distinction

The current Rich64 baseline clusters 64 tests into four summary groups and queries their 16 joint feedback coalitions. It is a group-level Shapley procedure. The **new experiment does not require a fixed number of groups**, does **not** pre-score 64 singleton teachers, and does not assume that independent teachers mix to the true joint teacher.

For candidate `i`, original response prefix `y_{i,<t}`, and raw execution-feedback set `S`, the true joint-prompt teacher is

\[
q^J_{S,it}(v)=\pi_r(v\mid x,F_{i,S},y_{i,<t}).
\]

For two hash-selected feedback items `A,B` from the same 64-item pool, the mixture control uses

\[
q^M_{it}(v)=\alpha_0q_{0,it}(v)+\alpha_Aq_{A,it}(v)+\alpha_Bq_{B,it}(v),\quad \alpha\in\Delta^2.
\]

The mixture **first combines normalized token probabilities**, not logits or KL losses. `alpha` is fitted on training-side C8 with a predeclared simplex grid. An additional diagnostics-only teacher is the log-additive fusion

\[
q^{\mathrm{add}}_{AB,it}\propto q_{A,it}q_{B,it}/q_{0,it},
\]

which is normalized over the entire vocabulary. A difference `qAB != qMix` alone is not evidence of useful learning; a difference from log-additive fusion alone is not proof of reasoning either.

Behavioral value is computed from candidate-level calibration outcomes:

\[
V_h(q)=\frac1G\sum_i (c_{ih}-\bar c_h)\frac1{L_i}\sum_t
\big[\log q_{it}(y_{it})-\log q_{0,it}(y_{it})\big].
\]

The policy criterion is `mean_h V_h - 0.5 SE_h(V_h)` (with an optional per-item cost in the budgeted search). Token weights remain the existing non-renormalized, confidence-preserving directional rule:

\[
w_{it}=|c_i-\bar c|\,\operatorname{clip}([
\operatorname{sign}(c_i-\bar c)\{\log q_{it}(y_{it})-\log q_{0,it}(y_{it})\}]_+/.15,0,1).
\]

The student uses the existing weighted reverse-KL objective, with `beta=0.02` on the no-feedback anchor and LoRA training. The actual parameter gradient need not have the sign suggested by the token-selection rule. No Shapley attribution is calculated by the new selector; retained Shapley is a **baseline**, not an alternative name for this proposal.

## Experiment 1 — frozen Joint vs calibrated Mix

Entry point: `scripts/run_joint_mix_experiment1.py`.

For each preselected distinct training problem, the same frozen checkpoint samples `G` runnable candidate codes. Their public F64 feedback is executed using mandatory bubblewrap. A **predeclared deterministic hash of source ID/seed/round** selects feedback `A,B` without consulting C8 or A8. The only teacher-subset queries for the primary contrast are `{}`, `{A}`, `{B}`, `{A,B}`; hence **no fixed 65G singleton pass**.

1. Compute sampled-token log-probabilities for those four actual prompts.
2. Fit the three-way nonnegative Mix weights on C8 (simplex grid, denominator 20, 231 options).
3. Compare true Joint and fitted Mix using **reserved A8**, which is never used for selection or mixture fitting. Preserve all cases, including zero-calibration-contrast cases.
4. Optionally, for `--probe-candidates 2 --probe-tokens 16`, score full-vocabulary distributions to report `KL(qAB || qMix)` and `KL(qAB || qAdd)`. These diagnostics are strictly a bounded token subset and cannot substitute for A8 behavioral value or downstream accuracy.

The primary observable is the **problem-level mean** `V_A8(qAB) - V_A8(qMix)`; it is not Pass@1. The pilot 128-problem size is a suggestion, and previous exposure of the same A8 cohort should be disclosed. Failure of A8 transfer is a valid negative result.

### Frozen pilot launch

```bash
python scripts/run_joint_mix_experiment1.py \
  --data-dir /ABS/PINNED_RICH64_COHORT \
  --init-adapter /ABS/SHARED_INITIAL_ADAPTER \
  --output /ABS/RUNS/exp1-joint-mix-seed1701 \
  --train-limit 128 --eval-limit 64 --rollouts 8 \
  --probe-candidates 2 --probe-tokens 16 --seed 1701 --dry-run

# Remove --dry-run after inspecting the sealed plan and confirming GPU / bwrap.
```

For training experiment outputs, use **unique output directories per arm/seed**, never overwrite previous runs.

## Experiment 2 — real five-round LoRA training

Entry point: `scripts/run_joint_mix_experiment2.py`.

- `joint_pair`: true joint teacher on the same hash-selected pair as `mix_pair`.
- `mix_pair`: C8-optimized arithmetic distribution mixture of q0/qA/qB on exactly that pair.
- `joint_search`: **new method**. An on-demand feedback subset proposer uses a finite query budget (`--query-budget`), explicitly considers true **pairs** even if every singleton is unhelpful, and alternates exploration with add/drop/swap local proposals. Each actual subset gets a **real joint prompt** evaluation on the frozen teacher. The best positive C8 criterion is used for distillation; otherwise abstain. No exhaustive coverage guarantee is claimed.
- `random_search`: independent feedback subset proposals (no reference Shapley update schedule), equal distinct-subset query budget, same C8 criterion and distillation rule.
- `no_update`: frozen reference on the same evaluation IDs.

For the matched `joint_pair`/`mix_pair` comparison the feedback pair is fixed across method arms as a function of data identity/seed/round, while each policy samples its own candidates after round 0. For search arms, query budgets, not teacher FLOPs or text-token exposure, are matched. **Report measured compute and feedback length alongside performance.**

Each actual training candidate executes the same F64 and C8; **A8 is not loaded by the experiment-2 trainer**. At each round the teacher is the frozen prior adapter. Each accepted training source performs at most one LoRA AdamW step on its weighted token-distribution loss. R0 through R5 run the unchanged fixed private evaluator; the final R5 is the prospectively chosen endpoint, not the best observed checkpoint.

### Real training launch (first check 64, then 512 distinct problems)

```bash
# Per seed, create the shared adapter once using existing create_learning_credit_init.py.
# Use the SAME --init-adapter among arms for this seed.
for arm in joint_pair mix_pair joint_search random_search no_update; do
  python scripts/run_joint_mix_experiment2.py \
    --data-dir /ABS/PINNED_RICH64_COHORT \
    --init-adapter /ABS/SEED1701_INIT_ADAPTER \
    --output "/ABS/RUNS/exp2-${arm}-1701" \
    --arm "$arm" --train-limit 64 --eval-limit 64 \
    --rounds 5 --rollouts 8 --query-budget 16 \
    --max-feedback-items 8 --seed 1701 --dry-run
done

# Once data, scorer, model and resource prerequisites are verified, remove
# --dry-run, use disjoint output directories, and repeat with 512 train
# problems, 199 test problems, seeds 1701/1702/1703.
```

After completed runs:

```bash
python scripts/summarize_joint_mix.py \
  --runs /ABS/RUNS/exp2-joint_pair-1701 /ABS/RUNS/exp2-mix_pair-1701 \
         /ABS/RUNS/exp2-joint_search-1701 /ABS/RUNS/exp2-random_search-1701 \
         /ABS/RUNS/exp2-no_update-1701 \
  --output /ABS/RUNS/paired-summary-seed1701.json
```

The collector verifies pinned dataset/model identifiers, paired evaluation problem IDs and within-seed identical initial adapter hashes. It reports each seed and a **problem-clustered bootstrap interval** conditional on the observed training seeds; three seeds do not create three times as many independent problems.

## Scorer and execution qualification: hard prerequisite for scientific claims

The existing `strict_execution` module requires bubblewrap without a soft fallback. However, the Rich64 execution comparator is not automatically identical to official RunBugRun (e.g. floating-point tolerance and certain input-newline conventions). **The newly wired runner uses this existing scorer unchanged** so that all new arms share one implementation. Before reporting official benchmark numbers, compare saved raw outputs against the pinned official matcher and fix any discrepancy consistently across *all* methods, including F64/C8 and final evaluation; label earlier strict-scorer results as local exploratory results if necessary. The new code does not claim official semantic equivalence yet.

No real H100 experiment was executed in this coding pass. Configuration syntax, pure mathematics, deterministic query budgeting and paired summary checks can be CPU tested; model-loading, bubblewrap behavior, GPU memory and end-to-end execution must be qualified on the target machine.

## Acceptance tests and expected reports

Run `pytest -q tests/test_joint_mix.py tests/test_joint_mix_summary.py` and `python -m compileall -q src scripts` locally. Inspect `plan.json`, experiment-1 `rows.jsonl`/`summary.json`/`completion.json`, or experiment-2 `round_*/training.jsonl`, `evaluation.json`, `summary.json`, `adapter/`, and `completion.json`.

Do **not** claim a win merely because Joint/Mix KL is positive, or because search found a better C8 criterion. Required scientific evidence is independent behavior on A8 **and** improved new-problem outcomes after actual training. Current Shapley Rich64, C00–C16 and official SDPO continue to exist as separate baseline lineages. The historical CI workflow is not changed.
