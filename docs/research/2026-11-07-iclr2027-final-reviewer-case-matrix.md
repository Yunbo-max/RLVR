# ICLR 2027 #46512 — Final Reviewer-by-Reviewer Rebuttal Case Matrix
**Frozen plan date: 2026-11-07. Results below are NOT measured outcomes.**
Primary manuscript remains the submitted **four-transition lexical Dirichlet EED**
with \(\alpha=0.1,\lambda=0.5,\kappa=16,\beta=0.03\) and common
forward KL. Token-aware correction credit is explicitly a NEW
reviewer-motivated extension, NOT the original method.

## Why the case matrix is needed
The submitted 12-cell one-round matrix contains 4 gains / 1 tie / 7 losses
with one fitting run per cell. DeepSeek/CodeARC 15.0% -> 20.4% demonstrates
a *complete update* vs frozen, not EED vs equal-weight or fixed-mass.
The old 3-seed matched-predictor NLL contrast changes sign. Do not claim
component efficacy before independently verifying matched training baselines.

Each case is separately launched, logged and hashed. Registry of case IDs:
configs/rebuttal/eesd_case_registry.json. Independent Case runner:
scripts/run_eesd_rebuttal_case.py. Evaluator bridge:
scripts/run_eesd_rebuttal_case_evaluation.py (original PBPF code).
No new real H100 results or reconstructed raw 3,000-label outcomes have been
created by these scripts in this repository. Archive all negative runs.

## Core priority case table

| ID | Training or audit | Isolates | Priority |
|---|---|---|---|
| C00_FROZEN | Original entering model (LoRA initialized, no update) | Does any training help? | P0 |
| C01_EQUAL | Correction CE weight 1 + the SAME KL | Is ordinary correction SFT sufficient? | P0 |
| C02_FIXED | Four-state Dirichlet with mass n + KL | Does adaptive mass E(p) matter? | P0 |
| C03_EED | Exact submitted EED + KL, uniform response-token weight | Submitted original method | P0 |
| D01_SOURCE_AUDIT | Raw source/problem-ID overlap, editable-pair coverage | Leakage/split provenance | P0 |
| D03_PREDICTION_PROTOCOL | Raw future labels, public/future test IDs, NLL/Brier/argmax over seeds | Predictor/protocol soundness | P0 *if raw labels exist* |
| D04_DESCRIPTOR_AND_REPAIR_AUDIT | Original descriptor/kernel definition, partial repairs, JToX counterexample | Relevance features and outcome semantics | P0 |
| C04_CONSTANT | Mean-train-weight constant + KL | Is EED only smaller CE strength? | P1 |
| C05_SHUFFLED | Permuted per-correction EED weights + KL | Is EED aligned to useful corrections? | P1 |
| C06_NO_UNCERTAINTY | Remove posterior uncertainty penalty | Is lambda*sigma necessary? | P1 |
| C07_FINAL_FILTER | Only fully passing corrections | Is the generated correction reliable? | P1 |
| C08_CONFIDENCE | Final public pass fraction weighted CE | Simple scalar public-correctness baseline | P1 |
| C09_TERMINAL_EED | Original EED * final public pass fraction | JToX final-quality repair of semantics | P1 |
| C10_FIXED_EDIT | Fixed Dirichlet plus edit-token gain | Token effect without adaptive mass | P1 |
| C11_EED_EDIT | Original EED plus edit-token gain | Proposed *extension*: correction vs token granularity | P1 |
| C12_EED_RANDOM_TOKEN | EED plus matched same-count random token gain | Does edit specificity matter? | P1 |
| D02_RELEVANCE_DIAGNOSTIC | CPU only kappa 8/16/32, E(p), weight zero/mean, public pass fraction | Relevance sensitivity | P0 |
| C15_UNIFORM_RELEVANCE | Uniform public-test relevance then original 4-state EED | Is lexical kernel useful at all? | P1 |
| C16_PERMUTED_RELEVANCE | Same relevance multiset permuted across public tests | Is relevance-to-test association useful? | P1 |
| C13_KAPPA8 / C14_KAPPA32 | Recompute original-kernel relevance with kappa 8 or 32 | Downstream temperature robustness | P2 |

**Do NOT run all 21 IDs immediately.** P0 is C00/C01/C02/C03 + D01/D02/D04,
plus D03 if the original predictor traces can be recovered. P1 is C04/C05,
C09, and C10/C11/C12, plus C15/C16 if the lexical kernel is still an
independent claim after P0. C06/C07/C08 are cheap important controls but
can be prioritized after the principal matrix. P2 temperature GPU runs only
if D02 shows meaningful change or reviewers request downstream sensitivity.

## Reviewer kZZR — Rating 1 / Presentation 1 / Soundness 1

| Reviewer concern | Experimental or writing response | What it can actually answer |
|---|---|---|
| "Impossible to read"; overly abstract Dirichlet prose | Rewrite abstract/intro in plain correction-level terms; retain existing worked 4-test example; D04 audit | Readability is a **writing fix**, not a new GPU claim. Explicitly label p as distribution over TESTS and w as whole-CORRECTION CE coefficient. |
| "Just a heuristic weight for coding" | C01/C02/C03; C04/C05; C15/C16 | Tests whether the supposed mechanism beats trivial weighting, fixed mass, mean shrinkage, shuffled weights, and no semantic relevance. Merely deriving a Dirichlet posterior is not enough. |
| Assumption "concentrated relevance should get less weight" unjustified | D02 (E(p) bins/zero fraction), D03 matched prediction on genuine future labels, C02 vs C03, C05/C16 | If low E(p) correlates with worse future outcomes or harmful updates, supports the assumption. Otherwise explain it as conservative inductive bias, not truth. |
| "Not self-improvement" | C00 vs C01/C02/C03 on **one** fresh independent generation following real LoRA update; revise positioning | This demonstrates correction-based policy adaptation, NOT recursive autonomous RSI. To claim recursive self-improvement would require a separate multi-round study (not feasible as a rebuttal replacement). |

## Reviewer GRLz — Rating 2 / Soundness 1

| Explicit concern/question | Cases | Required outcome |
|---|---|---|
| No equal-weight correction training; no fixed-mass training (both with matched KL) | **C01/C02/C03 + C00** | Identical correction records, entering adapter, actual response tokens, optimizer schedule, beta=0.03 and evaluator. The EED-specific effect is C03-C02, not C03-C00. |
| The original full matrix has more negative than positive cells incl. Gemma | C00-C03 on DeepSeek/CodeARC and a negative Gemma cell; optional C09/C11 | Report ALL 12 historical cells, every new cell run, all regressions, and fit-seed variance. Do not selectively report a rescue cell. |
| Table 10 mechanism changes sign across seeds | **D03** | Read the genuine raw predictor labels+probabilities for seeds 1701/1702/1703, report per-seed NLL, Brier, argmax, CI clustered by problem ID. If raw traces missing, disclose the historical three signed NLL differences and DO NOT claim stable benefit. |
| Unknown training/selection/evaluation source overlap | **D01 + frozen evaluation Case bridge** | Explicit source and problem IDs, train/dev/test separated at problem level; if identities are absent, report "unverified" rather than asserting independent transfer. |
| "Future outcome" label space/construction ambiguous, "matched effective-arm" not defined | **D03** + protocol text | Record future test IDs NOT visible, class order/definitions, probability vectors, same predictor lock within a seed. Historical raw source documents must establish HOW these labels were obtained, not just model predictions. |
| Missing numeric training budget | All Cxx training logs and immutable case.json | Exact response tokens and optimizer steps per arm, shared initial adapter hash, trainable LoRA layers and KL. Historical budget cannot be recovered from the paper CSV. |
| How chosen kappa=16 and temperature sensitivity? | **D02**, C13/C14 if justified, C15/C16 | 8/16/32 mass distributions, zero-weight fraction, matched train/dev alignment and potential downstream changes, no hidden selection. |
| Does kernel concentration actually affect training? | C02/C03/C04/C05/C16 | Need actual heldout transfer, not in-sample weight diagrams. |

## Reviewer JToX — Rating 2 / Soundness 2

| Explicit concern/question | Cases | Required result/clarification |
|---|---|---|
| SFT/fixed-mass/uncertainty penalty baseline absent | **C01/C02/C03/C06**, C04/C05 | Same CE response targets, beta, tokens, init; isolate each component. |
| RQ1 most of NLL decrease comes from more history; same for fixed | **D03**, existing paper/data/history.csv | Separate n=1 to 8 history effect from fixed-vs-EED within a given n; do not attribute entire 55–59% drop to effective mass. |
| Relevance-weighted mass not shown to predict correction reliability | **D02, D03, C05/C15/C16**, C09 | Show raw alignment/sensitivity and heldout learning. If signal does not help beyond uniform/shuffled relevance, reduce novelty claim. |
| Unclear a1 generation and whether unsuccessful corrections are included | **D04**, original PBPF code receipt | Original author code generates 1 new correction from the model using PUBLIC feedback, extracts Python, re-executes it, and stores before/after outputs. Failed/partially failed revisions are not excluded by EED automatically; quantify how often they have positive weight. |
| 1 FIX+3 UNRESOLVED vs 1 FIX+3 PRESERVE both get same weight | **D04, C07/C08/C09** | Verify the actual numerical counterexample. Token-aware C11 does **not** fix this semantic defect; C09 tests a transparent final-correctness extension. |
| Details of 256-dimensional hashing and execution descriptor: before or after? | **D04** (optional exact recomputation from pinned old PBPF source) | Original generator uses unified_diff(original,a1) and signed-hash FrozenTextEncoder(256). Descriptor includes **before-correction** public input, expected, actual, stderr and outcome; after outcomes are obtained by re-executing a1 and supply four transition labels. Document clipping / missing source descriptors honestly. |
| Training/evaluation overlap and "future outcome" separation | **D01/D03** | Require immutable source bank and true future-only labels, no hidden outcomes used to weight CE. |
| Did worse backbones regress, and can granularity explain it? | **C10/C11/C12 + C00-C03** on previously negative cell | Identical comparison on a true original→correction pair. If C11 improves, test whether improvement is from EED (C11-C10) or token emphasis (C11-C12). |

## Pinned input and output protocol (per CASE, never batch overwrite)

1. Find the original **raw correction JSONL** and exact public/evaluator data roots
from yuhanlydia/PBPF research/eesd-seed1701-resume-20260922.
Original frozen source commit:
0200283849274b49e303638b91f142acd8bba984.
The repository's paper/data/downstream.csv is NOT sufficient to regenerate
training corrections or per-source predictions. User must supply actual raw
data and base adapter/initialization. Generated tests are not created to
fill missing records.
2. Freeze training/development identities, and keep every row including
w_EED=0 (KL remains). Use PINNED model YAML in configs/rebuttal.
3. For each CASE and seed choose a NEW immutable output directory.
Preparing a case does not launch a GPU. Running a prepared case
updates its status from "prepared" to "trained_awaiting_independent_evaluation".
A failed case writes its own log/receipt and cannot silently overwrite.
4. Fresh evaluation is a separate, single-CASE job invoking the original
PBPF one-greedy-candidate RunBugRun/CodeARC generator + evaluator inside the
original sandbox. Case evaluation never returns an in-sample training metric.
Gemma/APPS Replay uses a different original replay evaluator and is NOT
claimed supported by this direct bridge.
5. Six core scientific arm reports are collected only after REAL heldout
evaluation, with a user-supplied, source-originating independent problem-ID
map. Historical PBPF task IDs MUST NOT be silently promoted to problem IDs
for cluster confidence intervals.

### List the cases

    export PYTHONPATH="$PWD/src"
    python scripts/run_eesd_rebuttal_case.py --list

### Prepare and run ONE case (C03 Original EED example)

    python scripts/run_eesd_rebuttal_case.py \
      --case C03_EED --study deepseek_codearc --seed 1701 \
      --corrections /sealed/codearc-deepseek-corrections.jsonl \
      --model-config configs/rebuttal/deepseek_coder_6p7b.yaml \
      --init-adapter /sealed/one-shared-initial-lora \
      --budget 524288 \
      --eval-ids /sealed/evaluation-source-problem-ids.jsonl \
      --require-problem-disjoint \
      --out-root runs/rebuttal2027 --action prepare

    python scripts/run_eesd_rebuttal_case.py \
      --case C03_EED --study deepseek_codearc --seed 1701 \
      --out-root runs/rebuttal2027 --action run

To run another case, only change --case and keep exactly the same data,
model, seed, initial adapter and budget. Run C00_FROZEN separately for
the no-update comparison. For a diagnostic case use e.g.
--case D04_DESCRIPTOR_AND_REPAIR_AUDIT --action both with --corrections,
optionally --pbpf-root /checked-out/PBPF for kernel recomputation.
D03_PREDICTION_PROTOCOL requires --prediction-rows pointing to RAW 3,000
observation-level records; missing records produce a "blocked" receipt,
NOT an invented result.

**524288 is the existing prospective configuration budget, not proof
of the original 2026-09 paper's consumed token count.** If missing,
report this experiment as a NEW matched-budget protocol, not an exact
replay of historical optimizer steps.

### Real PBPF original evaluator for ONE completed case

    python scripts/run_eesd_rebuttal_case_evaluation.py \
      --case-dir runs/rebuttal2027/deepseek_codearc/C03_EED/seed1701 \
      --pbpf-root /checked-out/PBPF \
      --public-root /sealed/codearc-public \
      --evaluator-root /sealed/codearc-evaluator \
      --domain codearc --action prepare

    python scripts/run_eesd_rebuttal_case_evaluation.py \
      --case-dir runs/rebuttal2027/deepseek_codearc/C03_EED/seed1701 \
      --action run

The PBPF checkout must be CLEAN and at the pinned original source SHA,
with local HF weights and a working strict execution sandbox.
Evaluate C01,C02,C03,C10,C11,C12 and C00 in distinct directories.
No result may be selected by inspecting hidden test success.

### Collect and summarize 6 independently evaluated cases

    python scripts/collect_eesd_rebuttal_case_evals.py \
      --cases-root runs/rebuttal2027 --study deepseek_codearc \
      --seed 1701 \
      --problem-map /sealed/evaluation-source-problem-ids.jsonl \
      --output runs/rebuttal2027/paired-deepseek-s1701

    python scripts/summarize_submitted_eed_token_factorial.py \
      --eval-dir runs/rebuttal2027/paired-deepseek-s1701 \
      --output runs/rebuttal2027/paired-deepseek-s1701/paired_summary.json

The second script reports source/cluster Pass@1, paired bootstrap CI,
fixes/regressions, adaptive mass effect, token effect, interaction,
and edit-vs-random specificity. Repeat seeds 1702/1703 only with
independent frozen initialization/candidate receipts; do not call source
bootstrap a training-seed interval.

## Final author-response decision rules

- If C03 > C02 and C01 on independent matched training/evaluation:
  EED's correction-level adaptive mass has evidence. Explain the effect
  and report negative cells.
- If C11 > C03 but C11 = C10, token granularity helps; the original
  adaptive mass still may not be independently useful.
- If C11 > C12, edited positions have incremental merit over equal-size
  random token emphasis; if not, frame it as sparsity/concentration.
- If C09 > C03, terminal correctness is important but the original
  PRESERVE/UNRESOLVED ambiguity remains a real submitted-method limitation.
- If D03 sign flips across seeds or cannot be reconstructed, DO NOT claim
  robust positive predictor-mechanism transfer.
- If C15/C16 approximate C03, do not call the lexical encoder a
  validated reliability estimator.
- Do NOT turn Submission 46512 into a Group Shapley/GRPO paper in rebuttal.

## Remaining real prerequisites (never infer from the summary tables)

- Actual correction train/dev JSONL, with unmodified prompt and original a0.
- Independent official source/problem mapping and private evaluator roots.
- Model weights, shared LoRA initialization, GPU training, sandbox.
- Actual raw predictor history rows for D03 and old fitting metadata.
- Exact old numerical optimizer/token budget, if claiming replication.
- Original Replay evaluator for Gemma/APPS cell if using that cell.
- 3 fit seeds and completed six-arm heldout evaluations.

These are **input requirements**, not completed scientific measurements.
