# ICLR 2027 #46512 — Dirichlet EED × Token-Aware Correction Credit
**2026-11-07. Prospective rebuttal extension, NOT reported experimental results.**

## Hypothesis to test

The submitted 4-transition lexical Dirichlet EESD maps a *whole correction*
to one conservative coefficient w(h) and trains CE uniformly over its target
tokens. Even if the evidence reliability hypothesis is useful, copying many
unchanged source tokens can dilute gradients on changed code. This is a
**hypothesis**, NOT a proven explanation of the heterogeneous 12-cell matrix.

The revised objective retains ALL of the submitted statistics (z_i in
FIX/REGRESS/PRESERVE/UNRESOLVED, lexical s_i=exp(16*cos),
r_k=sum p_i*1[z_i=k], E(p)=1/sum p_i^2, alpha=0.1,
lambda=0.5 and beta=0.03):

    L = w_EED(h) / L_h * sum_t g(h,t)*CE(h,t)
        + 0.03*KL_forward(pi_old||pi_new)

Define edit mask m_t=1 if a target token is inserted or replaced when
matching original program a0 to correction a1 using the pinned tokenizer's
SequenceMatcher edit alignment; copied tokens m_t=0, EOS m_t=0.
For a fixed nonnegative edit intensity eta=2,

    g_edit,t = (1 + eta*m_t) / (1 + eta*mean_t m_t)

The average g_t equals **exactly one on every scheduled response prefix**.
This preserves per-correction CE scale; it is not a global learning-rate
change. When no edit or all edit, g=1. The reference teacher and KL anchor
remain UNCHANGED and unweighted by g. This is only a *token allocation*
ablation; it does not prove edited tokens are correct.

The RANDOM control chooses the same number of high-weight positions among
valid output tokens, excluding actual EOS, without looking at outcomes;
it uses the same g formula. All controls see identical corrections, prompts,
before/after PUBLIC execution records, seed, feedback, optimizer, LoRA
configuration, forward KL, schedule and number of target response tokens.

## Core factorial: six arms, no new model architectures yet

| Arm | Whole-correction coefficient | Token multipliers | Main reason |
|-----|------------------------------|-------------------|-------------|
| U | 1 | Uniform | Equal-weight Correction SFT + SAME KL |
| F | Fixed-mass Dirichlet | Uniform | Original required reviewer baseline |
| E | Submitted adaptive Dirichlet EED | Uniform | The submitted method, independently isolate evidence |
| F+T | Fixed mass | Edit-aware | Does granularity matter without adaptive mass? |
| E+T | Adaptive EED | Edit-aware | Proposed reviewer-motivated extension |
| E+R | Adaptive EED | Random matched to edit count | Is identifying changed locations necessary? |

Only change these TWO crossed dimensions for F/F+T/E/E+T. U and E+R
are external controls. The three primary contrasts:

- Evidence: E - F under identical uniform token supervision
  (isolates adaptive effective mass).
- Granularity: E+T - E, and F+T - F
  (tests gradient relocation to changed source tokens).
- Interaction: (E+T - F+T) - (E - F)
  (does the adaptive mass help *more* with edit-aware supervision?).
- Specificity: E+T - E+R
  (better than emphasizing the same number of random tokens?).

Differences are based on paired held-out per-problem all-tests Pass@1,
not training loss or reused public tests. Report absolute Pass@1, mean
test-pass fraction, fixes, regressions, source-cluster bootstrap CI,
training seeds, and actual measured GPU cost. Across training seeds
estimate training variance; do not treat 500 sampled sources from one
training run as repeated model fits.

## Dataset preflight (MUST PASS before running GPU)

1. Recover original a0 and correction a1 from the SEALED correction bank.
   Raw trajectory field 'original' is ideal; for old repaired-program
   prompts the exact JSON 'program' can be parsed and checked.
   If only a task statement is available (some CodeARC or replay synthesis
   modes), an edit mask cannot be identified. Do not invent a0, truncate
   to favorable rows, or silently drop the missing subset.
2. Freeze SAME train/development correction identities and same official
   evaluation task IDs for all arms, with source and problem-level overlap
   audited. If no auditable independent eval exists, explicitly say
   there is no confirmed transfer result.
3. Count edit tokens with tokenizer for each backbone. Report fraction
   with 0 edits, all edits, 0<m<L, median edit fraction, and truncation
   fraction. If all/none masks dominate, edit weighting is experimentally
   **not identifiable**; changing eta cannot rescue it.
4. Keep **zero EED weights in the training schedule**, because the original
   paper says those trajectories still contribute forward KL. Default old
   train paths omit them; this branch supplies --include-zero-weight-kl.
5. Use pinned exact DeepSeek/CodeARC correction bank and evaluation
   ancestry if accessible. If CodeARC cannot supply a0, choose an
   independently source-disjoint true repair cohort and label it NEW;
   keep the original CodeARC result as historical, not comparable.
6. Do not expose hidden evaluation results to source filtering, token
   boost tuning or arm choice.

## Minimal training order and cost

P0 (directly addresses the reviewers): U, F, E at one frozen cell.
If E beats F AND U, the original effective-mass learning claim becomes
plausible. If not, report the null and reduce the claim to NLL calibration.

P1 (one explanation of heterogeneity): F+T, E+T, E+R on SAME cell.
Fix eta=2 before opening held-out evaluation. Diagnose performance by
original final-public pass fraction and edit fraction on train/dev only.

P2: Repeat three seeds 1701,1702,1703 on one principal cell, plus a
negative-control previously observed cell. Choices DeepSeek/CodeARC
(original +5.4pp) and Gemma/APPS Replay (original -6.8pp) are **post-hoc
diagnostics** selected after viewing the old result matrix. An untouched
source-disjoint test or an additional predeclared cohort is needed for
confirmatory generalization.

Only if time remains: eta in {1,2,4}, tuned exclusively on development
sources, and one direct terminal-public-quality control. Stronger
backbones and new datasets are lower priority than fair baselines.

## Reviewer-specific interpretation

GRLz: U vs F vs E with identical response-token budget and KL is
mandatory; the token-aware extension alone does NOT answer this review.

JToX: one FIX plus three PRESERVE and one FIX plus three UNRESOLVED have
the same original Dirichlet utility. Edit-aware token supervision
**does not by itself repair this outcome-semantics problem**. Report the
limitation honestly; as an independent diagnostic, terminal-quality
conditioning (w_EED * final-public-pass-fraction) is already available
in the weight generator. Never assert token edit credit resolves it.

kZZR: one plain sentence first: "The original rule decides how much to
trust a repaired program; our additional study checks whether applying
that trust to every copied code token is unnecessarily coarse."
Do not rebrand the submission as a completely different Shapley or
token-level RL paper.

## Reproducible scripts and gates

    export PYTHONPATH="$PWD/src"

    python scripts/audit_eesd_token_aware_inputs.py \
      --corrections /sealed/corrections.jsonl \
      --output runs/ta-data-audit.json --require-complete

    python scripts/prepare_submitted_dirichlet_rebuttal_weights.py \
      --input /sealed/corrections.jsonl \
      --output runs/ta-weights --alpha 0.1 --lambda-penalty 0.5

    python scripts/materialize_submitted_dirichlet_training_inputs.py \
      --weights runs/ta-weights/weights.jsonl \
      --output runs/ta-inputs \
      --arms equal_weight fixed_mass adaptive_eed

Use scripts/run_eesd_weighted_sft.py with --rule eesd_full
--include-zero-weight-kl --anchor-beta 0.03 on each frozen input,
and --token-weighting uniform/edit/random as appropriate.
For fixed arm use --token-weighting edit to create F+T, and for
adaptive arm use --token-weighting random to create E+R.
Set --token-edit-boost 2.0 on ALL arms, although uniform ignores it.

Check a fresh shared LoRA adapter initialization is identical for every
arm, compare trainer reports and full response-token budget.
The existing paper/data/downstream.csv contains ONLY historical
aggregate results, not the raw correction program pairs or a proven
disjoint evaluation manifest; we cannot produce real new test
accuracy from that CSV alone.

## Decision tree after results

- E>F and E>U, E+T>E and E+T>E+R:
  potentially supports BOTH the original evidence effect and token
  allocation. Clearly report the effect sizes and seed-level uncertainty.
- E≈F but E+T>F+T or both edit models improve:
  granularity may help but original adaptive mass NOT independently supported.
- E+T≈E+R:
  edit-location specificity NOT supported; likely a supervision
  concentration/optimization effect.
- E+T<=E or all equal:
  reject the proposed granularity explanation; keep as negative ablation.
- Very few nontrivial source pairs:
  the experiment cannot test edit-aware token allocation reliably.

**Paper status:** paper/main.tex is a WORKING rebuttal draft with a
method/protocol addition and no fabricated measurements. Do not upload
a draft with unfilled experimental table or claim new GPU gains.
