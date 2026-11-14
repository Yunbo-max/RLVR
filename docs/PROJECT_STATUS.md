# Project status — November 14, 2026
## Update on November 14, 2026 — new code

**Joint/Mix software is now included:** `src/eesd/joint_mix.py`, `scripts/run_joint_mix_experiment1.py`, `scripts/run_joint_mix_experiment2.py`, and `scripts/summarize_joint_mix.py`. The new study compares a genuine pairwise joint teacher with a C8-calibrated mixture, and adds an ungrouped, budgeted actual-joint subset search and independent random-search control. Eleven new CPU tests passed; **no GPU run, scorer-semantic certification or scientific result has been completed**. Read [the complete new protocol](research/2026-11-14-joint-mix-experiments.md) (from docs) or `docs/research/2026-11-14-joint-mix-experiments.md` (from repository root). The remainder of this document describes the historical November 12 consolidation and must not be interpreted as the new method's current implementation status.


## Decision

Use **Yunbo-max/RLVR / main** for subsequent work. Consolidate the existing
Rich64 implementation and preserve historical evidence. Stop treating the
original adaptive-Dirichlet ICLR claim as the primary development target.
This decision does not submit a withdrawal or change any OpenReview record.

The current research question is:

> Can a teacher that jointly reads execution feedback produce useful learning
> signals beyond teachers conditioned on each feedback item independently,
> and can this improve unseen-problem performance at a stated compute budget?

This is a hypothesis, not a demonstrated contribution or a completed novelty
audit. Merely showing that two teacher distributions differ is insufficient.

## What is available

The Rich64 implementation was integrated from
`c44efb97332e0432023d32e7e34d484dce2c54f6`. It includes multiple candidate
programs, public feedback, independent calibration and audit roles, actual
joint group-coalition teachers, behavior-dependent token weights, recursive
LoRA updates, simpler controls, isolated execution and an official SDPO bridge.

The current runner still constructs **four groups** and scores their 16
coalitions. It is the baseline implementation from which the next work can
proceed; it is not the proposed ungrouped, on-demand teacher-query algorithm.
The existing random-subset arm consumes a reference schedule. An independent
random-feedback control for the proposed study remains to be specified and
implemented.

## Existing evidence and limits

- The Rich64 pilot used one seed, 64 training bugs, G=4, five rounds and 64
  evaluation problems: Shapley 31/64, random subset 31/64, DirectMax 30/64,
  frozen 29/64. It has not established independent Shapley value.
- The older pilot document contains a stronger causal interpretation of the
  Shapley/random tie. Treat that as historical commentary, not an established
  finding. Identical point estimates cannot establish equivalence or identify
  sparse updates as the cause of a gain.
- The complete Rich64 final cohort has 199 official test problems. Do not
  silently use 199 as the denominator for the 64-problem pilot.
- The official SDPO integration's full scientific rerun is pending. Prior
  simplified internal `sdpo` results cannot be assigned to it.
- The three completed C00-C16 rebuttal cells each used 500 distinct problems
  and training seed 1701. Adaptive-minus-fixed estimates were +0.4, -0.4 and
  -0.8 percentage points, all with paired descriptive 95% intervals crossing
  zero. These results concern original correction weighting, not Rich64.
- The DeepSeek decoding corruption described in the November 10 handoff is a
  pipeline failure, not a valid negative result for either method.

Sources: [Rich64 pilot](research/2026-11-03-rich64-pilot-results.md),
[Rich64 data protocol](research/2026-11-02-rich64-data-protocol.md),
[official SDPO record](research/2026-11-07-official-sdpo-reproduction.md),
[retained rebuttal evidence](../reports/rebuttal2027/README.md).

## Next scientific work — pending

The agreed comparison has two parts: a frozen-model comparison of genuine
joint teachers against calibrated mixtures, followed by a controlled learning
comparison if the first part supports useful behavioral information. A simple
log-additive evidence combination is also a relevant control. Final budgets,
data identities, selection rules, implementation, qualification and prospective
confirmation remain pending.

Retain DirectMax, independent random feedback, full feedback and official SDPO
as comparison questions. Treat Shapley as a mechanism whose added value must
be measured. Keep information-matched and actual-cost comparisons distinct.
No new experiment or compute allocation is launched by this consolidation.

## Maintenance acceptance

The source integration preserves the Rich64 learning implementation except for
seven default model-config path replacements. It removes unused old GRPO and
machine-specific preflight artifacts from the active tree while retaining their
original commits. See the [layout](REPOSITORY_LAYOUT.md) and
[archive index](archive/README.md).

Source inspection is the validation scope. Project tests, training, inference,
dataset preparation and scientific evaluation have not been run for this
maintenance change. The existing SDPO bridge tests require PyArrow in addition
to the old test extra; this setup gap is recorded in the Local runbook without
changing the owner's CI workflow or dependency lock.
