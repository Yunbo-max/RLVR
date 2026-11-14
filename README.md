# RLVR: feedback-conditioned self-distillation

**Use `Yunbo-max/RLVR`, branch `main`, for continued development.**
The former `Yunbo-max/EESD` address resolves to this same GitHub repository
(repository ID `1399772435`). It is not a second implementation to maintain.
The independent `yuhanlydia/PBPF` repository remains an upstream research and
historical-evidence source.

The current engineering foundation is **behavior-calibrated, token-aware
Rich64 self-distillation**, including its Shapley and simpler controls. The next
research question is whether **genuine joint-feedback teachers** provide useful
learning signals beyond independently conditioned teacher mixtures.
The new Joint/Mix **code has now been added** as separate experiments on top
of the Rich64 foundation. The mechanism comparison, five-round joint/mix
training, and budgeted true-joint feedback search are implemented, with eleven
new CPU tests passing. **No real GPU experiment has yet run.** The old Rich64
runner still uses four feedback groups.

## Start here

| Document | Purpose |
|---|---|
| [Project status](docs/PROJECT_STATUS.md) | Current research question, available evidence and missing work |
| [Repository layout](docs/REPOSITORY_LAYOUT.md) | Active code, shared dependencies and cleanup boundaries |
| [Local handoff](LOCAL_AGENT_RUNBOOK.md) | Safe checkout, existing setup limits and source acceptance |
| [Archive index](docs/archive/README.md) | Exact historical code, paper and result revisions |
| [Agent instructions](AGENTS.md) | Scope and evidence rules for future agents |

## Existing implementation

| Component | Entry point |
|---|---|
| Freeze the RunBugRun public/calibration/audit/test cohorts | `scripts/prepare_t64_runbugrun_cohort.py` |
| Create the shared initial adapter | `scripts/create_learning_credit_init.py` |
| Five-round Rich64 training | `scripts/run_t64_grouped_five_round.py` |
| **Joint/Mix experiment 1: frozen mechanism** | `scripts/run_joint_mix_experiment1.py` |
| **Joint/Mix experiment 2: five-round training** | `scripts/run_joint_mix_experiment2.py` |
| **New adaptive true-joint search / calibrated mixtures** | `src/eesd/joint_mix.py` |
| New paired evaluation aggregator | `scripts/summarize_joint_mix.py` |
| Protocol and launch commands | `docs/research/2026-11-14-joint-mix-experiments.md` |
| Existing comparison matrix | `scripts/run_t64_group_matrix.sh` |
| Paired Rich64 summaries | `scripts/summarize_t64_group_five_round.py` |
| Behavioral calibration and token weights | `src/eesd/behavioral_shapley.py` |
| Grouped feedback and coalition values | `src/eesd/feedback_group_credit.py` |
| Shapley / DirectMax / control selection | `src/eesd/learning_credit_selection.py` |
| Isolated program execution | `src/eesd/strict_execution.py`, `src/eesd/recursive_execution.py` |
| Official SDPO bridge | `scripts/run_sdpo_official.py`, `src/eesd/sdpo_official.py` |
| Pinned Qwen model configuration | `configs/models/qwen25_7b.yaml` |

`run_t64_grouped_five_round.py` reuses `run_learning_credit_five_round.py`, which
reuses model-loading, LoRA and prompt utilities in
`run_recursive_shapley_sdpo.py`. Those earlier files are required dependencies.
The old internal `sdpo` option in the earlier runner is not the authors' SDPO.
The current Rich64 matrix uses the pinned official implementation separately.

The existing comparison config is
[`configs/experiments/eesd_t64_budget_matched.yaml`](configs/experiments/eesd_t64_budget_matched.yaml).
Its status remains `official_sdpo_integrated_scientific_rerun_pending`.
Its schedule-matched random control is not yet the proposed independent-random
control for the new study.

## Evidence status

| Lineage | What the retained evidence supports |
|---|---|
| Rich64 pilot, seed 1701, 64 evaluation problems | Shapley 31/64, random subset 31/64, DirectMax 30/64, frozen 29/64; no established independent Shapley advantage |
| Original correction-weight EED, three completed rebuttal cells | Adaptive mass has not established an advantage over fixed mass; this is a different method and protocol |
| Official SDPO | Source integration and earlier engineering qualification are recorded; the full matched scientific rerun remains pending |
| Joint teachers versus mixtures | Implementation and CPU-unit checks added; GPU validation and scientific results pending |

Read the [Rich64 pilot record](docs/research/2026-11-03-rich64-pilot-results.md)
alongside the [current interpretation](docs/PROJECT_STATUS.md).
Historical point-score equality does not establish equivalence or identify the
cause of the gain. The pilot used 64 evaluation problems; the later full frozen
test cohort contains 199. Neither is the 500-problem-per-cell rebuttal protocol.

The original ICLR manuscript in `paper/`, the records in `results/recorded/`,
and the summaries in `reports/rebuttal2027/` are historical evidence. They are
not a draft or results for the new Joint-versus-Mix paper.

## Setup and execution

The November 12 consolidation changed repository organization and model-config
paths without modifying the Rich64 learning rule. The November 14 addition
implements the **separate Joint/Mix research protocol** with independent entry
points and unit tests; it does not report or certify any new GPU result. Follow [LOCAL_AGENT_RUNBOOK.md](LOCAL_AGENT_RUNBOOK.md)
for acceptance and the
[official SDPO setup](docs/research/2026-11-07-official-sdpo-reproduction.md)
for that separate environment. The existing GitHub CI workflow is preserved.

## Provenance

The Rich64 foundation comes from
[`c44efb97`](https://github.com/Yunbo-max/RLVR/tree/c44efb97332e0432023d32e7e34d484dce2c54f6).
The complete C00-C16 rebuttal implementation and its later fixes remain at
[`b8974183`](https://github.com/Yunbo-max/RLVR/tree/b8974183142a99574cca0b74daf57cca2035eebf).
The original namespace extraction came from PBPF
[`d00969ce`](https://github.com/yuhanlydia/PBPF/tree/d00969ce49bde6108aa4f53dccb09eeb150db601).
See [PROVENANCE.md](PROVENANCE.md) and the [archive index](docs/archive/README.md).
