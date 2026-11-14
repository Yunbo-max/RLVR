
## Update on November 14, 2026 — new code

**Joint/Mix software is now included:** `src/eesd/joint_mix.py`, `scripts/run_joint_mix_experiment1.py`, `scripts/run_joint_mix_experiment2.py`, and `scripts/summarize_joint_mix.py`. The new study compares a genuine pairwise joint teacher with a C8-calibrated mixture, and adds an ungrouped, budgeted actual-joint subset search and independent random-search control. Eleven new CPU tests passed; **no GPU run, scorer-semantic certification or scientific result has been completed**. Read [the complete new protocol](research/2026-11-14-joint-mix-experiments.md) (from docs) or `docs/research/2026-11-14-joint-mix-experiments.md` (from repository root). The remainder of this document describes the historical November 12 consolidation and must not be interpreted as the new method's current implementation status.

# Repository layout and consolidation

## Canonical repository

`Yunbo-max/RLVR` is the active destination, on `main`.
GitHub metadata for `Yunbo-max/EESD` and `Yunbo-max/RLVR` resolves to repository
ID `1399772435`. The separate `yuhanlydia/PBPF` repository has ID `1360907661`.
PBPF retains independent PBPF/A-PBPF work and original EESD evidence.

## Active foundation

| Location | Role |
|---|---|
| `src/eesd/behavioral_shapley.py` | Behavioral calibration and token weights |
| `src/eesd/feedback_group_credit.py` | Existing Rich64 grouping and joint coalition values |
| `src/eesd/learning_credit_selection.py` | Selection and control semantics |
| `src/eesd/strict_execution.py`, `recursive_execution.py` | Program execution |
| `src/eesd/sdpo_official.py` | Official SDPO data/reward/result bridge |
| `scripts/run_t64_grouped_five_round.py` | Existing Rich64 runner |
| `scripts/run_learning_credit_five_round.py` | Shared training implementation |
| `scripts/run_recursive_shapley_sdpo.py` | Required model, LoRA and prompt utilities; earlier runner also retained |
| `scripts/prepare_t64_runbugrun_cohort.py` | Official RunBugRun cohort materialization |
| `scripts/run_sdpo_official.py` | Pinned external baseline launcher |
| `configs/models/qwen25_7b.yaml` | Shared pinned model configuration |
| `configs/experiments/eesd_t64_budget_matched.yaml` | Existing Rich64 comparison protocol, not the proposed Joint/Mix protocol |
| `tests/` | Existing tests carried from the Rich64 source revision; not rerun during cleanup |

Earlier preparation helpers are also shared: `prepare_learning_credit_data.py`
and `prepare_behavioral_runbugrun.py` import `prepare_recursive_runbugrun.py`.
Keep these dependencies intact.

## Removed from the active tree

- Old `rlvr/` GRPO/SFT runners, diagnostic scripts and their saved scratch
  outputs. Their model YAML is retained at `configs/models/qwen25_7b.yaml`.
- Machine-specific `logs/` and generated `preflight/` files.
- `scripts/download_assets.sh`, `scripts/preflight_gpu.sh` and
  `scripts/prepare_preflight_data.py`, which served that older preflight path.

Every removed path and its original blob is listed in
[`research/2026-11-12-repository-consolidation.json`](research/2026-11-12-repository-consolidation.json).
Restore the original tree from the commits in the archive index when replaying
history; do not mix old artifacts with a new run identity.

## Model configuration migration

The YAML bytes, model ID and revision are unchanged. These seven default paths
now refer to `configs/models/qwen25_7b.yaml`:

1. `scripts/create_learning_credit_init.py`
2. `scripts/probe_learning_credit_delta_theta.py`
3. `scripts/probe_t64_group_delta_theta.py`
4. `scripts/run_behavioral_shapley.py`
5. `scripts/run_learning_credit_five_round.py`
6. `scripts/run_recursive_shapley_sdpo.py`
7. `scripts/run_t64_grouped_five_round.py`

The ignore rule explicitly allows this configuration directory; large model
weights remain ignored. External scripts that explicitly passed the former
path should update it, or use the historical checkout for old runs.

## Historical material retained

- `results/recorded/`: original records and locks, unchanged.
- `paper/`: original ICLR manuscript and data, unchanged from the Rich64
  baseline; the later rebuttal draft is available at `b8974183`.
- `reports/rebuttal2027/`: exact retained summaries from `b8974183`, plus an
  index clarifying that the corresponding full implementation is historical.
- Dated `docs/research/` records: original research decisions and results.
- `scripts/run_stage.py` and `configs/experiments/eesd_final.yaml`: historical
  planner compatibility and the unchanged CI command, not the new study.

The full C00-C16 implementation and its shared-trainer changes remain at
`b8974183142a99574cca0b74daf57cca2035eebf`. They are not silently transplanted
into the Rich64 learning rule. All research branch heads are preserved.

## Workflow preservation

The `main` workflow `.github/workflows/ci.yml` is retained byte-for-byte with
blob SHA `f597a77d81eb1d0cc4c0ff59a12460f148407bef`. No historical workflow run
is repaired, rerun or cancelled. Dependency installation and scientific
execution are not part of repository cleanup.
