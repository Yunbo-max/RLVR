# RLVR project instructions

## Current destination and scope

- Continue this project in `Yunbo-max/RLVR`, literal branch `main`.
- `Yunbo-max/EESD` is the former address of the same repository, not a second
  project. `yuhanlydia/PBPF` is a separate upstream research/evidence repository.
- Read `README.md`, `docs/PROJECT_STATUS.md`, `docs/REPOSITORY_LAYOUT.md` and
  `LOCAL_AGENT_RUNBOOK.md` before choosing a runner or changing code.
- The November 12, 2026 cleanup selects Rich64 as the engineering foundation.
  Joint-versus-Mix remains proposed. Do not report it as implemented or tested.

## Preserve the owner's constraints

- Do not edit `.github/workflows/ci.yml` or repair, rerun or cancel the historical
  failed CI workflow as part of this cleanup. The owner explicitly asked to
  leave that flow alone.
- Do not delete, force-update or rename historical research branches, rewrite
  history, alter repository visibility or delete PBPF as part of consolidation.
- Use a current expected-head update for `main`; preserve concurrent work and
  active execution checkouts. One writer publishes an integration.
- Keep `results/recorded/`, original paper data, current retained result reports,
  source commits and original dataset/scorer identities traceable.

## Method and evidence identity

- Original correction-level Dirichlet EED, edit-token EED controls, behavioral
  Shapley token-aware Rich64, and proposed Joint-versus-Mix are distinct lineages.
- A four-group Rich64 coalition is not exact individual-test Shapley over 64
  tests. The future method is not required to retain four groups.
- The earlier internal `sdpo` arm is not official SDPO. The official bridge is
  pinned in `configs/experiments/eesd_t64_budget_matched.yaml`.
- Equal pilot point estimates do not prove equivalence or a causal mechanism.
  Preserve negative and null results; do not upgrade them to positive claims.
- Keep training/calibration/audit/final-test roles and problem-ID separation.

## Source maintenance

- Rich64 depends on the earlier learning-credit and recursive runners. Inspect
  imports before deleting apparently old files.
- The shared Qwen YAML is now `configs/models/qwen25_7b.yaml`. Its content is
  unchanged from the former `rlvr/models/qwen25_7b.yaml`.
- The retained `run_stage.py` and `eesd_final.yaml` serve the old planner and
  the unchanged CI command; they are not the new study's launch contract.
- Source review, software tests, real training and scientific validation are
  separate statuses. This repository cleanup performs source review only.
