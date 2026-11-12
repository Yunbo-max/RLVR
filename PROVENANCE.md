# Provenance and scientific status

## Current consolidation — November 12, 2026

- Canonical development destination: `Yunbo-max/RLVR`, branch `main`.
- `Yunbo-max/EESD` resolves to the same repository ID `1399772435`.
- Existing Rich64 implementation source: `c44efb97332e0432023d32e7e34d484dce2c54f6`.
- Original-EED rebuttal implementation and reports: `b8974183142a99574cca0b74daf57cca2035eebf`.
- The cleanup preserves historical commits and result blobs; seven model-config
  defaults move to `configs/models/qwen25_7b.yaml` without changing YAML content.
- Joint-versus-Mix is a proposed research direction, not implemented or validated
  by this repository consolidation. See `docs/PROJECT_STATUS.md`.
- Exact removed paths and retained identities are in
  `docs/research/2026-11-12-repository-consolidation.json`; historical replay
  instructions are in `docs/archive/README.md`.

The original extraction record below describes the earlier repository stage.

## Code

- Source repository: `yuhanlydia/PBPF`
- Final extraction source commit: `d00969ce49bde6108aa4f53dccb09eeb150db601`
- Earlier sealed result branch ancestry includes `0200283849274b49e303638b91f142acd8bba984`.
- Namespace-only extraction: `pbpf.eesd.*` → `eesd.*`.
- New files in this repository: strict experiment contracts, deterministic stage planner, final design/plan, standalone packaging, and CI.

## Paper

- Source archive: `EESD_Overleaf_FullText_Reviewed_9pages.zip`
- The included manuscript reports the recorded lexical-relevance EED experiments.
- Exact Shapley relevance is a prospective method extension and has no completed scientific validation in this repository at publication time.

## Result labels

- `results/recorded/`: immutable historical records and locks.
- Future final-method runs must use schema `eesd-stage-plan-v1` and method status `prospective_shapley` until their gate decision is complete.
- Recorded result schema `eesd-recorded-lexical-v1` must never be relabeled as Shapley evidence.

