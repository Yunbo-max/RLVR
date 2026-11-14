
## Update on November 14, 2026 — new code

**Joint/Mix software is now included:** `src/eesd/joint_mix.py`, `scripts/run_joint_mix_experiment1.py`, `scripts/run_joint_mix_experiment2.py`, and `scripts/summarize_joint_mix.py`. The new study compares a genuine pairwise joint teacher with a C8-calibrated mixture, and adds an ungrouped, budgeted actual-joint subset search and independent random-search control. Eleven new CPU tests passed; **no GPU run, scorer-semantic certification or scientific result has been completed**. Read [the complete new protocol](research/2026-11-14-joint-mix-experiments.md) (from docs) or `docs/research/2026-11-14-joint-mix-experiments.md` (from repository root). The remainder of this document describes the historical November 12 consolidation and must not be interpreted as the new method's current implementation status.

# Local handoff: repository consolidation

## Purpose and scope

Use `Yunbo-max/RLVR / main` for subsequent development. This maintenance round
brings in the existing Rich64 foundation, changes seven default model-config
paths, removes unused old GRPO/preflight files, and retains historical results.
It does not implement Joint-versus-Mix or start an experiment.

Read `docs/PROJECT_STATUS.md` and `docs/REPOSITORY_LAYOUT.md` first.
Exact source identities and removed paths are recorded in
`docs/research/2026-11-12-repository-consolidation.json`.

## Checkout

Use a fresh checkout for continued development; keep running jobs on their
existing commit. A fresh directory avoids overwriting an active local branch:

```bash
git clone --branch main https://github.com/Yunbo-max/RLVR.git rlvr-current
cd rlvr-current
git status --short
git rev-parse HEAD
```

Record the actual commit before any subsequent run. To inspect old C00-C16
experiments, use the exact `b8974183` worktree in `docs/archive/README.md`.

## Source acceptance

The consolidation uses source inspection only. No project test, training,
inference, data preparation or scientific evaluation was executed here.
The next Local acceptance should confirm:

1. `configs/models/qwen25_7b.yaml` retains Git blob
   `472ced3132460d611bbc9ceb081eb1fbc8efc68f` from its former path.
2. All seven runner defaults resolve to that new path.
3. The Rich64 -> learning-credit -> recursive runner dependency chain remains
   intact, along with official SDPO files and the isolated executor.
4. `.github/workflows/ci.yml` retains blob
   `f597a77d81eb1d0cc4c0ff59a12460f148407bef`.
5. Historical result JSON blobs match the consolidation manifest.

Useful read-only source checks include:

```bash
git diff --check
git hash-object .github/workflows/ci.yml
git hash-object configs/models/qwen25_7b.yaml
git grep -n 'configs/models/qwen25_7b.yaml' -- scripts
```

## Existing environment limits

The base package and dependency lock are retained. The Rich64 SDPO bridge tests
directly import `pyarrow`, while the old `test` extra only declares `pytest`.
This is an existing setup gap, not a scientific result. The CI workflow is
preserved at the owner's request; do not patch it to close this gap.

For a separately authorized Local software acceptance, use the established
project environment with PyArrow available. The current lock already contains
PyArrow 25.0.1 through the ML dependency graph. An explicit supplemental command
for that acceptance is:

```bash
uv sync --extra test
uv run --with pyarrow==25.0.1 pytest -q
```

These commands are supplied for Local acceptance and were not executed in this
cleanup. Use the project's existing native execution policy for any test that
executes code. Report actual results rather than copying historical test claims.

For GPU use, preserve the existing model and dataset revisions and follow
`docs/research/2026-11-07-official-sdpo-reproduction.md` for the separate official
SDPO environment. Earlier environment paths in that document are historical
examples; inspect actual paths and available resources before use.

## Next scientific handoff

The two Joint/Mix comparisons discussed with the owner still need a qualified
implementation, fixed data roles, comparison rules, measured budget and
prospective acceptance. Do not launch the old Rich64 matrix and label its
outputs as the new study. Existing four-group and schedule-matched controls
remain baselines with their original semantics.
