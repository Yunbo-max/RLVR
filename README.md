# EESD: Effective-Evidence Self-Distillation

## Current Rich64 experiments

The current study is **EESD-Rich64** on Qwen2.5-Coder-7B / RunBugRun.
The old simplified `sdpo` training option has been removed from the Rich64
runner and default matrix. Its replacement is **`sdpo_official_verl_rich64`**,
using the pinned authors' SDPO trainer. At T64, `scripts/run_t64_group_matrix.sh`
preflights and runs this baseline in the separate `SDPO_PYTHON` environment,
sharing frozen rows, initial adapter, seed and stage rollout count.

Read [the official baseline setup and commands](docs/research/2026-11-07-official-sdpo-reproduction.md).
Official results use `scripts/summarize_sdpo_official.py`; internal controls use
`scripts/summarize_t64_group_five_round.py`. BF16/NF4 and online/frozen-round
differences remain explicit. Historical `sdpo` scores must not be attributed to
the official replacement. The sections below describe the earlier EESD formulation.

EESD is a testable method for deciding **how strongly a coding agent should learn from its own execution-guided correction**. A correction can look successful on visible tests yet carry weak, concentrated, or misleading evidence. EESD separates the direction of feedback from its effective amount before using it for self-distillation.

## Final method

For four public executions, EESD defines a coalition value from the corrected program's edited-token log-probability advantage over the original program. Exact Shapley values attribute that contrast to individual executions. Their absolute magnitudes define normalized relevance `p`; the Rényi-2 effective support

```text
E(p) = 1 / sum_i p_i^2
```

sets the total pseudo-count mass of a Jeffreys Dirichlet posterior over `improved / unchanged / regressed`. The positive posterior mean advantage weights correction SFT, while a forward KL penalty anchors the student to its pre-update policy.

## Evidence status — read this first

| Component | Status |
|---|---|
| Dirichlet/effective-mass mathematics | implemented and unit-tested |
| Exact four-execution Shapley attribution | implemented and unit-tested |
| Recorded lexical-EED mechanism study | recorded exploratory evidence |
| Recorded 12-cell downstream matrix | 4 positive, 1 tie, 7 negative |
| Final Shapley–EESD signal/mechanism/downstream gates | **not yet run** |

The recorded DeepSeek-Coder-6.7B + CodeARC result (15.0%→20.4%, +5.4 pp; source-bootstrap 95% CI [+2.8,+8.0]) belongs to the earlier lexical-relevance EED pipeline. It is motivation and a diagnostic-cell choice, **not evidence that the final Shapley method succeeds**.

## Frozen experiment order

```text
Software → Signal (64 trajectories/cell) → Mechanism → Matched downstream → Expansion
```

The first three diagnostic cells are:

| Role | Model (pinned) | Dataset | Why |
|---|---|---|---|
| Positive preservation | `deepseek-ai/deepseek-coder-6.7b-instruct` | CodeARC | strongest prior positive cell |
| Moderate signal | `Qwen/Qwen2.5-Coder-7B-Instruct` | RunBugRun | positive but uncertain prior cell |
| Harm prevention | `google/gemma-3-4b-it` | RunBugRun | prior negative cell |

Each downstream comparison uses 500 fresh primary sources, identical candidate/evaluation ancestry, seed 1701 first, 10,000 paired source-cluster bootstrap draws, and the same response-token budget for every training arm. Confirmation seeds 1702/1703 run only after the first gate passes.

### Required downstream baselines

`no_update`, `equal_weight`, `final_correctness`, `scalar_confidence`, `fixed_mass_dirichlet`, `lexical_eed`, `shapley_eed_no_kl`, and `eesd_full`.

This is essential: comparing only against `no_update` cannot isolate adaptive evidence mass or Shapley attribution.

## Benchmarks

- **RunBugRun** — executable program repair; core hidden-test generalization benchmark.
- **CodeARC** — interactive inductive program synthesis; core execution-feedback benchmark.
- **EvalPlus (HumanEval+/MBPP+)** — external functional-correctness transfer after core gates pass.
- **LiveCodeBench release_v6** — temporally filtered external coding transfer after core gates pass.
- **APPS/CodeContests Replay** — controlled replay stress tests only; never reported as official benchmark accuracy.

Exact model and benchmark revisions, quantization, LoRA rank, seeds, budgets, metrics, and arms live in [`configs/experiments/eesd_final.yaml`](configs/experiments/eesd_final.yaml).

## Install and validate

```bash
uv sync --extra test
uv run pytest -q
uv run python scripts/run_stage.py \
  --config configs/experiments/eesd_final.yaml \
  --stage signal --seed 1701 --dry-run
```

GPU dependencies are optional:

```bash
uv sync --extra test --extra ml
```

## Run order

1. Materialize pinned datasets and model snapshots; record them in an input manifest.
2. Inspect the dry-run plan for a stage.
3. Run the worker scripts listed in that plan, one cell at a time, preserving every artifact.
4. Do not start expansion until the prior gate's decision file says `PASS`.

Example planner invocation:

```bash
uv run python scripts/run_stage.py \
  --config configs/experiments/eesd_final.yaml \
  --stage downstream --seed 1701 --dry-run > downstream-plan.json
```

The research design and PASS/REVISE/KILL logic are in [`docs/research/2026-10-02-eesd-final-design.md`](docs/research/2026-10-02-eesd-final-design.md). The latest ICLR paper source is under [`paper/`](paper/); it still reports the recorded lexical-EED evidence and must be revised only after the prospective Shapley gates are run.

## Repository provenance

The mathematical core and experiment utilities were extracted from `yuhanlydia/PBPF` commit `d00969ce49bde6108aa4f53dccb09eeb150db601`. The paper source is the reviewed `EESD_Overleaf_FullText_Reviewed_9pages` archive. See [`PROVENANCE.md`](PROVENANCE.md).

