# Official SDPO baseline for the latest Rich64 study

This integration starts from RLVR research commit
`60af331cd98d35d43c17b12f9205c888df30dddf`, not the old main-branch proposal.
It uses the authors' unmodified trainer at
[`lasgroup/SDPO@7c457fc`](https://github.com/lasgroup/SDPO/tree/7c457fc1b1f636ae794eb0362ba37d4743b06fbc).
The baseline name is **`sdpo_official_verl_rich64`**.

The simplified `--arm sdpo` option has now been removed from the current Rich64
runner and its default summary. The T64 matrix automatically preflights and
launches the official replacement. Pilot uses G=4; mechanism and scale use G=8
via the explicit `--rollouts` override. T4/8/16/32 internal sweeps do not launch
or relabel the full-T64 external baseline. Dated reports remain historical
evidence; compatibility helpers used by older archived workflows are not current
Rich64 training options.

## What the historical `sdpo` result means

The `sdpo` arm in the November 3 Rich64 report is an internal grouped-feedback
control. Its reported R5 score of 46.88% is **not an official SDPO reproduction**.
Keep historical result files unchanged and label that arm "grouped-feedback
distillation control (historical sdpo)" in new comparisons. The 48.44% Shapley
pilot result does not establish an advantage over official SDPO, which has no
completed scientific result in this workspace yet.

| Detail | Historical Rich64 `sdpo` | New baseline |
|---|---|---|
| Training implementation | Local recursive distillation loop | Actual pinned upstream VERL trainer |
| Teacher | Previous-round checkpoint | EMA teacher, update rate 0.01 |
| Teacher prompt | Grouped feedback summaries | Upstream successful same-prompt peer demonstration selection, otherwise execution feedback |
| Distillation | Full-vocabulary loss plus separate anchor | Reverse KL, student-selected top-20 support plus tail bucket, no extra reference KL |
| Tokens receiving loss | Extracted/normalized code | Original generated response tokens |
| Importance correction | Historical local implementation | Upstream token rollout correction and clipped distillation IS, threshold 2 |
| Source of execution feedback | Group summaries | Official feedback formatter over mandatory bubblewrap outcomes |

The authoritative settings are the pinned
[rich-feedback launch script](https://github.com/lasgroup/SDPO/blob/7c457fc1b1f636ae794eb0362ba37d4743b06fbc/experiments/rich_feedback/run_sdpo.sh),
[SDPO config](https://github.com/lasgroup/SDPO/blob/7c457fc1b1f636ae794eb0362ba37d4743b06fbc/verl/trainer/config/sdpo.yaml),
and [actor defaults](https://github.com/lasgroup/SDPO/blob/7c457fc1b1f636ae794eb0362ba37d4743b06fbc/verl/trainer/config/actor/actor.yaml).
The success threshold is 0.5 in the actual config; our 0/1 all-tests-pass rewards
make this equivalent to requiring reward 1. `adv_estimator=grpo` disables the
critic in upstream; the actor objective is explicitly `loss_mode=sdpo`.

## Scope and comparison limits

This is official **algorithm code adapted to our benchmark**, not reproduction
of the paper's published LiveCodeBench/Qwen3 scores. Both profiles below use
RunBugRun Python repair, pinned Qwen2.5-Coder-7B-Instruct, rank-16 LoRA, the shared
initial adapter, and greedy one-response evaluation. Paper experiments used
Qwen3-8B, LCBv6, full parameter training, and stochastic four-response validation.

| Setting | `rich64-pilot` | `upstream-hparams` |
|---|---:|---:|
| Train batch | 1 | 32 |
| Rollouts per prompt | 4 | 8 |
| Learning rate | 1e-5 | 1e-6 |
| Weight decay | 0 | 0.01 |
| Sampling temperature / top-p | 0.7 / 0.95 | 1.0 / 1.0 |
| Prompt / response cap | 8192 / 256 | 2048 / 8192 |
| Teacher reprompt cap | 10240 | 10240 |
| Epochs | 5 | 30 |

`upstream-hparams` preserves selected published optimization settings; it is not
an exact paper-reproduction profile. Current pilot prompts reach 2514 tokens,
so that profile's 2048 cap deliberately fails preflight. Do not silently drop or
truncate those examples: establish a separate declared length protocol before
running it on this cohort.

The official adaptation uses BF16 base weights; historical Rich64 uses NF4.
Online per-batch EMA updates also differ from Rich64 frozen-per-round updates.
Identical source IDs alone do not establish equal compute, identical sampled
trajectories, equal optimizer steps, or precision parity. Report generated tokens,
test executions, updates and GPU time alongside accuracy; rerun our arm with an
explicitly matched protocol before making a controlled superiority claim.

`score`/`reward` is binary whole-program success. Upstream's `acc` diagnostic is
the fraction of individual tests passed and may appear as `val-core` in logs.
**Use mean binary `score` from saved validation JSONL for the Rich64 accuracy
table, never the `acc` diagnostic.** Evaluation feedback is empty and is never
used for teacher reprompting or training.

## Prepared WSL paths

- Research worktree: `/root/ai-work/RLVR-sdpo`, branch
  `codex/sdpo-official-reproduction-20261107`.
- Unmodified official checkout: `/root/ai-work/SDPO-official`.
- Independent Python 3.12 environment: `/root/ai-venvs/sdpo-official`.
- Source DB: `/root/ai-assets/datasets/runbugrun-official-v2/runbugrun.db`.
- Frozen source: `data/rich64-wsl-20261107` (8000 bugs / 1033 train problems,
  64 independent meta problems, 199 independent test problems).
- Pilot export: `data/sdpo-rich64-pilot-1701` (first 64 train / first 64 test).
- Full export: `data/sdpo-rich64-full-1701` (8000 train / 199 test); prompt maxima
  are 6640 / 3581 tokens, both within the Rich64 profile's 8192 cap.
- Shared initial adapter: `artifacts/shared-init-1701`, with adjacent hash receipt.
- Windows source mirror: `C:\ai\RLVR-sdpo` (edit mirror, sync changes to WSL).

The complete official RunBugRun v2 SQL archive was missing from the older
Hugging Face shard download. It is now restored, with SQLite `quick_check=ok`.
Source: <https://github.com/giganticode/run_bug_run/releases/download/v2/runbugrun.sql.lrz>.
Local archive SHA256: `c46225277296add83a862e47c0bc0fb18c5d26ef8b6980bd41e5ef737f93896c`.
Restored DB SHA256: `c7949e276dcbfc3457bb40554d5d95151855d71bdc185ad711bb58f2fc1d8db0`.
These are locally computed checksums, not publisher-provided checksums.
Historical raw pilot outputs/manifests were not committed, so exact cohort
identity with the remote historical run has not been established.

The export checks source hashes, unique tests, and problem-level split isolation.
Only public T64 tests enter training parquet; C8 calibration, A8 audit, meta
tests and gold fixed code do not. Final evaluation uses the selected problems'
entire frozen test banks. A real-data Unicode JSONL boundary bug was found and
covered by a regression test.

## Running

To run the current T64 matrix including the replacement, use the original
Rich64 environment for our NF4 arms and the separate official environment for
SDPO (select an available GPU first):

```bash
cd /root/ai-work/RLVR-sdpo
source /root/ai-venvs/rlvr/bin/activate
export SDPO_PYTHON=/root/ai-venvs/sdpo-official/bin/python
CUDA_VISIBLE_DEVICES=1 bash scripts/run_t64_group_matrix.sh \
  data/rich64-wsl-20261107 runs/rich64-with-official-sdpo-1701 \
  64 pilot 1701 artifacts/shared-init-1701
```

The official result goes to `sdpo_official_verl_rich64/summary.json` within that
run; `paired_summary.json` covers the internal controls. These remain separate
because the official algorithm does not meet the internal summary's equal
group-coalition and frozen-round protocol. At scale, the official baseline makes
five passes over all 8000 examples whereas our method rotates panels; the run
must not be described as equal training exposure or compute.

From PowerShell, enter WSL with `wsl -u root -- bash`, then:

```bash
cd /root/ai-work/RLVR-sdpo
source /root/ai-venvs/sdpo-official/bin/activate
python scripts/run_sdpo_official.py check \
  --data-dir data/sdpo-rich64-pilot-1701 \
  --init-adapter artifacts/shared-init-1701 \
  --output runs/sdpo-official-pilot-1701

# Check nvidia-smi and assign an available GPU before starting the real study.
CUDA_VISIBLE_DEVICES=1 python scripts/run_sdpo_official.py run \
  --data-dir data/sdpo-rich64-pilot-1701 \
  --init-adapter artifacts/shared-init-1701 \
  --output runs/sdpo-official-pilot-1701

# After the final pilot checkpoint has produced all 64 evaluation records:
python scripts/summarize_sdpo_official.py \
  --data-dir data/sdpo-rich64-pilot-1701 \
  --validation runs/sdpo-official-pilot-1701/validation/320.jsonl
```

`plan` prints the exact command, upstream hashes, dataset hashes and deviations.
`check` resolves the actual upstream Hydra config, checks the shared initialization,
pinned model files, prompt lengths and working bubblewrap isolation. It does not
prove that a full-size GPU training run fits memory. `run` refuses an existing
output directory, preserves a resolved config and package inventory, writes
training logs/validation generations/checkpoints, and records the trainer exit
status. A zero exit status still requires review of evaluation counts and scores.
No W&B upload is enabled. No automatic resume is used because teacher EMA
checkpoint/resume fidelity has not been established.

For a new machine, install Linux `bubblewrap`, restore the official dataset,
clone SDPO at the exact commit above, create a separate Python 3.12 venv, install
`requirements/sdpo-official-wsl.txt`, then install both repositories with
`pip install --no-deps -e /path/to/SDPO -e /path/to/RLVR-sdpo`.
The WSL runtime uses Torch 2.9.0 / vLLM 0.12.0, SDPA with padding retained for
the actor, and `VLLM_ATTENTION_BACKEND=TRITON_ATTN` for rollout. Default vLLM
FlashAttention stalled during the bounded WSL/Blackwell qualification; use the
recorded Triton runtime and preserve that adaptation in comparisons.
NumPy 1.26.4 satisfies upstream setup metadata; upstream's own requirements and
setup metadata disagree on NumPy 2. Preserve the generated full package inventory.

Generate source data using `scripts/prepare_t64_runbugrun_cohort.py`, export via
`scripts/export_sdpo_rich64.py`, and create one shared untrained adapter with
`scripts/create_learning_credit_init.py`. That last script uses the original
Rich64 quantized-loading environment (`/root/ai-venvs/rlvr`) with
`PYTHONPATH=/root/ai-work/RLVR-sdpo/src`; do not generate separate initializations
for each comparison arm.

## Validation

Verified on 2026-11-07:

- 214 repository tests passed in the independent official environment after
  replacing the default baseline, including all three T64 matrix dispatches.
- `uv pip check` passed for all 228 installed packages.
- Pilot and full frozen datasets passed preflight, including tokenizer length
  checks and actual bubblewrap execution.
- Bounded random Qwen2 (~19.86M parameters), rank-16 LoRA, G=2, response cap 16:
  the unmodified official trainer completed rollout, public feedback, one
  optimizer step, private validation and checkpoint save with exit code 0.
  Feedback-used fraction was 1.0, extra KL loss 0, gradient norm
  `1.45336e-5`, and saved adapter change L2 norm `0.00110861`.
- The saved LoRA adapter reloaded and produced finite forward-pass logits.

Runtime receipt and exact command:
`runs/qualification/sdpo-tiny-runtime-v3/qualification.json`.
Source driver, full log and checkpoint are retained in the same directory.
This is a runtime qualification, **not a Qwen7B accuracy result**. The full-size
7B training footprint and the five-epoch scientific study remain unmeasured;
other AKR/SPECTRUM experiments were using both GPUs during preparation.

Run `python -m pytest tests -q` in the official environment. Tests call the actual
upstream loss and EMA routines; verify reverse-KL values and gradients, student
top-k/tail handling, success selection, masked gradients, split isolation,
public-only feedback, source/export hashes, and the actual upstream model schema.
Small random-model runtime qualification is engineering evidence only; its scores
must never be included in the scientific results table.
