# Official SDPO baseline integration

**Goal:** Run the authors' pinned SDPO trainer as a separately named baseline on the latest Rich64 cohort, with auditable differences from paper reproduction.

**Architecture:** Use unmodified `lasgroup/SDPO` commit `7c457fc1b1f636ae794eb0362ba37d4743b06fbc`. Export the existing frozen RunBugRun cohort to the upstream parquet schema; bridge its stdin/stdout public tests through mandatory bubblewrap. Use upstream reprompting, EMA, top-k/tail distillation and optimizer code. Preserve historical `sdpo` scores as a simplified control.

**Scope:** This is official SDPO adapted to RunBugRun/Qwen2.5-Coder. The published LCB/Qwen3-8B scores require different data/model/hardware and are not reproduced by this integration. Upstream-rich-feedback hyperparameters and Rich64-matched hyperparameters must have distinct profile names. Both training/evaluation selection and all deviations must be recorded.

## Tasks

- [x] Pin upstream source and runtime dependencies; install an independent WSL environment.
- [x] Test and implement frozen cohort export: hashes, disjoint problem IDs, exact public-only training payload, no overwrite, complete official evaluation tests.
- [x] Test and implement the upstream reward hook: sparse all-pass rewards, official feedback rendering, isolated execution, no private evaluation feedback returned.
- [x] Test and implement a launcher with source/model/data identity checks, resolved upstream Hydra config, CPU/sandbox preflight and explicit run mode.
- [x] Exercise the actual upstream loss and reprompt logic on CPU and validate the assembled runtime. Run a bounded GPU smoke if resources permit; distinguish this from a full baseline study.
- [x] Document verified state, data lineage, commands, limitations and the historical-control label.

Validation: 208 tests pass; installed dependency check passes; the synthetic
Qwen2/LoRA official-trainer run finishes one optimizer step, validation and
checkpoint save, with nonzero finite gradients and adapter changes. Checkpoint
reload and forward pass also pass. Real 7B scientific training has not started.

## Review focus

Fail on modified upstream training code, tampered parquet, train/eval/meta overlap, unknown splits, missing input tests, and unsupported model provenance. A private test must never appear in a training parquet or feedback. Do not overwrite existing results. Do not label a command printout or configuration check as completed training.
