# EESD rebuttal P0 execution run-state — 2026-11-08

Status: preparation complete; case matrices submitted; **no new scientific
results reported yet** (banks / prepare / corrections are sealed inputs, not
outcomes). This doc records the operational state for the continuation session;
it does not claim experimental results.

## Frozen revisions

- RLVR: `research/eesd-rebuttal-per-case-20261107` (this branch; `75febf3`
  adds `--batch-size` + Gemma3 `token_type_ids`; parent `bb4dcdb` is the
  per-case implementation revision).
- PBPF: `0200283849274b49e303638b91f142acd8bba984` (clean checkout, validated
  by the matched evaluator's `validate_checkout`).

## Execution-environment amendments (replacement machine, per RESTORE.md)

All five are operational amendments, not protocol changes; PBPF checkout files
are unmodified, and each is logged with the wrapper's sha256 in job output:

1. Bank generators require `--greedy --candidates 1` (transformers 5.x rejects
   `num_return_sequences>1` with greedy; correction prep requires one-candidate
   banks anyway).
2. New direct-execution lock `runs/eesd-setup/direct-execution-lock-20261108.json`
   (sha256 `f935cb53…`) built inside the qz container; the 20260921 lock binds
   the old host and is preserved as history.
3. Prepare/generate run with `python3` (the lock binds the interpreter binary).
4. `renameat2(RENAME_NOREPLACE)` is unsupported on GPFS (errno 22); wrappers
   `run_direct_stage.py` / `run_gemma_generate.py` substitute `os.rename` with
   the target-exists check preserved.
5. NEL-safe JSONL reads: pinned PBPF scripts use `splitlines()`, which breaks
   records containing raw U+0085/U+2028/U+2029; the wrappers replace those with
   JSON escapes at read time (data unchanged; file hashes unaffected).

## Inputs (all under GPFS, see eesd_rebuttal_inputs/paths.json)

- Models: three pinned revisions cached in hf_cache (verified).
- PBPF data restored from `humanlong/PBPF` (4.89 GB; sha256 305/305; artifact
  verify 22,162 + 22,568 files).
- Eval source→problem ID maps derived for the 4 historical-bridge cells from
  sealed D03 caches + public/evaluator materializations (500 rows each).
- Gemma bank generation reconstructed outside the pinned checkout (lost
  old-host artifacts): model proof rebuilt from the pinned snapshot; prompt
  source reimplemented from the pinned gemma user-only chat format; a
  development-split variant added. Documented in the inputs-root RUNBOOK.

## Submitted so far (2026-11-08)

- 6/6 cell bank generation jobs completed (sealed).
- 6/6 prepare jobs submitted (3 succeeded at first submit).
- Corrections generation: codearc qwen 479 rows, runbugrun qwen 321 rows,
  codearc gemma 515 rows done; 3 cells in flight.
- Full 17-case matrices (C00–C16 + D01) submitted for 3 cells
  (codearc qwen running; runbugrun qwen, codearc gemma queued after a project
  quota move). Remaining 3 cells follow their corrections.
- Batch sizes (H100 probe, reserved GiB): qwen B=2 (46.0), deepseek B=8 (49.7),
  gemma B=2 (65.8); exported via `EESD_TRAIN_BATCH_SIZE`.

## Open items

- Replay cells (apps_replay/codecontests_replay × 3 models): blocked — train
  reserve re-materialization is a separate upstream pipeline.
- Seed 1702/1703 confirmation rounds: only after the 1701 P0 gates.
- Honest reporting: zero new GPU scientific results until P0 arms + matched
  evaluation complete.
