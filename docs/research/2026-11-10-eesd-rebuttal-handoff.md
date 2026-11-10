# EESD rebuttal handoff — 2026-11-10 (project switch)

State at pause: three cells fully complete (results committed in
`2026-11-09-eesd-rebuttal-p0-results.md`); one cell trained but not yet
evaluated; two cells require data regeneration due to a tokenizer-decoding
regression discovered on 2026-11-10.

## Completed (results on GitHub)

| Cell | Status |
|---|---|
| codearc__qwen25_7b | 17/17 trained + evaluated + summarized |
| runbugrun__qwen25_7b | 17/17 trained + evaluated + summarized |
| codearc__gemma3_4b | 17/17 trained + evaluated + summarized |

Headline findings: EED weighting is protective (equal-weight training
collapses Gemma on CodeARC: 1.4% vs 11.4% frozen; significant); learned
relevance beats uniform relevance on RunBugRun/Qwen (+1.6pp, CI excludes 0);
adaptive vs fixed mass is null on every completed cell. Consistent with the
archived D03 re-analysis.

## Pending work (paused by user instruction)

1. **runbugrun__gemma3_4b**: 16/17 arms trained (C13_KAPPA8 job failed and
   needs a retry); evaluations NOT yet submitted. Estimated 4–6 hours once
   resumed.
2. **DeepSeek cells (codearc + runbugrun) — data regeneration required**:
   transformers 5.3 decodes DeepSeek-Coder byte-fallback tokens as literal
   U+010A (`Ċ`) / U+0120 (`Ġ`) characters instead of newline/space. Every
   generated candidate and correction for both deepseek cells contains these
   characters, so all 612 (codearc) and 721 (runbugrun) corrections are
   COMPILE_ERROR on every test; EED weights are all zero and the case arms
   landed in `non_estimable_zero_positive_training_weight` (codearc) or will
   do so (runbugrun). These are sealed artifacts and cannot be patched in
   place — the bank → prepare → generate chain must be re-run under new
   round directories with a decode-restoration wrapper (same pattern as the
   NEL/rename wrappers: monkey-patch `pbpf.apbpf.rbr_prompt.decode_completion`
   before the pinned generator runs, restoring Ċ→\n, Ġ→space, ĉ→\r), then the
   case matrices re-prepared against the new corrections and re-evaluated.
   Estimated 1.5–2 days on resume.
3. Seeds 1702/1703: not started (gated on 1701 results).

## Environment amendments (all documented in the inputs-root RUNBOOK)

1. `--greedy --candidates 1` (transformers 5.x rejects greedy with
   num_return_sequences>1).
2. New direct-execution lock `direct-execution-lock-20261108.json`
   (`f935cb53…`); old lock preserved as history.
3. `python3` for direct stages (lock binds the interpreter binary).
4. GPFS lacks renameat2(RENAME_NOREPLACE) → wrappers use `os.rename`.
5. NEL-safe JSONL reads (U+0085/U+2028/U+2029 escaped at read time).
6. Gemma PEFT 0.14 canonical LoRA target names accepted (two sites:
   adapter validation + matched generation).
7. Sealed per-cell identity manifest for the dataset_identity gap.
8. bwrap shims: flag translation (--proc/--dev → ro-bind, --tmpfs → --dir),
   bwrap.real wrapper layer that self-installs LD_PRELOAD (executor Popen
   resets the environment), noslave.so that fakes the root MS_SLAVE marking
   only on EPERM. Pods are heterogeneous (proc-mount blocked / MS_SLAVE
   blocked / fully permissive).
9. Pinned-scorer parse overflow (pathological candidate literals) recorded as
   WRONG_OUTPUT with a `parse_overflow_recorded` marker.
10. Node kernel heterogeneity: evaluation plans bind a per-run sandbox lock
    sha; all arms of a cell must share it (collector requirement) — re-run
    minority-kernel arms until convergence.

## Resume procedure

1. DeepSeek decode wrapper: restore byte-fallback characters after
   `tokenizer.decode`; regenerate `correction-original-banks` for the two
   deepseek cells under `round-rebuttal2` (train+dev), re-run prepare +
   generate under round2 dirs, publish corrections to the inputs root,
   regenerate identity manifests.
2. Clean the two deepseek cells' case directories (their receipts bind the
   corrupted corrections) and resubmit D01 + cases + ablations.
3. Retry runbugrun gemma C13; submit the 17 evaluations for
   runbugrun__gemma3_4b.
4. Collect + summarize each finished cell; append results to the results doc
   and push.
5. Replay cells remain blocked (train-reserve re-materialization).
