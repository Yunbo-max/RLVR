# Historical source and evidence index

This index preserves complete old implementations and adverse evidence after
the active tree is cleaned. Historical branch names are retained. Use exact
commits to reproduce an old protocol.

| Snapshot | Exact source | Purpose |
|---|---|---|
| Previous RLVR main | [b773c71d](https://github.com/Yunbo-max/RLVR/tree/b773c71dfb33ad492f25503809fc781e5fddac6a) | Old default tree, including GRPO diagnostics, preflight files and original CI |
| Rich64 foundation | [c44efb97](https://github.com/Yunbo-max/RLVR/tree/c44efb97332e0432023d32e7e34d484dce2c54f6) | Behavioral Shapley/token-aware implementation and official SDPO integration |
| Original EED rebuttal | [b8974183](https://github.com/Yunbo-max/RLVR/tree/b8974183142a99574cca0b74daf57cca2035eebf) | Complete C00-C16/D01-D04 code, later trainer fixes, result reports and handoff |
| Behavioral Shapley, earlier | [fb6c4062](https://github.com/Yunbo-max/RLVR/tree/fb6c4062a4d99bc2cba52381f05f20fc6069a457) | Earlier three-round route |
| Five-round learning credit | [2993e15c](https://github.com/Yunbo-max/RLVR/tree/2993e15cb7f08336f2a25396deda409bc938ae73) | Original five-round design and implementation |
| PBPF extraction origin | [d00969ce](https://github.com/yuhanlydia/PBPF/tree/d00969ce49bde6108aa4f53dccb09eeb150db601) | Source of the independent `eesd` namespace extraction |
| PBPF sealed original pipeline | [02002838](https://github.com/yuhanlydia/PBPF/tree/0200283849274b49e303638b91f142acd8bba984) | Original lexical-EED prediction records and pinned evaluator dependencies |

The new consolidation commit retains previous main and the complete rebuttal
snapshot in its parent history. Removing old files from the active tree does
not erase their commits. No branch is deleted or force-updated.

## Restore an old implementation separately

From a clean local RLVR checkout, use a separate, new worktree path:

```bash
git fetch origin
git worktree add --detach ../rlvr-rebuttal-20261110 b8974183142a99574cca0b74daf57cca2035eebf
git worktree add --detach ../rlvr-rich64-20261107 c44efb97332e0432023d32e7e34d484dce2c54f6
```

These are restoration instructions, not commands run during cleanup. Keep an
already running job on its original checkout and immutable inputs.

## Evidence locations

- [Rich64 pilot](../research/2026-11-03-rich64-pilot-results.md).
- [Retained rebuttal summaries](../../reports/rebuttal2027/README.md).
- [Full November 10 handoff](https://github.com/Yunbo-max/RLVR/blob/b8974183142a99574cca0b74daf57cca2035eebf/docs/research/2026-11-10-eesd-rebuttal-handoff.md).
- [Original reviewer-to-case mapping](https://github.com/Yunbo-max/RLVR/blob/b8974183142a99574cca0b74daf57cca2035eebf/docs/research/2026-11-07-iclr2027-final-reviewer-case-matrix.md).
- [PBPF original prediction archives](https://github.com/yuhanlydia/PBPF/tree/0200283849274b49e303638b91f142acd8bba984/artifacts/eesd-eval-20260921).
- [PBPF dataset and runtime migration record](https://github.com/yuhanlydia/PBPF/blob/0200283849274b49e303638b91f142acd8bba984/docs/EESD_GPU_MIGRATION_20260921.md).

PBPF is not deleted or repurposed: it also contains independent PBPF/A-PBPF
research. Its data backups and original scorer/provenance records are not
replaced by this code consolidation. No large backup has been revalidated here.
