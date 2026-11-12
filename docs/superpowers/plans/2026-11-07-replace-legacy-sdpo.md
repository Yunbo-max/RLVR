# Replace the legacy SDPO baseline

Goal: replace the old baseline in the latest Rich64 workflow, as explicitly
selected by the user; do not delete the RLVR checkout or shared method code.

Implementation:
- [x] Remove `sdpo` from the current grouped runner and default paired summary.
- [x] Dispatch T64 matrices to the pinned official runner with the same frozen
  subset, shared initialization, seed and stage rollout count. Run its preflight
  before our training arms; keep its results in its own truthful schema.
- [x] Keep T4/8/16/32 sweeps on the internal controls: the external baseline
  currently implements the full T64 feedback protocol only.
- [x] Update current configuration and README; retain dated historical evidence.
- [x] Verify dispatch using subprocess recording without GPU training, reject the
  old CLI option, validate the official overrides, and run repository tests.

Do not relabel historical scores as official results. BF16 vs NF4, online EMA
vs frozen rounds, and scale-stage repeated vs rotated data remain explicit
protocol differences. Official SDPO must not enter the internal paired summary's
equal-group-coalition checks under a fabricated schema.

Validation: 214 tests passed. All three T64 stage dispatches select the official
runner with G4/G8 as appropriate, and T4 never invokes the full-T64 baseline.
The removed CLI arm is rejected. Actual frozen-data Shapley dry-run and actual
official Hydra/sandbox/tokenizer preflight with G8 passed. Shell syntax and git
whitespace checks passed. No new scientific GPU study was started.
