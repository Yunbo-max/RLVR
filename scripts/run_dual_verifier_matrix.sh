#!/usr/bin/env bash
# Distinct arm/seed directories; do not launch on a shared GPU without resource planning.
set -euo pipefail
: "${DATA_DIR:?Set DATA_DIR to frozen RunBugRun Rich64 cohort}"
: "${INIT_ADAPTER:?Set INIT_ADAPTER to the immutable seed-specific LoRA path}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a dedicated run directory root}"
SEED=${SEED:-1701}
TRAIN_LIMIT=${TRAIN_LIMIT:-32}
EVAL_LIMIT=${EVAL_LIMIT:-32}
ROUNDS=${ROUNDS:-5}
SOLVER_ROLLOUTS=${SOLVER_ROLLOUTS:-4}
VERIFIER_PROPOSALS=${VERIFIER_PROPOSALS:-2}
DRY_RUN=${DRY_RUN:-1}
# Explicit CPU-only preflight is the default, not a GPU launch.
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
  echo 'DRY_RUN must be 0 or 1' >&2; exit 2
fi
mkdir -p "${OUTPUT_ROOT}"
for arm in dual solver_fixed solver_oracle_verdict solver_static_verifier solver_regenerated_verifier verifier_only; do
  out="${OUTPUT_ROOT}/${arm}-seed${SEED}-N${TRAIN_LIMIT}-R${ROUNDS}"
  args=(--data-dir "${DATA_DIR}" --init-adapter "${INIT_ADAPTER}"
        --output "${out}" --arm "${arm}" --seed "${SEED}"
        --train-limit "${TRAIN_LIMIT}" --eval-limit "${EVAL_LIMIT}"
        --rollouts "${SOLVER_ROLLOUTS}" --verifier-proposals "${VERIFIER_PROPOSALS}"
        --rounds "${ROUNDS}")
  if [[ "${DRY_RUN}" == "1" ]]; then
    args+=(--dry-run)
  fi
  echo "arm=${arm} output=${out} dry_run=${DRY_RUN}"
  python scripts/run_dual_verifier_opsd.py "${args[@]}"
done
