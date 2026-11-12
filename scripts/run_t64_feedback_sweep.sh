#!/usr/bin/env bash
# Nested T=4/8/16/32/64 *only* Shapley and matched Direct-J.
# Same 64 executed tests/candidate, K=4 group players, B=16 coalition
# scoring calls, same frozen bug population and exact initial LoRA bytes.
set -euo pipefail
if [[ $# -ne 4 ]]; then
 echo "Usage: bash scripts/run_t64_feedback_sweep.sh FROZEN_DATA OUT_ROOT pilot|mechanism|scale SEED" >&2
 exit 2
fi
DATA="$1"; OUT="$2"; STAGE="$3"; SEED="$4"
if [[ -e "$OUT" ]]; then echo "Refuse overwrite: $OUT" >&2; exit 1; fi
export PYTHONPATH="$PWD/src"
mkdir -p "$OUT"
INIT="$OUT/shared_initial_adapter"
python scripts/create_learning_credit_init.py --output "$INIT" --seed "$SEED"
for T in 4 8 16 32 64; do
 bash scripts/run_t64_group_matrix.sh "$DATA" "$OUT/t$T" "$T" "$STAGE" "$SEED" "$INIT"
done
python scripts/aggregate_t64_feedback_sweep.py \
 --root "$OUT" --output "$OUT/feedback_richness_sweep.json"
