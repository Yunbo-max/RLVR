#!/usr/bin/env bash
# Seven-arm, five-update RunBugRun protocol. No model/evaluation fake data.
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
 echo "Usage: bash scripts/run_learning_credit_matrix.sh DATA_DIR OUTPUT_ROOT pilot|full [TRAIN_SEED]" >&2; exit 2
fi
DATA="$1"; OUT="$2"; SCALE="$3"
case "$SCALE" in
 pilot) TRAIN=8; EVAL=16; G=4 ;;
 full) TRAIN=64; EVAL=128; G=8 ;;
 *) echo "Unknown scale (pilot|full)" >&2; exit 2 ;;
esac
if [[ -e "$OUT" ]]; then
 echo "Refusing to reuse any experiment output: $OUT" >&2; exit 1
fi
export PYTHONPATH="$PWD/src"
mkdir -p "$OUT"
SEED=1701
if [[ $# -eq 4 ]]; then SEED="$4"; fi
INITIAL="$OUT/shared_initial_adapter"
python scripts/create_learning_credit_init.py --output "$INITIAL" --seed "$SEED"

# Need Shapley first ONLY to freeze TRAINING schedules, not final scores.
python scripts/run_learning_credit_five_round.py \
 --data-dir "$DATA" --output "$OUT/shapley" --arm shapley \
 --init-adapter "$INITIAL" --seed "$SEED" --rounds 5 \
 --rollouts "$G" --train-limit "$TRAIN" --eval-limit "$EVAL"
python scripts/make_learning_credit_schedule.py \
 --shapley-run "$OUT/shapley" --output "$OUT/shapley_schedule.json" --seed "$SEED"

for ARM in no_update sdpo audit_gate direct_max random_subset matched_random_updates; do
 if [[ "$ARM" == random_subset ]] || [[ "$ARM" == matched_random_updates ]]; then
  python scripts/run_learning_credit_five_round.py \
   --data-dir "$DATA" --output "$OUT/$ARM" --arm "$ARM" \
   --init-adapter "$INITIAL" --seed "$SEED" --rounds 5 \
   --rollouts "$G" --train-limit "$TRAIN" --eval-limit "$EVAL" \
   --schedule "$OUT/shapley_schedule.json"
 else
  python scripts/run_learning_credit_five_round.py \
   --data-dir "$DATA" --output "$OUT/$ARM" --arm "$ARM" \
   --init-adapter "$INITIAL" --seed "$SEED" --rounds 5 \
   --rollouts "$G" --train-limit "$TRAIN" --eval-limit "$EVAL"
 fi
done
python scripts/summarize_learning_credit_five_round.py \
 --root "$OUT" --output "$OUT/paired_summary.json"
for ARM in shapley direct_max random_subset; do
 python scripts/audit_learning_credit_five_round.py \
  --data-dir "$DATA" --experiment-dir "$OUT/$ARM" \
  --output "$OUT/$ARM/reserved_audit.json"
done
