#!/usr/bin/env bash
# Publication-oriented EXACT T=4 (16 coalitions) five-round matrix.
# Only ONE restored RunBugRun SQLite-derived, SHA256-frozen data population.
# If data census lacks requested distinct problem IDs, prepare_formal_runbugrun_t4
# rejects the sample rather than manufacturing a larger test set.
set -euo pipefail
if [[ $# -ne 4 ]]; then
 echo 'Usage: bash scripts/run_formal_t4_matrix.sh FROZEN_DATA OUTPUT_ROOT pilot|mechanism|full TRAIN_SEED' >&2
 exit 2
fi
DATA="$1"; OUT="$2"; SCALE="$3"; SEED="$4"
case "$SCALE" in
 pilot) TRAIN=32; EVAL=64; G=4 ;;
 mechanism) TRAIN=128; EVAL=128; G=8 ;;
 full) TRAIN=512; EVAL=256; G=8 ;;
 *) echo 'Use pilot, mechanism or full' >&2; exit 2 ;;
esac
if [[ -e "$OUT" ]]; then
 echo "Refuse to overwrite any existing experiment: $OUT" >&2; exit 1
fi
if [[ ! -f "$DATA/manifest.json" ]]; then
 echo "Missing frozen data manifest: $DATA" >&2; exit 1
fi
export PYTHONPATH="$PWD/src"
mkdir -p "$OUT"
INIT="$OUT/shared_initial_adapter"
python scripts/create_learning_credit_init.py --output "$INIT" --seed "$SEED"

# Reference schedule is frozen from Shapley TRAINING receipts ONLY. Neither
# held-out evaluation nor the later audit can inform random schedule choices.
python scripts/run_learning_credit_five_round.py \
 --data-dir "$DATA" --output "$OUT/shapley" --arm shapley \
 --init-adapter "$INIT" --seed "$SEED" --rounds 5 \
 --rollouts "$G" --train-limit "$TRAIN" --eval-limit "$EVAL"
python scripts/make_learning_credit_schedule.py \
 --shapley-run "$OUT/shapley" \
 --output "$OUT/shapley_schedule.json" --seed "$SEED"
for ARM in no_update sdpo audit_gate direct_max random_subset matched_random_updates; do
 if [[ "$ARM" == random_subset ]] || [[ "$ARM" == matched_random_updates ]]; then
  python scripts/run_learning_credit_five_round.py \
   --data-dir "$DATA" --output "$OUT/$ARM" --arm "$ARM" \
   --init-adapter "$INIT" --seed "$SEED" --rounds 5 \
   --rollouts "$G" --train-limit "$TRAIN" --eval-limit "$EVAL" \
   --schedule "$OUT/shapley_schedule.json"
 else
  python scripts/run_learning_credit_five_round.py \
   --data-dir "$DATA" --output "$OUT/$ARM" --arm "$ARM" \
   --init-adapter "$INIT" --seed "$SEED" --rounds 5 \
   --rollouts "$G" --train-limit "$TRAIN" --eval-limit "$EVAL"
 fi
done
python scripts/summarize_learning_credit_five_round.py \
 --root "$OUT" --output "$OUT/paired_summary.json"

# Protected audit happens ONLY AFTER all five model updates finish.
for ARM in shapley direct_max random_subset; do
 python scripts/audit_learning_credit_five_round.py \
  --data-dir "$DATA" --experiment-dir "$OUT/$ARM" \
  --output "$OUT/$ARM/reserved_audit.json"
done
