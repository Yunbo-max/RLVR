#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
 echo "Usage: $0 DATA_DIR OUTPUT_ROOT [pilot|full]" >&2; exit 2
fi
DATA="$1"; OUT="$2"; SCALE="$3"
case "$SCALE" in
 pilot) TRAIN=4; EVAL=8; G=4 ;;
 full) TRAIN=64; EVAL=64; G=8 ;;
 *) echo 'SCALE must be pilot or full' >&2; exit 2 ;;
esac
export PYTHONPATH="$PWD/src"
mkdir -p "$OUT"
for arm in no_update sdpo audit_gate shapley_select; do
 if [[ -e "$OUT/$arm" ]]; then echo "Refusing existing result: $OUT/$arm" >&2; exit 1; fi
 python scripts/run_behavioral_shapley.py --data-dir "$DATA" \
  --output "$OUT/$arm" --arm "$arm" --train-limit "$TRAIN" \
  --eval-limit "$EVAL" --rollouts "$G" --seed 1701 --rounds 3
done
python scripts/summarize_behavioral_shapley.py --root "$OUT" --output "$OUT/paired_results.json"

python scripts/audit_behavioral_shapley.py --data-dir "$DATA" \
 --experiment-dir "$OUT/shapley_select" --output "$OUT/reserved_audit.json"
