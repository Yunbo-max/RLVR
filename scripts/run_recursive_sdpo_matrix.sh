#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
 echo "Usage: $0 DATA_DIR OUTPUT_ROOT [pilot|full]" >&2; exit 2
fi
DATA="$1"; OUT="$2"; SCALE="pilot"
if [[ $# -eq 3 ]]; then SCALE="$3"; fi
case "$SCALE" in
 pilot) TRAIN=4; EVAL=8; G=4;;
 full) TRAIN=64; EVAL=64; G=8;;
 *) echo "scale must be pilot or full" >&2; exit 2;;
esac
export PYTHONPATH="$(pwd)/src"
mkdir -p "$OUT"
for arm in no_update sdpo loo shapley; do
 if [[ -e "$OUT/$arm" ]]; then
  echo "Refusing existing output $OUT/$arm" >&2; exit 1
 fi
 echo "ARM=$arm SCALE=$SCALE TRAIN=$TRAIN EVAL=$EVAL G=$G"
 python scripts/run_recursive_shapley_sdpo.py \
  --data-dir "$DATA" --output "$OUT/$arm" --arm "$arm" \
  --train-limit "$TRAIN" --eval-limit "$EVAL" \
  --rollouts "$G" --rounds 3 --seed 1701
done
python scripts/summarize_recursive_sdpo_matrix.py --root "$OUT" \
 --output "$OUT/paired_results.json"
