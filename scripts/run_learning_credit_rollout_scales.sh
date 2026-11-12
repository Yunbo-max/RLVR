#!/usr/bin/env bash
# Secondary scale experiment: G=8/16/32 with T=4 => ALWAYS 16 coalitions.
# Same shared frozen adapter. Compare only Shapley and direct J(S).
set -euo pipefail
if [[ $# -ne 3 ]]; then
 echo "Usage: $0 FROZEN_DATA OUTPUT_ROOT SHARED_INIT_ADAPTER" >&2; exit 2
fi
DATA="$1"; OUT="$2"; INITIAL="$3"
if [[ -e "$OUT" ]]; then echo "Refusing to overwrite rollout scale experiment" >&2; exit 1; fi
if [[ ! -f "$INITIAL/adapter_config.json" ]]; then echo "No shared LoRA init" >&2; exit 1; fi
export PYTHONPATH="$PWD/src"
mkdir -p "$OUT"
for G in 8 16 32; do
 for ARM in shapley direct_max; do
  python scripts/run_learning_credit_five_round.py \
   --data-dir "$DATA" --output "$OUT/g$G-$ARM" --arm "$ARM" \
   --init-adapter "$INITIAL" --seed 1701 --rounds 5 \
   --rollouts "$G" --train-limit 16 --eval-limit 32
 done
done
python scripts/summarize_learning_credit_rollouts.py \
 --root "$OUT" --output "$OUT/rollout_scaling.json"
