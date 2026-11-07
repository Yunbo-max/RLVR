#!/usr/bin/env bash
# Five-round nested T4/8/16/32/64. Each arm executes up to 64 public
# tests/candidate. Shapley/DirectMax use 16 GROUP coalition prompts/candidate.
# At T64, the external SDPO baseline runs in its own official environment.
set -euo pipefail
if [[ $# -lt 5 || $# -gt 6 ]]; then
 echo "Usage: bash scripts/run_t64_group_matrix.sh DATA_DIR OUT_ROOT T(4|8|16|32|64) pilot|mechanism|scale SEED [SHARED_INIT_ADAPTER]" >&2
 exit 2
fi
DATA="$1"; OUT="$2"; T="$3"; STAGE="$4"; SEED="$5"
case "$T" in 4|8|16|32|64) ;; *) echo "T must be 4/8/16/32/64" >&2; exit 2 ;; esac
case "$STAGE" in
 pilot) TRAIN=64; EVAL=64; G=4; PANEL=repeat ;;
 # eval pool = official test split (test-only) with >=10 unique input tests;
 # prepare_t64_runbugrun_cohort.py freezes 199 such problems (not 256, not 248).
 mechanism) TRAIN=512; EVAL=199; G=8; PANEL=repeat ;;
 scale) TRAIN=8000; EVAL=199; G=8; PANEL=rotate ;;
 *) echo "STAGE must be pilot, mechanism or scale" >&2; exit 2 ;;
esac
if [[ -e "$OUT" ]]; then echo "Refusing to overwrite $OUT" >&2; exit 1; fi
export PYTHONPATH="$PWD/src"
mkdir -p "$OUT"
if [[ $# -eq 6 ]]; then
 INIT="$6"
 if [[ ! -f "$INIT/adapter_config.json" ]]; then
  echo "Shared adapter missing: $INIT" >&2; exit 1
 fi
else
 INIT="$OUT/shared_initial_adapter"
 python scripts/create_learning_credit_init.py --output "$INIT" --seed "$SEED"
fi

if [[ "$STAGE" == "scale" ]]; then
 ARMS="shapley direct_max audit_gate no_update"
elif [[ "$T" == "64" || "$STAGE" == "pilot" ]]; then
 ARMS="shapley direct_max audit_gate random_subset matched_random_updates no_update"
else
 ARMS="shapley direct_max no_update"
fi

# The official implementation is a T64 external baseline. Do not mislabel it
# as a group-coalition arm or include it in the narrower-T sweep comparisons.
if [[ "$T" == "64" ]]; then
 SDPO_PYTHON="${SDPO_PYTHON:-/root/ai-venvs/sdpo-official/bin/python}"
 BASELINE="sdpo_official_verl_rich64"
 SDPO_DATA="$OUT/official_sdpo_data"
 "$SDPO_PYTHON" scripts/export_sdpo_rich64.py \
  --data-dir "$DATA" --output "$SDPO_DATA" --train-limit "$TRAIN" --eval-limit "$EVAL"
 "$SDPO_PYTHON" scripts/run_sdpo_official.py check \
  --data-dir "$SDPO_DATA" --init-adapter "$INIT" \
  --output "$OUT/$BASELINE" --seed "$SEED" --rollouts "$G" \
  > "$OUT/official_sdpo_preflight.json"
fi

python scripts/run_t64_grouped_five_round.py \
 --data-dir "$DATA" --output "$OUT/shapley" --arm shapley \
 --init-adapter "$INIT" --exposed-feedback "$T" --panel-mode "$PANEL" \
 --seed "$SEED" --rounds 5 --rollouts "$G" \
 --train-limit "$TRAIN" --eval-limit "$EVAL"
python scripts/make_t64_group_schedule.py \
 --shapley-run "$OUT/shapley" --output "$OUT/reference_schedule.json"

for ARM in $ARMS; do
 if [[ "$ARM" == shapley ]]; then continue; fi
 if [[ "$ARM" == random_subset || "$ARM" == matched_random_updates ]]; then
  python scripts/run_t64_grouped_five_round.py \
   --data-dir "$DATA" --output "$OUT/$ARM" --arm "$ARM" \
   --init-adapter "$INIT" --exposed-feedback "$T" --panel-mode "$PANEL" \
   --seed "$SEED" --rounds 5 --rollouts "$G" \
   --train-limit "$TRAIN" --eval-limit "$EVAL" \
   --schedule "$OUT/reference_schedule.json"
 else
  python scripts/run_t64_grouped_five_round.py \
   --data-dir "$DATA" --output "$OUT/$ARM" --arm "$ARM" \
   --init-adapter "$INIT" --exposed-feedback "$T" --panel-mode "$PANEL" \
   --seed "$SEED" --rounds 5 --rollouts "$G" \
   --train-limit "$TRAIN" --eval-limit "$EVAL"
 fi
done

if [[ "$T" == "64" ]]; then
 "$SDPO_PYTHON" scripts/run_sdpo_official.py run \
  --data-dir "$SDPO_DATA" --init-adapter "$INIT" \
  --output "$OUT/$BASELINE" --seed "$SEED" --rollouts "$G"
 "$SDPO_PYTHON" scripts/summarize_sdpo_official.py \
  --data-dir "$SDPO_DATA" \
  --validation "$OUT/$BASELINE/validation/$((5 * TRAIN)).jsonl" \
  > "$OUT/$BASELINE/summary.json"
fi
python scripts/summarize_t64_group_five_round.py \
 --root "$OUT" --output "$OUT/paired_summary.json" --arms $ARMS

for ARM in shapley direct_max; do
 python scripts/audit_t64_group_feedback.py \
  --data-dir "$DATA" --experiment-dir "$OUT/$ARM" \
  --output "$OUT/$ARM/reserved_audit.json"
done
