#!/usr/bin/env bash
# One-seed six-arm original Dirichlet EED x token granularity ablation.
# GPU launch only. This does NOT perform final held-out evaluation.
set -euo pipefail
if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo "Usage: bash scripts/run_submitted_eed_token_factorial.sh CORRECTIONS.jsonl OUT_DIR MODEL_CONFIG SEED RESPONSE_TOKEN_BUDGET SHARED_INIT_ADAPTER [EVAL_IDS.jsonl]" >&2
  exit 2
fi

SRC="$1"
OUT="$2"
MODEL_CONFIG="$3"
SEED="$4"
BUDGET="$5"
INIT="$6"

if [[ -e "$OUT" ]]; then echo "Refuse overwrite: $OUT" >&2; exit 1; fi
if [[ ! -f "$SRC" || ! -f "$MODEL_CONFIG" || ! -f "$INIT/adapter_config.json" ]]; then
  echo "Missing correction bank, pinned model config or frozen shared adapter" >&2
  exit 1
fi
if [[ ! "$SEED" =~ ^[0-9]+$ || ! "$BUDGET" =~ ^[1-9][0-9]*$ ]]; then
  echo "SEED and BUDGET must be positive integers" >&2; exit 2
fi

export PYTHONPATH="$PWD/src"
mkdir -p "$OUT"
if [[ $# -eq 7 ]]; then
  python scripts/audit_eesd_token_aware_inputs.py --corrections "$SRC" \
    --output "$OUT/origin_audit.json" --require-complete --evaluation-ids "$7"
else
  python scripts/audit_eesd_token_aware_inputs.py --corrections "$SRC" \
    --output "$OUT/origin_audit.json" --require-complete
fi
python scripts/prepare_submitted_dirichlet_rebuttal_weights.py \
  --input "$SRC" --output "$OUT/four_state_weights" \
  --alpha 0.1 --lambda-penalty 0.5 --seed "$SEED"
python scripts/materialize_submitted_dirichlet_training_inputs.py \
  --weights "$OUT/four_state_weights/weights.jsonl" \
  --output "$OUT/matched_inputs" \
  --arms equal_weight fixed_mass adaptive_eed

# All arms share the exact initial LoRA adapter, input ancestry, KL, optimizer,
# correction order and target-response-token budget. The --rule name is a common
# anchored execution SLOT, not a claim about the scientific weight strategy.
for row in \
  "equal_uniform|equal_weight|uniform" \
  "fixed_uniform|fixed_mass|uniform" \
  "adaptive_uniform|adaptive_eed|uniform" \
  "fixed_edit|fixed_mass|edit" \
  "adaptive_edit|adaptive_eed|edit" \
  "adaptive_random|adaptive_eed|random"
do
  IFS='|' read -r name evidence granularity <<< "$row"
  echo "===== $name: $evidence x $granularity ====="
  python scripts/run_eesd_weighted_sft.py \
    --input "$OUT/matched_inputs/$evidence/scored-corrections.jsonl" \
    --model-config "$MODEL_CONFIG" \
    --rule eesd_full --include-zero-weight-kl \
    --token-weighting "$granularity" --token-edit-boost 2.0 \
    --previous-adapter "$INIT" \
    --seed "$SEED" --response-token-budget "$BUDGET" \
    --anchor-beta 0.03 \
    --output "$OUT/models/$name"
done

python scripts/check_submitted_eed_token_factorial.py \
  --root "$OUT/models" --output "$OUT/training_parity.json"

echo "Six-arm training and parity checks finished. No heldout accuracy has been measured."
