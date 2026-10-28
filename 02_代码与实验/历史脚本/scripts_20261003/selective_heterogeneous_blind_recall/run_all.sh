#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${HERE}/frozen_config.json"
RUN_ID="${SHBR_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="/data/uav/shbr_20261003_${RUN_ID}"

if [[ -e "${RUN_ROOT}" ]]; then
  printf 'Refusing to overwrite existing run directory: %s\n' "${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}/00_preflight"

json_status() {
  python - "$1" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding='utf-8'))['status'])
PY
}

finalize_stop() {
  local status="$1"
  local reason="$2"
  if [[ ! -e "${RUN_ROOT}/FINAL_DECISION.json" ]]; then
    python "${HERE}/finalize_decision.py" --run-root "${RUN_ROOT}" --status "${status}" --reason "${reason}" || true
  fi
}

on_error() {
  local code=$?
  local line=${BASH_LINENO[0]:-unknown}
  trap - ERR
  finalize_stop "STOP_INTEGRITY" "Command failed at run_all.sh line ${line}; inspect the preceding terminal output."
  exit "${code}"
}
trap on_error ERR

printf 'SHBR run root: %s\n' "${RUN_ROOT}"
python -m py_compile "${HERE}"/*.py
python "${HERE}/test_shbr_contracts.py"

python "${HERE}/freeze_assets.py" \
  --config "${CONFIG}" \
  --output "${RUN_ROOT}/00_preflight/ARTIFACT_RECEIPT.json"

GPU_INDEX="$(python - "${RUN_ROOT}/00_preflight/ARTIFACT_RECEIPT.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding='utf-8'))['selected_gpu']['index'])
PY
)"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

mkdir -p "${RUN_ROOT}/01_source_cache"
for model in a05 armb donor; do
  python "${HERE}/build_prediction_cache.py" \
    --config "${CONFIG}" \
    --preflight "${RUN_ROOT}/00_preflight/ARTIFACT_RECEIPT.json" \
    --scope source \
    --model "${model}" \
    --output-dir "${RUN_ROOT}/01_source_cache/source_${model}"
done

python "${HERE}/discover_and_validate_gate.py" \
  --config "${CONFIG}" \
  --cache-root "${RUN_ROOT}/01_source_cache" \
  --output-dir "${RUN_ROOT}/02_gate"

SOURCE_STATUS="$(json_status "${RUN_ROOT}/02_gate/GATE_DECISION.json")"
if [[ "${SOURCE_STATUS}" != "PROCEED_SOURCE_GATE" ]]; then
  finalize_stop "${SOURCE_STATUS}" "The frozen donor rule did not pass the source complementarity gate."
  exit 0
fi

mkdir -p "${RUN_ROOT}/03_target_predictions"
for model in armb donor; do
  python "${HERE}/build_prediction_cache.py" \
    --config "${CONFIG}" \
    --preflight "${RUN_ROOT}/00_preflight/ARTIFACT_RECEIPT.json" \
    --scope target \
    --model "${model}" \
    --output-dir "${RUN_ROOT}/03_target_predictions/target_${model}"
done

python "${HERE}/build_target_cache.py" \
  --config "${CONFIG}" \
  --cache-root "${RUN_ROOT}/03_target_predictions" \
  --frozen-gate "${RUN_ROOT}/02_gate/FROZEN_GATE.json" \
  --source-decision "${RUN_ROOT}/02_gate/GATE_DECISION.json" \
  --output-dir "${RUN_ROOT}/03_target_support"

TARGET_STATUS="$(json_status "${RUN_ROOT}/03_target_support/TARGET_SUPPORT.json")"
if [[ "${TARGET_STATUS}" != "PROCEED_TARGET_SUPPORT" ]]; then
  finalize_stop "${TARGET_STATUS}" "The frozen source rule is outside its pre-registered test2 support range."
  exit 0
fi

python "${HERE}/calibrate_fixed_beta.py" \
  --config "${CONFIG}" \
  --preflight "${RUN_ROOT}/00_preflight/ARTIFACT_RECEIPT.json" \
  --target-cache "${RUN_ROOT}/03_target_support" \
  --output "${RUN_ROOT}/03_target_support/FIXED_BETA_RECEIPT.json"

BETA_STATUS="$(json_status "${RUN_ROOT}/03_target_support/FIXED_BETA_RECEIPT.json")"
if [[ "${BETA_STATUS}" != "PROCEED_TREATMENT_SCALE" ]]; then
  finalize_stop "${BETA_STATUS}" "The fixed beta=0.25 treatment is too weak or too strong; beta was not retuned."
  exit 0
fi

mkdir -p "${RUN_ROOT}/04_training"
for role in control treat; do
  python "${HERE}/train_shbr.py" \
    --config "${CONFIG}" \
    --preflight "${RUN_ROOT}/00_preflight/ARTIFACT_RECEIPT.json" \
    --target-cache "${RUN_ROOT}/03_target_support" \
    --beta-receipt "${RUN_ROOT}/03_target_support/FIXED_BETA_RECEIPT.json" \
    --role "${role}" \
    --max-steps 4 \
    --output-dir "${RUN_ROOT}/04_training/smoke_${role}"
done

python "${HERE}/train_shbr.py" \
  --config "${CONFIG}" \
  --preflight "${RUN_ROOT}/00_preflight/ARTIFACT_RECEIPT.json" \
  --target-cache "${RUN_ROOT}/03_target_support" \
  --beta-receipt "${RUN_ROOT}/03_target_support/FIXED_BETA_RECEIPT.json" \
  --role treat \
  --max-steps 2000 \
  --output-dir "${RUN_ROOT}/04_training/formal_treat"

python "${HERE}/evaluate_source_health.py" \
  --config "${CONFIG}" \
  --preflight "${RUN_ROOT}/00_preflight/ARTIFACT_RECEIPT.json" \
  --treat-checkpoint "${RUN_ROOT}/04_training/formal_treat/checkpoint_step2000.pt" \
  --output "${RUN_ROOT}/04_training/SOURCE_HEALTH.json"

HEALTH_STATUS="$(json_status "${RUN_ROOT}/04_training/SOURCE_HEALTH.json")"
if [[ "${HEALTH_STATUS}" != "PASS_SOURCE_HEALTH" ]]; then
  finalize_stop "${HEALTH_STATUS}" "The formal TREAT failed the source catastrophic-collapse guard."
  exit 0
fi

python "${HERE}/package_and_audit.py" \
  --config "${CONFIG}" \
  --preflight "${RUN_ROOT}/00_preflight/ARTIFACT_RECEIPT.json" \
  --checkpoint "${RUN_ROOT}/04_training/formal_treat/checkpoint_step2000.pt" \
  --source-health "${RUN_ROOT}/04_training/SOURCE_HEALTH.json" \
  --output-dir "${RUN_ROOT}/05_package"

PACKAGE_STATUS="$(json_status "${RUN_ROOT}/05_package/PACKAGE_MANIFEST.json")"
if [[ "${PACKAGE_STATUS}" != "READY_FOR_FINALIZATION" ]]; then
  finalize_stop "${PACKAGE_STATUS}" "The final package was identical to CONTROL or exceeded the behavior outlier bounds."
  exit 0
fi

python "${HERE}/finalize_decision.py" \
  --run-root "${RUN_ROOT}" \
  --status "READY_FOR_MANUAL_PAIRED_SUBMISSION" \
  --reason "All pre-registered engineering and donor-value gates passed. Online score sign remains unknown."

printf '\nFinished. Read only these two top-level files first:\n'
printf '  %s\n' "${RUN_ROOT}/FINAL_DECISION.json"
printf '  %s\n' "${RUN_ROOT}/给队友看的最终结论.md"
printf 'No platform submission was performed.\n'
