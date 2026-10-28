#!/usr/bin/env bash
# ============================================================================
# ATDA-71 unattended launch.
#
#   gate 1: GPU2 must be free (no foreign compute process above the idle floor)
#   gate 2: exactly ONE 200-step smoke; all nine checks must pass
#   gate 3: the real run starts immediately after, no second validator
#
# The smoke output directory is never reused: a passing smoke leaves its
# SMOKE_REPORT.json behind as the acceptance record.
#
# Physical GPU2 only. CUDA_VISIBLE_DEVICES=2, CUDA_DEVICE_ORDER=PCI_BUS_ID.
# ============================================================================
set -uo pipefail

ROOT=/home/jhy/rare_diag_20260915
ATDA=$ROOT/atda_20260918
SNAP=/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1
PY=/home/jhy/.venvs/uavseg/bin/python
CFG=$SNAP/configs/atda/atda_s2s3.yaml
SRC_IDS=${ATDA_SRC_IDS:-$ATDA/source_ids_5597.txt}
SRC_ROOT=${ATDA_SRC_ROOT:-/home/jhy/datasets/2026-低空图像语义分割赛道-训练集}
LOG=$ATDA/launch.log

SMOKE=$ATDA/smoke_200
REAL=$ATDA/run_s2s3_seed42

exec >>"$LOG" 2>&1
echo "=== ATDA LAUNCH START $(date '+%F %T') ==="

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SNAP:/home/jhy/gpu2_day_queue"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
export UAVSEG_MODEL_SOURCE=/home/jhy/models/segformer-b3-ade512
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SNAP" || exit 9

# ---- gate 1: nobody else is on GPU2 ---------------------------------------
USED=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
echo "GPU2 memory used: ${USED} MiB"
if [ "${USED:-99999}" -gt 2000 ]; then
    echo "ABORT: GPU2 is not free (${USED} MiB in use). Not starting."
    exit 2
fi
# The bracket keeps this test from matching the launcher's own command line.
if pgrep -f "[_]atda_train[.]py" >/dev/null 2>&1; then
    echo "ABORT: an ATDA trainer is already running."
    pgrep -af "[_]atda_train[.]py"
    exit 2
fi
if [ ! -s "$SRC_IDS" ]; then
    echo "ABORT: source id list missing or empty: $SRC_IDS"
    exit 2
fi
if [ "$(wc -l < "$SRC_IDS")" -ne 5597 ]; then
    echo "ABORT: source id list must hold exactly 5597 stems, has $(wc -l < "$SRC_IDS")"
    exit 2
fi

# ---- gate 2: the single 200-step smoke ------------------------------------
rm -rf "$SMOKE"
echo "--- SMOKE 200 steps $(date '+%F %T') ---"
"$PY" -u "$ATDA/_atda_train.py" \
    --config "$CFG" \
    --source-root "$SRC_ROOT" \
    --source-ids "$SRC_IDS" \
    --output-dir "$SMOKE" \
    --device cuda:0 \
    --max-steps 200 --gate-steps 200 --smoke
SMOKE_RC=$?
echo "SMOKE exit code $SMOKE_RC $(date '+%F %T')"

if [ $SMOKE_RC -ne 0 ]; then
    echo "SMOKE FAILED (rc=$SMOKE_RC). The long run is NOT started."
    exit 3
fi
if [ ! -f "$SMOKE/SMOKE_REPORT.json" ]; then
    echo "SMOKE produced no report. Not starting the long run."
    exit 3
fi
if ! grep -q '"pass": true' "$SMOKE/SMOKE_REPORT.json"; then
    echo "SMOKE report says pass=false. Not starting the long run."
    grep -A3 '"failed"' "$SMOKE/SMOKE_REPORT.json"
    exit 3
fi
SPS=$("$PY" -c "import json;print(json.load(open('$SMOKE/ATDA_RUN_SUMMARY.json'))['sec_per_step'])")
echo "SMOKE PASSED. sec/step=$SPS -> 8000 steps ~= $( "$PY" -c "print(round(8000*float('$SPS')/3600,2))" ) h"

# ---- gate 3: the real run --------------------------------------------------
if [ -d "$REAL" ]; then
    echo "ABORT: $REAL already exists; refusing to overwrite a real run."
    exit 2
fi
echo "--- REAL RUN 8000 steps $(date '+%F %T') ---"
"$PY" -u "$ATDA/_atda_train.py" \
    --config "$CFG" \
    --source-root "$SRC_ROOT" \
    --source-ids "$SRC_IDS" \
    --output-dir "$REAL" \
    --device cuda:0 \
    --max-steps 8000 --gate-steps 1000 2000 4000 6000 8000
echo "REAL RUN exit code $? $(date '+%F %T')"
echo "=== ATDA LAUNCH DONE $(date '+%F %T') ==="
