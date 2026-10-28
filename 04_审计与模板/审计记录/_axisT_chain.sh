#!/usr/bin/env bash
# ============================================================================
# Axis T unattended prep chain.
#
#   1. wait for the Phase 0 driver to finish (it owns GPU2)
#   2. REPLAY GATE: regenerate the historical pseudo500 with the ORIGINAL Exp23
#      teacher and diff it bit-for-bit against pseudo_labels_conf09/masks_official.
#      Fail -> stop everything, write a marker, generate nothing.
#   3. v1: same pipeline, teacher = champ237 (0.702574). The single variable.
#   4. assemble the Axis T dataset root (never touches pseudo_train_6796).
#
# Does NOT start training and does NOT produce any submission package.
# ============================================================================
set -uo pipefail

ROOT=/home/jhy/rare_diag_20260915
E=/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1
PY=/home/jhy/.venvs/uavseg/bin/python
LOG=$ROOT/axisT_prep_chain.log
DS=/home/jhy/datasets/2026-低空图像语义分割赛道-训练集
OUTBASE=$ROOT/axisT_20260918

exec >>"$LOG" 2>&1
echo "=== AXIS_T CHAIN START $(date '+%F %T') ==="

# ---- 1. wait for Phase 0 ----------------------------------------------------
while pgrep -f "_run_phase0.sh" >/dev/null 2>&1; do
    echo "  waiting for phase0 driver $(date '+%H:%M:%S')"
    sleep 60
done
while pgrep -f "_eval_soup_family" >/dev/null 2>&1; do sleep 30; done
echo "--- phase0 done, GPU2 free $(date '+%F %T') ---"
nvidia-smi -i 2 --query-gpu=memory.used,utilization.gpu --format=csv,noheader

# ---- 2. replay gate ---------------------------------------------------------
echo "--- REPLAY GATE $(date '+%F %T') ---"
if ! bash "$ROOT/_axisT_run.sh" replay_exp23; then
    echo "REPLAY_GATE FAIL: generator did not reproduce the historical pseudo500."
    echo "CHAIN ABORTED, v1 NOT generated."
    echo "ABORTED_REPLAY_GATE $(date '+%F %T')" > "$OUTBASE/CHAIN_STATUS.txt"
    exit 1
fi
echo "REPLAY_GATE PASS $(date '+%F %T')"

# ---- 3. v1 labels -----------------------------------------------------------
echo "--- V1 CHAMPION LABELS $(date '+%F %T') ---"
if ! bash "$ROOT/_axisT_run.sh" v1_champion; then
    echo "V1 LABEL GENERATION FAILED"
    echo "FAILED_V1_GEN $(date '+%F %T')" > "$OUTBASE/CHAIN_STATUS.txt"
    exit 2
fi

# ---- 4. assemble the Axis T dataset root ------------------------------------
echo "--- ASSEMBLE DATASET $(date '+%F %T') ---"
TSHA=$("$PY" -c "import hashlib;h=hashlib.sha256();f=open('$ROOT/donor_gate2_20260917/S_semeda_M.pt','rb');[h.update(b) for b in iter(lambda: f.read(1<<20), b'')];print(h.hexdigest())")
if ! "$PY" "$ROOT/_axisT_assemble.py" \
        --real-root     "$DS" \
        --real-ids      "$E/splits/train.txt" \
        --test-root     "$DS" \
        --pseudo-masks  "$OUTBASE/v1_champion/masks_official" \
        --out           /home/jhy/datasets/pseudo_train_axisT_v1 \
        --label-tag     v1_champion \
        --teacher-sha256 "$TSHA"; then
    echo "DATASET ASSEMBLY FAILED"
    echo "FAILED_ASSEMBLE $(date '+%F %T')" > "$OUTBASE/CHAIN_STATUS.txt"
    exit 3
fi

echo "READY_TRAIN $(date '+%F %T')" > "$OUTBASE/CHAIN_STATUS.txt"
echo "=== AXIS_T CHAIN DONE, ready to train $(date '+%F %T') ==="
