#!/usr/bin/env bash
# ============================================================================
# Axis S unattended launch.
#   wait for Phase 0 and the Axis T chain to release GPU2
#   -> 20-step SMOKE (verifies init-from, stage2-only scope, frozen-buffer
#      snapshot, SEMEDA criterion, and measures real s/step)
#   -> real run: p1_semeda_s2_init209, 16k horizon, gates 1k/2k/4k/8k/16k
#
# Deviations from the frozen P1 spec, both forced by the hard GPU deadline:
#   * horizon 32k -> 16k, so the run completes its own lr schedule instead of
#     being cut mid-decay.
#   * dual start (209 vs 237) -> single arm A (209). One run's delta serves both
#     compositions (theta209+... and theta237+lambda*delta), halving GPU cost.
# ============================================================================
set -uo pipefail

ROOT=/home/jhy/rare_diag_20260915
E=/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1
PY=/home/jhy/.venvs/uavseg/bin/python
DS=/home/jhy/datasets/2026-低空图像语义分割赛道-训练集
LOG=$ROOT/axisS_launch.log
RUN=$ROOT/p1_runs
SMOKE=$RUN/_smoke_init209
REAL=$RUN/p1_semeda_s2_init209_seed42

exec >>"$LOG" 2>&1
echo "=== AXIS_S LAUNCH START $(date '+%F %T') ==="

while pgrep -f "_run_phase0.sh"  >/dev/null 2>&1; do sleep 60; done
while pgrep -f "_eval_soup_family" >/dev/null 2>&1; do sleep 30; done
while pgrep -f "_axisT_chain.sh"  >/dev/null 2>&1; do sleep 30; done
while pgrep -f "_axisT_run.sh"    >/dev/null 2>&1; do sleep 30; done
echo "--- GPU2 free $(date '+%F %T') ---"
nvidia-smi -i 2 --query-gpu=memory.used,utilization.gpu --format=csv,noheader

mkdir -p "$RUN"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$E"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 UAVSEG_PHYSICAL_GPU_ID=2
export UAVSEG_MODEL_SOURCE=/home/jhy/models/segformer-b3-ade512
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$E" || exit 9

# ---- smoke ----------------------------------------------------------------
rm -rf "$SMOKE"
SMOKELOG=$RUN/_smoke_init209.log
echo "--- SMOKE $(date '+%F %T') ---"
"$PY" -u scripts/_p1_surgical_train.py \
    --config configs/p1/p1_semeda_s2_init209.yaml \
    --dataset-root "$DS" --output-dir "$SMOKE" \
    --device cuda:0 --max-steps 20 --max-val-samples 8 > "$SMOKELOG" 2>&1
tail -20 "$SMOKELOG"

# The P1 SURGICAL lines go to stdout, so they must be read from the captured
# log -- not from the output dir (2026-09-18: reading $SMOKE made a passing
# smoke look like a failure and cost a launch).
if ! grep -q "P1 SURGICAL init_from=" "$SMOKELOG"; then
    echo "SMOKE FAILED: no init_from line. NOT starting the real run."
    echo "SMOKE_FAILED $(date '+%F %T')" > "$RUN/LAUNCH_STATUS.txt"
    exit 1
fi
if ! grep -q "P1 SURGICAL scope=" "$SMOKELOG"; then
    echo "SMOKE FAILED: no trainable-scope line. NOT starting the real run."
    echo "SMOKE_FAILED_SCOPE $(date '+%F %T')" > "$RUN/LAUNCH_STATUS.txt"
    exit 1
fi
echo "--- SMOKE PASS $(date '+%F %T') ---"

# ---- real run -------------------------------------------------------------
if [ -e "$REAL" ] && [ -n "$(ls -A "$REAL" 2>/dev/null)" ]; then
    echo "OUTPUT_COLLISION $REAL"; echo "COLLISION" > "$RUN/LAUNCH_STATUS.txt"; exit 3
fi
echo "--- REAL RUN START $(date '+%F %T') ---"
"$PY" -u scripts/_p1_surgical_train.py \
    --config configs/p1/p1_semeda_s2_init209.yaml \
    --dataset-root "$DS" --output-dir "$REAL" \
    --device cuda:0 2>&1

echo "REAL_RUN_EXIT=$? $(date '+%F %T')" | tee "$RUN/LAUNCH_STATUS.txt"
echo "=== AXIS_S LAUNCH END $(date '+%F %T') ==="
