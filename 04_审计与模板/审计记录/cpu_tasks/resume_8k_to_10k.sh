#!/usr/bin/env bash
# ============================================================================
# ATDA-71 EXACT RESUME 8k -> 10k (READY, DO NOT RUN without GPU2 access).
#
# Resumes from run_s2s3_base261/checkpoint_step8000.pt and continues to the
# original 10k horizon. total_steps stays 10000 so the polynomial LR schedule
# continues from where it stopped (lr ~2.0e-6 at step 8000 -> 0 at 10000).
#
# Resume completeness verified (see cpu_resume/RESUME_READINESS.json):
#   - model  (644 tensors) + ema (644)  : yes
#   - optimizer (462 states, lr recorded): yes
#   - scaler (scale=131072)              : yes
#   - global step (8000)                 : yes
#   - config (incl scheduler total_steps=10000) : yes
#   - RNG / dataloader state             : NOT saved (matches the original
#     trainer; resume re-seeds, so the source/target stream order differs from
#     a hypothetical uninterrupted run -- this is the ONLY non-exact part)
#
# The ATDA trainer is stateless per-process: it rebuilds student/ref/EMA from
# theta261, then applies the checkpoint. To resume we need the trainer to accept
# --resume; this is the ONE missing trainer capability, flagged in
# RESUME_READINESS.json as "trainer_resume_flag_needed".
# ============================================================================
set -uo pipefail

ATDA=/home/jhy/rare_diag_20260915/atda_20260918
SNAP=/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1
PY=/home/jhy/.venvs/uavseg/bin/python
CFG=$SNAP/configs/atda/atda_s2s3.yaml
SRC_IDS=$ATDA/source_ids_5597.txt
SRC_ROOT=/home/jhy/datasets/2026-低空图像语义分割赛道-训练集
CKPT=$ATDA/run_s2s3_base261/checkpoint_step8000.pt
OUT=$ATDA/run_resume_10k

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SNAP:/home/jhy/gpu2_day_queue"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
export UAVSEG_MODEL_SOURCE=/home/jhy/models/segformer-b3-ade512
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SNAP" || exit 9

echo "RESUME 8k->10k from $CKPT"
"$PY" -u "$ATDA/_atda_train.py" \
    --config "$CFG" \
    --source-root "$SRC_ROOT" \
    --source-ids "$SRC_IDS" \
    --output-dir "$OUT" \
    --device cuda:0 \
    --max-steps 10000 \
    --gate-steps 10000 \
    --resume "$CKPT"
echo "RESUME exit $?"
