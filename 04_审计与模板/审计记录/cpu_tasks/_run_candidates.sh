#!/usr/bin/env bash
# ============================================================================
# Evaluate the 15 pre-built CPU candidates on GPU2 (D4 TTA8 + Bstar package).
# Order = priority: D (alpha narrow) -> E (late extrap) -> F (layerwise) -> C (EMA).
# Each candidate is a complete state dict; we use --eval-only (no composition).
# ============================================================================
set -uo pipefail

ATDA=/home/jhy/rare_diag_20260915/atda_20260918
SNAP=/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1
PY=/home/jhy/.venvs/uavseg/bin/python
CAND=$ATDA/cpu_candidates
OUT=$ATDA/candidate_eval
LOG=$ATDA/candidate_eval.log

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SNAP:/home/jhy/gpu2_day_queue"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
export UAVSEG_MODEL_SOURCE=/home/jhy/models/segformer-b3-ade512
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SNAP" || exit 9
mkdir -p "$OUT"

exec >>"$LOG" 2>&1
echo "=== CANDIDATE EVAL START $(date '+%F %T') ==="

# order = priority
for tag in \
  d_s8k_a1p1 d_s8k_a1p15 d_s8k_a1p2 d_s8k_a1p3 d_s8k_a1p35 \
  e_X025 e_X050 \
  f_s2p125_s3p100 f_s2p100_s3p125 f_s2p125_s3p125 f_s2p110_s3p135 \
  c_ema8k_direct c_05s_05e c_075s_025e c_261p125ema ; do
  ckpt="$CAND/$tag.pt"
  if [ ! -f "$ckpt" ]; then echo "MISSING $ckpt"; continue; fi
  if [ -f "$OUT/${tag}_Bstar.zip" ]; then echo "SKIP $tag"; continue; fi
  sub="$OUT/sub_$tag"
  rm -rf "$sub"; mkdir -p "$sub"
  echo "=== $tag $(date '+%H:%M:%S') ==="
  "$PY" -u "$ATDA/_atda_export.py" \
      --outdir "$sub" --eval-only "$ckpt" --tag "$tag" --device cuda:0
  rc=$?
  if [ $rc -ne 0 ]; then echo "  FAILED $tag rc=$rc"; continue; fi
  mv "$sub"/${tag}_Bstar.zip "$OUT"/ 2>/dev/null
  cp "$sub/EXPORT_META.json" "$OUT/EXPORT_META_${tag}.json" 2>/dev/null
  rm -rf "$sub"
  echo "  ok $tag $(date '+%H:%M:%S')"
done
echo "=== CANDIDATE EVAL DONE $(date '+%F %T') ==="
ls -la "$OUT"/*.zip 2>/dev/null
