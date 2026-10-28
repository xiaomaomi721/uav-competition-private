#!/usr/bin/env bash
# Layer-wise merge evaluation. Each candidate gets one D4-TTA8; Bstar included.
# Runs the same validated evaluator, so the L0 control should reproduce the
# global champion's 0.698641 through this code path.
set -uo pipefail
R=/home/jhy/rare_diag_20260915
S=$R/layerwise_20260917
O=$R/layerwise_eval_20260917
PY=/home/jhy/.venvs/uavseg/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2

SOUPS="L0_all020_ctrl L1_head000 L2_late030_head000 L3_late030_head020 L4_early030_head000 L5_head040 L6_mid030_head010"

mkdir -p "$O"
cd "$R"
rm -rf "$O"/sub_* 2>/dev/null

for s in $SOUPS; do
  if ls "$O"/${s}_Bstar.zip >/dev/null 2>&1; then echo "SKIP $s"; continue; fi
  SUB="$O/sub_$s"
  rm -rf "$SUB"
  echo "=== $s $(date +%H:%M:%S) ==="
  "$PY" -u "$R/_eval_soup_family.py" --outdir "$SUB" --soup "$S/$s.pt" --device cuda:0
  rc=$?
  if [ $rc -ne 0 ]; then echo "  FAILED rc=$rc; keeping $SUB"; continue; fi
  mv "$SUB"/*.zip "$O"/ 2>/dev/null
  cp "$SUB/EVAL_META.json" "$O/EVAL_META_$s.json" 2>/dev/null
  rm -rf "$SUB"
  echo "  ok $(date +%H:%M:%S) free: $(df -h /home/jhy | tail -1 | awk '{print $4}')"
done

echo "LAYERWISE_DONE $(date +%H:%M:%S)"
ls "$O"/*_Bstar.zip | xargs -n1 basename
