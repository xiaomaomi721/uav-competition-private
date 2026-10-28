#!/usr/bin/env bash
# Layer-wise round 2 evaluation. Sequential, Bstar included.
set -uo pipefail
R=/home/jhy/rare_diag_20260915
S=$R/layerwise2_20260917
O=$R/layerwise2_eval_20260917
PY=/home/jhy/.venvs/uavseg/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2

SOUPS="M0_ctrl_L4 M1_early040 M6_monotone M4_late000 M2_early050 M5_early040_late000 M7_mid030 M3_early060"

mkdir -p "$O"
cd "$R"
for s in $SOUPS; do
  [ -z "$s" ] && continue
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
echo "LAYERWISE2_DONE $(date +%H:%M:%S)"
ls "$O"/*_Bstar.zip | xargs -n1 basename
