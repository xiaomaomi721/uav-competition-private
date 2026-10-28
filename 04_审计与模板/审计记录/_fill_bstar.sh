#!/usr/bin/env bash
# Fill in Bstar for the s58<->s64 line (wA/wB/wC/wD started before Bstar existed).
# wA == soupC is already known at Bstar (0.696865), so only wB/wC/wD need a forward.
set -uo pipefail
R=/home/jhy/rare_diag_20260915
S=$R/weighted_soups_20260917
O=$R/weighted_soup_eval_20260917
PY=/home/jhy/.venvs/uavseg/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2

cd "$R"
for s in wB_58_64_40_60 wC_58_64_60_40 wD_58_64_30_70; do
  if ls "$O"/${s}_Bstar.zip >/dev/null 2>&1; then echo "SKIP $s"; continue; fi
  SUB="$O/sub2_$s"
  rm -rf "$SUB"
  echo "=== $s Bstar $(date +%H:%M:%S) ==="
  "$PY" -u "$R/_eval_soup_family.py" --outdir "$SUB" --soup "$S/$s.pt" --device cuda:0
  rc=$?
  if [ $rc -ne 0 ]; then echo "  FAILED rc=$rc"; continue; fi
  mv "$SUB"/*.zip "$O"/ 2>/dev/null
  cp "$SUB/EVAL_META.json" "$O/EVAL_META_${s}_bstar.json" 2>/dev/null
  rm -rf "$SUB"
  echo "  ok $(date +%H:%M:%S) free: $(df -h /home/jhy | tail -1 | awk '{print $4}')"
done
echo "BSTAR_FILL_DONE $(date +%H:%M:%S)"
ls "$O"/*_Bstar.zip | xargs -n1 basename
