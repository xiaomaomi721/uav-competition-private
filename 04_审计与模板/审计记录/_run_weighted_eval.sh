#!/usr/bin/env bash
# GPU main line: evaluate the weighted soup grid, one D4-TTA8 each.
#
# soupC (equal 50/50 of s58+s64) scored 0.696818 and beat both the old
# single-model record (0.693096) and the illegal logit-ensemble reference
# (0.695637). Only two member sets had ever been tried, both EQUAL-weighted.
# The parameter-averaging map is non-linear enough (soupB kept 68% of its
# members' mean response, soupC kept 93%) that the weights are a real axis.
#
# _eval_soup_family.py refuses an existing --outdir, so each soup gets its own
# fresh directory; the packages are moved up afterwards and the 16 GB score
# cache is deleted so ten caches never coexist on a ~200 GB filesystem.
set -uo pipefail
R=/home/jhy/rare_diag_20260915
S=$R/weighted_soups_20260917
O=$R/weighted_soup_eval_20260917
PY=/home/jhy/.venvs/uavseg/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2

SOUPS=${SOUPS:-"wA_58_64_50_50 wB_58_64_40_60 wC_58_64_60_40 wD_58_64_30_70 wE_58_64_23b_45_45_10 wF_58_64_23b_40_40_20 wG_58_64_55b_45_45_10 wH_58_64_55b_40_40_20 wI_58_64_62_45_45_10 wJ_58_64_23b_33each"}

mkdir -p "$O"
cd "$R"

for s in $SOUPS; do
  if ls "$O"/${s}_ZERO.zip >/dev/null 2>&1; then echo "SKIP $s: already packed"; continue; fi
  SUB="$O/sub_$s"
  if [ -e "$SUB" ]; then rm -rf "$SUB"; fi
  echo "=== $s  $(date +%H:%M:%S) ==="
  "$PY" -u "$R/_eval_soup_family.py" --outdir "$SUB" --soup "$S/$s.pt" --device cuda:0
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "  FAILED $s rc=$rc; keeping $SUB for inspection"
    continue
  fi
  mv "$SUB"/*.zip "$O"/ 2>/dev/null
  cp "$SUB/EVAL_META.json" "$O/EVAL_META_$s.json" 2>/dev/null
  rm -rf "$SUB"
  echo "  ok $(date +%H:%M:%S)  free: $(df -h /home/jhy | tail -1 | awk '{print $4}')"
done

echo "WEIGHTED_EVAL_DONE $(date +%H:%M:%S)"
ls -la "$O"/*.zip | awk '{print $5, $9}'
