#!/usr/bin/env bash
# Round-2 coefficient search: extend the three live parameter rays and test the
# two-member combination. Sequential, Bstar included (the evaluator now emits it).
set -uo pipefail
R=/home/jhy/rare_diag_20260915
S=$R/round2_soups_20260917
O=$R/round2_eval_20260917
PY=/home/jhy/.venvs/uavseg/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2

SOUPS="r2_55b_a30 r2_55b_a40 r2_62_a20 r2_62_a30 r2_23b_a50 r2_55b_a50 r2_55b_23b_25_25 r2_55b_23b_35_15"

mkdir -p "$O"
cd "$R"

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

echo "ROUND2_DONE $(date +%H:%M:%S)"
ls "$O"/*_Bstar.zip | xargs -n1 basename
