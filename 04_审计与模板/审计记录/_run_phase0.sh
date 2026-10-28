#!/usr/bin/env bash
# Phase 0 evaluation driver: 10 candidates -> TTA8 -> ZERO/C/13/Bstar packages.
# Same contract as _run_bc.sh. Physical GPU2 only.
set -uo pipefail
R=/home/jhy/rare_diag_20260915
PY=/home/jhy/.venvs/uavseg/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2

OUT=$R/phase0_eval_20260918
CAND=$R/phase0_candidates_20260918
STAGE=$R/phase0_stage_20260918
mkdir -p "$OUT"

for f in "$CAND/A_cleanSEM_M_k050_th209.pt" \
         "$CAND/A_cleanSEM_M_k100_th209.pt" \
         "$CAND/B_nodK_D_k100_th237.pt" \
         "$CAND/B_nodK_M_k100_th237.pt" \
         "$CAND/B_nodK_E_k100_th237.pt" \
         "$CAND/B_nodK_H_k100_th237.pt" \
         "$CAND/C_rareExp_E_k100_th237.pt" \
         "$CAND/C_rareExp_M_k100_th237.pt" \
         "$CAND/C_rareExp_D_k100_th237.pt" \
         "$STAGE/RE_v2_standalone.pt" ; do
  s=$(basename "$f" .pt)
  if ls "$OUT"/"${s}"_Bstar.zip >/dev/null 2>&1; then echo "SKIP $s"; continue; fi
  if [ ! -f "$f" ]; then echo "MISSING $f"; continue; fi
  SUB="$OUT/sub_$s"; rm -rf "$SUB"
  echo "=== $s $(date +%H:%M:%S) ==="
  "$PY" -u "$R/_eval_soup_family.py" --outdir "$SUB" --soup "$f" --device cuda:0
  if [ $? -ne 0 ]; then echo "  FAILED $s"; rm -rf "$SUB"; continue; fi
  mv "$SUB"/*.zip "$OUT"/ 2>/dev/null
  cp "$SUB/EVAL_META.json" "$OUT/EVAL_META_$s.json" 2>/dev/null
  rm -rf "$SUB"
  echo "  ok $(date +%H:%M:%S)"
done
echo "PHASE0_EVAL_DONE $(date +%H:%M:%S)"
