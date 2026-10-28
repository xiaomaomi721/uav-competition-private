#!/usr/bin/env bash
# Package a list of Axis S candidate checkpoints into submission zips.
# Same contract as _eval_soup_family.py / _run_bc.sh. Physical GPU2.
#
# usage: _axisS_pack.sh <outdir> <ckpt1.pt> [ckpt2.pt ...]
set -uo pipefail
R=/home/jhy/rare_diag_20260915
PY=/home/jhy/.venvs/uavseg/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2

OUT=$1; shift
mkdir -p "$OUT"

for f in "$@"; do
    s=$(basename "$f" .pt)
    if ls "$OUT"/"${s}"_Bstar.zip >/dev/null 2>&1; then echo "SKIP $s"; continue; fi
    [ -f "$f" ] || { echo "MISSING $f"; continue; }
    SUB="$OUT/sub_$s"; rm -rf "$SUB"
    echo "=== $s $(date +%H:%M:%S) ==="
    "$PY" -u "$R/_eval_soup_family.py" --outdir "$SUB" --soup "$f" --device cuda:0
    if [ $? -ne 0 ]; then echo "  FAILED $s"; rm -rf "$SUB"; continue; fi
    mv "$SUB"/*.zip "$OUT"/ 2>/dev/null
    cp "$SUB/EVAL_META.json" "$OUT/EVAL_META_$s.json" 2>/dev/null
    rm -rf "$SUB"
    echo "  ok $(date +%H:%M:%S)"
done
echo "PACK_DONE $(date +%H:%M:%S)"
