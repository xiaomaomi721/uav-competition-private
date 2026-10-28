#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?mode smoke|formal}"
OUT="${2:?output directory}"
STEPS="${3:?max steps}"
ROOT=/home/jhy/rare_diag_20260915/tekd_final_20260925
TRAINER="$ROOT/code/_r2_tekd_train.py"
PY=/home/jhy/.venvs/uavseg/bin/python
LOCK=/home/jhy/gpu3_tekd_final.lock
EXPECTED_UUID=GPU-32853ccf-4c4f-1328-8d75-04efe9d5298f
EXPECTED_BUS=00000000:CA:00.0

if [[ "$MODE" != smoke && "$MODE" != formal ]]; then
  echo "bad mode: $MODE" >&2; exit 2
fi
if [[ "$MODE" == smoke && "$STEPS" != 4 ]]; then
  echo "smoke requires 4 steps" >&2; exit 2
fi
if [[ "$MODE" == formal && "$STEPS" != 1000 ]]; then
  echo "formal requires 1000 steps" >&2; exit 2
fi
if [[ -e "$OUT" ]]; then
  echo "output already exists: $OUT" >&2; exit 3
fi
[[ -f "$TRAINER" ]] || { echo "missing trainer" >&2; exit 4; }

row=$(nvidia-smi --query-gpu=index,uuid,pci.bus_id,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '$1==3 {print}')
[[ "$row" == *"$EXPECTED_UUID"* && "$row" == *"$EXPECTED_BUS"* ]] || { echo "GPU3 identity mismatch: $row" >&2; exit 5; }
compute=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$EXPECTED_UUID" '$1==u {print}')
[[ -z "$compute" ]] || { echo "GPU3 occupied: $compute" >&2; exit 6; }
avail_kb=$(df -Pk /home/jhy/rare_diag_20260915 | awk 'NR==2 {print $4}')
(( avail_kb >= 12*1024*1024 )) || { echo "disk gate: only ${avail_kb} KiB free" >&2; exit 7; }

exec 9>"$LOCK"
flock -n 9 || { echo "TEKD GPU3 queue locked" >&2; exit 8; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
exec "$PY" "$TRAINER" --mode "$MODE" --output-dir "$OUT" --max-steps "$STEPS" --device cuda:0
