#!/usr/bin/env bash
# Wait for teacher cache, smoke-test C1, then launch both arms in parallel.
set -u
W=/home/jhy/aic2026_h76_20261026/code/reliable_kd_20261028
PY=/home/jhy/.venvs/uavseg/bin/python
cd "$W" || exit 1
mkdir -p logs

echo "[chain] waiting for teacher cache COMPLETE.json"
while [ ! -f "$W/teacher_d4/COMPLETE.json" ]; do
  if ! pgrep -f rkd_cache_teacher >/dev/null 2>&1; then
    echo "[chain] CACHE_EXITED_WITHOUT_COMPLETE"; exit 2
  fi
  sleep 30
done
echo "[chain] cache complete"

rm -rf "$W/train/C1_smoke"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 EXPECTED_GPU_INDEX=1 \
  "$PY" rkd_train.py --arm C1 --smoke 5 > logs/smoke_c1.log 2>&1
rc=$?
echo "[chain] smoke rc=$rc"
if [ "$rc" -ne 0 ]; then
  echo "[chain] SMOKE_FAILED"; tail -5 logs/smoke_c1.log; exit 3
fi
rm -rf "$W/train/C1_smoke"

echo "[chain] launching C0 on GPU0"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 EXPECTED_GPU_INDEX=0 \
  nohup "$PY" rkd_train.py --arm C0 --steps 3594 > logs/train_c0.log 2>&1 &
echo "C0_PID=$!"
echo "[chain] launching C1 on GPU1"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 EXPECTED_GPU_INDEX=1 \
  nohup "$PY" rkd_train.py --arm C1 --steps 3594 > logs/train_c1.log 2>&1 &
echo "C1_PID=$!"
echo "[chain] launched"
