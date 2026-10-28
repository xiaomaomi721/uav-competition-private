#!/usr/bin/env bash
# Eval + pack for the two M2F LR-reopen arms.
#   - re-pack iter76000 from its existing score cache (no TTA8)
#   - forward + pack iter82000
# Sequential; each stage is idempotent (existing good zips are skipped).
set -uo pipefail
R=/home/jhy/rare_diag_20260915
O=$R/lrreopen_eval_20260917
M=$R/m2f_gpu_20260915
PY=/home/jhy/.venvs/uavseg/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2

cd "$R"
rm -f "$O"/iter76000_ZERO.zip "$O"/iter76000_FG020.zip

echo "=== stage 1: repack iter76000 from cache ==="
"$PY" -u "$R/_m2f_lrreopen_eval.py" --outdir "$O" --pack-only "$O" --device cuda:0
echo "stage1 rc=$?"

echo "=== stage 2: forward+pack iter82000 ==="
"$PY" -u "$R/_m2f_lrreopen_eval.py" --outdir "$O" --pack-only "$O" \
  --checkpoint "$M/lrreopen_B_20260917/scores/score_iter82000.pt" --device cuda:0
echo "stage2 rc=$?"

echo "=== final ==="
ls -la "$O"/*.zip | awk '{print $5, $9}'
echo ALL_DONE
