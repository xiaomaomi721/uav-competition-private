#!/usr/bin/env bash
# Rebuild the 3-teacher soft-target cache that the 2026-09-16 disk cleanup
# deleted (kd_soft_fp16/probs, 500 x .npy, 7.81 GiB -- see 磁盘清理_20260916.txt
# item 2). Writes to a NEW directory so nothing existing is touched; the result
# is verified against the surviving PROBS_MANIFEST.json before it is put back.
set -uo pipefail
R=/home/jhy/rare_diag_20260915
PY=/home/jhy/.venvs/uavseg/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2

OUT="$R/kd_soft_fp16_rebuild"
if [ -e "$OUT" ]; then echo "refuse: $OUT already exists"; exit 1; fi

cd "$R"
setsid nohup "$PY" "$R/_kd_precompute_teachers.py" \
  --outdir "$OUT" \
  --names "$R/test_names.txt" \
  --test-dir "/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/images" \
  --teachers "$R/kd_teachers.json" \
  --temperature 2.0 --store fp16 --resolution native --device cuda:0 \
  > "$R/kd_soft_fp16_rebuild.log" 2>&1 < /dev/null &

echo "launched rebuild (pid $!)"
