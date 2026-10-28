#!/usr/bin/env bash
# Disposable smoke for _kd_train_soup_gate.py -- ~100 optimizer updates.
#
# Purpose is ONLY to confirm the gate-patched trainer still trains: forward /
# backward, finite CE+KD, AMP, no OOM, sane step time, and that --soft-dir is
# the rebuilt 500-file cache. It is NOT a performance judgement and it does NOT
# exercise the 3200 gate (--offload-at 0).
#
# Written as a file so that no supervisor command line ever contains the python
# script name (a previous `pkill -f` matched its own ssh command line).
set -uo pipefail
R=/home/jhy/rare_diag_20260915
B=/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1
PY=/home/jhy/.venvs/uavseg/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2

RUN=soup_gate_smoke_20260916
OUT="$R/$RUN"
# disposable by definition: always start from a clean directory
rm -rf "$OUT" "$R/$RUN.log"

cd "$R"
setsid nohup "$PY" -u "$R/_kd_train_soup_gate.py" \
  --outdir "$OUT" \
  --config "$B/stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml" \
  --init-checkpoint "$R/soups/soupB.pt" \
  --dataset-root "/home/jhy/datasets/2026-低空图像语义分割赛道-训练集" \
  --soft-dir "$R/kd_soft_fp16_rebuild" \
  --max-steps 100 --lambda-kd 0.35 --temperature 2.0 --w-min 0.3 \
  --kd-every 2 --batch-size 2 --grad-accum 2 --lr 3.0e-5 \
  --workers 4 --measure-grad-ratio --device cuda:0 \
  > "$R/$RUN.log" 2>&1 < /dev/null &

echo "launched $RUN pid $!"
