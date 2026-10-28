#!/usr/bin/env bash
# Launch the 3-teacher soft-KD run. Written as a file so that no supervisor
# command line ever contains the python script name (a previous `pkill -f`
# matched its own ssh command line and killed the session).
set -uo pipefail
R=/home/jhy/rare_diag_20260915
B=/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1
PY=/home/jhy/.venvs/uavseg/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2

RUN=${1:?usage: kd_launch.sh <run_name> <init_checkpoint> [extra args...]}
INIT=${2:?usage: kd_launch.sh <run_name> <init_checkpoint> [extra args...]}
shift 2
# KD_EVERY=1 -> one test soft-KD batch per optimizer step (higher target exposure)
# KD_EVERY=2 -> GT-heavy (2 real : 1 test) -- the agreed default
KD_EVERY=${KD_EVERY:-2}

OUT="$R/$RUN"
if [ -e "$OUT" ]; then echo "refuse: $OUT already exists"; exit 1; fi

cd "$R"
setsid nohup "$PY" "$R/_kd_train.py" \
  --outdir "$OUT" \
  --config "$B/stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml" \
  --init-checkpoint "$INIT" \
  --dataset-root "/home/jhy/datasets/2026-低空图像语义分割赛道-训练集" \
  --soft-dir "$R/kd_soft_fp16" \
  --max-steps 8000 --lambda-kd 0.35 --temperature 2.0 --w-min 0.3 \
  --kd-every "$KD_EVERY" --batch-size 2 --grad-accum 2 --lr 3.0e-5 \
  --workers 4 --measure-grad-ratio --device cuda:0 "$@" \
  > "$R/$RUN.log" 2>&1 < /dev/null &

echo "launched $RUN (pid $!)"
