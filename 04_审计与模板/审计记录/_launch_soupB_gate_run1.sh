#!/usr/bin/env bash
# Formal Run1: soupB weight-average init -> evidence-matched KD continuation ->
# 3200-optimizer-update platform gate.
#
# Frozen recipe (do not edit mid-run):
#   max_steps=8000 (scheduler total stays 8000; 3200 is a gate, not the end)
#   offload_at=3200 (CPU-offload gate, same process, no resume framework)
#   lambda_kd=0.35 T=2.0 w_min=0.3 kd_every=2
#   batch=2 grad_accum=2 lr=3e-5 AdamW wd=0.01 clip=1.0 AMP on CWFA forced ACTIVE
#
# Fail-closed: refuses to start if the outdir already exists. Never rm -rf.
# Written as a file so no supervisor command line contains the script name.
set -euo pipefail
R=/home/jhy/rare_diag_20260915
B=/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1
PY=/home/jhy/.venvs/uavseg/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2

RUN=soupB_kd_gate_run1_20260916
OUT="$R/$RUN"
INIT="$R/soups/soupB.pt"
INIT_SHA=966139d11a62447c60518bfd27caed3244ccf08860859fd1ae9c7a79225f2f75
SOFT="$R/kd_soft_fp16_rebuild"

# ---- preflight: init weights ------------------------------------------------
[ -f "$INIT" ] || { echo "ABORT: missing $INIT"; exit 1; }
GOT_SHA=$(sha256sum "$INIT" | awk '{print $1}')
[ "$GOT_SHA" = "$INIT_SHA" ] || { echo "ABORT: soupB sha256 $GOT_SHA != $INIT_SHA"; exit 1; }
echo "soupB sha256 OK: $GOT_SHA"

# ---- preflight: teacher cache ----------------------------------------------
[ -d "$SOFT/probs" ] || { echo "ABORT: missing $SOFT/probs"; exit 1; }
N=$(ls "$SOFT/probs"/*.npy 2>/dev/null | wc -l)
[ "$N" -eq 500 ] || { echo "ABORT: expected 500 probs, found $N"; exit 1; }
echo "teacher cache OK: $N probs under $SOFT"

# ---- preflight: fail-closed on outdir --------------------------------------
if [ -e "$OUT" ]; then echo "ABORT: outdir exists, refusing to overwrite: $OUT"; exit 1; fi
if [ -e "$R/$RUN.log" ]; then echo "ABORT: log exists: $R/$RUN.log"; exit 1; fi

# ---- preflight: GPU2 must be free ------------------------------------------
BUSY=$(nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$BUSY" -eq 0 ] || { echo "ABORT: GPU2 already has $BUSY compute process(es)"; exit 1; }
echo "GPU2 free"

cd "$R"
setsid nohup "$PY" -u "$R/_kd_train_soup_gate.py" \
  --outdir "$OUT" \
  --config "$B/stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml" \
  --init-checkpoint "$INIT" \
  --dataset-root "/home/jhy/datasets/2026-低空图像语义分割赛道-训练集" \
  --soft-dir "$SOFT" \
  --max-steps 8000 --offload-at 3200 \
  --lambda-kd 0.35 --temperature 2.0 --w-min 0.3 --kd-every 2 \
  --batch-size 2 --grad-accum 2 --lr 3.0e-5 \
  --workers 4 --device cuda:0 \
  > "$R/$RUN.log" 2>&1 < /dev/null &

echo "launched $RUN pid $!"
