#!/usr/bin/env bash
# M2F Run1b LR-reopen arms: continue from the exact 56k scored checkpoint with a
# non-negligible LR, annealed to zero.
#
# Why: M2F Run1b's ZERO is still rising (20k 0.680570 -> 40k 0.683170 ->
# 56k 0.684133) but its terminal LR is ~1.6e-6, i.e. it stopped moving. The
# calibration-response trajectory measured locally is flat (FG020 flip 1.3749 ->
# 1.1646 -> 1.1409%), so this is a ZER0-side bet, not an R-side one.
#
# The intervention is the LR at the first tail step. Both arms re-enter the
# stock schedule at --start-iter 56001 with a LARGER horizon, so
#     lr(56001) = 1e-4 * (1 - 56001/steps)
# is 2.63e-5 for steps=76000 and 3.17e-5 for steps=82000, instead of the ~1.6e-6
# the parent run actually had. Since `step` alone determines the data seed, the
# epoch permutation and the LR, starting at 56001 reproduces the post-56000
# state exactly -- no replay.
#
# Independent per arm: fresh AdamW, fresh scaler. No resume framework.
set -euo pipefail
R=/home/jhy/rare_diag_20260915
M=$R/m2f_gpu_20260915
PY=/home/jhy/.venvs/uavseg/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2

INIT=$M/formal_run1b/scores/score_iter56000.pt
INIT_SHA=707a582df4cf5a22227ed73db9cf4bdad7e1689f72ae9f99f71fd09e22629272

[ -f "$INIT" ] || { echo "ABORT: missing $INIT"; exit 1; }
GOT=$(sha256sum "$INIT" | awk '{print $1}')
[ "$GOT" = "$INIT_SHA" ] || { echo "ABORT: 56k sha256 $GOT != $INIT_SHA"; exit 1; }
echo "56k checkpoint OK: $GOT"

BUSY=$(nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$BUSY" -eq 0 ] || { echo "ABORT: GPU2 has $BUSY compute process(es)"; exit 1; }
echo "GPU2 free"

cd "$M/py"
ARMS=${ARMS:-"A B"}
for ARM in $ARMS; do
  case "$ARM" in
    A) STEPS=76000 ;;
    B) STEPS=82000 ;;
    *) echo "unknown arm $ARM"; exit 1 ;;
  esac
  OUT=$M/lrreopen_${ARM}_$(date +%Y%m%d)
  if [ -e "$OUT" ]; then echo "SKIP $ARM: $OUT exists"; continue; fi
  if [ -e "$M/lrreopen_${ARM}.log" ]; then echo "SKIP $ARM: log exists"; continue; fi
  SCORES=""
  for ((s=60000; s<STEPS; s+=5000)); do SCORES="$SCORES $s"; done
  SCORES="$SCORES $STEPS"
  # sequential: each arm must finish (or die) before the next one takes GPU2
  echo "START arm $ARM steps=$STEPS out=$OUT $(date +%H:%M:%S)"
  setsid nohup "$PY" -u b3_m2f_head/_m2f_tail_train_entry.py \
    --outdir "$OUT" --stage formal_run1 \
    --steps "$STEPS" --start-iter 56001 \
    --init-from "$INIT" \
    --score-checkpoint-steps $SCORES \
    --score-checkpoint-dir "$OUT/scores" \
    --device cuda:0 \
    > "$M/lrreopen_${ARM}.log" 2>&1 < /dev/null &
  PID=$!
  wait $PID 2>/dev/null || true
  echo "END arm $ARM rc=$? $(date +%H:%M:%S)"
  sleep 20
done
echo "all arms finished"
