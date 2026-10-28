#!/usr/bin/env bash
# Waits for KD Run 1 to finish, then produces the s62 combo4 / agri-0.8 probe
# packages on GPU2. Written as a file so no supervisor command line contains a
# pattern that could match its own process (an earlier `pkill -f` killed its own
# ssh session that way).
set -uo pipefail
R=/home/jhy/rare_diag_20260915
B=/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1
PY=/home/jhy/.venvs/uavseg/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
exec >>"$R/combo_queue.log" 2>&1
say(){ echo "[$(date +%H:%M:%S)] $*"; }

say "waiting for KD run 1 to finish"
for i in $(seq 1 480); do
  grep -q KD_TRAIN_DONE "$R/kd_run1_s62init.log" 2>/dev/null && break
  pgrep -f 'kd_run1_s62init' >/dev/null 2>&1 || { say "kd process gone"; break; }
  sleep 60
done
say "KD finished (or gone); giving GPU 30s to release"
sleep 30

if [ ! -f "$R/kd_run1_s62init/checkpoint_step8000.pt" ]; then
  say "WARN: final KD checkpoint missing -- still running the probe"
fi

if [ -e "$R/s62_combo_v1" ]; then say "SKIP: s62_combo_v1 exists"; say COMBO_DONE; exit 0; fi

say "launching combo probe"
"$PY" "$R/_s62_combo_probe.py" \
  --outdir "$R/s62_combo_v1" \
  --names "$R/test_names.txt" \
  --test-dir "/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/images" \
  --config "$B/stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml" \
  --checkpoint "$B/stage62_runs/stage62_w1_resume55b_pseudo500_seed42/best_miou.pt" \
  --device cuda:0
say "combo probe rc=$?"
say COMBO_DONE
