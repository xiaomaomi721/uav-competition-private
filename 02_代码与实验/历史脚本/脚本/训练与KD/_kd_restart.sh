#!/usr/bin/env bash
# Stop any running KD trainer and relaunch with the corrected loss.
#
# IMPORTANT: this lives in a file on purpose. Running `pgrep -f <run_name>` or
# `pkill -f <run_name>` directly from an ssh command line makes the pattern match
# the ssh shell's OWN command line, which kills the session before the restart
# completes. From inside this script the pattern '_kd_train.py' does not appear in
# the script's own command line, so it is safe.
set -uo pipefail
R=/home/jhy/rare_diag_20260915
B=/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1
RUN=kd_run1_s62init

echo "--- stopping existing KD trainer ---"
PIDS=$(pgrep -f '_kd_train[.]py' | tr '\n' ' ')
if [ -n "$PIDS" ]; then
  # shellcheck disable=SC2086
  kill $PIDS 2>/dev/null
  sleep 8
  PIDS2=$(pgrep -f '_kd_train[.]py' | tr '\n' ' ')
  if [ -n "$PIDS2" ]; then
    # shellcheck disable=SC2086
    kill -9 $PIDS2 2>/dev/null
    sleep 3
  fi
fi
LEFT=$(pgrep -f '_kd_train[.]py' | wc -l)
echo "remaining: $LEFT"
[ "$LEFT" -eq 0 ] || { echo "ABORT: trainer still alive"; exit 1; }

echo "--- clearing stale artefacts ---"
rm -rf "$R/$RUN" "$R/$RUN.log" \
       "$R/${RUN}_VOID_double_softmax" "$R/${RUN}_VOID_double_softmax.log"

echo "--- relaunching ---"
KD_EVERY=1 bash "$R/_kd_launch.sh" "$RUN" \
  "$B/stage62_runs/stage62_w1_resume55b_pseudo500_seed42/best_miou.pt"

sleep 45
echo "--- new run ---"
grep -E 'initialised|CWFA|step=' "$R/$RUN.log" 2>/dev/null | tail -3 || tail -5 "$R/$RUN.log"
