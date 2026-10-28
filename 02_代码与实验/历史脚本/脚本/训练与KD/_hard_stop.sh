#!/usr/bin/env bash
# Stop the KD trainer + combo watchdog by PID, then clear the aborted run dir.
# File-based so the pattern can never match this shell's own command line.
set -uo pipefail
R=/home/jhy/rare_diag_20260915

echo "--- stopping combo watchdog ---"
for p in $(pgrep -f 'combo_queue' || true); do
  [ "$p" = "$$" ] && continue
  kill "$p" 2>/dev/null || true
done

echo "--- stopping KD trainer ---"
for p in $(pgrep -f '_kd_train[.]py' || true); do
  [ "$p" = "$$" ] && continue
  kill "$p" 2>/dev/null || true
done
sleep 8
for p in $(pgrep -f '_kd_train[.]py' || true); do
  [ "$p" = "$$" ] && continue
  kill -9 "$p" 2>/dev/null || true
done
sleep 3

echo "kd procs left: $(pgrep -f '_kd_train[.]py' | wc -l)"
echo "combo procs left: $(pgrep -f 'combo_queue' | wc -l)"

rm -rf "$R/kd_run1_s62init" "$R/kd_run1_s62init.log" "$R/combo_queue.log"
echo "cleared"

nvidia-smi -i 2 --query-compute-apps=pid,used_memory --format=csv,noheader
echo "(blank above = GPU2 free)"
df -h / | tail -1
