#!/usr/bin/env bash
set -euo pipefail
cd /home/jhy/aic2026_claude_diag_20261029
P=/home/jhy/.venvs/uavseg/bin/python
until [ -f cache_C1/CACHE_META.json ] && ! pgrep -f "cache_raw.py --ckpt .*C1" >/dev/null; do sleep 20; done
CUDA_VISIBLE_DEVICES=0 $P mm_fit.py --cache cache_C1/scores --init bstar --out mm/C1_mm_bstar.json | tail -1
python3 - <<'PY'
import json
d=json.load(open("mm/C1_mm_bstar.json"))
json.dump({"name":"C1_MM","description":"C1 (reliable KD) bias fitted so Target3594 per-class area (native256) matches H75+Bstar champion; mm_fit.py","bias":[round(x,5) for x in d["fitted_bias"]]},open("bias/C1_MM.json","w"),indent=1)
print(open("bias/C1_MM.json").read())
PY
./pack.sh C1_MM /home/jhy/aic2026_h76_20261026/code/reliable_kd_20261028/train/C1/checkpoint_step3594.pt c37cd0138c008ea98fe35eb7f8d2d93d22b850d2747b0e8af454216e77503c36 /home/jhy/aic2026_claude_diag_20261029/bias/C1_MM.json 0
