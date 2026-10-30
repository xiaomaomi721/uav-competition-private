#!/usr/bin/env bash
# After both seeds of one arm finish: soup -> Target raw cache -> marginal-matching bias -> package -> Source check.
# Usage: arm_chain.sh ARM GPU   (ARM = MIX_L02 or MIX_L10)
set -euo pipefail
ARM=$1; GPU=$2
D=/home/jhy/aic2026_claude_diag_20261029; cd $D
P=/home/jhy/.venvs/uavseg/bin/python
until [ -f train/${ARM}_s1/SUMMARY.json ] && [ -f train/${ARM}_s2/SUMMARY.json ]; do sleep 30; done
$P soup.py train/${ARM}_SOUP.pt train/${ARM}_s1/checkpoint_step3594.pt train/${ARM}_s2/checkpoint_step3594.pt | tee logs/${ARM}_soup.json
SHA=$(sha256sum train/${ARM}_SOUP.pt | awk '{print $1}')
EXPECTED_GPU_INDEX=$GPU CUDA_VISIBLE_DEVICES=$GPU $P cache_raw.py --ckpt train/${ARM}_SOUP.pt --sha $SHA --out cache_${ARM}
CUDA_VISIBLE_DEVICES=$GPU $P mm_fit.py --cache cache_${ARM}/scores --init bstar --out mm/${ARM}_mm_bstar.json | tail -1
python3 - "$ARM" <<'PY'
import json, sys
a = sys.argv[1]; d = json.load(open(f"mm/{a}_mm_bstar.json"))
json.dump({"name": f"{a}_SOUP_MM", "description": f"{a} 2-seed soup; bias fitted so Target3594 area matches H75+Bstar champion",
           "bias": [round(x, 5) for x in d["fitted_bias"]]}, open(f"bias/{a}_SOUP_MM.json", "w"), indent=1)
PY
./pack.sh ${ARM}_SOUP_MM $D/train/${ARM}_SOUP.pt $SHA $D/bias/${ARM}_SOUP_MM.json $GPU
EXPECTED_GPU_INDEX=$GPU CUDA_VISIBLE_DEVICES=$GPU $P src_eval.py train/${ARM}_SOUP.pt logs/${ARM}_src_eval.json \
  /home/jhy/aic2026_ctrl3594_teacher_20261027/configs/bias_bstar.json bias/${ARM}_SOUP_MM.json
rm -rf cache_${ARM}
echo ARM_CHAIN_DONE $ARM
