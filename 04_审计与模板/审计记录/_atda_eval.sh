#!/usr/bin/env bash
# ============================================================================
# ATDA-71 evaluation driver.
#
# Every package here is produced under the champion's own inference protocol
# (D4 TTA8, FP32, autocast off, cwfa disabled) with the Bstar deployment
# operator only. The ZERO/C/13 reference set is deliberately not generated.
#
# Sequence, per the ATDA-71 spec:
#   2k  --theta_2k direct (alpha=1.0) and half-delta (alpha=0.5)   [user submits]
#   4k  --alpha 0.5 and alpha 1.0                                  [formal gate]
#   8k  --alpha 0.75, 1.0, 1.25                                    [winner]
#
# Physical GPU2 only.
# ============================================================================
set -uo pipefail

ATDA=/home/jhy/rare_diag_20260915/atda_20260918
RUN=$ATDA/run_s2s3_base261
PY=/home/jhy/.venvs/uavseg/bin/python
EXP=$ATDA/export
LOG=$ATDA/eval.log

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1:/home/jhy/gpu2_day_queue
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
export UAVSEG_MODEL_SOURCE=/home/jhy/models/segformer-b3-ade512
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec >>"$LOG" 2>&1
echo "=== ATDA EVAL START $(date '+%F %T') ==="
mkdir -p "$EXP"

run_variant () {   # $1=gate step  $2=alpha  $3=tag
    local step=$1 alpha=$2 tag=$3
    local ckpt=$RUN/checkpoint_step${step}.pt
    if [ ! -f "$ckpt" ]; then echo "MISSING $ckpt -- skipping $tag"; return 1; fi
    if [ -f "$EXP/${tag}_Bstar.zip" ]; then echo "SKIP $tag (already packaged)"; return 0; fi
    echo "--- $tag  step=$step alpha=$alpha  $(date '+%H:%M:%S') ---"
    "$PY" -u "$ATDA/_atda_export.py" \
        --outdir "$EXP" --student "$ckpt" --alpha "$alpha" --tag "$tag" --device cuda:0
    local rc=$?
    if [ $rc -ne 0 ]; then echo "  FAILED $tag rc=$rc"; return 1; fi
    echo "  ok $tag $(date '+%H:%M:%S')"
    "$PY" - "$EXP" "$tag" <<'PYEOF'
import json, sys
from pathlib import Path
outdir, tag = Path(sys.argv[1]), sys.argv[2]
meta = json.loads((outdir / "EXPORT_META.json").read_text())
r = meta["results"].get(tag)
if r:
    print(f"  {tag}: {r['packages']['zip_path']} "
          f"share={r['bstar_class_share_pct']} {r['elapsed_sec']}s")
PYEOF
}

# ---- 2k: the two the user submits by hand ---------------------------------
run_variant 2000 1.0  atda_2k_a1p0
run_variant 2000 0.5  atda_2k_a0p5

# ---- 4k: the formal gate --------------------------------------------------
run_variant 4000 0.5  atda_4k_a0p5
run_variant 4000 1.0  atda_4k_a1p0

# ---- 8k: the winner, three alpha ------------------------------------------
run_variant 8000 0.75 atda_8k_a0p75
run_variant 8000 1.0  atda_8k_a1p0
run_variant 8000 1.25 atda_8k_a1p25

echo "=== ATDA EVAL DONE $(date '+%F %T') ==="
ls -la "$EXP"/*.zip 2>/dev/null
