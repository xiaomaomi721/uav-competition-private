#!/usr/bin/env bash
# ============================================================================
# Axis T driver: the historical Pseudo500 pipeline, with ONE variable changed.
#
# The generative logic is the ORIGINAL /home/jhy/gen_pseudo_labels.py, called
# unmodified. This driver only picks the teacher and then runs the post-step
# (train-ID -> official + manifest + optional replay diff).
#
# Historical chain being replicated (verified 2026-09-18):
#   gen_pseudo_labels.py --mode region --threshold 0.9
#     teacher = stage23_runs/.../best_miou.pt        (Exp23 / Stage23B)
#   convert_pseudo_to_official.py    (255->0 ; 0-7 -> 1-8)
#   prepare_pseudo_train.sh          (symlink 7000..7499, build the split file)
#
# Modes:
#   replay_exp23   teacher = the ORIGINAL Exp23 checkpoint  -> must reproduce
#                  /home/jhy/pseudo_labels_conf09/masks_official bit-for-bit.
#                  This is the single gate that proves the pipeline is faithful.
#   v1_champion    teacher = 237_S_semeda_M (0.702574). The ONLY difference vs
#                  the historical run.
# ============================================================================
set -uo pipefail

PY=/home/jhy/.venvs/uavseg/bin/python
E=/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1
ROOT=/home/jhy/rare_diag_20260915
DS=/home/jhy/datasets/2026-低空图像语义分割赛道-训练集
OUTBASE=$ROOT/axisT_20260918
POST=$ROOT/_axisT_post.py
HIST_REF=/home/jhy/pseudo_labels_conf09/masks_official

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
# gen_pseudo_labels.py does `from src.models import ...`, so it must run with the
# bundle on sys.path and from the bundle root (2026-09-18: this was missing and
# the first chain run died with ModuleNotFoundError: No module named 'src').
export PYTHONPATH="$E"
cd "$E" || exit 9

MODE=${1:-}
case "$MODE" in
  replay_exp23)
    EXPDIR=$E/stage23_runs/stage23_w1_exp23_b3_1024_cwfa_ce_formal_seed42
    CKPT=$EXPDIR/best_miou.pt
    CMP=$HIST_REF
    ;;
  v1_champion)
    EXPDIR=$E/stage62_runs/stage62_w1_resume55b_pseudo500_seed42
    CKPT=$ROOT/donor_gate2_20260917/S_semeda_M.pt
    CMP=
    ;;
  *)
    echo "usage: $0 {replay_exp23|v1_champion}" >&2
    exit 2
    ;;
esac

OUT=$OUTBASE/$MODE
if [ -e "$OUT" ]; then echo "output dir exists: $OUT" >&2; exit 3; fi
mkdir -p "$OUT" "$OUTBASE/logs"

[ -f "$EXPDIR/config.yaml" ] || { echo "missing $EXPDIR/config.yaml" >&2; exit 4; }
[ -f "$CKPT" ]              || { echo "missing checkpoint $CKPT" >&2; exit 4; }

echo "=== AxisT mode=$MODE  teacher=$CKPT  $(date +%H:%M:%S) ==="

# ---- step 1: ORIGINAL generator, unmodified ---------------------------------
"$PY" -u /home/jhy/gen_pseudo_labels.py \
    --experiment-dir "$EXPDIR" \
    --checkpoint     "$CKPT" \
    --dataset-root   "$DS" \
    --output-dir     "$OUT" \
    --mode region --threshold 0.9 \
    --device cuda 2>&1 | tee "$OUTBASE/logs/${MODE}_gen.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "GENERATOR FAILED" >&2; exit 5; }

# ---- step 2: train-ID -> official + manifest (+ replay diff) ----------------
CKPT_SHA=$("$PY" - <<PYEOF
import hashlib,sys
h=hashlib.sha256()
with open("$CKPT","rb") as f:
    for b in iter(lambda: f.read(1<<20), b""): h.update(b)
print(h.hexdigest())
PYEOF
)

ARGS=( "$PY" -u "$POST"
       --src-region "$OUT/masks_region_t0p90"
       --out-official "$OUT/masks_official"
       --manifest "$OUT/AXIS_T_MANIFEST.json"
       --teacher-checkpoint "$CKPT"
       --teacher-sha256 "$CKPT_SHA"
       --label-tag "$MODE"
       --mode region --threshold 0.9 )
[ -n "$CMP" ] && ARGS+=( --compare-to "$CMP" )
"${ARGS[@]}" 2>&1 | tee "$OUTBASE/logs/${MODE}_post.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "POST STEP FAILED" >&2; exit 6; }

echo "=== AxisT $MODE DONE $(date +%H:%M:%S) -> $OUT ==="
