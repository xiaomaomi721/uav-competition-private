#!/usr/bin/env bash
# Thin wrapper around the frozen, already-scored packager (package_semifinal_bias.py SHA 206cc80d...).
# Usage: pack.sh TAG CKPT CKPT_SHA BIAS_JSON GPU
set -euo pipefail
TAG=$1; CKPT=$2; SHA=$3; BIAS=$4; GPU=$5
R=/home/jhy/aic2026_ctrl3594_teacher_20261027
D=/home/jhy/aic2026_claude_diag_20261029
PY=/home/jhy/.venvs/uavseg/bin/python
h() { sha256sum "$1" | awk '{print $1}'; }
[ "$(h $R/code/package_semifinal_bias.py)" = 206cc80d6bc2a88012f6dd06b02829ca72a8ecef245ed06d6f160ce63d7723a1 ] || { echo WRAPPER_SHA; exit 3; }
[ "$(h $R/manifests/semifinal_3594.csv)" = 76463ff99a495afe82085bb59530032f13f9efbff2cc1936f09178285e6cdd19 ] || { echo MANIFEST_SHA; exit 3; }
[ "$(h $CKPT)" = "$SHA" ] || { echo CKPT_SHA; exit 3; }
OUT=$D/pack/$TAG
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU EXPECTED_CUDA_VISIBLE_DEVICES=$GPU \
  $PY $R/code/package_semifinal_bias.py --work $R/work --checkpoint $CKPT \
  --images-dir /home/jhy/aic2026_h76_20261026/data/test3/images --manifest $R/manifests/semifinal_3594.csv \
  --output-dir $OUT --tag $TAG --bias-config $BIAS --expected-count 3594 --expected-checkpoint-sha $SHA
rm -rf $OUT/predictions
$PY /home/jhy/aic2026_ziponly_20261028/code/qa_submission.py --manifest $R/manifests/semifinal_3594.csv \
  --zip $OUT/${TAG}_semifinal.zip --expected-count 3594 > $OUT/SUBMISSION_QA.json
sha256sum $OUT/${TAG}_semifinal.zip | tee $OUT/ZIP_SHA256.txt
echo PACK_DONE $TAG
