#!/usr/bin/env bash
# CPU only. H75_AGRI_2P82 from the H75 FP32 raw cache with the teammate-verified replay script, then QA.
set -u
D=/home/jhy/aic2026_claude_diag_20261029; Z=/home/jhy/aic2026_ziponly_20261028
P=/home/jhy/.venvs/uavseg/bin/python
REF=/home/jhy/aic2026_ctrl3594_teacher_20261027/reference/H75_BSTAR_20261026/H75_BSTAR_semifinal.zip
O=$D/pack/H75_AGRI_2P82
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
mkdir -p $O; s=$(date +%s)
$P $Z/code/replay_from_scores.py --cache $Z/01_h75_cache/target3594_fp32 --manifest $Z/manifests/semifinal_3594.csv \
  --bias-config $D/bias/H75_AGRI_2P82.json --device cpu --compare $REF \
  --zip $O/H75_AGRI_2P82_20261029.zip --out-json $O/REPLAY_REPORT.json > $D/logs/agri_replay.log 2>&1
echo "replay_rc=$? elapsed=$(( $(date +%s)-s ))s" >> $D/logs/agri_replay.log
$P $Z/code/qa_submission.py --manifest $Z/manifests/semifinal_3594.csv --zip $O/H75_AGRI_2P82_20261029.zip --expected-count 3594 > $O/SUBMISSION_QA.json 2> $D/logs/agri_qa.err
echo "qa_rc=$? total_elapsed=$(( $(date +%s)-s ))s" >> $D/logs/agri_replay.log
sha256sum $O/H75_AGRI_2P82_20261029.zip | tee $O/ZIP_SHA256.txt >> $D/logs/agri_replay.log
echo AGRI_DONE >> $D/logs/agri_replay.log
