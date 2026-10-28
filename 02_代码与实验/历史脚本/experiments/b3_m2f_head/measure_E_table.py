"""E = d_mIoU_pp / flip% on every calibration pair whose platform score is known.

CPU only. Reads prediction PNGs that are already downloaded. No GPU, no new
packages, no submissions.

The operator differs between bodies (old = raw-logit bias, M2F = final-semantic-
score bias) so E is NOT a like-for-like causal quantity across bodies.  Within a
body the operator is fixed and E is meaningful.  That distinction is reported
rather than hidden.
"""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

BASE = Path(r'C:\Users\Lenovo\OneDrive\Documents\桌面\无人机\提交包')
OUT = Path(r'C:\Users\Lenovo\OneDrive\Documents\桌面\无人机\审计记录\calib_geometry_20260916')
NAMES = [f'test1_{i}.png' for i in range(1, 501)]
NAMES8 = ['Background', 'Building', 'Road', 'Water', 'Barren', 'Vegetation',
          'Agricultural', 'Vehicle']
ZERO8 = [0.0] * 8

HIST = BASE / '历史ZERO补包_20260916'
MZF = BASE / 'M2F_Run1b_三档ZERO_20260916'
FG = BASE / 'M2F56_FG正向五包_20260916'
CP = BASE / 'M2F56_CalibProbe_8包_20260916'
MS = BASE / 'M2F56_multiscale探针_20260916'

# (group, body, ref_zip, ref_miou, var_zip, var_miou, note)
JOBS = [
    ('MATCHED_PAIR', 'Stage59_52k', HIST / '05_s52000_R_TTA8.zip', 0.670442,
     HIST / '09_s52000_R_TTA8_FIXED8D.zip', 0.683325,
     'same ckpt+TTA, bias only -> the only clean old-model lever'),

    ('M2F_bias_only', 'M2F56', MZF / '01_M2F_Run1b_56k_TTA8_ZERO.zip', 0.684133,
     CP / '01_M2F56_A_joint_local_best_TTA8.zip', 0.684348, 'A'),
    ('M2F_bias_only', 'M2F56', MZF / '01_M2F_Run1b_56k_TTA8_ZERO.zip', 0.684133,
     CP / '02_M2F56_B_joint_1d_best_TTA8.zip', 0.684302, 'B'),
    ('M2F_bias_only', 'M2F56', MZF / '01_M2F_Run1b_56k_TTA8_ZERO.zip', 0.684133,
     CP / '03_M2F56_C_2x_joint_1d_best_TTA8.zip', 0.684327, 'C'),
    ('M2F_bias_only', 'M2F56', MZF / '01_M2F_Run1b_56k_TTA8_ZERO.zip', 0.684133,
     CP / '04_M2F56_D_opposite_joint_1d_best_TTA8.zip', 0.683859, 'D'),
    ('M2F_bias_only', 'M2F56', MZF / '01_M2F_Run1b_56k_TTA8_ZERO.zip', 0.684133,
     CP / '05_M2F56_E_uniform_neg_005_TTA8.zip', 0.683631, 'E'),
    ('M2F_bias_only', 'M2F56', MZF / '01_M2F_Run1b_56k_TTA8_ZERO.zip', 0.684133,
     CP / '06_M2F56_F_uniform_pos_005_TTA8.zip', 0.684599, 'F'),
    ('M2F_bias_only', 'M2F56', MZF / '01_M2F_Run1b_56k_TTA8_ZERO.zip', 0.684133,
     CP / '07_M2F56_G_uniform_neg_010_TTA8.zip', 0.683122, 'G'),
    ('M2F_bias_only', 'M2F56', MZF / '01_M2F_Run1b_56k_TTA8_ZERO.zip', 0.684133,
     CP / '08_M2F56_H_uniform_pos_010_TTA8.zip', 0.684903, 'H'),

    ('M2F_bias_only', 'M2F56', MZF / '01_M2F_Run1b_56k_TTA8_ZERO.zip', 0.684133,
     FG / '02_M2F56_FG_POS_020_TTA8.zip', 0.685100, 'FG_POS_020 (M2F best)'),

    ('OPERATOR_CHANGE', 'M2F56_ms', MZF / '01_M2F_Run1b_56k_TTA8_ZERO.zip', 0.684133,
     MS / '01_M2F56_A_S075_TTA8_ZERO.zip', 0.677264, 'A_S075 (scale, not bias)'),
    ('OPERATOR_CHANGE', 'M2F56_ms', MZF / '01_M2F_Run1b_56k_TTA8_ZERO.zip', 0.684133,
     MS / '02_M2F56_B_S125_TTA8_ZERO.zip', 0.673044, 'B_S125 (scale, not bias)'),
]


def load_zip(zpath):
    out = {}
    with zipfile.ZipFile(zpath) as z:
        e = set(z.namelist())
        for n in NAMES:
            if n not in e:
                raise RuntimeError(f'{zpath.name}: missing {n}')
            with Image.open(io.BytesIO(z.read(n))) as im:
                out[n] = np.asarray(im)
    return out


def compare(A, B):
    chg = 0
    tot = 0
    trans = np.zeros((9, 9), dtype=np.int64)
    for n in NAMES:
        a, b = A[n], B[n]
        m = a != b
        chg += int(m.sum())
        tot += a.size
        np.add.at(trans, (a[m].astype(np.int64), b[m].astype(np.int64)), 1)
    return chg, tot, trans


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = {}

    def get(z, miou):
        key = (str(z), miou)
        if key not in cache:
            cache[key] = load_zip(z)
        return cache[key]

    rows = []
    print(f'{"group":16s} {"body":11s} {"variant":24s} {"flip%":>8s} '
          f'{"dpp":>9s} {"E":>8s}  {"BG->FG share":>12s}', flush=True)
    for group, body, rz, rm, vz, vm, note in JOBS:
        A = get(rz, rm)
        B = get(vz, vm)
        chg, tot, trans = compare(A, B)
        flip = chg / tot * 100.0
        dm = (vm - rm) * 100.0
        E = dm / flip if flip else float('nan')
        bg_row = trans[1, :].sum()          # everything leaving Background
        bg_fg = trans[1, 1:].sum()          # Background -> any foreground
        share = bg_fg / chg * 100 if chg else float('nan')
        print(f'{group:16s} {body:11s} {note:24s} {flip:8.4f} {dm:9.4f} '
              f'{E:8.4f}  {share:11.1f}%', flush=True)
        idx = np.dstack(np.unravel_index(np.argsort(trans.ravel())[::-1][:8],
                                         trans.shape))[0]
        top = [{'from': NAMES8[i - 1], 'to': NAMES8[j - 1], 'px': int(trans[i, j])}
               for i, j in idx if trans[i, j] > 0]
        for t in top[:4]:
            print(f'      {t["from"]:13s} -> {t["to"]:13s} {t["px"]:10d} '
                  f'({t["px"]/chg*100:5.1f}%)', flush=True)
        rows.append(dict(group=group, body=body, note=note,
                         ref_zip=rz.name, ref_miou=rm,
                         var_zip=vz.name, var_miou=vm,
                         flipped_px=chg, total_px=tot, flip_pct=flip,
                         delta_pp=dm, E=float(E),
                         bg_to_fg_share_pct=float(share),
                         top_transitions=top))
    (OUT / 'E_TABLE.json').write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nE_TABLE_DONE', flush=True)


if __name__ == '__main__':
    main()
