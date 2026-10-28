"""Pre-registered CPU measurement (frozen 2026-09-16, BEFORE any of these numbers).

Loops over the old-model prediction packages ALREADY on local disk.  No GPU, no
re-inference, no new packages.

Stratum A (matched pairs, no GPU, no assumption about the operator):
    Stage59 52k  ZERO  vs  FIXED8D     <-- only pair with a platform d-mIoU
Stratum A? (same body, different vectors):
    Stage23 champion vs 7 prior-align vectors
    s62 champion vs 8 single-axis probes
    (platform d-mIoU NOT known for these -> flip% computed but E undefined)

Deliberately NOT done: any GPU re-inference (so the old-model span/median margin
measurement is deferred), any new package, any platform submission.

Everything here is measured from prediction PNGs only, so the biases that are
deltas from ZERO are exact and free of score-space assumptions.
"""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(r'C:\Users\Lenovo\OneDrive\Documents\桌面\无人机\提交包')
OUT = Path(r'C:\Users\Lenovo\OneDrive\Documents\桌面\无人机\审计记录\calib_geometry_20260916')
NAMES = [f'test1_{i}.png' for i in range(1, 501)]

# official class ids 1..8 -> names; predict_*.argmax(1)+1
NAMES8 = ['Background', 'Building', 'Road', 'Water', 'Barren', 'Vegetation',
          'Agricultural', 'Vehicle']

# exact vectors, from variants_*.json / RUN_META.json / target_transfer.py
CHAMP = [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5]
FIXED8D = [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5]
ZERO = [0.0] * 8

PAIRS = [
    # (group, tag_a, zip_a, vec_a, tag_b, zip_b, vec_b, platform_b, platform_a, same_body)
    ('stage59_52k_match', 'ZERO', '05_s52000_R_TTA8.zip', ZERO,
     'FIXED8D', '09_s52000_R_TTA8_FIXED8D.zip', FIXED8D,
     0.683325, 0.670442, True),
]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_zip_preds(zpath):
    """-> dict name -> uint8 (1024,1024), reading only the 500 test1 entries."""
    out = {}
    with zipfile.ZipFile(zpath) as z:
        entries = set(z.namelist())
        for n in NAMES:
            if n not in entries:
                raise RuntimeError(f'{zpath.name}: missing {n}')
            import io
            with Image.open(io.BytesIO(z.read(n))) as im:
                a = np.asarray(im)
            if a.shape != (1024, 1024):
                raise RuntimeError(f'{zpath.name}:{n} shape {a.shape}')
            out[n] = a
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    actual_dir = ROOT / '历史ZERO补包_20260916'
    rows = []
    print(f'{"group":22s} {"A->B":34s} {"flip%":>9s} {"d_mIoU(pp)":>11s} '
          f'{"E=pp/flip%":>11s}', flush=True)
    for group, ta, za, va, tb, zb, vb, pb, pa, same in PAIRS:
        A = load_zip_preds(actual_dir / za)
        B = load_zip_preds(actual_dir / zb)
        tot = 0
        chg = 0
        trans = np.zeros((9, 9), dtype=np.int64)   # [from][to], ids 1..8
        for n in NAMES:
            a, b = A[n], B[n]
            m = a != b
            chg += int(m.sum())
            tot += a.size
            np.add.at(trans, (a[m].astype(np.int64), b[m].astype(np.int64)), 1)
        flip_pct = chg / tot * 100.0
        dm = (pb - pa) * 100.0
        E = dm / flip_pct if flip_pct else float('nan')
        print(f'{group:22s} {ta+" -> "+tb:34s} {flip_pct:9.4f} {dm:11.4f} '
              f'{E:11.4f}', flush=True)
        # top transitions
        idx = np.dstack(np.unravel_index(np.argsort(trans.ravel())[::-1][:12],
                                         trans.shape))[0]
        top = [(NAMES8[i - 1], NAMES8[j - 1], int(trans[i, j]))
               for i, j in idx if trans[i, j] > 0]
        print('    top transitions (from -> to : px):', flush=True)
        for f, t, c in top[:8]:
            print(f'      {f:13s} -> {t:13s} {c:10d}  '
                  f'({c/chg*100:5.1f}% of flips)', flush=True)
        rows.append(dict(group=group, same_body=same, tag_a=ta, tag_b=tb,
                         vec_a=va, vec_b=vb, zip_a=za, zip_b=zb,
                         zip_a_sha256=sha(actual_dir / za),
                         zip_b_sha256=sha(actual_dir / zb),
                         flipped_px=chg, total_px=tot, flip_pct=flip_pct,
                         platform_miou_a=pa, platform_miou_b=pb,
                         delta_miou_pp=dm, E_pp_per_flip_pct=E,
                         transitions=[{'from': f, 'to': t, 'px': c}
                                      for f, t, c in top]))
    (OUT / 'PAIR_FLIP_RESULT.json').write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nPAIR_FLIP_DONE', flush=True)


if __name__ == '__main__':
    main()
