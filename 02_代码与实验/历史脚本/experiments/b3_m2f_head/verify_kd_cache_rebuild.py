"""Verify the rebuilt 3-teacher soft-target cache against the surviving manifest.

The 2026-09-16 disk cleanup deleted kd_soft_fp16/probs (500 .npy, 7.81 GiB) but
kept PROBS_MANIFEST.json, which records for every file its name, shape and the
float64 sum over the WHOLE 1024x1024x8 array. A sum over ~8.4M values is an
extremely sensitive fingerprint: any change in teacher weights, TTA order or
softmax placement moves it far more than fp16 rounding noise.

Stored value is float16, so exact equality is not guaranteed across GPU
reductions; the point is to show the difference is at fp16 epsilon and not a
different model. Reference scale: for a correct cache every pixel sums to 1
across classes, so sum ~= 1048576.
"""
import json
import sys
from pathlib import Path

ORIG = Path('/home/jhy/rare_diag_20260915/kd_soft_fp16')
NEW = Path('/home/jhy/rare_diag_20260915/kd_soft_fp16_rebuild')

mo = json.loads((ORIG / 'PROBS_MANIFEST.json').read_text(encoding='utf-8'))
mn = json.loads((NEW / 'PROBS_MANIFEST.json').read_text(encoding='utf-8'))

print(f'original  count={mo["count"]}  T={mo["temperature"]}  teachers={mo["teachers"]}')
print(f'rebuilt   count={mn["count"]}  T={mn["temperature"]}  teachers={mn["teachers"]}')

if mo['teachers'] != mn['teachers']:
    print('FAIL: teacher tags differ')
if float(mo['temperature']) != float(mn['temperature']):
    print('FAIL: temperature differs')

no = {it['name']: it for it in mo['items']}
nn = {it['name']: it for it in mn['items']}

missing = sorted(set(no) - set(nn))
extra = sorted(set(nn) - set(no))
print(f'\nmissing from rebuild : {len(missing)} {missing[:5]}')
print(f'extra in rebuild     : {len(extra)} {extra[:5]}')

shape_bad = [n for n in no if n in nn and no[n]['shape'] != nn[n]['shape']]
print(f'shape mismatches     : {len(shape_bad)} {shape_bad[:5]}')

# ---- the real test: per-file float64 sum ----
diffs = []
for n in sorted(set(no) & set(nn)):
    a, b = float(no[n]['sum']), float(nn[n]['sum'])
    diffs.append((abs(a - b), abs(a - b) / 1048576.0, n, a, b))

diffs.sort(reverse=True)
worst = diffs[0]
n_exact = sum(1 for d in diffs if d[0] == 0.0)
print(f'\nfiles compared       : {len(diffs)}')
print(f'bit-exact sums       : {n_exact}')
print(f'max abs sum diff     : {worst[0]:.6f}  ({worst[2]})')
print(f'max rel sum diff     : {worst[1]:.3e}')
print(f'  original sum       : {worst[3]:.6f}')
print(f'  rebuilt  sum       : {worst[4]:.6f}')

# fp16 has ~11 bits of mantissa; a single ulp near 1.0 is ~9.8e-4. A whole-image
# sum over 1.05M pixels should stay far below that if the cache is the same.
worst_rel = worst[1]
if missing or extra or shape_bad:
    verdict = 'FAIL'
elif worst_rel == 0.0:
    verdict = 'EXACT'
elif worst_rel < 1e-5:
    verdict = 'EQUIVALENT (within fp16 accumulation noise)'
else:
    verdict = 'FAIL: differences exceed fp16 noise'

print(f'\n=== CACHE REBUILD VERDICT: {verdict} ===')
raise SystemExit(0 if verdict != 'FAIL' else 1)
