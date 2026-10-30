"""Compare scored semifinal ZIPs against the champion ZIP (read-only, CPU).

Per candidate: per-class pixel area by subset, 8x8 transition matrix champion->candidate,
per-image presence counts at several area thresholds (1024-res pixels).
"""
import csv, io, json, sys, zipfile
from multiprocessing import Pool
import numpy as np
from PIL import Image

MANIFEST = "/home/jhy/aic2026_ziponly_20261028/manifests/semifinal_3594.csv"
CHAMP = "/home/jhy/aic2026_ctrl3594_teacher_20261027/reference/H75_BSTAR_20261026/H75_BSTAR_semifinal.zip"
THRESH = [1, 256, 1024, 4096]  # 1024-res pixel thresholds for presence

rows = list(csv.DictReader(open(MANIFEST)))
names = [r["output_name"] for r in rows]
subsets = [r["subset"] for r in rows]
SUBS = sorted(set(subsets))


def read(zf, name):
    a = np.asarray(Image.open(io.BytesIO(zf.read(name))))
    assert a.shape == (1024, 1024), (name, a.shape)
    return a.astype(np.int64) - 1  # 0..7


def work(args):
    cand, lo, hi = args
    zc, zk = zipfile.ZipFile(CHAMP), zipfile.ZipFile(cand)
    area = {s: np.zeros(8, np.int64) for s in SUBS}
    trans = {s: np.zeros((8, 8), np.int64) for s in SUBS}
    pres = np.zeros((len(THRESH), 8), np.int64)
    per_img = []
    for i in range(lo, hi):
        a, b = read(zc, names[i]), read(zk, names[i])
        bc = np.bincount(b.ravel(), minlength=8)
        area[subsets[i]] += bc
        trans[subsets[i]] += np.bincount((a * 8 + b).ravel(), minlength=64).reshape(8, 8)
        for t, th in enumerate(THRESH):
            pres[t] += bc >= th
        per_img.append(float((a != b).mean()))
    return area, trans, pres, per_img


def run(cand):
    chunks = [(cand, i, min(i + 60, len(names))) for i in range(0, len(names), 60)]
    with Pool(48) as p:
        res = p.map(work, chunks)
    area = {s: sum(r[0][s] for r in res) for s in SUBS}
    trans = {s: sum(r[1][s] for r in res) for s in SUBS}
    pres = sum(r[2] for r in res)
    per_img = [x for r in res for x in r[3]]
    return area, trans, pres, per_img


if __name__ == "__main__":
    out = {}
    for tag, path in json.loads(sys.argv[1]).items():
        area, trans, pres, per_img = run(path)
        tot_area = sum(area.values())
        tot_trans = sum(trans.values())
        out[tag] = {
            "zip": path,
            "area_pct_all": (100 * tot_area / tot_area.sum()).round(4).tolist(),
            "area_pct_by_subset": {s: (100 * area[s] / area[s].sum()).round(4).tolist() for s in SUBS},
            "flip_pct_all": round(100 * (1 - np.trace(tot_trans) / tot_trans.sum()), 4),
            "flip_pct_by_subset": {s: round(100 * (1 - np.trace(trans[s]) / trans[s].sum()), 4) for s in SUBS},
            "transitions_pct_of_all_pixels": (100 * tot_trans / tot_trans.sum()).round(4).tolist(),
            "presence_by_thresh": {str(th): pres[t].tolist() for t, th in enumerate(THRESH)},
            "images_flip_gt_5pct": int(sum(x > 0.05 for x in per_img)),
        }
        print(tag, "flip%", out[tag]["flip_pct_all"], "area", out[tag]["area_pct_all"], flush=True)
    json.dump(out, open(sys.argv[2], "w"), indent=1)
