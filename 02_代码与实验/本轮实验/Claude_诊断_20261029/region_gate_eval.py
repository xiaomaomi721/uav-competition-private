"""Region-gated bias: accept a Bstar flip (argmax(z+Bstar) != argmax(z)) only if its connected
component (8-connectivity, per new class, 1024 grid) has >= tau pixels; smaller flipped components
revert to the Zero prediction. CPU only, read-only on the H75 FP32 caches.

Source1399: precision of flips by component size (GT), mIoU per tau.
Target3594: class area and image-level presence per tau (no GT).
Variants: 'all' gates flips to any class; 'barren' gates only flips to Barren.
"""
import csv, json
from multiprocessing import Pool
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

torch.set_num_threads(1)
SC = "/home/jhy/aic2026_ziponly_20261028/01_h75_cache/source1399_fp32/scores/"
TC = "/home/jhy/aic2026_ziponly_20261028/01_h75_cache/target3594_fp32/scores/"
MASK = "/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/train/masks/"
MAN = "/home/jhy/aic2026_ziponly_20261028/manifests/semifinal_3594.csv"
VAL = [x.strip() for x in open("/home/jhy/uav_semantic_segmentation/splits/val.txt").read().split() if x.strip()]
VAL = [v if v.endswith(".png") else v + ".png" for v in VAL]
BSTAR = torch.tensor([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5])[:, None, None]
TAUS = [0, 64, 256, 1024, 4096, 16384]
VARIANTS = {"all": list(range(8)), "barren": [4]}
BINS = [0, 64, 256, 1024, 4096, 16384, 65536, 1 << 30]
ST = np.ones((3, 3))


def preds(path):
    z = torch.from_numpy(np.load(path).astype(np.float32))
    u = F.interpolate(z[None], size=(1024, 1024), mode="bilinear", align_corners=False)[0]
    return u.argmax(0).numpy(), (u + BSTAR).argmax(0).numpy()


def gated(pz, pb):
    flip = pz != pb
    comps = []
    for c in range(8):
        m = flip & (pb == c)
        if m.any():
            lab, _ = ndimage.label(m, structure=ST)
            comps.append((c, m, lab, np.bincount(lab.ravel())))
    out = {}
    for vn, cls in VARIANTS.items():
        for t in TAUS:
            p = pb.copy()
            if t:
                for c, m, lab, s in comps:
                    if c in cls:
                        small = s < t; small[0] = False
                        rev = small[lab] & m
                        p[rev] = pz[rev]
            out[(vn, t)] = p
    return out, comps


def src_job(name):
    pz, pb = preds(SC + name + ".npy")
    gt = np.asarray(Image.open(MASK + name)).astype(np.int64) - 1  # -1 = ignore
    v = gt >= 0
    out, comps = gated(pz, pb)
    cms = {k: np.bincount(gt[v] * 8 + p[v], minlength=64) for k, p in out.items()}
    cms["zero"] = np.bincount(gt[v] * 8 + pz[v], minlength=64)
    prec = np.zeros((8, len(BINS) - 1, 3), np.int64)  # new class, size bin, [gt==new, gt==old, other]
    for c, m, lab, s in comps:
        b = np.digitize(s, BINS) - 1
        pix_bin = b[lab]
        for col, cond in enumerate((gt == c, gt == pz, (gt != c) & (gt != pz))):
            sel = m & v & cond
            prec[c, :, col] += np.bincount(pix_bin[sel], minlength=len(BINS) - 1)[:len(BINS) - 1]
    return cms, prec


def tgt_job(r):
    pz, pb = preds(TC + r["output_name"] + ".npy")
    out, _ = gated(pz, pb)
    res = {k: np.bincount(p.ravel(), minlength=8) for k, p in out.items()}
    res["zero"] = np.bincount(pz.ravel(), minlength=8)
    return res


def miou(cm):
    cm = cm.reshape(8, 8); tp = np.diag(cm); u = cm.sum(0) + cm.sum(1) - tp
    return 100 * (tp / np.maximum(u, 1)).mean(), 100 * tp / np.maximum(u, 1)


if __name__ == "__main__":
    N = ["Bg", "Bld", "Road", "Water", "Barren", "Veg", "Agri", "Veh"]
    off = np.array([3523, 2329, 2288, 1754, 721, 3021, 1281, 752])
    rows = list(csv.DictReader(open(MAN)))
    with Pool(48) as P:
        S = P.map(src_job, VAL, chunksize=10)
        T = P.map(tgt_job, rows, chunksize=20)
    keys = list(S[0][0])
    out = {"source": {}, "target": {}, "flip_precision_by_size": {}}
    for k in keys:
        mi, iou = miou(sum(s[0][k] for s in S))
        out["source"][str(k)] = {"miou": round(mi, 4), "iou": iou.round(2).tolist()}
        area = sum(t[k] for t in T); pres = sum((t[k] > 0).astype(int) for t in T)
        out["target"][str(k)] = {"area_pct": (100 * area / area.sum()).round(4).tolist(),
                                 "presence": pres.tolist(), "presence_minus_official": (pres - off).tolist()}
    P_ = sum(s[1] for s in S)
    for c in (4, 6, 3, 5, 0):
        out["flip_precision_by_size"][N[c]] = [
            {"size_bin": f"[{BINS[i]},{BINS[i+1]})", "px": int(P_[c, i].sum()),
             "frac_gt_new": round(float(P_[c, i, 0] / max(P_[c, i].sum(), 1)), 3),
             "frac_gt_old": round(float(P_[c, i, 1] / max(P_[c, i].sum(), 1)), 3)} for i in range(len(BINS) - 1)]
    json.dump(out, open("/home/jhy/aic2026_claude_diag_20261029/region_gate_eval.json", "w"), indent=1)
    print("FLIP PRECISION BY COMPONENT SIZE (Source GT): px, frac correct after flip, frac correct before flip")
    for c, rowsz in out["flip_precision_by_size"].items():
        print(" ", c, [(r["size_bin"], r["px"], r["frac_gt_new"], r["frac_gt_old"]) for r in rowsz if r["px"]])
    print("variant tau | Source mIoU | Target area Bg/Barren/Agri/Water | Target presence-official Barren/Agri/Water/Veg")
    for k in ["zero"] + [k for k in keys if k != "zero"]:
        s, t = out["source"][str(k)], out["target"][str(k)]
        a = t["area_pct"]; d = t["presence_minus_official"]
        print(f"  {str(k):18s} {s['miou']:.3f} | {a[0]:.2f} {a[4]:.2f} {a[6]:.2f} {a[3]:.2f} | {d[4]} {d[6]} {d[3]} {d[5]}")
