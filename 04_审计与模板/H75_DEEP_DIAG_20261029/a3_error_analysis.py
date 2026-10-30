"""Task C: H75 error mechanism on Source1399 using the existing FP32 logits cache.

Reads /home/jhy/aic2026_ziponly_20261028/01_h75_cache/source1399_fp32/scores/*.npy
(raw native 256 D4-TTA8 mean, no bias, no upsample) + official GT.
No GPU, no new inference.
"""
import csv, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

CACHE = Path("/home/jhy/aic2026_ziponly_20261028/01_h75_cache/source1399_fp32")
MAN = Path("/home/jhy/aic2026_ziponly_20261028/manifests/source1399.csv")
OUT = Path("/home/jhy/aic2026_h75_deep_diag_20261029")
BSTAR = np.array([-.2, .1, .0709, .4, 3.6, .4, 2.1215, -.5], dtype=np.float32)
CLS = ["Background", "Building", "Road", "Water", "Barren", "Vegetation", "Agricultural", "Vehicle"]
OFFICIAL_TO_TRAIN = lambda a: np.where(a > 0, a.astype(np.int16) - 1, 255).astype(np.uint8)


def cm_from(pred, gt):
    M = np.zeros((8, 8), np.int64)
    v = gt != 255
    enc = (gt[v].astype(np.int64) * 8 + pred[v].astype(np.int64))
    M += np.bincount(enc, minlength=64).reshape(8, 8)
    return M


def stats(M):
    inter = np.diag(M).astype(np.float64)
    row = M.sum(1).astype(np.float64)   # gt count
    col = M.sum(0).astype(np.float64)   # pred count
    union = row + col - inter
    with np.errstate(all="ignore"):
        iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
        prec = np.where(col > 0, inter / np.maximum(col, 1), np.nan)
        rec = np.where(row > 0, inter / np.maximum(row, 1), np.nan)
    return iou, prec, rec


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "03_ERROR_ANALYSIS").mkdir(exist_ok=True)
    rows = list(csv.DictReader(open(MAN)))
    assert len(rows) == 1399, len(rows)
    b = torch.from_numpy(BSTAR)

    CM = {"raw": np.zeros((8, 8), np.int64), "bstar": np.zeros((8, 8), np.int64)}
    per_img = []
    corr_px = harm_px = 0
    flip_by_class = np.zeros((8, 8), np.int64)     # raw class -> bstar class
    # size buckets: accumulate (gt_px_in_component, correct_px) per class
    size_rec = {c: [] for c in range(8)}
    # boundary bands, per radius, for each class: (gt_px, correct_px) raw vs bstar
    RADII = [2, 4, 8]
    band = {r: {tag: np.zeros((8, 2), np.int64) for tag in ("raw", "bstar")} for r in RADII}

    for i, r in enumerate(rows, 1):
        name = r["input_name"]
        z = np.load(CACHE / "scores" / f"{name}.npy")           # (8,256,256) raw
        zt = torch.from_numpy(z)[None]
        gt = OFFICIAL_TO_TRAIN(np.asarray(Image.open(r["label_path"])))
        gt_t = torch.from_numpy(gt.astype(np.int64))[None]

        up_raw = F.interpolate(zt, size=(1024, 1024), mode="bilinear", align_corners=False)[0]
        up_bst = F.interpolate(zt + b[None, :, None, None], size=(1024, 1024),
                               mode="bilinear", align_corners=False)[0]
        p_raw = up_raw.argmax(0).numpy().astype(np.uint8)
        p_bst = up_bst.argmax(0).numpy().astype(np.uint8)

        CM["raw"] += cm_from(p_raw, gt)
        CM["bstar"] += cm_from(p_bst, gt)

        v = gt != 255
        ch = (p_raw != p_bst) & v
        corr_px += int((ch & (p_raw != gt) & (p_bst == gt)).sum())
        harm_px += int((ch & (p_raw == gt) & (p_bst != gt)).sum())
        m = ch
        if m.any():
            enc = (p_raw[m].astype(np.int64) * 8 + p_bst[m].astype(np.int64))
            flip_by_class += np.bincount(enc, minlength=64).reshape(8, 8)

        # per-image
        for tag, p in (("raw", p_raw), ("bstar", p_bst)):
            Mi = cm_from(p, gt)
            iou, _, _ = stats(Mi)
            per_img.append({"image": name, "subset": r["subset"], "variant": tag,
                            "mIoU": round(float(np.nanmean(iou)) * 100, 4),
                            "errors": int((p[v] != gt[v]).sum()),
                            "valid_px": int(v.sum())})

        # size buckets (class 2 Road, 4 Barren, 7 Vehicle) — use raw
        for c in (2, 4, 7):
            m = (gt == c)
            if not m.any():
                continue
            lab, n = ndimage.label(m, structure=np.ones((3, 3)))
            if n == 0:
                continue
            sizes = ndimage.sum(m, lab, range(1, n + 1))
            for k in range(1, n + 1):
                comp = (lab == k)
                size_rec[c].append((float(sizes[k - 1]), int((comp & (p_raw == c)).sum())))

        # boundary bands
        for rad in RADII:
            for tag, p in (("raw", p_raw), ("bstar", p_bst)):
                # boundary = pixels within rad of a GT class change
                bnd = np.zeros_like(gt, dtype=bool)
                d = (gt[:, :-1] != gt[:, 1:])
                bnd[:, :-1] |= d; bnd[:, 1:] |= d
                d = (gt[:-1, :] != gt[1:, :])
                bnd[:-1, :] |= d; bnd[1:, :] |= d
                bnd = ndimage.binary_dilation(bnd, iterations=rad)
                for c in range(8):
                    sel = v & (gt == c) & bnd
                    if sel.any():
                        band[rad][tag][c, 0] += int(sel.sum())
                        band[rad][tag][c, 1] += int((sel & (p == c)).sum())

        if i % 200 == 0:
            print(f"{i}/1399", flush=True)

    res = {"class_order": CLS, "n_images": len(rows), "variants": {}}
    for tag in ("raw", "bstar"):
        iou, prec, rec = stats(CM[tag])
        res["variants"][tag] = {
            "mIoU": round(float(np.nanmean(iou)) * 100, 4),
            "per_class": {CLS[c]: {"iou": round(float(iou[c]) * 100, 3),
                                   "precision": round(float(prec[c]) * 100, 3),
                                   "recall": round(float(rec[c]) * 100, 3),
                                   "gt_px": int(CM[tag][c].sum()),
                                   "pred_px": int(CM[tag][:, c].sum())} for c in range(8)},
            "confusion_matrix": CM[tag].tolist(),
        }
    res["bstar_effect"] = {"changed_px": int(flip_by_class.sum()),
                           "corrected_px": int(corr_px), "harmed_px": int(harm_px),
                           "net_px": int(corr_px - harm_px),
                           "corr_over_harm": round(corr_px / max(1, harm_px), 4),
                           "flip_matrix_raw_to_bstar": flip_by_class.tolist()}
    res["boundary_bands"] = {
        str(rad): {tag: {CLS[c]: {"gt_px": int(band[rad][tag][c, 0]),
                                  "correct_px": int(band[rad][tag][c, 1]),
                                  "iou_like": round(100.0 * band[rad][tag][c, 1] / max(1, band[rad][tag][c, 0]), 3)}
                          for c in range(8)} for tag in ("raw", "bstar")} for rad in RADII}
    # size buckets summary
    sz = {}
    for c, name in ((2, "Road"), (4, "Barren"), (7, "Vehicle")):
        arr = np.array(size_rec[c]) if size_rec[c] else np.zeros((0, 2))
        if len(arr) == 0:
            continue
        bins = [(0, 64), (64, 256), (256, 1024), (1024, 4096), (4096, 10 ** 9)]
        entry = []
        for lo, hi in bins:
            sel = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
            if sel.sum() == 0:
                continue
            entry.append({"size_lo": lo, "size_hi": None if hi > 10 ** 8 else hi,
                          "components": int(sel.sum()),
                          "gt_px": int(arr[sel, 0].sum()),
                          "correct_px": int(arr[sel, 1].sum()),
                          "pixel_recall": round(100.0 * arr[sel, 1].sum() / max(1, arr[sel, 0].sum()), 3)})
        sz[name] = entry
    res["object_size_recall"] = sz

    (OUT / "03_ERROR_ANALYSIS" / "A3_ERROR_ANALYSIS.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")

    # csv outputs
    with open(OUT / "03_ERROR_ANALYSIS" / "H75逐类混淆矩阵.csv", "w", encoding="utf-8") as f:
        for tag in ("raw", "bstar"):
            f.write(f"# variant={tag}\n")
            f.write("gt\\pred," + ",".join(CLS) + "\n")
            for c in range(8):
                f.write(CLS[c] + "," + ",".join(str(int(x)) for x in CM[tag][c]) + "\n")
    with open(OUT / "03_ERROR_ANALYSIS" / "H75逐图错误统计.csv", "w", encoding="utf-8") as f:
        f.write("image,subset,variant,mIoU,errors,valid_px\n")
        for r in per_img:
            f.write(f'{r["image"]},{r["subset"]},{r["variant"]},{r["mIoU"]},{r["errors"]},{r["valid_px"]}\n')
    with open(OUT / "03_ERROR_ANALYSIS" / "Bstar纠错与致错分析.csv", "w", encoding="utf-8") as f:
        f.write("raw_class,moved_to," + ",".join(CLS) + "\n")
        for c in range(8):
            f.write(CLS[c] + "," + CLS[c] + "," + ",".join(str(int(x)) for x in flip_by_class[c]) + "\n")

    print(json.dumps({"raw_mIoU": res["variants"]["raw"]["mIoU"],
                      "bstar_mIoU": res["variants"]["bstar"]["mIoU"],
                      "bstar_effect": res["bstar_effect"]}, indent=2))


if __name__ == "__main__":
    main()
