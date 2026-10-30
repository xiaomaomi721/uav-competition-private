"""H75 Selective Local-Global Refinement: source-side diagnostic + mechanism check.

Pre-registered design (frozen before running):
  DEV  = H300 first 200 images           -> rule selection ONLY
  EVAL = R1099 first 600 images          -> evaluation ONLY, never used for tuning

Global prediction: H75 D4-TTA8 + Bstar, native 256 FP32 mean, bilinear 1024, argmax+1
Local  prediction: 512x512 crop (4 crops, 2x2 grid) -> bilinear resize to 1024
                   -> same H75 D4-TTA8 + Bstar -> argmax -> nearest resize back to 512
Splice rules (all pre-registered):
  A_full      : replace the whole 512 crop region
  B_margin64  : replace only pixels >=64 px inside the crop (edge relies on global)
  C_m64_conf  : B_margin64 AND local!=global AND local confidence >= 0.90
No parameter is tuned on EVAL.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

ROOT = Path("/home/jhy/aic2026_h76_20261026")
SRC = ROOT / "code" / "h76_3594"
import sys
sys.path.insert(0, str(SRC))
import os
os.environ.setdefault("UAVSEG_MODEL_SOURCE", str(ROOT / "models" / "segformer-b3-ade512"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from src.models import build_model
from src.utils import load_config
from src.label_mapping import official_to_train
from src.transforms import normalize_to_tensor

H75 = ROOT / "models" / "H75.pt"
IMGD = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/train/images")
MSKD = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/train/masks")
OUT = Path("/home/jhy/aic2026_h75_local_refine_20261028")
BSTAR = np.array([-.2, .1, .0709, .4, 3.6, .4, 2.1215, -.5], dtype=np.float32)
CLS = ["Background", "Building", "Road", "Water", "Barren", "Vegetation", "Agricultural", "Vehicle"]
CROP = 512
MARGIN = 64
CONF_THR = 0.90
CROPS = [(0, 0), (0, 512), (512, 0), (512, 512)]


def d4(x, v):
    k, f = divmod(v, 2)
    if k: x = torch.rot90(x, k, (-2, -1))
    if f: x = x.flip(-1)
    return x


def inv_d4(x, v):
    k, f = divmod(v, 2)
    if f: x = x.flip(-1)
    if k: x = torch.rot90(x, -k, (-2, -1))
    return x.contiguous()


@torch.inference_mode()
def infer_logits(model, x1024, b):
    acc = None
    for v in range(8):
        z = inv_d4(model(pixel_values=d4(x1024, v)).logits.float(), v)
        acc = z if acc is None else acc + z
    raw = acc / 8.0
    return F.interpolate(raw + b, size=(1024, 1024), mode="bilinear", align_corners=False)


def cm_from(pred, gt):
    M = np.zeros((8, 8), np.int64)
    valid = gt != 255
    enc = (gt[valid].astype(np.int64) * 8 + pred[valid].astype(np.int64))
    M += np.bincount(enc, minlength=64).reshape(8, 8)
    return M


def miou(M):
    inter = np.diag(M).astype(np.float64)
    union = M.sum(1) + M.sum(0) - inter
    with np.errstate(all="ignore"):
        iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return float(np.nanmean(iou) * 100), iou


def size_buckets(gt):
    """Connected-component size buckets for selected classes."""
    out = {}
    for c in (2, 4, 7):  # Road, Barren, Vehicle
        m = (gt == c)
        if not m.any():
            continue
        lab, n = ndimage.label(m, structure=np.ones((3, 3)))
        if n == 0:
            continue
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        out[CLS[c]] = {"components": int(n), "px": int(m.sum()),
                       "median_px": float(np.median(sizes)),
                       "p10_px": float(np.percentile(sizes, 10)),
                       "p90_px": float(np.percentile(sizes, 90)),
                       "small_lt256_pct": float(100.0 * sizes[sizes < 256].sum() / max(1, sizes.sum()))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "eval"], required=True)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sp = json.loads((SRC / "splits" / "SOURCE_AUDIT_SPLIT.json").read_text())
    ids = (sp["H300"] if a.split == "dev" else sp["R1099"])[: a.n]

    cfg = load_config(str(SRC / "configs" / "base.yaml"))
    cfg["cwfa"] = {"enabled": False}; cfg["mixstyle"] = {"enabled": False}
    cfg["pretrained_model_path"] = str(ROOT / "models" / "segformer-b3-ade512")
    m = build_model(cfg)
    m.load_state_dict(torch.load(str(H75), map_location="cpu", weights_only=False).get("model"), strict=True)
    m = m.to("cuda:0").eval()
    b = torch.from_numpy(BSTAR).cuda()[None, :, None, None]

    variants = ["global", "A_full", "B_margin64", "C_m64_conf"]
    CM = {v: np.zeros((8, 8), np.int64) for v in variants}
    corr = {v: 0 for v in variants if v != "global"}
    harm = {v: 0 for v in variants if v != "global"}
    changed = {v: 0 for v in variants if v != "global"}
    total_px = 0
    size_acc = {}
    conf_hist = np.zeros(10, np.int64)
    local_conf_sum = 0.0
    local_conf_n = 0

    t0 = time.time()
    for i, stem in enumerate(ids, 1):
        rgb = np.asarray(Image.open(IMGD / f"{stem}.png").convert("RGB")).copy()
        gt = official_to_train(np.asarray(Image.open(MSKD / f"{stem}.png")).copy())
        x = normalize_to_tensor(rgb).unsqueeze(0).cuda()

        gl = infer_logits(m, x, b)                       # (1,8,1024,1024)
        gp = gl.argmax(1)[0].cpu().numpy().astype(np.uint8)

        preds = {"global": gp.copy(), "A_full": gp.copy(),
                 "B_margin64": gp.copy(), "C_m64_conf": gp.copy()}

        for (y0, x0) in CROPS:
            crop_rgb = rgb[y0:y0 + CROP, x0:x0 + CROP]
            up = np.asarray(Image.fromarray(crop_rgb).resize((1024, 1024), Image.BILINEAR))
            xu = normalize_to_tensor(up).unsqueeze(0).cuda()
            ll = infer_logits(m, xu, b)
            lp1024 = ll.argmax(1)[0].cpu().numpy().astype(np.uint8)
            lp = np.asarray(Image.fromarray(lp1024).resize((CROP, CROP), Image.NEAREST))
            conf = ll.softmax(1).max(1)[0].cpu().numpy()
            conf = np.asarray(Image.fromarray((conf * 255).astype(np.uint8))
                              .resize((CROP, CROP), Image.BILINEAR)).astype(np.float32) / 255.0

            gsub = gp[y0:y0 + CROP, x0:x0 + CROP]
            # A: full
            preds["A_full"][y0:y0 + CROP, x0:x0 + CROP] = lp
            # B: margin
            yy, xx = np.mgrid[0:CROP, 0:CROP]
            inner = ((yy >= MARGIN) & (yy < CROP - MARGIN) &
                     (xx >= MARGIN) & (xx < CROP - MARGIN))
            B = gsub.copy(); B[inner] = lp[inner]
            preds["B_margin64"][y0:y0 + CROP, x0:x0 + CROP] = B
            # C: margin + disagree + conf
            sel = inner & (lp != gsub) & (conf >= CONF_THR)
            C = gsub.copy(); C[sel] = lp[sel]
            preds["C_m64_conf"][y0:y0 + CROP, x0:x0 + CROP] = C

            conf_hist += np.histogram(conf.ravel(), bins=10, range=(0, 1))[0]
            local_conf_sum += float(conf.sum()); local_conf_n += conf.size

        valid = gt != 255
        for v in variants:
            CM[v] += cm_from(preds[v], gt)
        for v in ("A_full", "B_margin64", "C_m64_conf"):
            ch = (preds[v] != gp) & valid
            changed[v] += int(ch.sum())
            corr[v] += int((ch & (gp != gt) & (preds[v] == gt)).sum())
            harm[v] += int((ch & (gp == gt) & (preds[v] != gt)).sum())
        total_px += int(valid.sum())

        if i <= 50:
            for k, val in size_buckets(gt).items():
                size_acc.setdefault(k, []).append(val)
        if i % 50 == 0:
            print(f"{i}/{len(ids)} {round(time.time()-t0,1)}s", flush=True)

    res = {"split": a.split, "n_images": len(ids), "valid_px": total_px,
           "crop": CROP, "margin": MARGIN, "conf_thr": CONF_THR,
           "mean_local_conf": round(local_conf_sum / max(1, local_conf_n), 4),
           "variants": {}}
    for v in variants:
        mi, iou = miou(CM[v])
        d = {"mIoU": round(mi, 4),
             "per_class_iou": [None if not np.isfinite(x) else round(float(x) * 100, 3) for x in iou]}
        if v != "global":
            d.update({"changed_px": changed[v],
                      "changed_pct": round(changed[v] / total_px * 100, 4),
                      "corrected_px": corr[v], "harmed_px": harm[v],
                      "net_px": corr[v] - harm[v],
                      "delta_vs_global_pp": round(mi - res["variants"]["global"]["mIoU"], 4)
                      if "global" in res["variants"] else None})
        res["variants"][v] = d
    if size_acc:
        res["gt_object_sizes_first50"] = {
            k: {"components": int(np.sum([x["components"] for x in v])),
                "median_px": round(float(np.median([x["median_px"] for x in v])), 1),
                "small_lt256_area_pct": round(float(np.mean([x["small_lt256_pct"] for x in v])), 2)}
            for k, v in size_acc.items()}
    res["local_conf_histogram"] = conf_hist.tolist()

    (OUT / f"LR_{a.split.upper()}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
