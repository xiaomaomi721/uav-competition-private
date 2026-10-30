"""Stage-1 diagnostic B: Source1399 (has GT) per-scene per-class effect of Bstar.

Scene proxy (no official scene labels exist): derived from GT class composition.
Reports, per scene group and per class, raw vs Bstar TP/FP/FN, IoU, and
corrected/harmed pixel counts caused by Bstar.
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

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
OUT = Path("/home/jhy/aic2026_h75_scene_adaptive_20261028")
BSTAR = np.array([-.2, .1, .0709, .4, 3.6, .4, 2.1215, -.5], dtype=np.float32)
CLS = ["Background", "Building", "Road", "Water", "Barren", "Vegetation", "Agricultural", "Vehicle"]
SCENES = ["urban", "agricultural", "vegetation", "barren_dominant", "mixed"]


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


def scene_of(gt):
    """Scene proxy from GT composition over valid pixels."""
    v = gt != 255
    if not v.any():
        return "mixed"
    c = np.bincount(gt[v].ravel(), minlength=8) / v.sum()
    urban = c[1] + c[2] + c[7]
    if c[6] > 0.30:
        return "agricultural"
    if c[5] > 0.50:
        return "vegetation"
    if c[4] > 0.15:
        return "barren_dominant"
    if urban > 0.40:
        return "urban"
    return "mixed"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = json.loads((SRC / "splits" / "SOURCE_AUDIT_SPLIT.json").read_text())
    ids = ids["H300"] + ids["R1099"]

    cfg = load_config(str(SRC / "configs" / "base.yaml"))
    cfg["cwfa"] = {"enabled": False}; cfg["mixstyle"] = {"enabled": False}
    cfg["pretrained_model_path"] = str(ROOT / "models" / "segformer-b3-ade512")
    model = build_model(cfg)
    model.load_state_dict(torch.load(str(H75), map_location="cpu", weights_only=False).get("model"), strict=True)
    model = model.to("cuda:0").eval()
    b = torch.from_numpy(BSTAR).cuda()[None, :, None, None]

    cm = {sc: {k: np.zeros((8, 8), np.int64) for k in ("raw", "cal")} for sc in SCENES}
    counts = defaultdict(int)
    t0 = __import__("time").time()
    for i, stem in enumerate(ids, 1):
        rgb = np.asarray(Image.open(IMGD / f"{stem}.png").convert("RGB")).copy()
        gt = official_to_train(np.asarray(Image.open(MSKD / f"{stem}.png")).copy())
        sc = scene_of(gt)
        counts[sc] += 1
        gt_t = torch.from_numpy(gt.astype(np.int64)).cuda()[None]
        gt1024 = F.interpolate(gt_t[:, None].float(), size=(1024, 1024), mode="nearest")[:, 0].long()
        valid = gt1024 != 255
        x = normalize_to_tensor(rgb).unsqueeze(0).cuda()
        acc = None
        with torch.inference_mode():
            for v in range(8):
                z = inv_d4(model(pixel_values=d4(x, v)).logits.float(), v)
                acc = z if acc is None else acc + z
            raw = acc / 8.0
            for tag, bias in (("raw", 0.0), ("cal", b)):
                up = F.interpolate(raw + bias, size=(1024, 1024), mode="bilinear",
                                   align_corners=False).argmax(1)[0]
                t = gt1024[0][valid[0]]
                p = up[valid[0]]
                enc = (t * 8 + p).cpu().numpy()
                cm[sc][tag] += np.bincount(enc, minlength=64).reshape(8, 8)
        if i % 200 == 0:
            print(f"{i}/{len(ids)} {round(__import__('time').time()-t0,1)}s", flush=True)

    out = {"class_order": CLS, "scene_counts": dict(counts), "scenes": {}}
    rows = []
    for sc in SCENES:
        if counts[sc] == 0:
            continue
        ent = {}
        for tag in ("raw", "cal"):
            M = cm[sc][tag].astype(np.float64)
            inter = np.diag(M); union = M.sum(1) + M.sum(0) - inter
            iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
            ent[tag] = {"mIoU": float(np.nanmean(iou) * 100),
                        "iou": [None if not np.isfinite(v) else round(float(v) * 100, 3) for v in iou]}
        # corrected / harmed by Bstar
        Mraw, Mcal = cm[sc]["raw"], cm[sc]["cal"]
        corr = harm = 0
        for c in range(8):
            tp_raw = Mraw[c, c]; tp_cal = Mcal[c, c]
            if tp_cal > tp_raw:
                corr += int(tp_cal - tp_raw)
            else:
                harm += int(tp_raw - tp_cal)
        ent["bstar_corrected_px"] = corr
        ent["bstar_harmed_px"] = harm
        ent["net_px"] = corr - harm
        ent["delta_mIoU_pp"] = round(ent["cal"]["mIoU"] - ent["raw"]["mIoU"], 4)
        out["scenes"][sc] = ent
        rows.append((sc, counts[sc], ent["raw"]["mIoU"], ent["cal"]["mIoU"],
                     ent["delta_mIoU_pp"], corr, harm, ent["net_px"]))
        print(sc, counts[sc], "raw", round(ent["raw"]["mIoU"], 3), "cal", round(ent["cal"]["mIoU"], 3),
              "delta", ent["delta_mIoU_pp"], "corr", corr, "harm", harm, flush=True)

    (OUT / "T2_SOURCE_SCENE_DIAG.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    with open(OUT / "T2_SOURCE_SCENE_TABLE.csv", "w", encoding="utf-8") as f:
        f.write("scene,images,raw_mIoU,cal_mIoU,delta_pp,bstar_corrected_px,bstar_harmed_px,net_px\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
        f.write("scene,class,raw_iou,cal_iou,delta\n")
        for sc in out["scenes"]:
            for ci, cname in enumerate(CLS):
                a = out["scenes"][sc]["raw"]["iou"][ci]
                bb = out["scenes"][sc]["cal"]["iou"][ci]
                if a is None or bb is None:
                    continue
                f.write(f"{sc},{cname},{a},{bb},{round(bb-a,3)}\n")
    print("DONE")


if __name__ == "__main__":
    main()
