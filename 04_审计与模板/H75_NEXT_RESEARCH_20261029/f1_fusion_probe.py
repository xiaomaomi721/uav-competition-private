"""Offline single-checkpoint weight interpolation probe.

theta(alpha) = (1-a)*theta_CTRL + a*theta_TREAT  on float params of
segformer.stages.2/3 and decode_head; stages.0/1 copied (bitwise identical anyway).
Each alpha is evaluated as ONE synthesized checkpoint (never two models at inference).

Fixed alpha grid (pre-registered): 0, 0.25, 0.50, 0.75, 1.00
Evaluation: first 400 images of R1099 (fixed), real GT, D4-TTA8 + Bstar, bilinear 1024.
"""
import json
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

B = Path("/home/jhy/rare_diag_20260915/a05_blindrecall_pair_20260927")
CTRL = B / "05_formal_control/checkpoint_step2000.pt"
TREAT = B / "06_formal_treat/checkpoint_step2000.pt"
IMGD = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/train/images")
MSKD = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/train/masks")
OUT = Path("/home/jhy/aic2026_h75_next_research_20261029/01_复赛两模型融合")
BSTAR = np.array([-.2, .1, .0709, .4, 3.6, .4, 2.1215, -.5], dtype=np.float32)
CLS = ["Background", "Building", "Road", "Water", "Barren", "Vegetation", "Agricultural", "Vehicle"]
ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.00]
NIMG = 400
DEEP = ("segformer.stages.2.", "segformer.stages.3.", "decode_head.")
BUFFERY = ("running_mean", "running_var", "num_batches_tracked")


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


def state(p):
    o = torch.load(str(p), map_location="cpu", weights_only=False)
    for k in ("model", "student_model", "model_state_dict", "state_dict"):
        v = o.get(k) if isinstance(o, dict) else None
        if isinstance(v, dict) and v:
            return v
    return o


def blend(a, b, alpha):
    out = {}
    for k, x in a.items():
        y = b[k]
        if k.startswith(DEEP) and x.is_floating_point() and not any(k.endswith(s) for s in BUFFERY):
            out[k] = (x.double() + alpha * (y.double() - x.double())).to(x.dtype)
        else:
            out[k] = x.clone()
    return out


def miou(CM):
    inter = np.diag(CM).astype(np.float64)
    union = CM.sum(1) + CM.sum(0) - inter
    with np.errstate(all="ignore"):
        iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return float(np.nanmean(iou)) * 100, iou


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(SRC / "configs" / "base.yaml"))
    cfg["cwfa"] = {"enabled": False}; cfg["mixstyle"] = {"enabled": False}
    cfg["pretrained_model_path"] = str(ROOT / "models" / "segformer-b3-ade512")

    sp = json.loads((SRC / "splits" / "SOURCE_AUDIT_SPLIT.json").read_text())
    ids = sp["R1099"][:NIMG]

    sa, sb = state(CTRL), state(TREAT)
    assert set(sa) == set(sb)

    # cache images once
    cache = []
    for s in ids:
        rgb = np.asarray(Image.open(IMGD / f"{s}.png").convert("RGB")).copy()
        gt = official_to_train(np.asarray(Image.open(MSKD / f"{s}.png")).copy())
        cache.append((s, rgb, gt))
    print(f"cached {len(cache)} images", flush=True)

    b = torch.from_numpy(BSTAR).cuda()[None, :, None, None]
    results, preds = {}, {}
    for alpha in ALPHAS:
        st = blend(sa, sb, alpha)
        model = build_model(cfg)
        model.load_state_dict(st, strict=True)
        model = model.to("cuda:0").eval()
        CM = np.zeros((8, 8), np.int64)
        pm = {}
        for i, (s, rgb, gt) in enumerate(cache, 1):
            x = normalize_to_tensor(rgb).unsqueeze(0).cuda()
            acc = None
            with torch.inference_mode():
                for v in range(8):
                    z = inv_d4(model(pixel_values=d4(x, v)).logits.float(), v)
                    acc = z if acc is None else acc + z
                up = F.interpolate(acc / 8.0 + b, size=(1024, 1024), mode="bilinear",
                                   align_corners=False)
                p = up.argmax(1)[0].cpu().numpy().astype(np.uint8)
            pm[s] = p
            v = gt != 255
            CM += np.bincount((gt[v].astype(np.int64) * 8 + p[v]), minlength=64).reshape(8, 8)
            if i % 100 == 0:
                print(f"alpha={alpha} {i}/{len(cache)}", flush=True)
        m, iou = miou(CM)
        results[f"{alpha:.2f}"] = {"alpha": alpha, "mIoU": round(m, 4),
                                   "per_class_iou": [None if not np.isfinite(v) else round(float(v) * 100, 3) for v in iou],
                                   "confusion": CM.tolist()}
        preds[alpha] = pm
        del model
        torch.cuda.empty_cache()

    # disagreement & error complementarity vs alpha=0
    base = preds[0.0]
    extra = {}
    for alpha in ALPHAS:
        if alpha == 0.0:
            continue
        dis = comp = tot = 0
        for s, rgb, gt in cache:
            v = gt != 255
            p0 = base[s]; pa = preds[alpha][s]
            d = (p0 != pa) & v
            dis += int(d.sum()); tot += int(v.sum())
            # complementarity: pixels where exactly one of the two is correct
            c0 = (p0 == gt) & v
            ca = (pa == gt) & v
            comp += int((c0 ^ ca)[d].sum()) if d.any() else 0
        extra[f"{alpha:.2f}"] = {"disagree_px": dis, "disagree_pct": round(dis / tot * 100, 4),
                                 "complementary_px_in_disagreement": comp}

    res = {"pair": {"ctrl": str(CTRL), "treat": str(TREAT)},
           "alphas": ALPHAS, "n_images": len(cache), "subset": "R1099[:400]",
           "protocol": "D4-TTA8 + Bstar once + bilinear1024, single synthesized checkpoint",
           "results": results, "vs_alpha0": extra,
           "note": "CTRL=70.272000 and TREAT=70.867100 are round-2 online scores; "
                   "this pair is NOT confirmed to be the user's 70.3/70.13"}
    (OUT / "FUSION_PROBE.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    with open(OUT / "参数插值验证结果.csv", "w", encoding="utf-8") as f:
        f.write("alpha,mIoU," + ",".join(CLS) + ",disagree_pct,complementary_px\n")
        for k, v in results.items():
            ex = extra.get(k, {})
            f.write(f'{v["alpha"]},{v["mIoU"]},' +
                    ",".join("" if x is None else str(x) for x in v["per_class_iou"]) +
                    f',{ex.get("disagree_pct","")},{ex.get("complementary_px_in_disagreement","")}\n')
    print(json.dumps({k: {"mIoU": v["mIoU"]} for k, v in results.items()}, indent=2))
    print(json.dumps(extra, indent=2))


if __name__ == "__main__":
    main()
