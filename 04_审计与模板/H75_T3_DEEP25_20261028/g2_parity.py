"""G2.1 parity: replay H75 through the production packager path (WITH autocast)
on 10 fixed test3 images and require pixel-identical match against the
officially scored H75_BSTAR_semifinal.zip.
"""
import csv, hashlib, json, sys, zipfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path("/home/jhy/aic2026_h76_20261026")
SRC = ROOT / "code" / "h76_3594"
sys.path.insert(0, str(SRC))
import os
os.environ.setdefault("UAVSEG_MODEL_SOURCE", str(ROOT / "models" / "segformer-b3-ade512"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.models import build_model
from src.utils import load_config
from src.transforms import normalize_to_tensor

H75 = ROOT / "models" / "H75.pt"
OFFICIAL_ZIP = Path("/home/jhy/aic2026_ctrl3594_teacher_20261027/reference/H75_BSTAR_20261026/H75_BSTAR_semifinal.zip")
BSTAR = np.array([-.2, .1, .0709, .4, 3.6, .4, 2.1215, -.5], dtype=np.float32)
OUT = Path("/home/jhy/aic2026_h75_t3_deep25_20261028")
N = 10


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


def main():
    cfg = load_config(str(SRC / "configs" / "base.yaml"))
    cfg["cwfa"] = {"enabled": False}; cfg["mixstyle"] = {"enabled": False}
    cfg["pretrained_model_path"] = str(ROOT / "models" / "segformer-b3-ade512")
    model = build_model(cfg)
    model.load_state_dict(torch.load(str(H75), map_location="cpu", weights_only=False).get("model"), strict=True)
    model = model.to("cuda:0").eval()

    rows = list(csv.DictReader(open(ROOT / "manifests" / "semifinal_3594.csv")))[:N]
    b = torch.from_numpy(BSTAR).cuda()[None, :, None, None]
    diffs = []
    zf = zipfile.ZipFile(OFFICIAL_ZIP)
    for r in rows:
        img = np.asarray(Image.open(ROOT / "data" / "test3" / "images" / r["input_name"]).convert("RGB")).copy()
        tx = normalize_to_tensor(img).unsqueeze(0).cuda()
        acc = None
        with torch.inference_mode():
            for v in range(8):
                with torch.amp.autocast("cuda", enabled=True):   # SAME as packager
                    lz = model(pixel_values=d4(tx, v)).logits
                z = inv_d4(lz.float(), v)
                acc = z if acc is None else acc + z
            logits = acc / 8.0 + b
            pred = F.interpolate(logits, size=(1024, 1024), mode="bilinear",
                                 align_corners=False).argmax(1).add(1).to(torch.uint8)[0].cpu().numpy()
        with zf.open(r["output_name"]) as f:
            ref = np.array(Image.open(f))
        diffs.append(int((pred != ref).sum()))

    res = {"official_zip": str(OFFICIAL_ZIP),
           "official_zip_sha256": hashlib.sha256(open(OFFICIAL_ZIP, "rb").read()).hexdigest(),
           "images": N, "diff_pixels_per_image": diffs,
           "diff_total": int(sum(diffs)), "parity_pass": int(sum(diffs)) == 0}
    (OUT / "G2_PARITY.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
