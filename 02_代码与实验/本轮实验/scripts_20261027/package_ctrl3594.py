#!/usr/bin/env python3
"""CTRL@3594 test3 推理打包：一次前向，产出两个 bias 变体的 ZIP。

协议严格对齐已评分的 H75_BSTAR_semifinal.zip：
  D4-TTA8 [(r,hf) for r in range(4) for hf in (False,True)]
  正向先 rot90(r) 再水平 flip；逆向先水平 flip 再 rot90(-r)
  FP32 native(256x256) logits 八视图平均
  +Bstar 在 native 上施加一次
  bilinear 上采样 1024，align_corners=False
  argmax -> +1 -> 官方 1..8；mode L PNG；根目录平铺 ZIP
"""
import argparse, csv, hashlib, json, os, sys, time, zipfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path("/home/jhy/aic2026_h76_20261026")
CODE = ROOT / "code" / "h76_3594"
sys.path.insert(0, str(CODE))
os.environ.setdefault("UAVSEG_MODEL_SOURCE", str(ROOT / "models" / "segformer-b3-ade512"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.models import build_model
from src.utils import load_config
from src.transforms import normalize_to_tensor

BSTAR = np.array([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], dtype=np.float32)
EXPECTED_N = 3594
OFFICIAL_SIZE = (1024, 1024)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def d4_forward(x, view):
    k, hf = divmod(view, 2)
    if k:
        x = torch.rot90(x, k, (-2, -1))
    if hf:
        x = x.flip((-1,))
    return x


def d4_inverse(z, view):
    k, hf = divmod(view, 2)
    if hf:
        z = z.flip((-1,))
    if k:
        z = torch.rot90(z, -k, (-2, -1))
    return z.contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images-dir", default=str(ROOT / "data" / "test3" / "images"))
    ap.add_argument("--manifest", default=str(ROOT / "manifests" / "semifinal_3594.csv"))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bias-variants", default="bstar,agri0",
                    help="逗号分隔: bstar | agri0")
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")

    ckpt = Path(a.checkpoint)
    ckpt_sha = sha256_file(ckpt)

    biases = {}
    for v in a.bias_variants.split(","):
        v = v.strip()
        if v == "bstar":
            biases["BSTAR"] = BSTAR.copy()
        elif v == "agri0":
            b = BSTAR.copy(); b[6] = 0.0
            biases["AGRI0"] = b
        else:
            raise SystemExit(f"unknown bias variant {v}")

    cfg = load_config(str(CODE / "configs" / "base.yaml"))
    cfg["cwfa"] = {"enabled": False}; cfg["mixstyle"] = {"enabled": False}
    cfg["pretrained_model_path"] = str(ROOT / "models" / "segformer-b3-ade512")
    model = build_model(cfg)
    st = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    model.load_state_dict(st.get("model", st), strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rows = list(csv.DictReader(open(a.manifest)))
    names = [r["output_name"] for r in rows]
    inputs = [r["input_name"] for r in rows]
    assert len(names) == EXPECTED_N, f"manifest {len(names)} != {EXPECTED_N}"
    assert len(set(names)) == len(names), "duplicate output names"

    img_dir = Path(a.images_dir)
    pred_dirs = {}
    for tag in biases:
        d = out / f"predictions_{tag}"
        d.mkdir(parents=True, exist_ok=True)
        pred_dirs[tag] = d

    t0 = time.time()
    for i, (inp, outn) in enumerate(zip(inputs, names)):
        im = np.asarray(Image.open(img_dir / inp).convert("RGB")).copy()
        x = normalize_to_tensor(im).unsqueeze(0).to(device)
        acc = None
        with torch.inference_mode():
            for view in range(8):
                v = d4_forward(x, view)
                with torch.amp.autocast("cuda", enabled=True):
                    lg = model(pixel_values=v).logits
                lg = d4_inverse(lg.float(), view)
                acc = lg if acc is None else acc + lg
            logits = acc / 8.0  # native 256x256, FP32 mean
            for tag, b in biases.items():
                bt = torch.from_numpy(b).to(device)[None, :, None, None]
                biased = logits + bt
                up = F.interpolate(biased, size=OFFICIAL_SIZE, mode="bilinear", align_corners=False)
                pred = up.argmax(1).add(1).to(torch.uint8).cpu().numpy()[0]
                assert pred.min() >= 1 and pred.max() <= 8, f"illegal ids {pred.min()}-{pred.max()}"
                Image.fromarray(pred, mode="L").save(pred_dirs[tag] / outn, format="PNG")
        if (i + 1) % 200 == 0:
            print(f"{i+1}/{EXPECTED_N} elapsed={round(time.time()-t0,1)}s", flush=True)

    # QA + ZIP
    report = {"checkpoint": str(ckpt), "checkpoint_sha256": ckpt_sha,
              "images": EXPECTED_N, "elapsed_sec": round(time.time() - t0, 1),
              "variants": {}}
    for tag, d in pred_dirs.items():
        files = sorted(p.name for p in d.glob("*.png"))
        assert files == sorted(names), f"{tag} name set mismatch"
        bad = []
        for n in files[:]:
            with Image.open(d / n) as h:
                if h.mode != "L" or h.size != OFFICIAL_SIZE:
                    bad.append((n, h.mode, h.size))
        zp = out / f"{a.tag}_{tag}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for n in files:
                z.write(d / n, arcname=n)
        report["variants"][tag] = {
            "count": len(files), "bad_format": bad[:5], "bad_count": len(bad),
            "zip": str(zp), "zip_sha256": sha256_file(zp), "zip_bytes": zp.stat().st_size,
            "bias": biases[tag].tolist(),
        }
        print(f"[{tag}] {len(files)} png, bad={len(bad)}, zip={zp}", flush=True)
    (out / "PACKAGE_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
