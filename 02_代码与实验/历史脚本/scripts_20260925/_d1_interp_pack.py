#!/usr/bin/env python3
"""Run D1 D4 TTA8 on test2, interpolate with theta278 cache, package alpha=0.5/0.75."""
import sys, json, time, hashlib, zipfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SNAP = Path("/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1")
HELPER = Path("/home/jhy/rare_diag_20260915/round2_20260921/scripts")
sys.path.insert(0, str(SNAP))
sys.path.insert(0, str(HELPER))

from src.models import build_model as make_model
from src.utils import load_config
from src.transforms import normalize_to_tensor

D1_CKPT = Path("/home/jhy/rare_diag_20260915/round2_multiprobe_20260922/03_dart_fixed/D1/checkpoint_step2000.pt")
THETA278_CACHE = Path("/home/jhy/rare_diag_20260915/round2_20260921/cache_278_test2_raw/scores")
TEST2_DIR = Path("/home/jhy/datasets/AIC2026_UAV_semifinal/test_2/images")
OUT = Path("/home/jhy/rare_diag_20260915/d1_interp_20260925")

BSTAR = [-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5]
VIEWS = [(k, hf) for k in range(4) for hf in (False, True)]

def d4_fwd(t, i):
    r, hf = VIEWS[i]
    if r: t = torch.rot90(t, r, (-2, -1))
    if hf: t = t.flip((-1,))
    return t

def d4_inv(t, i):
    r, hf = VIEWS[i]; t = t.contiguous()
    if hf: t = t.flip((-1,))
    if r: t = torch.rot90(t, -r, (-2, -1))
    return t.contiguous()

def load_d1_state(path):
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for k in ("model","student_model","state_dict","model_state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt

def main():
    device = "cuda:0"
    print("Loading D1 model...", flush=True)
    cfg = load_config(str(SNAP/"stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml"))
    cfg["cwfa"] = {"enabled":False,"probability":0.3,"epsilon":4.0,"norm_eps":1e-12,"start_step":1}
    cfg.pop("mixstyle", None)
    model = make_model(cfg)
    sd = load_d1_state(D1_CKPT)
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()
    print("D1 loaded", flush=True)

    bstar = torch.tensor(BSTAR, dtype=torch.float32, device=device).view(1,8,1,1)
    images = sorted(TEST2_DIR.glob("*.png"), key=lambda p: int(p.stem.split("_")[-1]))
    print(f"{len(images)} images", flush=True)

    alphas = [0.5, 0.75]
    preds = {a: [] for a in alphas}
    d1_logits_list = []
    t0 = time.time()

    for idx, ip in enumerate(images):
        img = np.asarray(Image.open(ip).convert("RGB"))
        px = normalize_to_tensor(img).unsqueeze(0).to(device)
        acc = None
        with torch.no_grad():
            for vi in range(8):
                t = d4_fwd(px, vi)
                lo = model(pixel_values=t).logits
                canon = d4_inv(lo.float(), vi)[0]
                acc = canon if acc is None else acc + canon
        d1_logits = (acc / 8.0).cpu()  # [8,256,256]
        d1_logits_list.append(d1_logits)

        # Load theta278 cache
        stem = ip.stem  # test2_1
        w278 = np.load(THETA278_CACHE / f"{stem}.npy").astype(np.float32)  # [8,256,256]

        for alpha in alphas:
            interp = w278 + alpha * (d1_logits.numpy() - w278)
            lt = torch.from_numpy(interp).unsqueeze(0).to(device) + bstar
            lu = F.interpolate(lt, size=(1024,1024), mode="bilinear", align_corners=False)
            pred = lu[0].argmax(0).cpu().numpy().astype(np.uint8) + 1
            preds[alpha].append(pred)

        if (idx+1) % 200 == 0:
            print(f"  {idx+1}/1300 {time.time()-t0:.0f}s", flush=True)

    # Package
    for alpha in alphas:
        tag = f"alpha{int(alpha*100):03d}"
        pred_dir = OUT / tag / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        for i, (ip, pred) in enumerate(zip(images, preds[alpha])):
            Image.fromarray(pred, mode="L").save(pred_dir / ip.name)

        zp = OUT / f"D1_{tag}_test2.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
            for ip in sorted(pred_dir.glob("*.png"), key=lambda p: int(p.stem.split("_")[-1])):
                zf.write(ip, ip.name)
        h = hashlib.sha256()
        with open(zp, "rb") as f:
            for b in iter(lambda: f.read(1<<20), b""): h.update(b)

        # Class shares
        shares = np.zeros(8)
        for p in preds[alpha]:
            for c in range(1,9): shares[c-1] += int((p==c).sum())
        share_pct = (shares / shares.sum() * 100).tolist()
        print(f"\n{tag}: {zp.name}")
        print(f"  SHA256: {h.hexdigest()}")
        print(f"  shares: {[round(x,3) for x in share_pct]}")

    print(f"\nTotal time: {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
