#!/usr/bin/env python3
"""D1 direction interpolation: alpha sweep on H300/R1099."""
import sys, json, time, copy
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

THETA278 = Path("/home/jhy/rare_diag_20260915/atda_20260918/cpu_candidates/f_s2p100_s3p125.pt")
D1 = Path("/home/jhy/rare_diag_20260915/round2_multiprobe_20260922/03_dart_fixed/D1/checkpoint_step2000.pt")
GT_DIR = Path("/home/jhy/rare_diag_20260915/b2_val_20260917/gt")
SPLIT = Path("/home/jhy/rare_diag_20260915/round2_multiprobe_20260922/00_inventory/SOURCE_AUDIT_SPLIT.json")
# Holdout images: find them
IMG_DIR = Path("/home/jhy/rare_diag_20260915/b2_val_20260917")
BSTAR = [-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5]
OUT = Path("/home/jhy/rare_diag_20260915/d1_interp_20260925")
OUT.mkdir(exist_ok=True)

CLASS_NAMES = ["Bg","Bld","Rd","Wat","Bar","Veg","Ag","Veh"]

def load_state(path):
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for k in ("model","student_model","state_dict","model_state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt

def iou_from_confusion(conf):
    ious = []
    for c in range(8):
        tp = conf[c,c]
        fp = conf[:,c].sum() - tp
        fn = conf[c,:].sum() - tp
        ious.append(tp / max(tp+fp+fn, 1))
    return np.mean(ious), ious

def eval_model(model, stems, device, gt_dir):
    """Single-view eval at native resolution, +Bstar, bilinear to GT size."""
    conf = np.zeros((8,8), dtype=np.int64)
    model.eval()
    t0 = time.time()
    for i, stem in enumerate(stems):
        # Find image
        img_path = Path(f"/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/train/images/{stem}.png")
        if not img_path.exists():
            print(f"  MISSING image for {stem}")
            continue
        img = np.asarray(Image.open(img_path).convert("RGB"))
        gt = np.load(gt_dir / f"{stem}.npy")  # H,W uint8 1..8 or 0..7
        gt = gt.astype(int)
        if gt.max() > 7:
            gt = gt - 1  # if 1..8, shift to 0..7

        t = normalize_to_tensor(img).unsqueeze(0).to(device)
        with torch.no_grad():
            lo = model(pixel_values=t).logits  # [1,8,h,w]
        lo = lo[0].float()
        lo = lo + torch.tensor(BSTAR, device=device).view(8,1,1)
        # Resize logits to GT size
        lo_up = F.interpolate(lo.unsqueeze(0), size=gt.shape, mode="bilinear", align_corners=False)
        pred = lo_up[0].argmax(0).cpu().numpy()
        # Update confusion
        valid = gt >= 0
        for gc in range(8):
            for pc in range(8):
                conf[gc,pc] += int(((gt==gc) & (pred==pc)).sum())
        if (i+1) % 100 == 0:
            print(f"    {i+1}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
    miou, ious = iou_from_confusion(conf)
    return miou, ious, conf

def main():
    device = "cuda:0"
    print("Loading theta278 and D1 states...")
    s278 = load_state(THETA278)
    sD1 = load_state(D1)
    print(f"  theta278 keys: {len(s278)}, D1 keys: {len(sD1)}")

    # Load split
    split = json.loads(SPLIT.read_text())
    h300 = split.get("H300", split.get("source_split",{}).get("H300",[]))
    r1099 = split.get("R1099", split.get("source_split",{}).get("R1099",[]))
    print(f"  H300: {len(h300)}, R1099: {len(r1099)}")

    # Build model
    cfg = load_config(str(SNAP/"stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml"))
    cfg["cwfa"] = {"enabled":False,"probability":0.3,"epsilon":4.0,"norm_eps":1e-12,"start_step":1}
    cfg.pop("mixstyle", None)

    alphas = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5]
    results = {}

    for alpha in alphas:
        name = f"alpha{alpha:.2f}"
        print(f"\n=== {name} ===", flush=True)
        model = make_model(cfg)
        sd = {}
        for k in s278:
            if k in sD1:
                d = (sD1[k].float() - s278[k].float())
                sd[k] = (s278[k].float() + alpha * d)
            else:
                sd[k] = s278[k].float()
        model.load_state_dict(sd, strict=True)
        model = model.to(device).eval()

        # H300
        print(f"  H300...", flush=True)
        h300_miou, h300_ious, _ = eval_model(model, h300, device, GT_DIR)
        # R1099
        print(f"  R1099...", flush=True)
        r1099_miou, r1099_ious, _ = eval_model(model, r1099, device, GT_DIR)

        results[name] = {
            "alpha": alpha,
            "h300_miou": h300_miou,
            "r1099_miou": r1099_miou,
            "h300_per_class": h300_ious,
            "r1099_per_class": r1099_ious,
        }
        print(f"  H300: {h300_miou:.6f}, R1099: {r1099_miou:.6f}", flush=True)

        # Free GPU memory
        del model, sd
        torch.cuda.empty_cache()

    # Print summary
    print("\n\n=== INTERPOLATION SUMMARY ===")
    print(f"{'alpha':>8} {'H300':>10} {'R1099':>10}")
    for name, r in results.items():
        print(f"{r['alpha']:>8.2f} {r['h300_miou']:>10.6f} {r['r1099_miou']:>10.6f}")

    (OUT/"interp_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {OUT/'interp_results.json'}")

if __name__ == "__main__":
    main()
