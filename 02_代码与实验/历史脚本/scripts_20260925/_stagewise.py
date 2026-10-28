#!/usr/bin/env python3
"""Stage-wise alpha interpolation: separate alpha for stage2 and stage3."""
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

THETA278_CKPT = Path("/home/jhy/rare_diag_20260915/atda_20260918/cpu_candidates/f_s2p100_s3p125.pt")
D1_CKPT = Path("/home/jhy/rare_diag_20260915/round2_multiprobe_20260922/03_dart_fixed/D1/checkpoint_step2000.pt")
SPLIT_JSON = Path("/home/jhy/rare_diag_20260915/round2_multiprobe_20260922/00_inventory/SOURCE_AUDIT_SPLIT.json")
GT_DIR = Path("/home/jhy/rare_diag_20260915/b2_val_20260917/gt")
SOURCE_IMG_DIR = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/train/images")
OUT = Path("/home/jhy/rare_diag_20260915/stagewise_interp_20260925")
OUT.mkdir(exist_ok=True)

BSTAR = [-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5]
CLASS_NAMES = ["Background","Building","Road","Water","Barren","Vegetation","Agricultural","Vehicle"]

def load_state(p):
    ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for k in ("model","student_model","state_dict","model_state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt

def stage_of(name):
    n = name.lower()
    if "stages.2" in n: return "s2"
    if "stages.3" in n: return "s3"
    return "other"

def build_interp(s278, sd1, a2, a3):
    new = {}
    for k in s278:
        if k not in sd1: continue
        if "num_batches_tracked" in k:
            new[k] = s278[k]; continue
        if not torch.is_floating_point(s278[k]):
            new[k] = s278[k]; continue
        s = stage_of(k)
        if s == "s2":
            new[k] = s278[k].float() + a2 * (sd1[k].float() - s278[k].float())
        elif s == "s3":
            new[k] = s278[k].float() + a3 * (sd1[k].float() - s278[k].float())
        else:
            new[k] = s278[k].float()
    return new

def evaluate(model, items, gt_files, device, bstar_t):
    model.eval()
    ious = np.zeros((8,2))  # [intersection, union] per class
    for stem in items:
        img_path = SOURCE_IMG_DIR / f"{stem}.png"
        gt_path = gt_files.get(stem)
        if not img_path.is_file() or gt_path is None: continue
        img = np.asarray(Image.open(img_path).convert("RGB"))
        gt = np.load(gt_path).astype(np.int64)
        if gt.ndim == 3: gt = gt[0]
        gt_small = np.array(Image.fromarray(gt.astype(np.uint8)).resize((256,256), Image.NEAREST))
        px = normalize_to_tensor(img).unsqueeze(0).to(device)
        with torch.no_grad():
            lo = model(pixel_values=px).logits
        lo = lo.float() + bstar_t
        pred = lo[0].argmax(0).cpu().numpy()
        for c in range(8):
            ious[c,0] += ((pred==c)&(gt_small==c)).sum()
            ious[c,1] += ((pred==c)|(gt_small==c)).sum()
    miou = np.mean(ious[:,0]/np.maximum(ious[:,1],1))
    return miou

def main():
    device = "cuda:0"
    cfg = load_config(str(SNAP/"stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml"))
    cfg["cwfa"] = {"enabled":False,"probability":0.3,"epsilon":4.0,"norm_eps":1e-12,"start_step":1}
    cfg.pop("mixstyle", None)

    print("Loading states...", flush=True)
    s278 = load_state(THETA278_CKPT)
    sd1 = load_state(D1_CKPT)
    bstar_t = torch.tensor(BSTAR, device=device).view(1,8,1,1)

    split = json.loads(SPLIT_JSON.read_text())
    gt_files = {p.stem: p for p in GT_DIR.glob("*.npy")}

    # Combos: (name, alpha2, alpha3)
    combos = [
        ("A_s2=0.5_s3=0.5", 0.5, 0.5),
        ("B_s2=0.75_s3=0.5", 0.75, 0.5),
        ("C_s2=0.5_s3=0.25", 0.5, 0.25),
        ("D_s2=1.0_s3=0.5", 1.0, 0.5),
        ("E_s2=0.75_s3=0.25", 0.75, 0.25),
    ]

    results = []
    for name, a2, a3 in combos:
        print(f"\n=== {name} (a2={a2}, a3={a3}) ===", flush=True)
        interp = build_interp(s278, sd1, a2, a3)
        model = make_model(cfg)
        model.load_state_dict(interp, strict=True)
        model = model.to(device).eval()

        h300 = evaluate(model, split["H300"], gt_files, device, bstar_t)
        r1099 = evaluate(model, split["R1099"], gt_files, device, bstar_t)
        print(f"  H300: {h300:.5f}, R1099: {r1099:.5f}", flush=True)
        results.append({"name":name, "a2":a2, "a3":a3, "h300":h300, "r1099":r1099})
        del model; torch.cuda.empty_cache()

    print(f"\n=== STAGE-WISE SUMMARY ===")
    print(f"{'Combo':<25} {'a2':>5} {'a3':>5} {'H300':>10} {'R1099':>10}")
    for r in results:
        print(f"{r['name']:<25} {r['a2']:>5.2f} {r['a3']:>5.2f} {r['h300']:>10.5f} {r['r1099']:>10.5f}")

    (OUT/"stagewise_results.json").write_text(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
