#!/usr/bin/env python3
"""D1 pseudo-label audit on test1: teacher(theta278) vs student(D1)."""
import sys, json, time
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
TEST1_DIR = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/images")
OUT = Path("/home/jhy/rare_diag_20260915/d1_pseudo_audit_20260925")
OUT.mkdir(exist_ok=True)

BSTAR = [-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5]
VIEWS = [(k, hf) for k in range(4) for hf in (False, True)]
CLASS_NAMES = ["Background","Building","Road","Water","Barren","Vegetation","Agricultural","Vehicle"]

def d4_fwd(t, i):
    r,hf = VIEWS[i]
    if r: t = torch.rot90(t, r, (-2,-1))
    if hf: t = t.flip((-1,))
    return t
def d4_inv(t, i):
    r,hf = VIEWS[i]; t = t.contiguous()
    if hf: t = t.flip((-1,))
    if r: t = torch.rot90(t, -r, (-2,-1))
    return t.contiguous()

def load_state(path):
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for k in ("model","student_model","state_dict","model_state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt

def d4_mean_logits(model, px, device):
    acc = None
    with torch.no_grad():
        for vi in range(8):
            t = d4_fwd(px, vi)
            lo = model(pixel_values=t).logits
            canon = d4_inv(lo.float(), vi)[0]
            acc = canon if acc is None else acc + canon
    return acc / 8.0

def main():
    device = "cuda:0"
    print("Loading models...", flush=True)
    cfg = load_config(str(SNAP/"stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml"))
    cfg["cwfa"] = {"enabled":False,"probability":0.3,"epsilon":4.0,"norm_eps":1e-12,"start_step":1}
    cfg.pop("mixstyle", None)

    # Teacher = theta278
    teacher = make_model(cfg)
    teacher.load_state_dict(load_state(THETA278_CKPT), strict=True)
    teacher = teacher.to(device).eval()

    # Student = D1
    student = make_model(cfg)
    student.load_state_dict(load_state(D1_CKPT), strict=True)
    student = student.to(device).eval()
    print("Both models loaded", flush=True)

    bstar_t = torch.tensor(BSTAR, device=device).view(1,8,1,1)
    images = sorted(TEST1_DIR.glob("test1_*.png"), key=lambda p: int(p.stem.split("_")[-1]))
    print(f"{len(images)} test1 images", flush=True)

    # Accumulators
    total_pixels = 0
    conf_sum = np.zeros(8)  # per-class confidence sum
    class_pixels = np.zeros(8)  # per-class total pixels
    pseudo_high = np.zeros(8)  # per-class pixels with conf > 0.9
    pseudo_med = np.zeros(8)   # conf > 0.7
    conf_hist = np.zeros(6)  # 0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9, 0.9-0.95, 0.95-1.0
    flip_pixels = 0
    flip_in_pseudo = 0  # flips where teacher was confident (>0.9)
    flip_out_pseudo = 0  # flips where teacher was uncertain (<0.7)

    t0 = time.time()
    for idx, ip in enumerate(images):
        img = np.asarray(Image.open(ip).convert("RGB"))
        px = normalize_to_tensor(img).unsqueeze(0).to(device)

        # Teacher logits at 256x256
        t_logits = d4_mean_logits(teacher, px, device)  # [8,256,256]
        # Student logits
        s_logits = d4_mean_logits(student, px, device)

        # Apply Bstar for teacher prediction
        t_logits_b = t_logits + bstar_t[0]
        t_prob = F.softmax(t_logits_b, dim=0)  # [8,256,256]
        t_conf, t_pred = t_prob.max(0)  # [256,256]
        s_pred = (s_logits + bstar_t[0]).argmax(0)  # [256,256]

        t_conf_np = t_conf.cpu().numpy()
        t_pred_np = t_pred.cpu().numpy()
        s_pred_np = s_pred.cpu().numpy()

        total_pixels += t_conf_np.size
        flipped = (t_pred_np != s_pred_np)
        flip_pixels += flipped.sum()

        for c in range(8):
            mask = (t_pred_np == c)
            n = mask.sum()
            class_pixels[c] += n
            conf_sum[c] += t_conf_np[mask].sum()
            pseudo_high[c] += ((t_conf_np > 0.9) & mask).sum()
            pseudo_med[c] += ((t_conf_np > 0.7) & mask).sum()

        # Confidence histogram
        for i, (lo, hi) in enumerate([(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,0.95),(0.95,1.0)]):
            conf_hist[i] += ((t_conf_np >= lo) & (t_conf_np < hi)).sum()

        # Flip overlap
        flip_in_pseudo += (flipped & (t_conf_np > 0.9)).sum()
        flip_out_pseudo += (flipped & (t_conf_np < 0.7)).sum()

        if (idx+1) % 100 == 0:
            print(f"  {idx+1}/500 {time.time()-t0:.0f}s", flush=True)

    # Results
    result = {
        "total_target_pixels": int(total_pixels),
        "class_pixels": class_pixels.tolist(),
        "class_share_pct": (class_pixels/class_pixels.sum()*100).tolist(),
        "class_mean_conf": (conf_sum/np.maximum(class_pixels,1)).tolist(),
        "class_pseudo_high_0.9": pseudo_high.tolist(),
        "class_pseudo_high_0.9_pct": (pseudo_high/np.maximum(class_pixels,1)*100).tolist(),
        "class_pseudo_high_0.7_pct": (pseudo_med/np.maximum(class_pixels,1)*100).tolist(),
        "confidence_hist": {
            "bins": ["0.5-0.6","0.6-0.7","0.7-0.8","0.8-0.9","0.9-0.95","0.95-1.0"],
            "counts": conf_hist.tolist(),
            "pct": (conf_hist/conf_hist.sum()*100).tolist()
        },
        "flip_stats": {
            "total_flip_pixels": int(flip_pixels),
            "flip_rate": float(flip_pixels/total_pixels),
            "flip_in_high_conf_gt0.9": int(flip_in_pseudo),
            "flip_in_low_conf_lt0.7": int(flip_out_pseudo),
            "flip_high_conf_pct": float(flip_in_pseudo/max(flip_pixels,1)),
            "flip_low_conf_pct": float(flip_out_pseudo/max(flip_pixels,1)),
        }
    }

    print(f"\n=== PSEUDO-LABEL AUDIT ===")
    print(f"Total target pixels: {total_pixels:,}")
    print(f"\nClass shares (teacher predictions on test1):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<15}: {class_pixels[i]:>12,} ({class_pixels[i]/class_pixels.sum()*100:>6.2f}%)  "
              f"mean_conf={conf_sum[i]/max(class_pixels[i],1):.3f}  "
              f">0.9: {pseudo_high[i]/max(class_pixels[i],1)*100:.1f}%")

    print(f"\nConfidence histogram:")
    for i, b in enumerate(["0.5-0.6","0.6-0.7","0.7-0.8","0.8-0.9","0.9-0.95","0.95-1.0"]):
        print(f"  {b}: {conf_hist[i]:>12,} ({conf_hist[i]/conf_hist.sum()*100:.1f}%)")

    print(f"\nFlip (teacher vs D1 on test1):")
    print(f"  Total flip: {flip_pixels:,} ({flip_pixels/total_pixels*100:.3f}%)")
    print(f"  Flip in high-conf(>0.9): {flip_in_pseudo:,} ({flip_in_pseudo/max(flip_pixels,1)*100:.1f}%)")
    print(f"  Flip in low-conf(<0.7): {flip_out_pseudo:,} ({flip_out_pseudo/max(flip_pixels,1)*100:.1f}%)")

    (OUT/"pseudo_label_audit.json").write_text(json.dumps(result, indent=2))
    print(f"\nSaved to {OUT/'pseudo_label_audit.json'}")

if __name__ == "__main__":
    main()
