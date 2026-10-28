#!/usr/bin/env python3
"""Ensemble analysis: softmax fusion of theta278, D1, alpha050."""
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
SPLIT_JSON = Path("/home/jhy/rare_diag_20260915/round2_multiprobe_20260922/00_inventory/SOURCE_AUDIT_SPLIT.json")
GT_DIR = Path("/home/jhy/rare_diag_20260915/b2_val_20260917/gt")
SOURCE_IMG_DIR = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/train/images")
OUT = Path("/home/jhy/rare_diag_20260915/ensemble_20260925")
OUT.mkdir(exist_ok=True)

BSTAR = [-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5]

def load_state(p):
    ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for k in ("model","student_model","state_dict","model_state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt

def main():
    device = "cuda:0"
    cfg = load_config(str(SNAP/"stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml"))
    cfg["cwfa"] = {"enabled":False,"probability":0.3,"epsilon":4.0,"norm_eps":1e-12,"start_step":1}
    cfg.pop("mixstyle", None)

    print("Loading theta278...", flush=True)
    m278 = make_model(cfg)
    m278.load_state_dict(load_state(THETA278_CKPT), strict=True)
    m278 = m278.to(device).eval()
    print("Loading D1...", flush=True)
    md1 = make_model(cfg)
    md1.load_state_dict(load_state(D1_CKPT), strict=True)
    md1 = md1.to(device).eval()
    bstar_t = torch.tensor(BSTAR, device=device).view(1,8,1,1)

    split = json.loads(SPLIT_JSON.read_text())
    gt_files = {p.stem: p for p in GT_DIR.glob("*.npy")}

    for split_name in ["H300", "R1099"]:
        items = split[split_name]
        # Collect logits for both models
        all_p278 = []  # list of [8,256,256] numpy
        all_pd1 = []
        all_gt = []
        t0 = time.time()
        for i, stem in enumerate(items):
            img_path = SOURCE_IMG_DIR / f"{stem}.png"
            gt_path = gt_files.get(stem)
            if not img_path.is_file() or gt_path is None: continue
            img = np.asarray(Image.open(img_path).convert("RGB"))
            gt = np.load(gt_path).astype(np.int64)
            if gt.ndim == 3: gt = gt[0]
            gt_small = np.array(Image.fromarray(gt.astype(np.uint8)).resize((256,256), Image.NEAREST))

            px = normalize_to_tensor(img).unsqueeze(0).to(device)
            with torch.no_grad():
                lo278 = m278(pixel_values=px).logits.float()[0]  # [8,256,256]
                lod1 = md1(pixel_values=px).logits.float()[0]
            all_p278.append(lo278.cpu().numpy())
            all_pd1.append(lod1.cpu().numpy())
            all_gt.append(gt_small)
            if (i+1) % 200 == 0:
                print(f"  {split_name} {i+1}/{len(items)} {time.time()-t0:.0f}s", flush=True)

        # Now evaluate ensemble combos
        def miou_from_probs(probs_list):
            # probs_list: list of [8,256,256] numpy
            ious = np.zeros((8,2))
            for prob, gt in zip(probs_list, all_gt):
                pred = prob.argmax(0)
                for c in range(8):
                    ious[c,0] += ((pred==c)&(gt==c)).sum()
                    ious[c,1] += ((pred==c)|(gt==c)).sum()
            return float(np.mean(ious[:,0]/np.maximum(ious[:,1],1)))

        # Precompute softmax with Bstar
        probs_278 = []
        probs_d1 = []
        for lo in all_p278:
            lt = torch.from_numpy(lo).unsqueeze(0) + bstar_t.cpu()
            probs_278.append(F.softmax(lt, dim=1)[0].numpy())
        for lo in all_pd1:
            lt = torch.from_numpy(lo).unsqueeze(0) + bstar_t.cpu()
            probs_d1.append(F.softmax(lt, dim=1)[0].numpy())

        # alpha050 = 0.5*278 + 0.5*d1 in logit space, then softmax
        probs_a05 = []
        for lo278, lod1 in zip(all_p278, all_pd1):
            la = 0.5*lo278 + 0.5*lod1
            lt = torch.from_numpy(la).unsqueeze(0) + bstar_t.cpu()
            probs_a05.append(F.softmax(lt, dim=1)[0].numpy())

        combos = [
            ("theta278 only", probs_278),
            ("D1 only", probs_d1),
            ("alpha050 only", probs_a05),
            ("a05+D1 w=0.75", [0.75*a+0.25*d for a,d in zip(probs_a05, probs_d1)]),
            ("a05+D1 w=0.5",  [0.5*a+0.5*d for a,d in zip(probs_a05, probs_d1)]),
            ("a05+278 w=0.75",[0.75*a+0.25*t for a,t in zip(probs_a05, probs_278)]),
            ("a05+278 w=0.5", [0.5*a+0.5*t for a,t in zip(probs_a05, probs_278)]),
            ("278+D1 w=0.5",  [0.5*t+0.5*d for t,d in zip(probs_278, probs_d1)]),
        ]

        print(f"\n=== {split_name} Ensemble Results ===")
        print(f"{'Combo':<25} {'mIoU':>10}")
        for name, probs in combos:
            m = miou_from_probs(probs)
            print(f"{name:<25} {m:>10.5f}")

if __name__ == "__main__":
    main()
