#!/usr/bin/env python3
"""Fixed H300 theta278 geometry screen: G1 overlap crops and G2 1.25x.

G0 is always loaded from the exact theta278 source-holdout cache.  This script
never forwards G0 and never touches test2 or creates a submission package.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from _r2_contracts import (BSTAR, CONFIG, DATA, EXPECTED_FILE_SHA, OUT, SNAP,
                           THETA278, d4_forward, d4_inverse, load_state,
                           sha256_file, write_json)

TILE = 768
STARTS = (0, 256)
SCALE = 1.25


def build_model(device: str):
    from src.models import build_model as make_model
    from src.utils import load_config
    cfg = load_config(str(CONFIG))
    cfg["cwfa"] = {"enabled": False, "probability": .3, "epsilon": 4.,
                   "norm_eps": 1e-12, "start_step": 1}
    state, _ = load_state(THETA278)
    model = make_model(cfg); model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def d4_mean(model, px):
    import torch
    acc = None
    with torch.inference_mode():
        for i in range(8):
            with torch.amp.autocast("cuda", enabled=False):
                z = model(pixel_values=d4_forward(px, i)).logits.float()
            z = d4_inverse(z, i)[0]
            acc = z if acc is None else acc + z
    return acc / 8.0


def resize_logits(z, size):
    import torch.nn.functional as F
    return F.interpolate(z[None], size=size, mode="bilinear", align_corners=False)[0]


def confusion(pred, gt):
    valid = gt != 255
    x = gt[valid].astype(np.int64) * 8 + pred[valid].astype(np.int64)
    return np.bincount(x, minlength=64).reshape(8, 8)


def center_weight(device):
    import torch
    # Positive floor avoids division instability at outer image edges.
    one = torch.hann_window(TILE, periodic=False, device=device, dtype=torch.float32)
    return (.25 + .75 * (one[:, None] * one[None, :]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise SystemExit("geometry is scheduled on physical GPU3")
    for p in (str(SNAP), "/home/jhy/gpu2_day_queue"):
        if p not in sys.path: sys.path.insert(0, p)
    import torch
    from PIL import Image
    from src.transforms import normalize_to_tensor

    if sha256_file(THETA278) != EXPECTED_FILE_SHA["theta278"]:
        raise SystemExit("theta278 SHA mismatch")
    split = json.loads((OUT / "00_inventory/SOURCE_AUDIT_SPLIT.json").read_text())
    names = split["H300"]
    g0_dir = OUT / "01_holdout_cache/theta278/scores"
    if not (OUT / "01_holdout_cache/theta278/FORWARD_DONE.json").is_file():
        raise SystemExit("theta278 holdout cache prerequisite missing; G0 may not be re-forwarded")
    out = OUT / "04_geometry/H300"
    if out.exists(): raise SystemExit(f"outdir exists: {out}")
    for tag in ("G0", "G1", "G2"): (out / f"pred_{tag}").mkdir(parents=True)

    model = build_model(args.device); bstar = torch.from_numpy(BSTAR).to(args.device)[:, None, None]
    w = center_weight(args.device); records=[]; cms={x:np.zeros((8,8),np.int64) for x in ("G0","G1","G2")}
    image_dir=DATA/"train/images"; gt_dir=Path("/home/jhy/rare_diag_20260915/b2_val_20260917/gt")
    peak=0; t0=time.time()
    for j,stem in enumerate(names,1):
        with Image.open(image_dir/f"{stem}.png") as im: image=np.asarray(im.convert("RGB")).copy()
        gt=np.load(gt_dir/f"{stem}.npy",allow_pickle=False)
        whole=torch.from_numpy(np.load(g0_dir/f"{stem}.npy").astype(np.float32)).to(args.device)
        whole=resize_logits(whole,(1024,1024))

        canvas=torch.zeros((8,1024,1024),device=args.device); denom=torch.zeros((1024,1024),device=args.device)
        for y in STARTS:
            for x in STARTS:
                crop=image[y:y+TILE,x:x+TILE]
                arr=np.asarray(Image.fromarray(crop).resize((1024,1024),Image.Resampling.BILINEAR))
                px=normalize_to_tensor(arr).unsqueeze(0).to(args.device)
                tile=resize_logits(d4_mean(model,px),(TILE,TILE))
                canvas[:,y:y+TILE,x:x+TILE]+=tile*w
                denom[y:y+TILE,x:x+TILE]+=w
                del px,tile
        tile_canvas=canvas/denom.clamp_min(1e-6)[None]
        g1=.5*(whole+tile_canvas)

        scaled=np.asarray(Image.fromarray(image).resize((1280,1280),Image.Resampling.BILINEAR))
        px=normalize_to_tensor(scaled).unsqueeze(0).to(args.device)
        scale_logits=resize_logits(d4_mean(model,px),(1024,1024))
        g2=.5*(whole+scale_logits)
        per={}
        for tag,z in (("G0",whole),("G1",g1),("G2",g2)):
            pred=(z+bstar).argmax(0).to(torch.uint8).cpu().numpy()
            Image.fromarray(pred+1,mode="L").save(out/f"pred_{tag}/{stem}.png")
            cm=confusion(pred,gt);cms[tag]+=cm;per[tag]=cm.tolist()
        flip1=float(((g1+bstar).argmax(0)!=(whole+bstar).argmax(0)).float().mean())
        flip2=float(((g2+bstar).argmax(0)!=(whole+bstar).argmax(0)).float().mean())
        records.append({"stem":stem,"confusion":per,"flip_G1_vs_G0":flip1,"flip_G2_vs_G0":flip2})
        peak=max(peak,torch.cuda.max_memory_allocated());del whole,canvas,denom,tile_canvas,g1,g2,px,scale_logits
        if j%20==0: print(f"geometry {j}/300 {time.time()-t0:.0f}s",flush=True)
    meta={"created":datetime.now(timezone.utc).isoformat(),"scope":"THETA278_H300_FIXED_GEOMETRY_SCREEN",
          "checkpoint":str(THETA278),"checkpoint_sha256":EXPECTED_FILE_SHA["theta278"],
          "H300_sha256":split["H300_sha256"],"count":300,
          "G0":"reused whole-image native D4 cache; raw logits bilinear 1024",
          "G1":"whole + four 768 crops starts {0,256}; crops resized 1024; D4; restored 768; positive-floor Hann blend; equal whole/tile fusion",
          "G2":"whole + true 1.25x input (1280); D4; logit resize 1024; equal fusion",
          "common":"raw logits fused at 1024; old Bstar added once; argmax last",
          "fixed_parameters":{"tile":768,"starts":[0,256],"scale":1.25,"fusion_weight":.5},
          "records":records,"aggregate_confusions":{k:v.tolist() for k,v in cms.items()},
          "runtime_sec":round(time.time()-t0,1),"peak_vram_bytes":peak,"physical_gpu":3,
          "code_sha256":sha256_file(Path(__file__)),"test2_ground_truth_used":False,
          "platform_submission_performed":False,"test2_package_created":False}
    write_json(out/"META.json",meta);write_json(out/"FORWARD_DONE.json",{"count":300,"runtime_sec":meta["runtime_sec"]})
    print(json.dumps({"out":str(out),"runtime_sec":meta["runtime_sec"],"peak_vram_bytes":peak},indent=2))

if __name__=="__main__":main()
