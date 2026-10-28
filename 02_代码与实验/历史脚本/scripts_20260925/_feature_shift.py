#!/usr/bin/env python3
"""D1 feature shift analysis: prototype distances + entropy."""
import sys, json, time, zipfile, io
from pathlib import Path
from collections import defaultdict
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
THETA278_CACHE = Path("/home/jhy/rare_diag_20260915/round2_20260921/cache_278_test2_raw/scores")
TEST2_DIR = Path("/home/jhy/datasets/AIC2026_UAV_semifinal/test_2/images")
GT_DIR = Path("/home/jhy/rare_diag_20260915/b2_val_20260917/gt")
SOURCE_IMG_DIR = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/train/images")
SPLIT_JSON = Path("/home/jhy/rare_diag_20260915/round2_multiprobe_20260922/00_inventory/SOURCE_AUDIT_SPLIT.json")
OUT = Path("/home/jhy/rare_diag_20260915/d1_feature_shift_20260925")
OUT.mkdir(exist_ok=True)

BSTAR = [-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5]
CLASS_NAMES = ["Background","Building","Road","Water","Barren","Vegetation","Agricultural","Vehicle"]

def load_state(path):
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for k in ("model","student_model","state_dict","model_state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt

def get_stage2_feat(model, px):
    """Hook stage2 output: model.segformer.encoder.stages[2]"""
    feats = {}
    def hook(mod, inp, out):
        feats["s2"] = out
    h = model.segformer.stages[2].register_forward_hook(hook)
    with torch.no_grad():
        logits = model(pixel_values=px).logits
    h.remove()
    return feats["s2"], logits

def main():
    device = "cuda:0"
    cfg = load_config(str(SNAP/"stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml"))
    cfg["cwfa"] = {"enabled":False,"probability":0.3,"epsilon":4.0,"norm_eps":1e-12,"start_step":1}
    cfg.pop("mixstyle", None)

    print("Loading theta278...", flush=True)
    teacher = make_model(cfg)
    teacher.load_state_dict(load_state(THETA278_CKPT), strict=True)
    teacher = teacher.to(device).eval()
    print("Loading D1...", flush=True)
    student = make_model(cfg)
    student.load_state_dict(load_state(D1_CKPT), strict=True)
    student = student.to(device).eval()
    bstar_t = torch.tensor(BSTAR, device=device).view(1,8,1,1)

    # === Analysis 1 & 2: test2 class distribution + entropy ===
    print("\n=== Analysis 1&2: test2 distribution + entropy ===", flush=True)
    test2_imgs = sorted(TEST2_DIR.glob("*.png"), key=lambda p: int(p.stem.split("_")[-1]))
    # Use existing theta278 cache for entropy
    conf_sum = np.zeros(8); ent_sum = np.zeros(8); cls_count = np.zeros(8)
    t0 = time.time()
    for i, ip in enumerate(test2_imgs):
        w = np.load(THETA278_CACHE / f"{ip.stem}.npy").astype(np.float32)
        wt = torch.from_numpy(w).unsqueeze(0).to(device) + bstar_t
        prob = F.softmax(wt, dim=1)[0].cpu().numpy()  # [8,256,256]
        conf = prob.max(0)  # [256,256]
        pred = prob.argmax(0)  # [256,256]
        ent = -(prob * np.log(prob + 1e-12)).sum(0)
        for c in range(8):
            m = (pred == c)
            cls_count[c] += m.sum()
            conf_sum[c] += conf[m].sum()
            ent_sum[c] += ent[m].sum()
        if (i+1) % 400 == 0:
            print(f"  {i+1}/1300 {time.time()-t0:.0f}s", flush=True)

    print("\n  Class | pixels | mean_conf | mean_entropy")
    for c in range(8):
        n = max(cls_count[c],1)
        print(f"  {CLASS_NAMES[c]:<15} {cls_count[c]:>10.0f} {conf_sum[c]/n:.4f} {ent_sum[c]/n:.4f}")

    # === Analysis 3&4: Prototype distances ===
    print("\n=== Analysis 3&4: Feature prototype distances ===", flush=True)
    split = json.loads(SPLIT_JSON.read_text())
    gt_files = {p.stem: p for p in GT_DIR.glob("*.npy")}

    # Source prototypes: sample 200 H300 images with GT
    src_items = split["H300"][:200]
    # Target prototypes: sample 200 test2 images
    tgt_items = test2_imgs[:200]

    def collect_prototypes(model, items, source_mode):
        """Collect per-class mean feature vectors. source_mode=True uses GT, else uses model pred."""
        class_feats = defaultdict(list)
        for stem in items:
            if source_mode:
                img_path = SOURCE_IMG_DIR / f"{stem}.png"
                gt_path = gt_files.get(stem)
                if not img_path.is_file() or gt_path is None: continue
                gt = np.load(gt_path).astype(np.int64)
                if gt.ndim == 3: gt = gt[0]
            else:
                img_path = stem  # it's a Path
                gt = None

            img = np.asarray(Image.open(img_path).convert("RGB"))
            px = normalize_to_tensor(img).unsqueeze(0).to(device)
            feat, logits = get_stage2_feat(model, px)  # [1,C,hf,wf]
            feat = feat[0].cpu().numpy()  # [C,hf,wf]
            C, hf, wf = feat.shape

            if source_mode:
                # Resize GT to feat spatial size
                gt_small = np.array(Image.fromarray(gt.astype(np.uint8)).resize((wf,hf), Image.NEAREST))
                labels = gt_small
            else:
                # Use model prediction
                lo = logits.float() + bstar_t
                lo_up = F.interpolate(lo, size=(1024,1024), mode="bilinear", align_corners=False)
                pred = lo_up[0].argmax(0).cpu().numpy().astype(np.uint8)
                labels = np.array(Image.fromarray(pred).resize((wf,hf), Image.NEAREST))

            for c in range(8):
                m = (labels == c)
                if m.sum() > 10:
                    class_feats[c].append(feat[:, m].mean(1))  # [C]

        # Average per class
        protos = {}
        for c in range(8):
            if class_feats[c]:
                protos[c] = np.stack(class_feats[c]).mean(0)
        return protos

    def proto_distance(protos, c1, c2):
        if c1 not in protos or c2 not in protos: return None
        return float(np.linalg.norm(protos[c1] - protos[c2]))

    pairs = [(5,1,"Vegetation-Building"), (0,5,"Background-Vegetation"),
             (0,6,"Background-Agricultural"), (0,1,"Background-Building"),
             (0,2,"Background-Road"), (6,4,"Agricultural-Barren"),
             (5,6,"Vegetation-Agricultural")]

    print("  Computing source prototypes (theta278)...", flush=True)
    src_protos_278 = collect_prototypes(teacher, src_items, source_mode=True)
    print("  Computing target prototypes (theta278)...", flush=True)
    tgt_protos_278 = collect_prototypes(teacher, tgt_items, source_mode=False)
    print("  Computing target prototypes (D1)...", flush=True)
    tgt_protos_d1 = collect_prototypes(student, tgt_items, source_mode=False)

    print(f"\n  {'Pair':<25} {'Source(278)':>12} {'Target(278)':>12} {'Target(D1)':>12} {'D1 change':>10}")
    results = []
    for c1, c2, name in pairs:
        ds = proto_distance(src_protos_278, c1, c2)
        dt278 = proto_distance(tgt_protos_278, c1, c2)
        dtd1 = proto_distance(tgt_protos_d1, c1, c2)
        chg = (dtd1 - dt278) / dt278 * 100 if dt278 and dtd1 else None
        chg_s = f"{chg:+.1f}%" if chg is not None else "N/A"
        print(f"  {name:<25} {ds:>12.2f} {dt278:>12.2f} {dtd1:>12.2f} {chg_s:>10}")
        results.append({"pair":name, "source":ds, "target_278":dt278, "target_d1":dtd1, "change_pct":chg})

    print(f"\nTotal time: {time.time()-t0:.0f}s")

    # Save
    summary = {
        "entropy_by_class": {CLASS_NAMES[c]: {"pixels":int(cls_count[c]),
            "mean_conf":float(conf_sum[c]/max(cls_count[c],1)),
            "mean_entropy":float(ent_sum[c]/max(cls_count[c],1))} for c in range(8)},
        "prototype_distances": results
    }
    (OUT/"feature_shift.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved to {OUT/'feature_shift.json'}")

if __name__ == "__main__":
    main()
