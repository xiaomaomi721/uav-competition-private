#!/usr/bin/env python3
"""ETKD-v2 exact single-model D4 inference, behavior audit, H300 guard, ZIP.
Modified from v1: GPU2 binding instead of GPU3."""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, time, zipfile
from pathlib import Path
import numpy as np

ROOT = Path("/home/jhy/rare_diag_20260915")
SNAP = Path("/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1")
CONFIG = SNAP / "stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml"
TEST2 = Path("/home/jhy/datasets/AIC2026_UAV_semifinal/test_2/images")
DATA = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集")
THETA_CACHE = ROOT / "round2_20260921/cache_278_test2_raw/scores"
D1_CACHE = ROOT / "round2_20260923/D1_2k_probe/raw_cache/scores"
D1_CKPT = ROOT / "round2_multiprobe_20260922/03_dart_fixed/D1/checkpoint_step2000.pt"
D1_PRED = ROOT / "round2_20260923/D1_2k_probe/predictions"
TEACHER_ZIP = ROOT / "d1_interp_20260925/D1_alpha050_test2.zip"
SOURCE_SPLIT = ROOT / "round2_multiprobe_20260922/00_inventory/SOURCE_AUDIT_SPLIT.json"
SOURCE_GT = ROOT / "b2_val_20260917/gt"
EXPECTED_GPU_UUID = "GPU-cacb7500-f2e9-c482-d85c-53c7f8d1def9"
EXPECTED_GPU_BUS = "00000000:B1:00.0"
BSTAR = np.asarray([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], np.float32)
VIEWS = [(k, hf) for k in range(4) for hf in (False, True)]
CLASS_NAMES = ["Background", "Building", "Road", "Water", "Barren", "Vegetation", "Agricultural", "Vehicle"]
TEMPERATURE = 2.0

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()

def write_json(path, v): path.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding="utf-8")

def load_state(path):
    import torch
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for k in ("model","student_model","model_state_dict","state_dict"):
            v = obj.get(k)
            if isinstance(v, dict) and v and all(isinstance(t, torch.Tensor) for t in v.values()): return v
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()): return obj
    raise RuntimeError(f"no state in {path}")

def assert_gpu2():
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID" or os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise SystemExit("GPU2 binding required")
    rows = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,pci.bus_id,name","--format=csv,noheader"], text=True).splitlines()
    m = [x.strip() for x in rows if x.strip().startswith("2,")]
    if len(m) != 1 or EXPECTED_GPU_UUID not in m[0]: raise SystemExit(f"GPU2 mismatch: {m}")
    return {"index":2,"uuid":EXPECTED_GPU_UUID,"pci_bus":EXPECTED_GPU_BUS,"row":m[0]}

def d4_fwd(t, i):
    import torch
    r,hf = VIEWS[i]
    if r: t = torch.rot90(t, r, (-2,-1))
    if hf: t = t.flip((-1,))
    return t

def d4_inv(t, i):
    import torch
    r,hf = VIEWS[i]
    if hf: t = t.flip((-1,))
    if r: t = torch.rot90(t, -r, (-2,-1))
    return t.contiguous()

def build_model(ckpt, device):
    from src.models import build_model as mm
    from src.utils import load_config
    cfg = load_config(str(CONFIG))
    cfg["cwfa"] = {"enabled":False,"probability":0.3,"epsilon":4.0,"norm_eps":1e-12,"start_step":1}
    cfg.pop("mixstyle", None)
    m = mm(cfg)
    m.load_state_dict(load_state(ckpt), strict=True)
    return m.to(device).eval()

def d4_mean(model, px):
    import torch
    acc = None
    with torch.inference_mode():
        for i in range(8):
            with torch.amp.autocast("cuda", enabled=False):
                lg = model(pixel_values=d4_fwd(px, i)).logits.float()
            canon = d4_inv(lg, i)[0]
            acc = canon if acc is None else acc + canon
    return acc / 8.0

def update_transition(mat, ref, cand):
    mat += np.bincount((ref.astype(np.int64)*8+cand.astype(np.int64)).ravel(), minlength=64).reshape(8,8)

def iou_from_cm(cm):
    u = cm.sum(1)+cm.sum(0)-np.diag(cm)
    v = np.divide(np.diag(cm), u, out=np.zeros(8,dtype=np.float64), where=u>0)
    return v, float(v.mean())

def eval_h300(ckpt, device):
    import torch, torch.nn.functional as F
    from PIL import Image
    from src.transforms import normalize_to_tensor
    names = json.loads(SOURCE_SPLIT.read_text())["H300"]
    model = build_model(ckpt, device)
    cm = np.zeros((8,8), np.int64)
    bias = torch.from_numpy(BSTAR).to(device)[:,None,None]
    for idx, stem in enumerate(names, 1):
        with Image.open(DATA/f"train/images/{stem}.png") as im:
            rgb = np.asarray(im.convert("RGB")).copy()
        gt = np.load(SOURCE_GT/f"{stem}.npy", allow_pickle=False)
        px = normalize_to_tensor(rgb).unsqueeze(0).to(device)
        raw = d4_mean(model, px)
        dep = F.interpolate((raw+bias)[None], size=gt.shape, mode="bilinear", align_corners=False)[0]
        pred = dep.argmax(0).byte().cpu().numpy()
        valid = gt != 255
        cm += np.bincount((gt[valid].astype(np.int64)*8+pred[valid]).ravel(), minlength=64).reshape(8,8)
    pc, miou = iou_from_cm(cm)
    del model; torch.cuda.empty_cache()
    return {"miou":miou, "per_class_iou":pc.tolist()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-h300", action="store_true")
    args = ap.parse_args()
    gpu = assert_gpu2()
    ckpt = Path(args.checkpoint)
    out = Path(args.output_dir)
    if out.exists(): raise SystemExit(f"exists: {out}")
    out.mkdir(parents=True)
    for p in (str(SNAP), str(ROOT/"round2_multiprobe_20260922")):
        if p not in sys.path: sys.path.insert(0, p)
    import torch, torch.nn.functional as F
    from PIL import Image
    from src.transforms import normalize_to_tensor

    model = build_model(ckpt, args.device)
    bias = torch.from_numpy(BSTAR).to(args.device)[:,None,None]
    names = [f"test2_{i}.png" for i in range(1,1301)]
    t_hist = np.zeros(8,np.int64); d1_hist = np.zeros(8,np.int64); th_hist = np.zeros(8,np.int64)
    m_d1 = np.zeros((8,8),np.int64); m_th = np.zeros((8,8),np.int64)
    pi_d1=[]; pi_th=[]; kl_c=0.0; kl_d1=0.0; conf_sum=0.0
    start=time.time()
    with zipfile.ZipFile(TEACHER_ZIP) as tz:
        for idx, name in enumerate(names, 1):
            with Image.open(TEST2/name) as im:
                rgb = np.asarray(im.convert("RGB")).copy()
            px = normalize_to_tensor(rgb).unsqueeze(0).to(args.device)
            raw = d4_mean(model, px)
            dep = F.interpolate((raw+bias)[None], size=(1024,1024), mode="bilinear", align_corners=False)[0]
            prob = dep.softmax(0)
            pred = prob.max(0)[1].byte().cpu().numpy()
            conf_sum += float(prob.max(0)[0].mean())
            d1p = np.asarray(Image.open(D1_PRED/name), dtype=np.uint8)-1
            with tz.open(name) as f:
                thp = np.asarray(Image.open(f), dtype=np.uint8)-1
            update_transition(m_d1, d1p, pred); update_transition(m_th, thp, pred)
            pi_d1.append(float(np.mean(d1p!=pred))); pi_th.append(float(np.mean(thp!=pred)))
            t_hist += np.bincount(pred.ravel(), minlength=8)
            d1_hist += np.bincount(d1p.ravel(), minlength=8)
            th_hist += np.bincount(thp.ravel(), minlength=8)
            stem = Path(name).stem
            th_n = torch.from_numpy(np.load(THETA_CACHE/f"{stem}.npy").astype(np.float32)).to(args.device)
            d1_n = torch.from_numpy(np.load(D1_CACHE/f"{stem}.npy").astype(np.float32)).to(args.device)
            teach = 0.5*(th_n+d1_n)
            tp = ((teach+bias)/TEMPERATURE).softmax(0)
            cl = ((raw+bias)/TEMPERATURE).log_softmax(0)
            dl = ((d1_n+bias)/TEMPERATURE).log_softmax(0)
            kl_c += float(F.kl_div(cl, tp, reduction="none").sum(0).mean()*TEMPERATURE**2)
            kl_d1 += float(F.kl_div(dl, tp, reduction="none").sum(0).mean()*TEMPERATURE**2)
            if idx % 200 == 0: print(f"test2 {idx}/1300 {time.time()-start:.0f}s", flush=True)

    per_d1=np.asarray(pi_d1); per_th=np.asarray(pi_th)
    total=int(t_hist.sum())
    flip_d1=int(m_d1.sum()-np.trace(m_d1))
    flip_th=int(m_th.sum()-np.trace(m_th))
    audit = {
        "single_model": True, "checkpoint": str(ckpt), "checkpoint_sha256": sha256_file(ckpt),
        "gpu": gpu, "candidate_teacher_kl_T2": kl_c/1300, "d1_teacher_kl_T2": kl_d1/1300,
        "teacher_kl_reduction": (kl_d1-kl_c)/max(kl_d1,1e-12),
        "candidate_vs_d1_flip": flip_d1/total,
        "candidate_vs_d1_per_image": {"mean":float(per_d1.mean()),"median":float(np.median(per_d1)),"p90":float(np.quantile(per_d1,.9)),"p95":float(np.quantile(per_d1,.95)),"max":float(per_d1.max())},
        "candidate_vs_teacher_flip": flip_th/total,
        "candidate_vs_teacher_per_image": {"mean":float(per_th.mean()),"median":float(np.median(per_th)),"p90":float(np.quantile(per_th,.9)),"p95":float(np.quantile(per_th,.95)),"max":float(per_th.max())},
        "class_share_delta_pp": ((t_hist-th_hist)/total*100).tolist(),
        "mean_confidence": conf_sum/1300,
        "runtime_sec": round(time.time()-start,1),
    }
    write_json(out/"TEST2_BEHAVIOR.json", audit)
    print(json.dumps(audit, indent=2), flush=True)

    h300_result = None
    if not args.skip_h300:
        del model; torch.cuda.empty_cache()
        cand_h = eval_h300(ckpt, args.device)
        d1_h = eval_h300(D1_CKPT, args.device)
        h300_result = {"candidate_miou":cand_h["miou"],"d1_miou":d1_h["miou"],"delta_pp":100*(cand_h["miou"]-d1_h["miou"]),
                       "candidate_per_class":cand_h["per_class_iou"],"d1_per_class":d1_h["per_class_iou"]}
        write_json(out/"H300_GUARD.json", h300_result)

    # Gate
    fails = []
    if kl_c/1300 >= kl_d1/1300: fails.append("teacher KL not improved")
    if flip_th/total >= flip_d1/total: fails.append("not closer to teacher than D1")
    if flip_d1/total > 0.02: fails.append(f"vs D1 flip > 2%: {flip_d1/total:.4f}")
    if per_d1.max() > 0.30: fails.append(f"per-image max flip > 30%: {per_d1.max():.4f}")
    if h300_result and h300_result["delta_pp"] < -0.5: fails.append(f"H300 delta < -0.5pp: {h300_result['delta_pp']:.3f}")
    gate = {"status":"PASS" if not fails else "FAIL", "failures":fails,
            "candidate_teacher_kl":kl_c/1300, "d1_teacher_kl":kl_d1/1300,
            "kl_reduction_pct":100*(kl_d1-kl_c)/max(kl_d1,1e-12)}
    write_json(out/"P1_GATE.json", gate)
    print(json.dumps(gate, indent=2), flush=True)

if __name__ == "__main__":
    main()
