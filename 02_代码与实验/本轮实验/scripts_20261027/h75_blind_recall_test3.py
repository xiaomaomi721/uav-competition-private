#!/usr/bin/env python3
"""H75 Blind-Class-Recall soft-KL on test3 (迁移历史已验证正杠杆 TREAT-CONTROL +0.5951pp)."""
import argparse, csv, hashlib, json, os, random, sys, time
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
from src.label_mapping import official_to_train
from src.transforms import normalize_to_tensor

BSTAR = np.array([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], dtype=np.float32)
EXPECTED_H75_SHA = "b5eea2795d8a6d90c803a15b2ca66cf0abcf62322ef1aacd0d4d0ec82098a28a"
TRAIN_PREFIX = ("segformer.stages.2.", "segformer.stages.3.")
HEAD_PREFIX = "decode_head."
MAX_STEPS = 2000
BASE_LR = 1e-5
HEAD_MULT = 0.05
WD = 0.01
RAMP = 500
TARGET_COEFF = 0.5
CONF_NORMAL = 0.98
CONF_BLIND = 0.50
BLIND_CLASSES = (3, 4, 6)
SEED = 20260922

H75 = ROOT / "models" / "H75.pt"
CACHE_DIR = ROOT / "pseudo" / "teacher_test3"
IMAGES_DIR = ROOT / "data" / "test3" / "images"
SOURCE_IMG = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/train/images")
SOURCE_MSK = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/train/masks")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def is_train(name):
    return name.startswith(TRAIN_PREFIX) or name.startswith(HEAD_PREFIX)


def strong_photo(x):
    mean = torch.tensor([.485, .456, .406], device=x.device, dtype=x.dtype)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=x.device, dtype=x.dtype)[None, :, None, None]
    x = (x * std + mean).clamp(0, 1)
    br = 1 + random.uniform(-.15, .15)
    ct = 1 + random.uniform(-.15, .15)
    sat = 1 + random.uniform(-.12, .12)
    gam = random.uniform(.9, 1.1)
    x = x * br
    mu = x.mean((-2, -1), keepdim=True)
    x = (x - mu) * ct + mu
    gray = x.mean(1, keepdim=True)
    x = (x - gray) * sat + gray
    x = x.clamp(0, 1).pow(gam)
    x = x + torch.randn_like(x) * random.uniform(0, .015)
    if random.random() < .25:
        x = F.avg_pool2d(x, 3, 1, 1)
    return (x.clamp(0, 1) - mean) / std


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


def load_teacher_prob(stem, device, bstar):
    d = np.load(CACHE_DIR / f"{stem}.npz")
    prob = d["prob"].astype(np.float32)
    prob = torch.from_numpy(prob).to(device).clamp(1e-8, 1.0)
    cal = torch.softmax(prob.log() + bstar[:, None, None], dim=0)
    return cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--role", choices=["control", "treat"], default="treat")
    a = ap.parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")

    assert sha256_file(H75) == EXPECTED_H75_SHA, "H75 SHA mismatch"

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    cfg = load_config(str(CODE / "configs" / "base.yaml"))
    cfg["cwfa"] = {"enabled": False}; cfg["mixstyle"] = {"enabled": False}
    cfg["pretrained_model_path"] = str(ROOT / "models" / "segformer-b3-ade512")
    model = build_model(cfg)
    ckpt = torch.load(str(H75), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt), strict=True)
    model = model.to(device)
    for n, p in model.named_parameters():
        p.requires_grad_(is_train(n))

    base_params = [p for n, p in model.named_parameters() if n.startswith(TRAIN_PREFIX)]
    head_params = [p for n, p in model.named_parameters() if n.startswith(HEAD_PREFIX)]
    opt = torch.optim.AdamW([
        {"params": base_params, "lr": BASE_LR},
        {"params": head_params, "lr": BASE_LR * HEAD_MULT},
    ], weight_decay=WD)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    bstar = torch.from_numpy(BSTAR).to(device)

    src_ids = [x.strip() for x in (CODE / "splits" / "train.txt").read_text().split() if x.strip()]
    assert len(src_ids) == 5597
    rows = list(csv.DictReader(open(ROOT / "manifests" / "semifinal_3594.csv")))
    test3_ids = [r["output_name"].replace(".png", "") for r in rows]
    assert len(test3_ids) == 3594

    random.shuffle(src_ids)
    random.shuffle(test3_ids)

    t0 = time.time()
    hist = []
    amp_skips = []
    si = 0; ti = 0
    for step in range(1, a.max_steps + 1):
        for gi, g in enumerate(opt.param_groups):
            g["lr"] = (BASE_LR if gi == 0 else BASE_LR * HEAD_MULT) * (1 - (step - 1) / 10000)
        opt.zero_grad(set_to_none=True)

        stem = src_ids[si % len(src_ids)]; si += 1
        sx = np.asarray(Image.open(SOURCE_IMG / f"{stem}.png").convert("RGB")).copy()
        sy = official_to_train(np.asarray(Image.open(SOURCE_MSK / f"{stem}.png")))
        sx = normalize_to_tensor(sx).unsqueeze(0).to(device)
        sy = torch.from_numpy(sy.astype(np.int64)).unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", enabled=True):
            sz = model(pixel_values=sx).logits
        lsrc = F.cross_entropy(F.interpolate(sz.float(), size=sy.shape[-2:], mode="bilinear", align_corners=False), sy, ignore_index=255)

        stem = test3_ids[ti % len(test3_ids)]; ti += 1
        tx_img = np.asarray(Image.open(IMAGES_DIR / f"{stem}.png").convert("RGB")).copy()
        tx = normalize_to_tensor(tx_img).unsqueeze(0).to(device)
        view = random.randrange(8)
        aug = strong_photo(d4_forward(tx, view))
        with torch.amp.autocast("cuda", enabled=True):
            uz = model(pixel_values=aug).logits
        z = d4_inverse(uz.float(), view)
        z = F.interpolate(z, size=(512, 512), mode="bilinear", align_corners=False)

        pa = load_teacher_prob(stem, device, bstar).unsqueeze(0)
        conf, ya = pa.max(1)
        blind = (ya == 3) | (ya == 4) | (ya == 6)
        mask = (conf >= CONF_NORMAL) | (blind & (conf >= CONF_BLIND))
        logp_s = z.log_softmax(1)
        kl = (pa * (pa.log() - logp_s)).sum(1)
        lt = kl[mask].mean() if mask.any() else lsrc.new_zeros(())
        cov = float(mask.float().mean())

        ramp = min(1.0, step / RAMP)
        coeff = 0.0 if a.role == "control" else TARGET_COEFF * ramp
        loss = lsrc + coeff * lt

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        sb = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        stepped = scaler.get_scale() >= sb
        if not stepped:
            amp_skips.append(step)

        row = {"step": step, "source": round(float(lsrc),4), "target_kl": round(float(lt),4), "coeff": round(coeff,4),
               "loss": round(float(loss),4), "mask_cov": round(cov,4), "stepped": stepped, "elapsed": round(time.time()-t0,1)}
        hist.append(row)
        if step % 50 == 0 or step == 1:
            print(json.dumps(row), flush=True)

        if step in (1000, 2000):
            torch.save({"model": model.state_dict(), "step": step, "role": a.role,
                        "h75_sha": EXPECTED_H75_SHA, "target_recipe": "soft_kl_blind_recall",
                        "conf_normal": CONF_NORMAL, "conf_blind": CONF_BLIND, "blind_classes": list(BLIND_CLASSES)},
                       out / f"checkpoint_step{step}.pt")

    summary = {"steps": a.max_steps, "elapsed_sec": round(time.time()-t0,1), "role": a.role,
               "h75_sha": EXPECTED_H75_SHA, "target_recipe": "soft_kl_blind_recall",
               "attempted_loops": a.max_steps, "successful_updates": a.max_steps - len(amp_skips),
               "amp_skips": amp_skips, "history_tail": hist[-10:]}
    (out / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
