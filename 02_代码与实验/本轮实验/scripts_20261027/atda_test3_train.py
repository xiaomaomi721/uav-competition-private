#!/usr/bin/env python3
"""ATDA 迁移到 test3：Anchored Target-Domain Adaptation。
历史配方（test1 上 +0.33pp 正收益）：
  起点/ref/ema = 已收敛 CTRL (ckpt_005400, TTA8 79.29%)
  loss = loss_source + ramp * (0.5 * loss_hard + 0.25 * loss_cons)
  loss_hard = 加权 CE 到参考教师 argmax (w = ref 置信度)
  loss_cons = 加权 KL(学生 || EMA教师)  ← 自监督一致性
参考缓存 = native-256 D4 TTA8 logits (pseudo/teacher_test3_native256)
"""
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
NUM_CLASSES = 8
LOGIT_GRID = 256
IGNORE_INDEX = 255
TRAIN_PREFIX = ("segformer.stages.2.", "segformer.stages.3.")
HEAD_PREFIX = "decode_head."
MAX_STEPS = 2000
BASE_LR = 1e-5
HEAD_MULT = 0.05
WD = 0.01
RAMP_STEPS = 500
LAMBDA_HARD = 0.5
LAMBDA_CONS = 0.25
EMA_DECAY = 0.999
SEED = 20260922

CKPT = ROOT / "experiments" / "ctrl" / "checkpoints" / "ckpt_005400.pt"
REF_CACHE = ROOT / "pseudo" / "teacher_test3_native256"
IMAGES_DIR = ROOT / "data" / "test3" / "images"
SOURCE_IMG = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/train/images")
SOURCE_MSK = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/train/masks")


def is_train(name):
    return name.startswith(TRAIN_PREFIX) or name.startswith(HEAD_PREFIX)


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


def load_model(device):
    cfg = load_config(str(CODE / "configs" / "base.yaml"))
    cfg["cwfa"] = {"enabled": False}; cfg["mixstyle"] = {"enabled": False}
    cfg["pretrained_model_path"] = str(ROOT / "models" / "segformer-b3-ade512")
    m = build_model(cfg)
    st = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    m.load_state_dict(st.get("model", st), strict=True)
    return m.to(device)


def build_ref(cache_dir, names, device):
    bstar = torch.tensor(BSTAR, dtype=torch.float32).view(1, NUM_CLASSES, 1, 1).to(device)
    y_ref = torch.empty((len(names), LOGIT_GRID, LOGIT_GRID), dtype=torch.uint8)
    w_ref = torch.empty((len(names), LOGIT_GRID, LOGIT_GRID), dtype=torch.float16)
    for i, stem in enumerate(names):
        arr = np.load(cache_dir / f"{stem}.npy").astype(np.float32)
        z = torch.from_numpy(arr).unsqueeze(0).to(device)
        probs = torch.softmax(z + bstar, dim=1)[0]
        conf = probs.max(0).values
        y_ref[i] = probs.argmax(0).to(torch.uint8)
        w_ref[i] = conf.to(torch.float16)
    return y_ref, w_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--role", choices=["treat", "control"], default="treat")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    rows = list(csv.DictReader(open(ROOT / "manifests" / "semifinal_3594.csv")))
    names = [r["output_name"].replace(".png", "") for r in rows]

    student = load_model(device)
    model_ref = load_model(device); model_ref.eval()
    model_ema = load_model(device); model_ema.eval()
    for m in (model_ref, model_ema):
        for p in m.parameters():
            p.requires_grad_(False)
    for n, p in student.named_parameters():
        p.requires_grad_(is_train(n))

    base_params = [p for n, p in student.named_parameters() if n.startswith(TRAIN_PREFIX)]
    head_params = [p for n, p in student.named_parameters() if n.startswith(HEAD_PREFIX)]
    opt = torch.optim.AdamW([
        {"params": base_params, "lr": BASE_LR},
        {"params": head_params, "lr": BASE_LR * HEAD_MULT},
    ], weight_decay=WD)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    bstar = torch.tensor(BSTAR, dtype=torch.float32).view(1, NUM_CLASSES, 1, 1).to(device)

    y_ref, w_ref = build_ref(REF_CACHE, names, device)

    src_ids = [x.strip() for x in (CODE / "splits" / "train.txt").read_text().split() if x.strip()]
    random.shuffle(src_ids)
    random.shuffle(names)

    t0 = time.time()
    hist = []
    amp_skips = []
    si = 0; ti = 0
    max_steps = 3 if a.smoke else a.max_steps
    for step in range(1, max_steps + 1):
        for gi, g in enumerate(opt.param_groups):
            g["lr"] = (BASE_LR if gi == 0 else BASE_LR * HEAD_MULT) * (1 - (step - 1) / 10000)
        opt.zero_grad(set_to_none=True)

        # source
        stem = src_ids[si % len(src_ids)]; si += 1
        sx = np.asarray(Image.open(SOURCE_IMG / f"{stem}.png").convert("RGB")).copy()
        sy = official_to_train(np.asarray(Image.open(SOURCE_MSK / f"{stem}.png")))
        sx = normalize_to_tensor(sx).unsqueeze(0).to(device)
        sy = torch.from_numpy(sy.astype(np.int64)).unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", enabled=True):
            sz = student(pixel_values=sx).logits
        loss_source = F.cross_entropy(F.interpolate(sz.float(), size=sy.shape[-2:], mode="bilinear", align_corners=False), sy, ignore_index=IGNORE_INDEX)

        # target
        stem = names[ti % len(names)]; ti += 1
        idx = names.index(stem) if stem in names else 0  # 用 name 找 ref 索引
        idx = next(i for i, n in enumerate(names) if n == stem)
        tx_img = np.asarray(Image.open(IMAGES_DIR / f"{stem}.png").convert("RGB")).copy()
        tx = normalize_to_tensor(tx_img).unsqueeze(0).to(device)
        view = random.randrange(8)
        with torch.amp.autocast("cuda", enabled=True):
            student_logits = student(pixel_values=d4_forward(strong_photo(tx), view)).logits
        z_s = d4_inverse(student_logits.float(), view)
        biased = z_s + bstar

        y_r = y_ref[idx].to(device).long().unsqueeze(0)
        w_a = w_ref[idx].to(device).float().unsqueeze(0)
        ce = F.cross_entropy(biased, y_r, reduction="none")
        w_sum = w_a.sum().clamp_min(1e-6)
        loss_hard = (w_a * ce).sum() / w_sum

        # EMA consistency
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True):
                ema_logits = model_ema(pixel_values=tx).logits
            p_ema = torch.softmax(ema_logits.float() + bstar, dim=1)
        p_s = torch.softmax(biased, dim=1)
        kl = F.kl_div(torch.log(p_s.clamp_min(1e-12)), p_ema, reduction="none").sum(dim=1)
        loss_cons = (w_a * kl).sum() / w_sum

        ramp = min(1.0, step / RAMP_STEPS)
        loss = loss_source + ramp * (LAMBDA_HARD * loss_hard + LAMBDA_CONS * loss_cons)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        sb = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        stepped = scaler.get_scale() >= sb
        if not stepped:
            amp_skips.append(step)

        # EMA update
        with torch.no_grad():
            for name, param in model_ema.named_parameters():
                if name.startswith(TRAIN_PREFIX) or name.startswith(HEAD_PREFIX):
                    sp = student.get_parameter(name)
                    param.copy_(EMA_DECAY * param + (1.0 - EMA_DECAY) * sp)

        row = {"step": step, "source": round(float(loss_source),4), "hard": round(float(loss_hard),4),
               "cons": round(float(loss_cons),4), "loss": round(float(loss),4), "stepped": stepped,
               "elapsed": round(time.time()-t0,1)}
        hist.append(row)
        if step % 50 == 0 or step == 1:
            print(json.dumps(row), flush=True)

        if step in (1000, 2000):
            torch.save({"model": student.state_dict(), "step": step, "role": a.role,
                        "recipe": "atda_test3", "ema_decay": EMA_DECAY},
                       out / f"checkpoint_step{step}.pt")

    summary = {"steps": max_steps, "role": a.role, "recipe": "atda_test3",
               "successful_updates": max_steps - len(amp_skips), "amp_skips": amp_skips,
               "elapsed_sec": round(time.time()-t0,1), "history_tail": hist[-10:]}
    (out / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
