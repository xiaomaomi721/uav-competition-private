"""Matched C0 / C1 reliable-teacher KD training on H75.

Only difference between arms is the per-pixel reliability weight on the target KL:
  C0: uniform over valid target pixels
  C1: w = a^2 * c , a = D4 view agreement, c = 1 - H_norm ; w *= 0.5 where
      argmax(z_raw) != argmax(z_raw + Bstar)

L_total = CE_src + 0.1 * KL(H75_src || student_src) + lam_t(step) * L_target
lam_t: linear warmup over first 20% of steps to 0.2, then constant.

Deterministic: same initial weights, same pair order, same augmentation rng per step.
"""
import argparse
from rkd_common import *

OUTBASE = WORK / "train"
SOURCE_IMG = SOURCE_DATA / "train" / "images"
SOURCE_MSK = SOURCE_DATA / "train" / "masks"
LAM_ANCHOR = 0.1
LAM_T_MAX = 0.2
WARM_FRAC = 0.2
BASE_LR = 1e-5
HEAD_LR = 5e-7
WD = 0.01


def build_pairs(n):
    src_ids = [x.strip() for x in (SRC / "splits" / "train.txt").read_text().split() if x.strip()]
    assert len(src_ids) == 5597, len(src_ids)
    rows = target_rows()
    assert len(rows) == 3594, len(rows)
    sr = np.random.default_rng(SEED + 10001).permutation(len(src_ids))[:n]
    tr = np.random.default_rng(SEED + 10002).permutation(len(rows))[:n]
    return [(src_ids[int(s)], rows[int(t)]) for s, t in zip(sr, tr)]


def load_teacher_cache(stem):
    d = np.load(WORK / "teacher_d4" / f"{stem}.npz")
    z = torch.from_numpy(d["z_d4"].astype(np.float32))
    votes = torch.from_numpy(d["votes"].astype(np.float32))
    hn = torch.from_numpy(d["hn"].astype(np.float32))
    flip = torch.from_numpy(d["flip"].astype(np.float32))
    return z, votes, hn, flip


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def set_rng(d):
    random.setstate(d["python"]); np.random.set_state(d["numpy"])
    torch.set_rng_state(d["torch"].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in d["cuda"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["C0", "C1"], required=True)
    ap.add_argument("--steps", type=int, default=3594)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    guard(); seed_all()
    if not (WORK / "teacher_d4" / "COMPLETE.json").exists():
        raise RuntimeError("teacher cache incomplete")

    steps = a.smoke if a.smoke else a.steps
    out = OUTBASE / (f"{a.arm}_smoke" if a.smoke else a.arm)
    out.mkdir(parents=True, exist_ok=False)

    pairs = build_pairs(steps)
    order_sha = hashlib.sha256(json.dumps(
        [{"step": i + 1, "source": s, "target": r["input_name"], "subset": r["subset"]}
         for i, (s, r) in enumerate(pairs)], sort_keys=True).encode()).hexdigest()
    dump(out / "PAIR_ORDER.json", [{"step": i + 1, "source": s, "target": r["input_name"],
                                   "subset": r["subset"]} for i, (s, r) in enumerate(pairs)])

    from src.label_mapping import official_to_train
    from src.transforms import TrainTransform
    from src.rare_crop import RareClassCropper

    student = model_from()
    teacher = model_from(); teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student.eval()
    student.get_submodule("segformer.stages.2").train()
    student.get_submodule("segformer.stages.3").train()
    for n, p in student.named_parameters():
        p.requires_grad_(n.startswith(TRAIN_PREFIX) or n.startswith(HEAD_PREFIX))

    base = [p for n, p in student.named_parameters() if n.startswith(TRAIN_PREFIX)]
    head = [p for n, p in student.named_parameters() if n.startswith(HEAD_PREFIX)]
    opt = torch.optim.AdamW([{"params": base, "lr": BASE_LR},
                             {"params": head, "lr": HEAD_LR}], weight_decay=WD)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    b = torch.tensor(BSTAR, device="cuda")[None, :, None, None]
    tf = TrainTransform(crop_size=1024, config=get_cfg()["transforms"],
                        rare_cropper=RareClassCropper(1024, .5, (7, 4, 6), .25),
                        ignore_index=255)

    meta = {
        "arm": a.arm, "smoke": bool(a.smoke), "steps": steps,
        "h75_sha256": H75_SHA, "teacher_cache": str(WORK / "teacher_d4"),
        "pair_order_sha256": order_sha, "seed": SEED,
        "trainable_params": int(sum(p.numel() for p in student.parameters() if p.requires_grad)),
        "lam_anchor": LAM_ANCHOR, "lam_target_max": LAM_T_MAX, "warm_frac": WARM_FRAC,
        "base_lr": BASE_LR, "head_lr": HEAD_LR, "wd": WD,
        "c1_weight": "w = a^2 * (1-H_norm); *= 0.5 where argmax(raw)!=argmax(raw+Bstar)",
        "c0_weight": "uniform over valid target pixels",
        "gpu_index": EXPECTED_GPU_INDEX, "torch": torch.__version__,
        "started_unix": time.time(),
    }
    dump(out / "RUN_CONFIG.json", meta)
    emit(stage="train_start", **{k: meta[k] for k in ("arm", "steps", "trainable_params")})

    t0 = time.time()
    hist = open(out / "history.jsonl", "w", buffering=1)
    ess_stats = []
    try:
        for i, (stem, row) in enumerate(pairs, 1):
            step = i
            if step % 20 == 1:
                guard()
            warm = max(1, int(WARM_FRAC * steps))
            lam_t = LAM_T_MAX * min(1.0, step / warm)

            # ---- source ----
            rgb = read_rgb(SOURCE_IMG / f"{stem}.png")
            with Image.open(SOURCE_MSK / f"{stem}.png") as f:
                lab = official_to_train(np.asarray(f).copy())
            sx, sy = tf(rgb, lab, rng=random.Random(SEED + 100000 + step))
            sx = sx.unsqueeze(0).cuda(); sy = sy.unsqueeze(0).cuda()

            # ---- target ----
            tx = normalize_to_tensor(read_rgb(TARGET_IMAGES / row["input_name"])).unsqueeze(0).cuda()
            zT, votes, hn, flipf = load_teacher_cache(Path(row["input_name"]).stem)
            zT = zT.unsqueeze(0).cuda(); votes = votes.unsqueeze(0).cuda()
            hn = hn.unsqueeze(0).cuda(); flipf = flipf.unsqueeze(0).cuda()
            q = (zT + b).softmax(1).detach()

            view = random.Random(SEED + 200000 + step).randrange(8)
            gen = torch.Generator(device="cuda").manual_seed(SEED + 300000 + step)
            aug = strong_photo(d4(tx, view), random.Random(SEED + 200000 + step), gen)

            before = rng_state()
            for retry in range(8):
                set_rng(before)
                opt.zero_grad(set_to_none=True)
                for gi, g in enumerate(opt.param_groups):
                    g["lr"] = (BASE_LR if gi == 0 else HEAD_LR) * (1 - (step - 1) / 10000)

                with torch.amp.autocast("cuda"):
                    sz = student(pixel_values=sx).logits
                sz = F.interpolate(sz.float(), size=sy.shape[-2:], mode="bilinear", align_corners=False)
                l_src = F.cross_entropy(sz, sy, ignore_index=255)

                with torch.no_grad():
                    with torch.amp.autocast("cuda"):
                        tz = teacher(pixel_values=sx).logits
                    tz = F.interpolate(tz.float(), size=sy.shape[-2:], mode="bilinear", align_corners=False)
                    pt = tz.softmax(1)
                valid = (sy != 255).float()
                kl_a = (pt * (pt.clamp_min(1e-8).log() - sz.log_softmax(1))).sum(1)
                l_anchor = (kl_a * valid).sum() / valid.sum().clamp_min(1.0)

                with torch.amp.autocast("cuda"):
                    ts = student(pixel_values=aug).logits
                ts = inv_d4(ts.float(), view)
                p = (ts + b).log_softmax(1)
                kl_pix = (q * (q.clamp_min(1e-8).log() - p)).sum(1)

                if a.arm == "C0":
                    w = torch.ones_like(kl_pix)
                else:
                    av = (votes / 8.0)
                    c = (1.0 - hn).clamp(0, 1)
                    w = av.pow(2) * c
                    w = w * torch.where(flipf > 0.5, 0.5, 1.0)
                denom = w.sum().clamp_min(1e-6)
                l_tgt = (w * kl_pix).sum() / denom

                total = l_src + LAM_ANCHOR * l_anchor + lam_t * l_tgt
                scaler.scale(total).backward()
                scaler.unscale_(opt)
                sb = scaler.get_scale()
                scaler.step(opt); scaler.update()
                if scaler.get_scale() >= sb:
                    break
                emit(stage="AMP_RETRY", step=step, retry=retry)
            else:
                raise RuntimeError("repeated non-finite gradients")
            if not torch.isfinite(total):
                raise RuntimeError("non-finite objective")

            ess = float((w.sum() ** 2) / (w.pow(2).sum().clamp_min(1e-9)))
            ess_frac = ess / float(w.numel())
            ess_stats.append(ess_frac)
            row_out = {"step": step, "arm": a.arm, "source": stem, "target": row["input_name"],
                       "subset": row["subset"], "lam_t": round(lam_t, 5),
                       "loss_src": round(float(l_src), 6), "loss_anchor": round(float(l_anchor), 6),
                       "loss_target": round(float(l_tgt), 6), "loss_total": round(float(total), 6),
                       "ess_frac": round(ess_frac, 5), "mean_w": round(float(w.mean()), 5),
                       "view": view, "elapsed": round(time.time() - t0, 1),
                       "gpu_mem_mib": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)}
            hist.write(json.dumps(row_out) + "\n")
            if step % 50 == 0 or step == steps:
                emit(stage="train", **{k: row_out[k] for k in
                                       ("step", "arm", "loss_src", "loss_anchor", "loss_target", "ess_frac")})
                dump(out / "PROGRESS.json", row_out)
            if not a.smoke and (step in (1000, 2000) or step == steps):
                torch.save({"model": student.state_dict(), "step": step, "arm": a.arm,
                            "h75_sha256": H75_SHA, "pair_order_sha256": order_sha,
                            "bstar": BSTAR, "recipe": "reliable_kd_c1" if a.arm == "C1" else "reliable_kd_c0"},
                           out / f"checkpoint_step{step}.pt")
        summary = {**meta, "status": "COMPLETED", "completed_steps": steps,
                   "runtime_sec": round(time.time() - t0, 1),
                   "ess_frac_mean": round(float(np.mean(ess_stats)), 5),
                   "ess_frac_min": round(float(np.min(ess_stats)), 5),
                   "peak_mem_mib": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)}
        dump(out / "SUMMARY.json", summary)
        emit(stage="train_complete", arm=a.arm, steps=steps, ess_mean=summary["ess_frac_mean"])
    finally:
        hist.close()


if __name__ == "__main__":
    main()
