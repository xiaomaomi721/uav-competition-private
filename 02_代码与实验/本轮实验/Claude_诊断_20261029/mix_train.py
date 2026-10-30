"""C1 + bias-consistent cross-domain ClassMix with FDA-styled pasted source (Track B).

Identical to reliable_kd C1 (rkd_train.py, scored 70.907678) except the target branch:
  C1 target input : strong_photo(d4(target))                    loss = w-weighted KL(q || softmax(z+b))
  here            : mix = M*FDA(src->tgt) + (1-M)*target, then strong_photo(d4(mix))
                    loss on ~M : same w-weighted KL as C1
                    loss on  M : CE(z+b, y_src)  (pasted pixels carry source GT, weight 1)
M = ClassMix mask: random half of the classes present in the source GT (DACS, Tranheden et al. 2021).
Style = per-channel mean/std transfer toward the paired target image (DC-only limit of FDA, Yang & Soatto 2020).
Pasted CE uses z+b because the pasted pixels sit inside a target image and are scored after +b at deploy.
Source branch (CE on raw z, anchor KL 0.1) is unchanged and sees unstyled source images.
"""
import argparse, sys
sys.path.insert(0, "/home/jhy/aic2026_h76_20261026/code/reliable_kd_20261028")
from rkd_common import *  # noqa: E402,F401
from rkd_train import build_pairs, load_teacher_cache, rng_state, set_rng  # noqa: E402

OUTBASE = Path("/home/jhy/aic2026_claude_diag_20261029/train")
SOURCE_IMG = SOURCE_DATA / "train" / "images"
SOURCE_MSK = SOURCE_DATA / "train" / "masks"
LAM_ANCHOR, WARM_FRAC, BASE_LR, HEAD_LR, WD = 0.1, 0.2, 1e-5, 5e-7, 0.01
MEAN = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
STD = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)


def fda(src, tgt, beta=None):
    """Per-channel mean/std transfer src->tgt (FDA DC-only limit + contrast), src,tgt in [0,1].

    Full FDA (beta=0.01) was rejected after visual check: swapping low-frequency amplitudes between
    unrelated UAV/satellite scenes produced large coloured blotches. Std ratio clamped to [0.5, 2]
    (Source/Target gray-std 10-90% quantiles span ~21-52, ratio <~2.5).
    """
    ms, ss = src.mean((-2, -1), keepdim=True), src.std((-2, -1), keepdim=True).clamp_min(1e-3)
    mt, st = tgt.mean((-2, -1), keepdim=True), tgt.std((-2, -1), keepdim=True).clamp_min(1e-3)
    return ((src - ms) * (st / ss).clamp(0.5, 2.0) + mt).clamp(0, 1)


def classmix_mask(y, rng):
    cls = [int(c) for c in torch.unique(y) if int(c) != 255]
    if not cls:
        return torch.zeros_like(y, dtype=torch.bool)
    pick = rng.sample(cls, max(1, (len(cls) + 1) // 2))
    return torch.isin(y, torch.tensor(pick, device=y.device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--lam-t", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=3594)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    guard(); seed_all(a.seed)
    steps = a.smoke or a.steps
    out = OUTBASE / (a.name + ("_smoke" if a.smoke else "")); out.mkdir(parents=True, exist_ok=False)
    import rkd_train; rkd_train.SEED = a.seed
    pairs = build_pairs(steps)

    from src.label_mapping import official_to_train
    from src.transforms import TrainTransform
    from src.rare_crop import RareClassCropper
    student, teacher = model_from(), model_from(); teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student.eval()
    student.get_submodule("segformer.stages.2").train(); student.get_submodule("segformer.stages.3").train()
    for n, p in student.named_parameters():
        p.requires_grad_(n.startswith(TRAIN_PREFIX) or n.startswith(HEAD_PREFIX))
    base = [p for n, p in student.named_parameters() if n.startswith(TRAIN_PREFIX)]
    head = [p for n, p in student.named_parameters() if n.startswith(HEAD_PREFIX)]
    opt = torch.optim.AdamW([{"params": base, "lr": BASE_LR}, {"params": head, "lr": HEAD_LR}], weight_decay=WD)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    b = torch.tensor(BSTAR, device="cuda")[None, :, None, None]
    mean, std = MEAN.cuda(), STD.cuda()
    tf = TrainTransform(crop_size=1024, config=get_cfg()["transforms"],
                        rare_cropper=RareClassCropper(1024, .5, (7, 4, 6), .25), ignore_index=255)
    meta = {"name": a.name, "seed": a.seed, "steps": steps, "lam_t": a.lam_t, "beta": a.beta,
            "h75_sha256": H75_SHA, "base_recipe": "reliable_kd C1 (rkd_train.py)",
            "change": "target input = ClassMix(FDA(src->tgt), tgt); CE(z+b,y_src) on pasted, C1 KL elsewhere"}
    dump(out / "RUN_CONFIG.json", meta)
    hist = open(out / "history.jsonl", "w", buffering=1); t0 = time.time()
    for step, (stem, row) in enumerate(pairs, 1):
        if step % 20 == 1:
            guard()
        lam_t = a.lam_t * min(1.0, step / max(1, int(WARM_FRAC * steps)))
        rgb = read_rgb(SOURCE_IMG / f"{stem}.png")
        with Image.open(SOURCE_MSK / f"{stem}.png") as f:
            lab = official_to_train(np.asarray(f).copy())
        sx, sy = tf(rgb, lab, rng=random.Random(a.seed + 100000 + step))
        sx = sx.unsqueeze(0).cuda(); sy = sy.unsqueeze(0).long().cuda()
        tx = normalize_to_tensor(read_rgb(TARGET_IMAGES / row["input_name"])).unsqueeze(0).cuda()
        zT, votes, hn, flipf = (t.unsqueeze(0).cuda() for t in load_teacher_cache(Path(row["input_name"]).stem))
        q = (zT + b).softmax(1).detach()
        mrng = random.Random(a.seed + 400000 + step)
        with torch.no_grad():
            M = classmix_mask(sy[0], mrng)[None, None]                       # (1,1,1024,1024)
            s01, t01 = (sx * std + mean).clamp(0, 1), (tx * std + mean).clamp(0, 1)
            styled = (fda(s01, t01, a.beta) - mean) / std
            mix = torch.where(M, styled, tx)
            y256 = sy[:, ::4, ::4]; M256 = M[:, 0, ::4, ::4] & (y256 != 255)
        view = random.Random(a.seed + 200000 + step).randrange(8)
        gen = torch.Generator(device="cuda").manual_seed(a.seed + 300000 + step)
        aug = strong_photo(d4(mix, view), random.Random(a.seed + 200000 + step), gen)
        av = votes / 8.0; w = av.pow(2) * (1.0 - hn).clamp(0, 1) * torch.where(flipf > 0.5, 0.5, 1.0)
        w = w * (~M256).float()
        before = rng_state()
        for retry in range(8):
            set_rng(before); opt.zero_grad(set_to_none=True)
            for gi, g in enumerate(opt.param_groups):
                g["lr"] = (BASE_LR if gi == 0 else HEAD_LR) * (1 - (step - 1) / 10000)
            with torch.amp.autocast("cuda"):
                sz = student(pixel_values=sx).logits
            sz = F.interpolate(sz.float(), size=sy.shape[-2:], mode="bilinear", align_corners=False)
            l_src = F.cross_entropy(sz, sy, ignore_index=255)
            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    tz = teacher(pixel_values=sx).logits
                pt = F.interpolate(tz.float(), size=sy.shape[-2:], mode="bilinear", align_corners=False).softmax(1)
            valid = (sy != 255).float()
            l_anchor = ((pt * (pt.clamp_min(1e-8).log() - sz.log_softmax(1))).sum(1) * valid).sum() / valid.sum().clamp_min(1.0)
            with torch.amp.autocast("cuda"):
                ts = student(pixel_values=aug).logits
            p = (inv_d4(ts.float(), view) + b).log_softmax(1)
            kl_pix = (q * (q.clamp_min(1e-8).log() - p)).sum(1)
            ce_pix = F.nll_loss(p, y256.clamp(0, 7), reduction="none")
            n_tgt, n_mix = w.sum(), M256.float().sum()
            l_tgt = ((w * kl_pix).sum() + (ce_pix * M256.float()).sum()) / (n_tgt + n_mix).clamp_min(1e-6)
            total = l_src + LAM_ANCHOR * l_anchor + lam_t * l_tgt
            scaler.scale(total).backward(); scaler.unscale_(opt)
            sb = scaler.get_scale(); scaler.step(opt); scaler.update()
            if scaler.get_scale() >= sb:
                break
            emit(stage="AMP_RETRY", step=step, retry=retry)
        else:
            raise RuntimeError("repeated non-finite gradients")
        if not torch.isfinite(total):
            raise RuntimeError("non-finite objective")
        r = {"step": step, "lam_t": round(lam_t, 5), "loss_src": round(float(l_src), 5),
             "loss_anchor": round(float(l_anchor), 5), "loss_target": round(float(l_tgt), 5),
             "mix_frac": round(float(M256.float().mean()), 4), "elapsed": round(time.time() - t0, 1)}
        hist.write(json.dumps(r) + "\n")
        if step % 100 == 0 or step == steps:
            emit(stage="train", **r)
    hist.close()
    ck = out / f"checkpoint_step{steps}.pt"
    torch.save({"model": student.state_dict(), "step": steps, **meta}, ck)
    dump(out / "SUMMARY.json", {**meta, "status": "COMPLETED", "runtime_sec": round(time.time() - t0, 1),
                                "checkpoint": str(ck), "checkpoint_sha256": sha256(ck),
                                "peak_mem_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)})
    emit(stage="done", ckpt=str(ck))


if __name__ == "__main__":
    main()
