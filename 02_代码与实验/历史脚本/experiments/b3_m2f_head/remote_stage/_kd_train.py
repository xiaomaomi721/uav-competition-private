#!/usr/bin/env python3
"""Three-teacher zero-bias soft-KD with strong-student consistency.

What this implements
--------------------
Main line agreed on 2026-09-15 (see 决策记录_20260915_第二轮评审定案.md):

  * Teachers : stage23 B + s62 + s58, ZERO bias, TTA8 ensemble soft
               probabilities -- precomputed by _kd_precompute_teachers.py.
               No leaderboard calibration is baked into the targets.
  * Student  : initialised from a supplied checkpoint (s62 by default).
  * GT       : the official 5597 train split keeps plain hard cross-entropy.
  * test1    : the 500 unlabelled test images contribute soft KD ONLY --
               never argmax'd into a hard pseudo mask.

Teacher-guided consistency
--------------------------
The student sees a randomly augmented view; the teacher target is carried
through the SAME geometric ops (flip / rot90 / translation) so the target stays
aligned. That is what lets the student learn a decision function that is stable
under perturbation rather than merely re-fitting the ensemble. The teacher
probabilities are stored channels-last [H, W, C] precisely so the project's own
`apply_geometric_ops(image, mask)` can move image and target together.

Disagreement handling
---------------------
Low-confidence teacher pixels are DOWN-WEIGHTED, never dropped: the genuinely
complementary information between the three teachers lives in the disagreement
region, so hard filtering would collapse this back into the already-refuted
confidence-filtered hard pseudo-label regime.

  w(p) = w_min + (1 - w_min) * (p_max - 1/C) / (1 - 1/C),   clipped to [w_min, 1]

Loss
----
  L = CE(student, GT)  +  lambda * mean_pixels[ w * T^2 * KL(q_teacher || p_student) ]

Checkpoints are written at 40 / 60 / 80 / 100 % of max_steps, because
test-domain KD can improve early and then start chasing teacher noise.

Writes only under --outdir.

Resume (added 2026-09-16, for the soupB-init continuation)
----------------------------------------------------------
--save-full-at N writes checkpoint_full_step<N>.pt and exits 0, where the
checkpoint carries model + optimizer + scheduler + AMP scaler + the global step
counter + the number of batches already consumed from the two shuffled loaders
and their generator states. --resume-from restores all of that, so a run parked
at a gate can continue on the ORIGINAL schedule (lr(step) is still defined over
the original total) instead of silently restarting the LR curve.

The loaders iterate a shuffled stream whose order comes from a seeded generator,
so no amount of state_dict juggling reproduces position k without replaying.
On resume the loaders are rebuilt with the same seed and fast-forwarded through
exactly `consumed[k]` batches with the model in eval() and under
torch.inference_mode() -- no dropout, no BatchNorm update, no graph, and nothing
from those batches touches the optimizer state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

NUM_CLASSES = 8
IGNORE_INDEX = 255
PHYSICAL_GPU_INDEX = 2
SNAP = "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def assert_gpu(device: str) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(PHYSICAL_GPU_INDEX):
        raise RuntimeError("require CUDA_VISIBLE_DEVICES=2")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("require CUDA_DEVICE_ORDER=PCI_BUS_ID")
    if device != "cuda:0":
        raise RuntimeError("require --device cuda:0")
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA device required")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    import yaml
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_loader(config, dataset_root, project_root, split, transform, img_dir, mask_dir,
                 prob_dir=None, maximum=None, seed=42, shuffle=True, batch_size=1,
                 workers=4, use_probs=False):
    import torch
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    from src.transforms import normalize_to_tensor, apply_geometric_ops
    from src.label_mapping import official_to_train

    class KDDataset(Dataset):
        def __init__(self):
            self.stems = [l.strip() for l in (project_root / split).read_text().split()
                          if l.strip()]
            if maximum:
                rng = random.Random(seed)
                self.stems = rng.sample(self.stems, min(maximum, len(self.stems)))
            self.img_dir = Path(img_dir)
            self.mask_dir = Path(mask_dir)
            self.prob_dir = Path(prob_dir) if prob_dir else None
            self.crop_size = int(config["crop_size"])

        def __len__(self):
            return len(self.stems)

        def _aug_params(self, rng):
            return dict(
                hflip=rng.random() < 0.5,
                vflip=rng.random() < 0.5,
                rot=rng.randrange(4) if rng.random() < 0.5 else 0,
            )

        def _apply_geom(self, image, aux, p):
            """Move image and aux through the exact same geometry."""
            if p["hflip"]:
                image = np.flip(image, axis=1)
                aux = np.flip(aux, axis=1)
            if p["vflip"]:
                image = np.flip(image, axis=0)
                aux = np.flip(aux, axis=0)
            if p["rot"] % 4:
                image = np.rot90(image, k=p["rot"] % 4)
                aux = np.rot90(aux, k=p["rot"] % 4)
            return np.ascontiguousarray(image), np.ascontiguousarray(aux)

        def __getitem__(self, index):
            stem = self.stems[index]
            with Image.open(self.img_dir / f"{stem}.png") as h:
                image = np.asarray(h.convert("RGB")).copy()

            if use_probs:
                # target-domain image: soft teacher distribution, no GT at all
                probs = np.load(self.prob_dir / f"{stem}.npy")  # [H,W,C]
                p = self._aug_params(random)
                image, probs = self._apply_geom(image, probs, p)
                tensor = normalize_to_tensor(image)
                # uint8 caches store round(p*255); fp16 caches store p directly.
                arr = probs.astype(np.float32)
                if probs.dtype == np.uint8:
                    arr = arr / 255.0
                target = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
                return {"kind": "soft", "pixel_values": tensor,
                        "soft_target": target, "stem": stem}

            with Image.open(self.mask_dir / f"{stem}.png") as h:
                mask = official_to_train(np.asarray(h).copy())
            p = self._aug_params(random)
            image, mask = self._apply_geom(image, mask, p)
            tensor = normalize_to_tensor(image)
            label = torch.from_numpy(mask.astype(np.int64, copy=True))
            return {"kind": "hard", "pixel_values": tensor,
                    "labels": label, "stem": stem}

    def collate(batch):
        out = {"kind": batch[0]["kind"], "stem": [b["stem"] for b in batch]}
        out["pixel_values"] = torch.stack([b["pixel_values"] for b in batch])
        if out["kind"] == "hard":
            out["labels"] = torch.stack([b["labels"] for b in batch])
        else:
            out["soft_target"] = torch.stack([b["soft_target"] for b in batch])
        return out

    ds = KDDataset()
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=True, drop_last=False, collate_fn=collate,
                      generator=torch.Generator().manual_seed(seed))


# --------------------------------------------------------------------------
# loss
# --------------------------------------------------------------------------
def soft_kd_loss(student_logits, teacher_prob, temperature, w_min):
    """Weighted, temperature-scaled KL(teacher || student) in logit space.

    The student logits are upsampled to the teacher's (full) grid rather than
    the teacher being downsampled: the soft target carries 1024-res detail and
    downsampling it would throw that away.
    """
    import torch
    import torch.nn.functional as F

    if student_logits.shape[-2:] != teacher_prob.shape[-2:]:
        student_logits = F.interpolate(student_logits, size=teacher_prob.shape[-2:],
                                       mode="bilinear", align_corners=False)

    # teacher_prob is ALREADY softmax(teacher_logits / T) -- taken straight from
    # the cache. Taking log_softmax of it again would silently re-shape the
    # distribution (softmax of a probability vector is much flatter) and change
    # the objective. log(q) is the correct log-probability here.
    # Invariant 1: the KD target must be a legal probability distribution.
    # uint8 caches store round(p*255)/255, which sums to ~0.9991; renormalising
    # is free and makes KL well-defined.
    teacher_prob = teacher_prob.clamp_min(0.0)
    teacher_prob = teacher_prob / teacher_prob.sum(dim=1, keepdim=True).clamp_min(1e-12)

    log_q = torch.log(teacher_prob.clamp_min(1e-12))
    log_p = F.log_softmax(student_logits / temperature, dim=1)

    kl = (log_q.exp() * (log_q - log_p)).sum(dim=1)            # [B,H,W]

    p_max = teacher_prob.max(dim=1).values
    c = teacher_prob.shape[1]
    w = w_min + (1.0 - w_min) * (p_max - 1.0 / c) / (1.0 - 1.0 / c)
    w = w.clamp(w_min, 1.0)

    return (w * kl).mean() * (temperature ** 2), float(w.mean())


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--config", required=True, help="base config yaml (architecture/crop etc)")
    ap.add_argument("--init-checkpoint", required=True, help="student initialisation")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--soft-dir", required=True, help="dir with probs/*.npy teacher targets")
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--lambda-kd", type=float, default=0.35)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--w-min", type=float, default=0.3)
    ap.add_argument("--kd-every", type=int, default=2,
                    help="insert ONE test1 soft-KD batch every N optimizer steps. "
                         "N=1 -> 1 real : 1 test; N=2 -> 2 real : 1 test (GT-heavy). "
                         "NOTE: a previous name (--real-per-test) made this "
                         "ambiguous; the semantics are the ones documented here.")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3.0e-5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-interval", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--measure-grad-ratio", action="store_true",
                    help="one-shot diagnostic: lambda*||g_KD|| / ||g_CE||")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume-from", default=None,
                    help="continue an interrupted run from a full-state checkpoint. "
                         "Everything else on the CLI must be byte-identical to the "
                         "original launch (checked against resume_state.json): the "
                         "loader order, the optimizer/scheduler/scaler state and the "
                         "global step counter all have to continue, not restart.")
    ap.add_argument("--save-full-at", default=None,
                    help="comma-separated global step numbers at which to ALSO write "
                         "checkpoint_full_step<N>.pt (model+optimizer+scheduler+"
                         "scaler+global_step+consumed loader position) and then exit "
                         "cleanly. Used to park a run at a gate and resume it later "
                         "without restarting the LR schedule.")
    args = ap.parse_args()

    # ---- resume input validation (before any output is created) --------------
    resume_ck = None
    if args.resume_from:
        import torch  # module-level imports deliberately defer torch (see assert_gpu)
        resume_ck = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if resume_ck.get("schema") != "KD_FULL_STATE_V1":
            raise SystemExit(f"not a full-state checkpoint: {args.resume_from}")
        resume_outdir = Path(resume_ck["outdir"])
        resume_meta_path = resume_outdir / "resume_state.json"
        if not resume_meta_path.is_file():
            raise SystemExit(f"missing {resume_meta_path} (needed to verify the "
                             f"recipe is unchanged)")
        recorded = json.loads(resume_meta_path.read_text(encoding="utf-8"))
        # The schedule is defined over the ORIGINAL total; --max-steps passed on
        # the resume CLI has to be that same total or the LR curve silently
        # becomes a different curve.
        if int(recorded["max_steps"]) != int(args.max_steps):
            raise SystemExit(f"max_steps changed: recorded "
                             f"{recorded['max_steps']} vs CLI {args.max_steps}; the "
                             f"LR schedule would restart. Refusing.")
        if str(Path(resume_outdir).resolve()) != str(Path(args.outdir).resolve()):
            raise SystemExit(f"--outdir {args.outdir} != recorded {resume_outdir}")
        for key, cli in (("lambda_kd", args.lambda_kd),
                         ("temperature", args.temperature),
                         ("kd_every", args.kd_every),
                         ("batch_size", args.batch_size),
                         ("grad_accum", args.grad_accum),
                         ("lr", args.lr),
                         ("w_min", args.w_min),
                         ("workers", args.workers),
                         ("seed", args.seed)):
            if recorded["args"].get(key) != cli:
                raise SystemExit(f"recipe drift on {key}: recorded "
                                 f"{recorded['args'].get(key)!r} vs CLI {cli!r}")
        for key in ("init_checkpoint", "config", "dataset_root", "soft_dir"):
            if Path(str(recorded["args"].get(key))).resolve() != Path(str(getattr(args, key))).resolve():
                raise SystemExit(f"recipe drift on {key}: recorded "
                                 f"{recorded['args'].get(key)!r} vs CLI "
                                 f"{getattr(args, key)!r}")
        print(f"[resume] full state {args.resume_from}: global_step="
              f"{resume_ck['global_step']} consumed={resume_ck['consumed']}", flush=True)

    save_full_at = set()
    if args.save_full_at:
        save_full_at = {int(s) for s in str(args.save_full_at).split(",") if s.strip()}

    outdir = Path(args.outdir)
    if outdir.exists() and not args.resume_from:
        raise SystemExit(f"outdir exists: {outdir}")

    assert_gpu(args.device)
    import torch
    import torch.nn.functional as F
    for p in (SNAP, "/home/jhy/gpu2_day_queue"):
        if p not in sys.path:
            sys.path.insert(0, p)
    from src.models import build_model
    from src.utils import set_seed
    from src.inference import evaluate_model

    outdir.mkdir(parents=True)
    set_seed(args.seed)

    config = load_config(Path(args.config))
    config["experiment_id"] = outdir.name
    project_root = Path(SNAP)
    dataset_root = Path(args.dataset_root)
    crop = int(config["crop_size"])
    ignore_index = int(config.get("ignore_index", IGNORE_INDEX))
    temperature = float(args.temperature)

    train_dir = dataset_root / "train" / "images"
    mask_dir = dataset_root / "train" / "masks"
    # test1 images live at <root>/images, NOT under train/ (the official tarball
    # layout is misleading: `images/` holds the 500 unlabelled test1 patches)
    test_dir = dataset_root / "images"
    if not test_dir.is_dir():
        raise SystemExit(f"missing test1 image dir: {test_dir}")
    prob_dir = Path(args.soft_dir) / "probs"
    if not prob_dir.is_dir():
        raise SystemExit(f"missing teacher probs: {prob_dir}")

    test_names = sorted(p.stem for p in prob_dir.glob("*.npy"))
    if len(test_names) != 500:
        raise SystemExit(f"expected 500 teacher targets, found {len(test_names)}")
    (outdir / "test_names.txt").write_text("\n".join(test_names), encoding="utf-8")

    # the test1 split file must exist before the loader is constructed; it is
    # written under outdir so the loader resolves it against project_root=outdir
    (outdir / "splits_test1.txt").write_text("\n".join(test_names), encoding="utf-8")

    real_loader = build_loader(config, dataset_root, project_root, "splits/train.txt",
                               None, train_dir, mask_dir, batch_size=args.batch_size,
                               workers=args.workers, seed=args.seed)
    test_loader = build_loader(config, dataset_root, outdir, "splits_test1.txt", None,
                               test_dir, mask_dir, prob_dir=prob_dir,
                               batch_size=args.batch_size, workers=args.workers,
                               seed=args.seed + 1, use_probs=True)

    val_split = str(config.get("validation_split", "splits/val.txt"))
    val_loader = build_loader(config, dataset_root, project_root, val_split, None,
                              train_dir, mask_dir, batch_size=1, workers=args.workers,
                              shuffle=False, seed=args.seed + 2)

    print(f"real batches/epoch={len(real_loader)}  test batches={len(test_loader)}  "
          f"val={len(val_loader)}", flush=True)

    device = torch.device(args.device)
    model = build_model(config).to(device)
    from src.models import load_model_checkpoint, get_cwfa_module, set_cwfa_active
    ck = load_model_checkpoint(model, Path(args.init_checkpoint), device)
    print(f"student initialised from {args.init_checkpoint} "
          f"(step={ck.get('step')}, best_miou={ck.get('best_miou')})", flush=True)

    # CWFA is only active on steps >= cwfa.start_step. The base config carries
    # start_step=28001 from the original 56000-step schedule, which would leave
    # CWFA off for this short fine-tune. Pin the module ACTIVE for the whole KD
    # run so the student trains under the same regime it will be evaluated in
    # (the s62 checkpoint itself was produced with CWFA active).
    cwfa_cfg = config.get("cwfa", {})
    cwfa_module = get_cwfa_module(model)
    if bool(cwfa_cfg.get("enabled", False)):
        if cwfa_module is None:
            raise RuntimeError("cwfa.enabled=true but model has no CWFA module")
        config["cwfa"]["start_step"] = 1
        set_cwfa_active(model, True)
        print(f"CWFA forced ACTIVE for all KD steps "
              f"(p={cwfa_cfg.get('probability')}, eps={cwfa_cfg.get('epsilon')})", flush=True)
    else:
        if cwfa_module is not None:
            set_cwfa_active(model, False)
        print("CWFA disabled", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr),
                                  weight_decay=float(config.get("optimizer", {})
                                                     .get("weight_decay", 0.01)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: max(0.0, 1.0 - min(s, args.max_steps) / args.max_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    max_steps = int(args.max_steps)
    ckpt_fracs = [0.4, 0.6, 0.8, 1.0]
    ckpt_steps = sorted({max(1, int(round(f * max_steps))) for f in ckpt_fracs})

    # ---- restore full training state (model was already loaded from
    # --init-checkpoint, which for a resume is the same soupB as at step 0) ----
    init_step = 0
    if resume_ck is not None:
        # weights_only=False: these are plain python/JSON values plus tensors.
        model.load_state_dict(resume_ck["model"])
        optimizer.load_state_dict(resume_ck["optimizer"])
        scheduler.load_state_dict(resume_ck["scheduler"])
        scaler.load_state_dict(resume_ck["scaler"])
        init_step = int(resume_ck["global_step"])
        print(f"[resume] state restored at global_step={init_step}; "
              f"scheduler.last_epoch={scheduler.last_epoch} "
              f"lr={scheduler.get_last_lr()[0]:.6e} "
              f"scaler_scale={scaler.get_scale()}", flush=True)

    grad_ratio = None  # filled in by the one-shot diagnostic below

    meta = {
        "schema": "KD_TRAIN_V1",
        "created": datetime.now(timezone.utc).isoformat(),
        "init_checkpoint": args.init_checkpoint,
        "init_checkpoint_sha256": sha256_file(Path(args.init_checkpoint)),
        "soft_dir": args.soft_dir,
        "max_steps": max_steps,
        "lambda_kd": args.lambda_kd,
        "grad_ratio_lambdaKD_over_CE": grad_ratio,
        "temperature": temperature,
        "w_min": args.w_min,
        "kd_every": args.kd_every,
        "sampling_ratio_note": "1 test-KD batch per kd_every optimizer steps",
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "checkpoint_steps": ckpt_steps,
        "teacher_bias": "ZERO",
        "note": "test1 contributes soft KD only; never argmax'd into hard pseudo masks",
    }
    (outdir / "KD_META.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    # ---- one-shot gradient-scale diagnostic ----------------------------------
    # lambda is a loss weight, NOT a guarantee about the relative size of the
    # gradients. Measure r = lambda*||grad L_KD|| / ||grad L_CE|| once so we know
    # whether 500 test images are steering the update or merely nudging it.
    grad_ratio = None
    if args.measure_grad_ratio:
        try:
            rb = next(iter(real_loader))
            tb = next(iter(test_loader))
            px = rb["pixel_values"].to(device); lb = rb["labels"].to(device)
            tpx = tb["pixel_values"].to(device); tgt = tb["soft_target"].to(device)

            def _grad_norm(fn):
                model.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=True):
                    loss = fn()
                loss.backward()
                sq = 0.0
                for prm in model.parameters():
                    if prm.grad is not None:
                        sq += float(prm.grad.detach().float().pow(2).sum())
                model.zero_grad(set_to_none=True)
                return math.sqrt(sq)

            g_ce = _grad_norm(lambda: F.cross_entropy(
                F.interpolate(model(pixel_values=px).logits, size=lb.shape[-2:],
                              mode="bilinear", align_corners=False),
                lb, ignore_index=ignore_index))
            g_kd = _grad_norm(lambda: soft_kd_loss(
                model(pixel_values=tpx).logits, tgt, temperature, args.w_min)[0])
            if g_ce > 0:
                grad_ratio = (args.lambda_kd * g_kd) / g_ce
                print(f"[diag] ||g_CE||={g_ce:.6f}  ||g_KD||={g_kd:.6f}  "
                      f"lambda*||g_KD||/||g_CE|| = {grad_ratio:.4f}", flush=True)
            del px, lb, tpx, tgt
            torch.cuda.empty_cache()
        except Exception as exc:  # diagnostic must never kill the run
            print(f"[diag] gradient ratio measurement failed: {exc}", flush=True)

    # The two iter() calls above consumed a permutation from each loader's
    # generator -- these are the permutations the training loop will use. Record
    # the resulting generator states so a resume can PROVE it replayed to the
    # right position instead of merely running the same code.
    diag_generator_states = {
        "real": real_loader.generator.get_state(),
        "test": test_loader.generator.get_state(),
    }

    history = []
    if (outdir / "validation_history.json").is_file():
        history = json.loads((outdir / "validation_history.json").read_text(encoding="utf-8"))
    real_iter = iter(real_loader)
    test_iter = iter(test_loader)

    # ---- fast-forward the loader streams on resume ---------------------------
    # The DataLoader shuffles with a seeded generator (torch.Generator().manual_seed(seed)),
    # so the stream is a fixed permutation; position k can only be recovered by
    # replaying the first k batches. Rebuild with the SAME seed, then pull them
    # through with the model in eval() under inference_mode: no dropout, no
    # BatchNorm running-stat update, no autograd graph, and nothing here feeds
    # the optimizer.
    #
    # ORDER MATTERS: the one-shot gradient diagnostic above also calls
    # iter(real_loader) / iter(test_loader), and each iter() draws a FRESH
    # permutation from the generator. The original run therefore trained on the
    # permutation drawn by the SECOND iter() of each loader. Fast-forwarding
    # before the diagnostic would consume an extra permutation and silently put
    # training on a different stream, so this has to sit after it and act on the
    # very iterators the training loop uses.
    consumed = {"real": 0, "test": 0}
    if resume_ck is not None:
        consumed = {k: int(v) for k, v in resume_ck["consumed"].items()}
        t_ff = time.time()
        model.eval()
        with torch.inference_mode():
            for _it, _key in ((real_iter, "real"), (test_iter, "test")):
                for _ in range(consumed[_key]):
                    next(_it)
                print(f"[resume] fast-forwarded {_key}: {consumed[_key]} batches "
                      f"({time.time()-t_ff:.0f}s)", flush=True)
        model.train()
        # Proof, not intent: if the replayed stream is the same stream, the
        # generators must land on exactly the states recorded at park time.
        # (DataLoader wraps seed+1 into a fresh generator for its base_seed, so
        # the state that matters is the recorded one, not seed+1 verbatim.)
        _want = resume_ck.get("loader_generator_states") or {}
        for _ld, _key in ((real_loader, "real"), (test_loader, "test")):
            if _key not in _want:
                continue
            _got = _ld.generator.get_state()
            _exp = _want[_key]
            _ok = (_got.shape == _exp.shape) and bool(torch.equal(_got.long(), _exp.long()))
            print(f"[resume] {_key} generator state at park == after replay: {_ok}",
                  flush=True)
            if not _ok:
                raise SystemExit(
                    f"resume replay did NOT land on the parked stream for {_key}; "
                    f"refusing to continue on a different batch order")
        print(f"[resume] stream position restored: {consumed}", flush=True)

    t0 = time.time()
    accum = int(args.grad_accum)

    for step in range(init_step, max_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        ce_total = 0.0
        kd_total = 0.0
        kd_w = 0.0
        do_test = (step % int(args.kd_every) == 0)

        for micro in range(accum):
            try:
                rb = next(real_iter)
            except StopIteration:
                real_iter = iter(real_loader)
                rb = next(real_iter)
            consumed["real"] += 1
            px = rb["pixel_values"].to(args.device, non_blocking=True)
            lb = rb["labels"].to(args.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=True):
                logits = model(pixel_values=px).logits
                # SegFormer-B3 emits logits at 1/4 resolution; match the GT grid
                # exactly as src.inference.evaluate_model does.
                logits = F.interpolate(logits, size=lb.shape[-2:], mode="bilinear",
                                       align_corners=False)
                ce = F.cross_entropy(logits, lb, ignore_index=ignore_index)
            ce_total += float(ce.detach())

            kd_term = None
            if do_test:
                try:
                    tb = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_loader)
                    tb = next(test_iter)
                consumed["test"] += 1
                tpx = tb["pixel_values"].to(args.device, non_blocking=True)
                tgt = tb["soft_target"].to(args.device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=True):
                    tlogits = model(pixel_values=tpx).logits
                    kd, wm = soft_kd_loss(tlogits, tgt, temperature, args.w_min)
                kd_term = args.lambda_kd * kd
                kd_total += float(kd.detach()) * args.lambda_kd
                kd_w = wm

            # INVARIANT 2 (weighting): CE and KD must be divided by the SAME
            # accum, so that one optimizer step optimises exactly
            #     L = CE + lambda * KD
            # Dividing only one of them silently rescales lambda by accum.
            total = ce if kd_term is None else (ce + kd_term)
            scaler.scale(total / accum).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % 50 == 0 or step == max_steps - 1:
            print(f"step={step+1}/{max_steps} ce={ce_total/accum:.4f} "
                  f"kd={kd_total/accum:.4f} w={kd_w:.3f} "
                  f"lr={scheduler.get_last_lr()[0]:.3e} "
                  f"t={time.time()-t0:.0f}s", flush=True)

        if (step + 1) in ckpt_steps:
            vm = evaluate_model(model, val_loader, device, num_classes=NUM_CLASSES,
                                ignore_index=ignore_index, amp=True)
            # ---- diagnostic: did the student actually move toward the ensemble?
            # If agreement(student, ensemble) stays ~= agreement(s62, ensemble)
            # the student merely reproduced its own initialisation and absorbed
            # nothing from the other two teachers.
            agree = 0.0
            conf = 0.0
            nb = 0
            with torch.inference_mode():
                for tb in test_loader:
                    tpx = tb["pixel_values"].to(device, non_blocking=True)
                    tt = tb["soft_target"].to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=True):
                        sl = model(pixel_values=tpx).logits
                    sl = F.interpolate(sl, size=tt.shape[-2:], mode="bilinear",
                                       align_corners=False)
                    agree += float((sl.argmax(1) == tt.argmax(1)).float().mean())
                    conf += float(tt.max(1).values.mean())
                    nb += 1
                    if nb >= 25:
                        break
            diag = {"step": step + 1, "agree_with_ensemble": agree / max(nb, 1),
                    "teacher_mean_confidence": conf / max(nb, 1)}
            print(f"[diag] student-vs-ensemble argmax agreement = "
                  f"{diag['agree_with_ensemble']:.4f}  "
                  f"teacher_conf = {diag['teacher_mean_confidence']:.4f}", flush=True)
            local = float(vm.get("mIoU", 0.0))
            history.append({"step": step + 1, "local_miou": local, **diag})
            path = outdir / f"checkpoint_step{step+1}.pt"
            torch.save({"model": model.state_dict(), "config": config,
                        "step": step + 1, "local_miou": local,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "global_step": step + 1, "consumed": dict(consumed)},
                       path)
            print(f"[ckpt] step={step+1} local_miou={local:.6f} -> {path.name}", flush=True)
            (outdir / "validation_history.json").write_text(
                json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

        # ---- park the run at a gate: full state + where the loader streams are.
        # Deliberately AFTER the periodic checkpoint block so a gate that lands
        # on a ckpt_steps boundary produces both files.
        if (step + 1) in save_full_at:
            global_step = step + 1
            full = {
                "schema": "KD_FULL_STATE_V1",
                "global_step": global_step,
                "step_in_run": step,          # 0-based LAST completed step
                "consumed": dict(consumed),
                "diag_generator_states": diag_generator_states,
                "loader_generator_states": {
                    "real": real_loader.generator.get_state(),
                    "test": test_loader.generator.get_state(),
                },
                "outdir": str(outdir),
                # the CLI `seed` is what determines the stream order; the
                # scheduler is defined over the ORIGINAL total.
                "args": {k: getattr(args, k) for k in (
                    "init_checkpoint", "config", "dataset_root", "soft_dir",
                    "max_steps", "lambda_kd", "temperature", "w_min", "kd_every",
                    "batch_size", "grad_accum", "lr", "workers", "val_interval",
                    "seed", "device")},
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
            }
            p = outdir / f"checkpoint_full_step{global_step}.pt"
            torch.save(full, p)
            (outdir / "resume_state.json").write_text(
                json.dumps({k: v for k, v in full.items()
                            if k not in ("model", "optimizer", "scheduler", "scaler")},
                           ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            (outdir / "PARKED.json").write_text(json.dumps({
                "status": "parked", "global_step": global_step,
                "consumed": dict(consumed), "checkpoint": p.name,
                "resume_cmd": f"--resume-from {p} (same outdir, same CLI)",
                "scheduler_last_epoch": scheduler.last_epoch,
                "lr_at_park": scheduler.get_last_lr()[0],
                "scaler_scale_at_park": scaler.get_scale(),
                "parked": datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[park] global_step={global_step} consumed={consumed} "
                  f"lr={scheduler.get_last_lr()[0]:.6e} "
                  f"scaler_scale={scaler.get_scale()} -> {p.name}", flush=True)
            print("KD_TRAIN_PARKED", flush=True)
            return 0

    (outdir / "COMPLETED.json").write_text(json.dumps({
        "status": "completed", "steps": max_steps, "history": history,
        "finished": datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("KD_TRAIN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
