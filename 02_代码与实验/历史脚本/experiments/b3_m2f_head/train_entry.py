"""Independent M2F training entry -- does NOT touch src/trainer.py.

Scope (authorised 2026-09-15, GPT-gated):
  s62 MiT-B3 encoder  +  frozen M2F-style head
  encoder small LR, random head large LR; warmup; AMP; grad clipping.
  Explicitly NOT included in v1: label smoothing, focal, KD, new pseudo-labels,
  EMA, multi-loss ablations. First version must isolate ONE structural variable:
  the mask-classification / query-region paradigm.

Stages (run in order; each is a gate, not a hyper-parameter sweep):
  --stage smoke   forward -> Hungarian/set loss -> backward -> optimizer step
                  -> semantic inference. Asserts every loss/grad is finite
                  under AMP. Also records peak memory and step time.
  --stage small   8 images, ~40 steps, batch_size=1. Answers "did the entry
                  plumbing work" -- NOT "is the method good".
  --stage overfit tiny fixed subset, a few hundred steps. Answers ONE question:
                  can a random head drive train loss down and turn the
                  prediction from noise into semantic structure.
  --stage ckpt    measure a real full training checkpoint size (for disk budget).
  --stage formal_run1   M2F Run1. 5597 real-GT images, seeded per-epoch shuffle,
                  batch=1 sequential traversal, mainline Stage62/B3 augmentation
                  and CWFA(p=0.3 eps=4 start=28001). Recipe frozen 2026-09-15;
                  every value is recorded in RUN_META so the command line and the
                  executed config cannot drift apart.

Hard constraints carried over from the project:
  * physical GPU2 only; CUDA_DEVICE_ORDER=PCI_BUS_ID + CUDA_VISIBLE_DEVICES=2;
    logical device cuda:0. assert_gpu() enforces this and refuses to run
    otherwise -- it never falls back to another GPU.
  * Ctrl+C / SIGTERM are honoured: the loop polls a stop flag and exits at the
    next step boundary, writes a summary, and leaves no partial checkpoint.
  * writes only under --outdir (must not exist), plus a line in a stop file
    inside that dir. Never writes into the frozen bundle.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PHYSICAL_GPU_INDEX = 2
NUM_CLASSES = 8
IGNORE = 255
BUNDLE = "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1"
S62_CKPT = (BUNDLE + "/stage62_runs/"
            "stage62_w1_resume55b_pseudo500_seed42/best_miou.pt")
DATA_ROOT = "/home/jhy/datasets/2026-低空图像语义分割赛道-训练集"
MODEL_DIR = "/home/jhy/models/segformer-b3-ade512"

# ---------------------------------------------------------------------------
# M2F Run1 formal recipe -- FROZEN 2026-09-15.  Not derived from tiny-overfit.
#
# LR: official Mask2Former semantic seg uses AdamW base 1e-4 with a 0.1
# backbone multiplier, i.e. head 1e-4 / backbone 1e-5. Our encoder is an
# already-mature s62 B3 and is worth at least as much protection as an
# ImageNet backbone, so the 10x separation is kept rather than reverting to the
# B3 mainline's 6e-5.
#
# LR factor: min(1, step/50) * (1 - step/56000)**1.0 -- a 50-iteration linear
# warmup multiplied by the mainline polynomial decay. This matters because
# AdamW applies the LR OUTSIDE the normalised update:
#     theta_{t+1} = theta_t - eta_t * m_hat/(sqrt(v_hat)+eps) - eta_t*lam*theta_t
# so warmup=0 and warmup=50 really are different configurations. (An earlier
# revision of this file set warmup=0, arguing the LR cancels inside
# m_hat/sqrt(v_hat). That is wrong -- what cancels there is a rescaling of the
# GRADIENTS, not of eta -- and the run was aborted as
# ABORTED_CONFIG_DEVIATION_WARMUP.)
#
# clip=0.01 is the official full-model value. Global-norm clipping on a single
# step multiplies every gradient by one positive scalar and therefore preserves
# the update DIRECTION, changing only its magnitude; Adam's historical moments
# only approximately cancel such a scale under a fixed scaling factor, which is
# not the case here since the clipping factor varies per step. So this is a
# retained official design choice, not something justified by "exact
# cancellation".
# ---------------------------------------------------------------------------
FORMAL_STAGE = "formal_run1"
FORMAL_STEPS = 56000
FORMAL_WARMUP = 50
FORMAL_HEAD_LR = 1e-4
FORMAL_ENCODER_LR = 1e-5
FORMAL_BATCH_SIZE = 1
FORMAL_CWFA = {"enabled": True, "probability": 0.3, "epsilon": 4.0,
               "norm_eps": 1.0e-12, "start_step": 28001}

# The augmentation actually executed by the mainline, read back from the merged
# Stage62 config (configs/stage62/wave1_exp62_resume55b.yaml and its _base_
# chain) with src.utils.load_config. Recorded in RUN_META so the executed
# augmentation and the documented one cannot drift apart.
MAINLINE_AUG_CONFIG = {
    "horizontal_flip_probability": 0.5,
    "vertical_flip_probability": 0.5,
    "rotate90_probability": 0.5,
    "color_jitter": {"probability": 0.5, "brightness": 0.1, "contrast": 0.1,
                     "saturation": 0.1, "hue": 0.02},
    "domain_aug": {"enabled": False},
}
MAINLINE_RARE_CROP = {"enabled": True, "probability": 0.5,
                      "priority_train_ids": [7, 4, 6],
                      "max_offset_fraction": 0.25}
CROP_SIZE = 1024

_STOP = {"flag": False}


def _request_stop(signum, frame):        # noqa: ARG001
    _STOP["flag"] = True
    print(f"\n[stop] signal {signum} received; will exit at the next step boundary",
          flush=True)


def assert_gpu(device: str) -> None:
    """Refuse to run anywhere except physical GPU2 as logical cuda:0."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(PHYSICAL_GPU_INDEX):
        raise RuntimeError(
            f"require CUDA_VISIBLE_DEVICES={PHYSICAL_GPU_INDEX}, "
            f"got {os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("require CUDA_DEVICE_ORDER=PCI_BUS_ID")
    if device != "cuda:0":
        raise RuntimeError(f"logical device must be cuda:0, got {device!r}")


def write_stop(outdir: Path, note: str) -> None:
    (outdir / "STOP_REQUESTED").write_text(
        f"{datetime.now(timezone.utc).isoformat()}\n{note}\n", encoding="utf-8")


def build_encoder(device):
    """s62 MiT-B3 encoder, random decode head discarded."""
    import torch
    from transformers import AutoModelForSemanticSegmentation

    model = AutoModelForSemanticSegmentation.from_pretrained(
        MODEL_DIR, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True,
        local_files_only=True)
    checkpoint = torch.load(S62_CKPT, map_location="cpu", weights_only=False)
    state = checkpoint.get("model") or checkpoint.get("model_state_dict")
    if state is None:
        raise RuntimeError(f"no model weights in {S62_CKPT}")
    # keep only the encoder; the M2F head replaces the decode head entirely
    encoder_state = {k: v for k, v in state.items() if k.startswith("segformer.")}
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected encoder keys: {unexpected[:5]}")
    print(f"[init] loaded {len(encoder_state)} encoder tensors from s62 "
          f"(step={checkpoint.get('step')}, miou={checkpoint.get('best_miou')})",
          flush=True)
    return model.segformer.to(device)


def forward_logits(encoder, pixel_values, head):
    outputs = encoder(pixel_values=pixel_values, output_hidden_states=True)
    stages = outputs.hidden_states
    if stages is None:
        raise RuntimeError("output_hidden_states=True returned no hidden_states")
    return head(stages)


def build_batch(dataset_root, stems, crop, device, seed):
    """One real (image, label) batch at native resolution.

    The image goes to `device`; the LABEL deliberately stays on CPU so that
    `build_set_targets` runs wholly on one device. Mixing a CUDA valid map with
    CPU sample points fails inside `weight_at_points`; see
    `to_device_set_targets`.
    """
    import torch
    from PIL import Image
    sys.path.insert(0, BUNDLE)
    from src.transforms import normalize_to_tensor
    from src.label_mapping import official_to_train

    images, labels = [], []
    for stem in stems:
        img = np.asarray(Image.open(
            Path(dataset_root) / "train" / "images" / f"{stem}.png").convert("RGB")).copy()
        mask = np.asarray(Image.open(
            Path(dataset_root) / "train" / "masks" / f"{stem}.png")).copy()
        images.append(normalize_to_tensor(img))
        labels.append(torch.from_numpy(official_to_train(mask).astype(np.int64)))
    return torch.stack(images).to(device), torch.stack(labels)   # label stays on CPU


def build_train_items(stems, dev_seed):
    """Batch items for one M2F training iteration, via the MAINLINE data path.

    This deliberately constructs the real `TrainTransform` from the frozen B3
    bundle instead of re-implementing "something similar": the augmentation the
    formal run sees is then the one Stage62/Exp18 actually trained with, so a
    second measurement/data path cannot silently diverge from it.

    The augmentation RNG is seeded per item, so a given (epoch, position) yields
    the same crop for a given network-depth. Note that the number of
    `torch.rand` calls inside CWFA depends on the decoder layer, so different
    depths consume this stream differently and identical results across depths
    are NOT expected. What the fixed seed buys is per-depth determinism: each
    depth draws its augmentation from a fixed, reproducible stream.
    """
    import torch
    from PIL import Image
    sys.path.insert(0, BUNDLE)
    from src.rare_crop import RareClassCropper
    from src.transforms import TrainTransform
    from src.label_mapping import official_to_train

    cropper = RareClassCropper(crop_size=CROP_SIZE, rare_probability=0.5,
                               priority_ids=(7, 4, 6), max_offset_fraction=0.25)
    transform = TrainTransform(
        crop_size=CROP_SIZE,
        config=MAINLINE_AUG_CONFIG,
        rare_cropper=cropper, ignore_index=IGNORE)

    items = []
    for position, stem in enumerate(stems):
        image = np.asarray(Image.open(
            Path(DATA_ROOT) / "train" / "images" / f"{stem}.png"
        ).convert("RGB")).copy()
        raw_mask = np.asarray(Image.open(
            Path(DATA_ROOT) / "train" / "masks" / f"{stem}.png")).copy()
        label = official_to_train(raw_mask)
        generator = torch.Generator().manual_seed(
            (int(dev_seed) * 1_000_003 + position) % (2 ** 31 - 1))
        rng = random.Random(int(torch.randint(0, 2 ** 31 - 1, (1,),
                                              generator=generator).item()))
        pixel, labels, _ = transform(image, label, rng=rng, return_metadata=True)
        items.append({"pixel_values": pixel.unsqueeze(0),
                      "labels": labels.unsqueeze(0)})
    return items


def finite_or_die(tag, tensors, step):
    import torch
    for name, value in tensors.items():
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"[step {step}] non-finite {tag}.{name} = {value}")
        elif isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"[step {step}] non-finite {tag}.{name} = {value}")


def make_criterion():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from b3_m2f_head.loss import SetCriterion, build_set_targets
    return SetCriterion, build_set_targets


def to_device_set_targets(targets, points, weight, valid_map, device):
    """Move the set-prediction targets onto `device`.

    `build_set_targets` must run entirely on ONE device: it calls
    `weight_at_points` -> `sample_map` -> `F.grid_sample` internally, so a valid
    map on one device and sampled points on another fail immediately. The targets
    are built on CPU (the helper takes no device argument), then the finished
    tensors are moved here. This never shows up in the CPU-only parity tests --
    it is exactly the class of defect the GPU smoke exists to catch.
    """
    moved = [{"class_labels": t["class_labels"].to(device),
              "mask_labels": t["mask_labels"].to(device)} for t in targets]
    return moved, points.to(device), weight.to(device), valid_map.to(device)


def mean_iou_from_counts(confusion: "torch.Tensor", ignore_index: int = 0):
    """mIoU and per-class IoU from a (9, 9) [gt, pred] confusion matrix.

    The confusion matrix is indexed by OFFICIAL ids 0..8, where official 0 is
    Ignore -- NOT the internal label 255. The old default (`IGNORE` = 255) was
    outside the 0..8 range, so nothing was ever dropped: harmless while the
    ignore row happened to be empty, wrong the moment real Ignore pixels appear.

    Classes absent from the ground truth are excluded from the mean.

    NOTE: this is a DIAGNOSTIC metric, not a claim that it reproduces the
    competition's scoring definition. Use it to answer "did the head learn
    these images", nothing more.
    """
    import torch
    c = confusion.shape[0]
    keep = [i for i in range(c) if i != ignore_index]
    cm = confusion[keep][:, keep].double()
    inter = torch.diag(cm)
    union = cm.sum(1) + cm.sum(0) - inter
    present = cm.sum(1) > 0
    iou = torch.where(union > 0, inter / union.clamp_min(1e-12),
                      torch.full_like(inter, float("nan")))
    valid = present & torch.isfinite(iou)
    miou = float(iou[valid].mean()) if bool(valid.any()) else float("nan")
    return miou, [None if not bool(present[i]) else float(iou[i]) for i in range(cm.shape[0])]


def free_gib(path) -> float:
    """Free GiB on the filesystem holding `path`."""
    import shutil as _shutil
    return _shutil.disk_usage(str(path)).free / 2**30


def build_weight_dict(num_layers: int, aux_weight: float) -> dict:
    """Final-layer keys only, plus per-aux-layer keys.

    SetCriterion.total() looks each loss key up in weight_dict by its full name,
    and names aux terms `f"{key}_{i}"`. A bare key `mask` would therefore never
    reach its aux counterpart -- the weights have to be enumerated explicitly.
    """
    weights = {"class": 2.0, "mask": 5.0, "dice": 5.0}
    if aux_weight > 0:
        for index in range(max(num_layers - 1, 0)):
            for key in ("class", "mask", "dice"):
                weights[f"{key}_{index}"] = aux_weight
    return weights


def precision_context(amp: bool):
    import torch
    if amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    import contextlib
    return contextlib.nullcontext()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--stage", required=True,
                    choices=("smoke", "small", "overfit", "ckpt", FORMAL_STAGE))
    ap.add_argument("--crop", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--steps", type=int, default=0, help="0 = stage default")
    ap.add_argument("--num-queries", type=int, default=100)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-encoder", type=float, default=1e-5)
    ap.add_argument("--aux-weight", type=float, default=1.0,
                    help="per-aux-layer set-loss weight; 0 disables aux entirely "
                         "and leaves exactly class/mask/dice from the final layer")
    ap.add_argument("--bce-reduction", default="points",
                    choices=("reference", "points"),
                    help="criterion v2 mask-BCE reduction. 'points' (default, "
                         "production) = ignore-aware weighted point mean "
                         "sum(bce*w)/sum(w). 'reference' = the installed "
                         "implementation's plain point mean, kept ONLY for "
                         "official parity/debug -- it ignores valid weights, so "
                         "it must not be used for real data with Ignore pixels.")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--clip", type=float, default=0.01)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fixed-subset", action="store_true",
                    help="always use the first batch_size images (tiny-overfit); "
                         "without it the batch rotates through the subset")
    ap.add_argument("--save-checkpoint", action="store_true",
                    help="write a model-only score checkpoint at the end")
    ap.add_argument("--keep-checkpoint", action="store_true",
                    help="like --save-checkpoint but keep the file (needed for "
                         "post-hoc inference probes)")
    ap.add_argument("--resume-checkpoint", default=None,
                    help="path of the SINGLE rolling full-resume checkpoint "
                         "(overwritten on each save)")
    ap.add_argument("--resume-every", type=int, default=0,
                    help="save the rolling full-resume checkpoint every N data "
                         "iterations")
    ap.add_argument("--score-checkpoint-steps", type=int, nargs="*", default=[],
                    help="data iterations at which to write a model-only score "
                         "checkpoint (encoder + head + inference config)")
    ap.add_argument("--score-checkpoint-dir", default=None)
    ap.add_argument("--min-free-gib", type=float, default=1.5,
                    help="skip a checkpoint write when free space is already "
                         "below this floor. NOTE: this is a pre-write check, not "
                         "a guarantee that free space stays above the floor "
                         "AFTER the write -- verify the recorded free_gib_after "
                         "against the real file size in checkpoint_log")
    ap.add_argument("--permutation-test-seeds", type=int, nargs="*", default=[],
                    help="spatial-permutation sanity check seeds for the mask "
                         "logits (class logits untouched)")
    ap.add_argument("--eval-metrics", action="store_true",
                    help="measure semantic mIoU / pixel accuracy on the subset "
                         "after the first optimizer update and at the end")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if outdir.exists():
        raise SystemExit(f"outdir exists, refusing: {outdir}")
    outdir.mkdir(parents=True)
    assert_gpu(args.device)

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from b3_m2f_head.head import B3M2FStyleHead
    # criterion v2 (2026-09-15): v1's mask BCE summed over sampled points and
    # divided only by the mask count, giving ~8410 instead of ~0.68 for P=12544
    # and pushing the first autocast step into FP16 overflow. v2 restores the
    # reference point-mean reduction; see loss_v2.py for the evidence trail.
    from b3_m2f_head.loss_v2 import SetCriterionV2, build_set_targets
    from b3_m2f_head.semantic import semantic_scores, predict_masks_official_style

    defaults = {"smoke": 3, "small": 40, "overfit": 300, "ckpt": 2,
                FORMAL_STAGE: FORMAL_STEPS}
    steps = args.steps or defaults[args.stage]
    n_images = {"smoke": 2, "small": 8, "overfit": 4, "ckpt": 2,
                FORMAL_STAGE: None}[args.stage]
    is_formal = args.stage == FORMAL_STAGE

    if is_formal:
        # Max LR overrides only. The schedule itself is the shared loop's term
        # below: min(1, step/warmup) * (1 - step/steps)**1.0, i.e. a 50-iteration
        # linear warmup times the B3 mainline polynomial decay.
        args.lr_head = FORMAL_HEAD_LR
        args.lr_encoder = FORMAL_ENCODER_LR
        args.warmup = FORMAL_WARMUP
        args.batch_size = FORMAL_BATCH_SIZE

    if is_formal and args.fixed_subset:
        raise SystemExit("--fixed-subset is a tiny-overfit mechanism and is "
                         "forbidden in the formal stage")

    splits = Path(BUNDLE) / "splits/train.txt"
    all_stems = [l.strip() for l in splits.read_text().split() if l.strip()]
    if is_formal:
        # Run1 uses the 5597 REAL-label split only. The 500 pseudo-labelled
        # images are deliberately NOT re-added: s62 already carries whatever
        # that signal taught the encoder, and keeping the new data stream pure
        # GT is what makes "did swapping the head help?" attributable.
        stems = all_stems
        if len(stems) != 5597 or len(set(stems)) != len(stems):
            raise RuntimeError(f"formal stage expects 5597 unique stems, "
                               f"got {len(stems)} ({len(set(stems))} unique)")
        print(f"[data] formal stage: {len(stems)} real-GT stems "
              f"(no pseudo-labels) from {splits}", flush=True)
    else:
        stems = all_stems[:n_images]
    print(f"[cfg] stage={args.stage} steps={steps} crop={args.crop} "
          f"batch={args.batch_size} amp={args.amp} "
          f"images={len(stems) if is_formal else stems}", flush=True)

    meta = {"stage": args.stage, "steps": steps, "crop": args.crop,
            "batch_size": args.batch_size, "amp": bool(args.amp),
            "num_queries": args.num_queries, "hidden_dim": args.hidden_dim,
            "layers": args.layers, "lr_head": args.lr_head,
            "lr_encoder": args.lr_encoder, "warmup": args.warmup,
            "clip": args.clip, "seed": args.seed,
            "criterion_module": "b3_m2f_head.loss_v2",
            "criterion_class": "SetCriterionV2",
            "bce_reduction": args.bce_reduction,
            "s62_checkpoint": S62_CKPT, "started": datetime.now(timezone.utc).isoformat(),
            "torch": torch.__version__}
    if is_formal:
        # Everything below is what the run ACTUALLY does, written by the code
        # that does it -- not copied from the launch command.
        meta.update({
            "recipe_frozen": "2026-09-15",
            "n_images": len(stems),
            "train_split": str(splits),
            "uses_pseudo_labels": False,
            "optimizer": "AdamW(weight_decay=0.01)",
            "scheduler": {"name": "linear-warmup x polynomial-decay", "power": 1.0,
                          "total_steps": steps,
                          "note": "lr = base * min(1, step/50) * (1 - step/56000)**1.0; "
                                  "both groups share the same factor; 50-iteration "
                                  "linear warmup, then polynomial decay"},
            "grad_accumulation": 1,
            "augmentation_source": "mainline src/transforms.TrainTransform via "
                                   "build_train_items (Stage62/B3 Exp18 recipe)",
            "augmentation_config": MAINLINE_AUG_CONFIG,
            "rare_crop": MAINLINE_RARE_CROP,
            "cwfa": FORMAL_CWFA,
            "sampling": "per-epoch seeded shuffle of all stems; batch=1 "
                        "sequential traversal of that permutation; "
                        "seed=seed*1000+epoch",
            "ema": False, "swa": False, "kd": False, "focal": False,
            "label_smoothing": False, "new_pseudo_labels": False,
        })
    (outdir / "RUN_META.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    encoder = build_encoder(args.device)
    encoder.train()

    # CWFA, reused verbatim from the mainline module. Upstream attaches it as a
    # forward hook on the SegFormer decode head, i.e. on the point where all four
    # encoder stages are in hand; M2F consumes the same four stages directly, so
    # the hook goes on the encoder and augments the tuple the head would receive.
    # No M2F-specific variant is written.
    cwfa = None
    if is_formal and FORMAL_CWFA["enabled"]:
        sys.path.insert(0, BUNDLE)
        from src.cwfa import ChannelWiseFeatureAugmentation

        cwfa = ChannelWiseFeatureAugmentation(
            probability=FORMAL_CWFA["probability"],
            epsilon=FORMAL_CWFA["epsilon"],
            enabled=True, norm_eps=FORMAL_CWFA["norm_eps"]).to(args.device)

        def _cwfa_hook(module, hook_args, hook_kwargs, output):
            hidden = output.hidden_states
            if not isinstance(hidden, (tuple, list)) or len(hidden) != 4:
                raise RuntimeError(
                    "CWFA expected 4 encoder stages, got "
                    f"{type(hidden).__name__} of length "
                    f"{len(hidden) if isinstance(hidden, (tuple, list)) else 'n/a'}")
            for index, feature in enumerate(hidden):
                if not isinstance(feature, torch.Tensor):
                    raise RuntimeError(f"CWFA stage {index + 1} is not a Tensor")
                if feature.dim() != 4:
                    raise RuntimeError(f"CWFA stage {index + 1} expected BCHW, "
                                       f"got {tuple(feature.shape)}")
            augmented = tuple(cwfa(feature) for feature in hidden)
            if hasattr(output, "hidden_states"):
                output.hidden_states = augmented
                return output
            return (augmented,) + tuple(output[1:])

        cwfa_handle = encoder.register_forward_hook(_cwfa_hook, with_kwargs=True)

        def _set_cwfa_active(active):
            cwfa.set_active(bool(active))

        print(f"[cwfa] mainline ChannelWiseFeatureAugmentation attached "
              f"(p={FORMAL_CWFA['probability']} eps={FORMAL_CWFA['epsilon']} "
              f"start_step={FORMAL_CWFA['start_step']}, "
              f"state_dict entries={len(cwfa.state_dict())})", flush=True)
    else:
        cwfa_handle = None

        def _set_cwfa_active(active):
            return None

    head = B3M2FStyleHead(in_channels=(64, 128, 320, 512), num_classes=NUM_CLASSES,
                          hidden_dim=args.hidden_dim, num_queries=args.num_queries,
                          nhead=8, num_layers=args.layers,
                          dim_feedforward=args.hidden_dim * 4,
                          ignore_index=IGNORE).to(args.device)
    head.train()

    encoder_params = [p for p in encoder.parameters() if p.requires_grad]
    head_params = [p for p in head.parameters() if p.requires_grad]
    named_params = [("encoder." + n, p) for n, p in encoder.named_parameters()]
    named_params += [("head." + n, p) for n, p in head.named_parameters()]
    optimizer = torch.optim.AdamW(
        [{"params": encoder_params, "lr": args.lr_encoder, "name": "encoder"},
         {"params": head_params, "lr": args.lr_head, "name": "head"}],
        weight_decay=0.01)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp))
    # Distinguish data iterations from successful optimizer updates. GradScaler
    # legitimately skips early steps while it backs the scale off, so the two
    # counters must not be conflated. The LR schedule below is defined on DATA
    # ITERATIONS (unchanged); `updates` is recorded alongside it so the drift is
    # visible instead of implicit.
    updates_seen = []
    optimizer.register_step_post_hook(lambda *args, **kw: updates_seen.append(1))

    criterion = SetCriterionV2(num_classes=NUM_CLASSES,
                               weight_dict=build_weight_dict(args.layers, args.aux_weight),
                               bce_reduction=args.bce_reduction,
                               eos_coef=0.1).to(args.device)
    print(f"[criterion] loss_v2.SetCriterionV2 bce_reduction={args.bce_reduction!r}",
          flush=True)
    if args.aux_weight > 0:
        print(f"[loss] aux set-losses weighted {args.aux_weight} per extra layer "
              f"({args.layers - 1} aux layers)", flush=True)
    else:
        print("[loss] aux set-losses disabled (final layer only)", flush=True)

    print(f"[model] encoder={sum(p.numel() for p in encoder_params)/1e6:.1f}M "
          f"head={sum(p.numel() for p in head_params)/1e6:.1f}M", flush=True)

    history, t0 = [], time.time()
    torch.cuda.reset_peak_memory_stats(args.device)
    stopped = False

    eval_stems = stems[:args.batch_size]
    baseline_metrics = None
    metric_count = 0

    def subset_confusion(permute_mask_seed=None):
        """Semantic confusion over the fixed subset, official ids 0..8.

        Predictions come from `semantic_scores` (the mixture), never from an
        argmax over queries. Ground truth is converted to OFFICIAL ids so the
        ignore row (0) is comparable; 0 official == 255 internal.

        `permute_mask_seed` applies ONE shared random permutation to the spatial
        positions of every query mask before synthesis. Class logits are
        untouched and each query mask keeps its exact multiset of values, so the
        only thing destroyed is "which pixel is where". If mIoU survives that,
        the score never depended on mask spatial structure -- a clean causal test
        that is narrower than shuffling encoder features (which would also
        destroy positional/context semantics).
        """
        was_training = (encoder.training, head.training)
        encoder.eval(); head.eval()
        cwfa_was_active = cwfa.active if cwfa is not None else False
        _set_cwfa_active(False)          # evaluation must stay exactly clean
        cm = torch.zeros(9, 9, dtype=torch.float64, device=args.device)
        with torch.no_grad():
            for i in range(0, len(eval_stems), args.batch_size):
                chunk = eval_stems[i:i + args.batch_size]
                px, lb = build_batch(DATA_ROOT, chunk, args.crop, args.device, args.seed)
                # build_batch keeps labels on CPU on purpose (build_set_targets
                # must run on one device). The confusion matrix is built on the
                # GPU, so the label is moved here and only here.
                lb = lb.to(args.device)
                with precision_context(args.amp):
                    o = forward_logits(encoder, px, head)
                mask_logits = o["mask_logits"]
                if permute_mask_seed is not None:
                    flat = mask_logits.flatten(2)                 # (B,Q,hw)
                    gen = torch.Generator(device=flat.device)
                    gen.manual_seed(int(permute_mask_seed))
                    order = torch.randperm(flat.shape[-1], generator=gen,
                                           device=flat.device)
                    mask_logits = flat[..., order].view_as(mask_logits)
                # mask logits live at stride-4 (1024 -> 256). The mixture has to be
                # upsampled to the label grid before it can be compared with GT.
                scores = semantic_scores(o["class_logits"].float(),
                                         mask_logits.float(),
                                         upsample_to=lb.shape[-2:])
                pred = scores.argmax(1) + 1                                # 1..C
                gt = torch.where(lb == IGNORE, torch.zeros_like(lb), lb + 1)
                idx = gt.reshape(-1).long() * 9 + pred.reshape(-1).long()
                cm += torch.bincount(idx, minlength=81).reshape(9, 9).double()
                del o, px, lb, pred, gt, scores, mask_logits
        encoder.train(*[was_training[0]]); head.train(*[was_training[1]])
        _set_cwfa_active(cwfa_was_active)
        return cm

    score_dir = Path(args.score_checkpoint_dir) if args.score_checkpoint_dir else outdir
    score_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_log = []
    score_paths = []

    def save_full_resume(tag):
        """ONE rolling full-resume checkpoint: model + head + optimizer + AMP
        scaler state + both counters + config. Overwritten every time, so a long
        run never accumulates these."""
        if not args.resume_checkpoint:
            return
        target = Path(args.resume_checkpoint)
        target.parent.mkdir(parents=True, exist_ok=True)
        free = free_gib(target.parent)
        if free < args.min_free_gib:
            print(f"  [ckpt] SKIP full-resume:{tag}: only {free:.2f} GiB free "
                  f"(floor {args.min_free_gib} GiB)", flush=True)
            checkpoint_log.append({"kind": "full_resume", "tag": tag,
                                   "skipped": "disk_floor", "free_gib": free})
            return
        torch.save({"model": encoder.state_dict(), "head": head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "data_iteration": step, "optimizer_update": len(updates_seen),
                    "config": meta}, target)
        checkpoint_log.append({"kind": "full_resume", "tag": tag, "path": str(target),
                               "bytes": target.stat().st_size,
                               "free_gib_after": free_gib(target.parent)})

    def save_score(tag):
        """Model-only checkpoint for inference: encoder + head + config, no
        optimizer state, so it is much smaller. This is what validation/test1
        inference loads."""
        target = score_dir / f"score_{tag}.pt"
        free = free_gib(score_dir)
        if free < args.min_free_gib:
            print(f"  [ckpt] SKIP score:{tag}: only {free:.2f} GiB free "
                  f"(floor {args.min_free_gib} GiB)", flush=True)
            checkpoint_log.append({"kind": "score", "tag": tag,
                                   "skipped": "disk_floor", "free_gib": free})
            return None
        torch.save({"model": encoder.state_dict(), "head": head.state_dict(),
                    "data_iteration": step, "config": meta}, target)
        entry = {"kind": "score", "tag": tag, "path": str(target),
                 "bytes": target.stat().st_size,
                 "free_gib_after": free_gib(score_dir)}
        checkpoint_log.append(entry)
        score_paths.append(str(target))
        print(f"  [ckpt] score:{tag} -> {target} "
              f"({entry['bytes']/2**20:.1f} MiB, {entry['free_gib_after']:.2f} GiB free)",
              flush=True)
        return target

    epoch_perm = None
    # Iterations actually executed, tracked from the loop itself. Deriving this
    # from `history[-1]["data_iteration"]` is WRONG: history only gets a record
    # on logged iterations (every 10th, plus step 1 and any skipped step), so a
    # stop signal landing between two log points leaves the tail uncounted and
    # reports fewer iterations than updates -- which is impossible. Observed on
    # the aborted Run1: it reported 1670 iterations but 1672 updates.
    executed = 0
    for step in range(1, steps + 1):
        if _STOP["flag"]:
            stopped = True
            break
        executed = step

        if is_formal:
            # LR factor = min(1, step/warmup) * (1 - step/steps)**1.0
            #            = 50-iteration linear warmup x mainline polynomial decay.
            # Both param groups get the SAME factor; only the base LRs differ.
            # AdamW applies this outside m_hat/sqrt(v_hat), so the warmup is a
            # real part of the configuration, not a cosmetic ramp.
            scale = min(1.0, step / max(args.warmup, 1))
            poly = max(0.0, 1.0 - step / steps) ** 1.0
            for group, base in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base * scale * poly
        else:
            scale = min(1.0, step / max(args.warmup, 1))
            for group, base in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base * scale

        if is_formal:
            # One permutation per epoch over all 5597 stems, batch=1 sequential
            # traversal, reshuffled at each epoch boundary with a recorded seed.
            epoch = (step - 1) // len(stems)
            if epoch_perm is None or epoch != epoch_perm["epoch"]:
                order = list(range(len(stems)))
                random.Random(args.seed * 1000 + epoch).shuffle(order)
                epoch_perm = {"epoch": epoch, "order": order}
                print(f"  [epoch {epoch}] reshuffled {len(order)} stems "
                      f"(seed={args.seed * 1000 + epoch})", flush=True)
            position = (step - 1) % len(stems)
            batch_stems = [stems[epoch_perm["order"][position]]]
        else:
            batch_stems = [stems[(step + i) % len(stems)]
                           for i in range(args.batch_size)]

        if args.fixed_subset:
            # tiny-overfit uses ONE fixed batch, cycled: the same pixels every
            # step, so the only thing that can move the loss is the parameters.
            # Sample rotation would otherwise confound "did it fit" with "did the
            # batch change", which is exactly what this stage is meant to isolate.
            # Forbidden in the formal stage (checked above).
            batch_stems = stems[:args.batch_size]

        if is_formal:
            item = build_train_items(batch_stems, args.seed * 100000 + step)[0]
            pixel = item["pixel_values"].to(args.device)
            label = item["labels"]            # label stays on CPU, as build_batch does
        else:
            pixel, label = build_batch(DATA_ROOT, batch_stems, args.crop,
                                       args.device, args.seed)

        # CWFA: OFF through step 28000, ON from 28001 (mainline gate).
        cwfa_active = bool(is_formal and cwfa is not None
                           and step >= FORMAL_CWFA["start_step"])
        _set_cwfa_active(cwfa_active)
        if is_formal and cwfa is not None and step == FORMAL_CWFA["start_step"]:
            print(f"  [cwfa] activated at step {step} "
                  f"(p={FORMAL_CWFA['probability']})", flush=True)

        optimizer.zero_grad(set_to_none=True)

        # AMP wraps the MODEL FORWARD ONLY. The set criterion must run in FP32:
        # sigmoid_ce_sum/dice_sum accumulate over ~12k sampled points per image,
        # and in fp16 those reductions overflow to inf, which then poisons every
        # gradient (observed: grad norm = nan at step 1). This mirrors the
        # reference recipe, where autocast is not applied to the criterion.
        with precision_context(args.amp):
            out = forward_logits(encoder, pixel, head)

        class_logits = out["class_logits"].float()
        mask_logits = out["mask_logits"].float()
        aux = tuple({"class_logits": a["class_logits"].float(),
                     "mask_logits": a["mask_logits"].float()}
                    for a in out["aux_outputs"])

        targets, points, weight, valid_map = build_set_targets(
            label, NUM_CLASSES, ignore_index=IGNORE)
        targets, points, weight, valid_map = to_device_set_targets(
            targets, points, weight, valid_map, args.device)

        # Passing aux_outputs=() drops the aux terms entirely -> the loss has
        # exactly three components (class / mask / dice) from the final layer,
        # so nothing is unaccounted for in attribution.
        losses = criterion(class_logits, mask_logits, targets, points, weight,
                           aux if args.aux_weight > 0 else (), valid_map)
        total = criterion.total(losses)

        finite_or_die("loss", losses, step)
        if not bool(torch.isfinite(total)):
            raise RuntimeError(f"[step {step}] non-finite total loss {total}")

        scaler.scale(total).backward()
        scaler.unscale_(optimizer)

        # GradScaler's contract: an oversized initial scale makes the first
        # steps overflow on purpose, and the scaler recovers by skipping them and
        # halving the scale. Treating that as fatal aborts training before the
        # backoff can happen (measured: 65536 -> 32768 -> 16384 -> 8192, clean
        # from step 4). So a non-finite gradient here is recorded and handed to
        # `scaler.step`, which performs the skip itself -- we do NOT clip, do not
        # clean the gradients, and never force an update.
        nonfinite_params = [
            name for name, p in named_params
            if p.grad is not None and not bool(torch.isfinite(p.grad).all())]

        scale_before = scaler.get_scale()
        updates_before = len(updates_seen)
        grad_norm = None
        if not nonfinite_params:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                encoder_params + head_params, args.clip)
            if not bool(torch.isfinite(grad_norm)):
                raise RuntimeError(
                    f"[step {step}] gradient norm non-finite even though every "
                    f"parameter gradient was finite: {grad_norm}")

        scaler.step(optimizer)
        scaler.update()
        update_ran = len(updates_seen) > updates_before

        if step % 10 == 0 or step == 1 or not update_ran or steps <= 20:
            with torch.no_grad(), precision_context(args.amp):
                sem = semantic_scores(out["class_logits"], out["mask_logits"])
                pred = predict_masks_official_style(out["class_logits"],
                                                    out["mask_logits"])
            finite_or_die("semantic", {"scores": sem}, step)
            classes = torch.unique(pred).tolist()

            # Loss going down is not evidence the model got better; the same
            # subset has to be scored semantically. Measured right after the
            # FIRST optimizer update and again at the end.
            metrics = None
            if args.eval_metrics and update_ran and (
                    baseline_metrics is None or step == steps):
                cm = subset_confusion()
                miou, per_class = mean_iou_from_counts(cm)
                acc = float(cm.diag().sum() / cm.sum().clamp_min(1))
                metrics = {"miou": miou, "pixel_accuracy": acc,
                           "per_class_iou": per_class,
                           "confusion": cm.long().tolist()}
                metric_count += 1
                if baseline_metrics is None:
                    baseline_metrics = metrics
                    print(f"  [baseline] at first optimizer update: "
                          f"mIoU={miou:.4f} pixacc={acc:.4f} classes={classes}",
                          flush=True)
                else:
                    print(f"  [final]    mIoU={miou:.4f} pixacc={acc:.4f} "
                          f"(baseline mIoU={baseline_metrics['miou']:.4f}, "
                          f"delta={miou - baseline_metrics['miou']:+.4f})",
                          flush=True)
                del cm

            peak = torch.cuda.max_memory_allocated(args.device) / 2**20
            rec = {"data_iteration": step,
                   "optimizer_update": len(updates_seen),
                   "optimizer_updated_this_step": update_ran,
                   "lr_scale": scale,
                   "lr_encoder": optimizer.param_groups[0]["lr"],
                   "lr_head": optimizer.param_groups[1]["lr"],
                   "scale_before": scale_before,
                   "scale_after": scaler.get_scale(),
                   "nonfinite_param_grads": len(nonfinite_params),
                   "total": float(total),
                   # grad_norm is None on a skipped step: we deliberately do not
                   # clip, and the scaler is the one deciding to skip.
                   "grad_norm": None if grad_norm is None else float(grad_norm),
                   "class": float(losses["class"]),
                   "mask": float(losses["mask"]),
                   "dice": float(losses["dice"]),
                   "metrics": metrics,
                   "pred_classes": classes, "peak_MiB": round(peak, 1),
                   "sec": round(time.time() - t0, 1)}
            history.append(rec)
            tag = "UPDATE" if update_ran else "SKIPPED"
            grad_txt = "n/a" if grad_norm is None else f"{float(grad_norm):.3f}"
            print(f"  [iter {step:>4}/{steps}] {tag} upd={len(updates_seen)} "
                  f"total={float(total):.4f} "
                  f"cls={float(losses['class']):.3f} "
                  f"mask={float(losses['mask']):.3f} "
                  f"dice={float(losses['dice']):.3f} "
                  f"grad={grad_txt} nonfinite={len(nonfinite_params)} "
                  f"scale={scale_before:.0f}->{scaler.get_scale():.0f} "
                  f"peak={peak:.0f}MiB classes={classes} t={time.time()-t0:.0f}s",
                  flush=True)

        # Checkpoint policy: one rolling full-resume file (overwritten) plus a
        # small number of model-only score checkpoints. The disk floor is
        # enforced inside the savers, so a full disk stops checkpointing rather
        # than killing the run or evicting historical evidence.
        if args.resume_every and step % args.resume_every == 0:
            save_full_resume(f"iter{step}")
        if step in args.score_checkpoint_steps:
            # cap at 3: oldest model-only checkpoint is dropped first
            if len(score_paths) >= 3:
                oldest = Path(score_paths.pop(0))
                try:
                    oldest.unlink()
                    checkpoint_log.append({"kind": "score_evicted",
                                           "path": str(oldest)})
                except OSError:
                    pass
            save_score(f"iter{step}")

        del out, losses, total, pixel, label
        if step % 20 == 0:
            torch.cuda.empty_cache()

    peak_alloc = torch.cuda.max_memory_allocated(args.device) / 2**20
    peak_resv = torch.cuda.max_memory_reserved(args.device) / 2**20

    # If the final data iteration happened to be skipped by the scaler, the
    # in-loop "step == steps" trigger never fires. Measure the end state anyway
    # so the before/after comparison always exists.
    final_metrics = None
    if args.eval_metrics and baseline_metrics is not None and history:
        if metric_count < 2:
            cm = subset_confusion()
            miou, per_class = mean_iou_from_counts(cm)
            final_metrics = {"miou": miou,
                             "pixel_accuracy": float(cm.diag().sum() / cm.sum().clamp_min(1)),
                             "per_class_iou": per_class,
                             "confusion": cm.long().tolist()}
            print(f"  [final]    (post-loop) mIoU={miou:.4f} "
                  f"(baseline {baseline_metrics['miou']:.4f}, "
                  f"delta {miou - baseline_metrics['miou']:+.4f})", flush=True)
            del cm
        else:
            final_metrics = [r["metrics"] for r in history if r["metrics"]][-1]

    # Checkpoint size probe. The `ckpt` stage always measures; other stages do it
    # on request. The file is written and immediately deleted, but the measured
    # size is what the disk budget for a real run has to be derived from.
    ckpt_bytes = None
    ckpt_path = None
    if (args.stage == "ckpt" or args.save_checkpoint) and not args.score_checkpoint_steps:
        probe = outdir / ("final_checkpoint.pt" if args.keep_checkpoint
                          else "_ckpt_probe.pt")
        torch.save({"model": encoder.state_dict(), "head": head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "step": step,
                    "config": meta}, probe)
        ckpt_bytes = probe.stat().st_size
        if args.keep_checkpoint:
            ckpt_path = str(probe)
        else:
            probe.unlink()

    # Spatial-structure causal test: keep class logits and every query mask's
    # value distribution, destroy only WHERE each mask value sits. If mIoU holds
    # up under this, the score never came from mask spatial structure.
    permutation_test = None
    if args.permutation_test_seeds and args.eval_metrics and eval_stems:
        permutation_test = []
        for seed in args.permutation_test_seeds:
            cm = subset_confusion(permute_mask_seed=seed)
            miou, _ = mean_iou_from_counts(cm)
            # cm lives on CUDA; do the trace on the tensor, not through numpy.
            acc = float((torch.diagonal(cm).sum() / cm.sum()).item())
            permutation_test.append({"seed": int(seed), "miou": miou,
                                     "pixel_accuracy": acc})
            print(f"  [permute seed {seed}] mIoU={miou:.4f} pixacc={acc:.4f}",
                  flush=True)
            del cm
        if baseline_metrics is not None:
            drop = baseline_metrics["miou"] - max(t["miou"] for t in permutation_test)
            print(f"  [permute] best permuted mIoU {max(t['miou'] for t in permutation_test):.4f} "
                  f"vs final {final_metrics['miou'] if final_metrics else float('nan'):.4f} "
                  f"(drop {drop:+.4f})", flush=True)

    summary = {
        "stage": args.stage,
        "fixed_subset": bool(args.fixed_subset),
        # Report the ACTUAL cardinality. `eval_stems = stems[:batch_size]`, so
        # with batch_size=1 this is a ONE-image overfit regardless of how many
        # images were loaded from the split.
        "batch_size": args.batch_size,
        "evaluation_images": len(eval_stems),
        "train_stems_loaded": len(stems),
        "subset_note": ("one-image tiny-overfit" if len(eval_stems) == 1
                        else f"{len(eval_stems)}-image tiny-overfit"),
        "permutation_test": permutation_test,
        "log_records": len(history),
        "data_iterations": executed,
        "last_logged_iteration": history[-1]["data_iteration"] if history else 0,
        "optimizer_updates": len(updates_seen),
        "skipped_steps": executed - len(updates_seen),
        "baseline_metrics": baseline_metrics,
        "final_metrics": final_metrics,
        "lr_schedule_basis": ("polynomial decay on data_iteration, base lr "
                              "encoder/head = 1e-5/1e-4, warmup=0 (formal)"
                              if is_formal else
                              "linear warmup on data_iteration (unchanged; see RUN_META)"),
        "unique_train_stems": len(set(stems)) if is_formal else len(stems),
        "epochs_completed": ((executed // len(stems))
                             if (is_formal and history) else None),
        "final_lr": ({"encoder": optimizer.param_groups[0]["lr"],
                      "head": optimizer.param_groups[1]["lr"]} if is_formal else None),
        "cwfa": (FORMAL_CWFA if is_formal else None),
        "cwfa_state_dict_entries": (len(cwfa.state_dict())
                                    if cwfa is not None else None),
        "stopped_early": stopped,
        "peak_allocated_MiB": round(peak_alloc, 1),
        "peak_reserved_MiB": round(peak_resv, 1),
        "total_sec": round(time.time() - t0, 1),
        # Divide by iterations actually executed, not by the number of log
        # records: with `steps <= 20` logging every step but longer runs logging
        # sparsely, the record count makes the per-step time look ~9x too slow.
        "sec_per_iteration": round((time.time() - t0) / max(executed, 1), 3),
        "final_total": history[-1]["total"] if history else None,
        "first_total": history[0]["total"] if history else None,
        "ckpt_bytes": ckpt_bytes,
        "ckpt_path": ckpt_path,
        "checkpoint_policy": {
            "full_resume": args.resume_checkpoint,
            "resume_every": args.resume_every,
            "score_steps": list(args.score_checkpoint_steps),
            "score_max_kept": 3,
            "min_free_gib": args.min_free_gib,
        },
        "checkpoint_log": checkpoint_log,
        "history": history,
        "gpu": torch.cuda.get_device_name(args.device),
    }
    (outdir / "RUN_RESULT.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {summary['data_iterations']} data iterations, "
          f"{summary['optimizer_updates']} optimizer updates "
          f"({summary['skipped_steps']} skipped by scaler), "
          f"peak_alloc={peak_alloc:.0f}MiB peak_resv={peak_resv:.0f}MiB, "
          f"{summary['sec_per_iteration']}s/iteration"
          + (f", ckpt={ckpt_bytes/2**20:.0f}MiB" if ckpt_bytes else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
