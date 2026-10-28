#!/usr/bin/env python3
"""ATDA-71 export: alpha-compose an ATDA checkpoint onto theta237, then emit the
Bstar submission package under the champion's exact inference protocol.

Only the deployment operator (Bstar) is produced. The ZERO/C/13 reference set is
deliberately NOT generated -- the question tonight is only whether the ATDA
direction beats 0.702574 on the platform.

Protocol, byte-identical to _eval_soup_family.py (the script that produced the
submitted 237_S_semeda_M.zip):
    src.models.build_model with cwfa disabled, strict state load,
    tta_anatomy_v2.forward_views, autocast off, FP32 mean over the 8 D4 views,
    then + Bstar on the class axis, then argmax + 1.
Class axis: index i = train class i = official class i+1.

Physical GPU2 only. Writes only under --outdir.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

NUM_CLASSES = 8
PHYSICAL_GPU_INDEX = 2
SNAP = Path("/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1")
TEST = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/images")
CONFIG = SNAP / "stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml"
# --- theta261 (current champion, package 261, platform 0.702836) ---
CHAMPION = Path("/home/jhy/rare_diag_20260915/phase0_candidates_20260918/"
                "A_cleanSEM_M_k100_th209.pt")
CHAMPION_SHA256 = "9ffdd0beb8480a1efd361503d637d3d53cf716c808537c551fefae09f7fe2786"
CHAMPION_SCORE = 0.702836
BSTAR = [-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5]


def sha(p: Path) -> str:
    digest = hashlib.sha256()
    with Path(p).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_state(path: Path):
    import torch
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            candidate = checkpoint.get(key)
            if (isinstance(candidate, dict) and candidate
                    and all(isinstance(v, torch.Tensor) for v in candidate.values())):
                return candidate, checkpoint
        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint, {}
    raise SystemExit(f"no tensor state dict in {path}")


TRAINABLE_PREFIXES = ("segformer.stages.2.", "segformer.stages.3.")


def in_scope(key: str) -> bool:
    name = key[:-7] if key.endswith(".weight") or key.endswith(".bias") else key
    return any(name.startswith(p) for p in TRAINABLE_PREFIXES)


def compose(base: Path, student: Path, alpha: float, out: Path) -> dict:
    """theta237 + alpha * (theta_A - theta237), on every floating parameter."""
    import torch
    base_state, _ = load_state(base)
    student_state, _ = load_state(student)
    if set(base_state) != set(student_state):
        missing = set(base_state) ^ set(student_state)
        raise SystemExit(f"key mismatch between base and student: {sorted(missing)[:5]}")
    merged = {}
    worst_in, worst_out = 0.0, 0.0
    with torch.no_grad():
        for key, value in base_state.items():
            other = student_state[key]
            if value.is_floating_point():
                if alpha == 1.0:
                    # "direct" variant: the student checkpoint verbatim, no
                    # arithmetic (base + 1.0*(s-base) would round to 1 ulp off).
                    merged[key] = other.clone()
                    if in_scope(key):
                        worst_in = max(worst_in, float((other - value).float().abs().max()))
                    else:
                        worst_out = max(worst_out, float((other - value).float().abs().max()))
                    continue
                delta = other.float() - value.float()
                magnitude = float(delta.abs().max())
                if in_scope(key):
                    worst_in = max(worst_in, magnitude)
                else:
                    worst_out = max(worst_out, magnitude)
                merged[key] = (value.float() + alpha * delta).to(value.dtype)
            else:
                if not torch.equal(value, other):
                    raise SystemExit(f"non-floating tensor {key} differs; "
                                     "the trainer was supposed to leave it alone")
                merged[key] = value.clone()
    if worst_out != 0.0:
        raise SystemExit(f"frozen scope moved by {worst_out} between base and student; "
                         "refusing to compose")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": merged, "step": 0,
                "atda_export": {"base": str(base), "base_sha256": sha(base),
                                "student": str(student), "student_sha256": sha(student),
                                "alpha": alpha,
                                "max_abs_delta_in_scope": worst_in,
                                "max_abs_delta_out_of_scope": worst_out}},
               out)
    return {"path": str(out), "sha256": sha(out), "alpha": alpha,
            "max_abs_delta_in_scope": worst_in,
            "max_abs_delta_out_of_scope": worst_out,
            "frozen_scope_untouched": worst_out == 0.0}


def forward_all(model, names, device):
    import torch
    from PIL import Image

    from src.transforms import normalize_to_tensor
    from tta_anatomy_v2 import forward_views

    scores = {}
    with torch.inference_mode():
        for i, name in enumerate(names, 1):
            with Image.open(TEST / name) as im:
                img = np.asarray(im.convert("RGB")).copy()
            if img.shape != (1024, 1024, 3):
                raise SystemExit(f"bad shape {name}: {img.shape}")
            px = normalize_to_tensor(img).unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", enabled=False):
                sc = forward_views(model, px, 1).mean(dim=0, dtype=torch.float32)
            if not torch.isfinite(sc).all():
                raise SystemExit(f"nonfinite: {name}")
            if sc.shape != (NUM_CLASSES, 1024, 1024):
                raise SystemExit(f"bad score shape {sc.shape}")
            scores[name] = sc.cpu().numpy().astype(np.float32)
            del px, sc
            if i % 100 == 0:
                print(f"    {i}/500", flush=True)
    return scores


def pack(scores, names, outdir: Path, tag: str):
    from PIL import Image
    zp = outdir / f"{tag}_Bstar.zip"
    vec = np.asarray(BSTAR, dtype=np.float32)[:, None, None]
    digest = hashlib.sha256()
    with zipfile.ZipFile(zp, "x", compression=zipfile.ZIP_STORED) as zf:
        for name in names:
            pred = (scores[name] + vec).argmax(0).astype(np.uint8) + 1
            buf = io.BytesIO()
            Image.fromarray(np.ascontiguousarray(pred), mode="L").save(buf, format="PNG")
            data = buf.getvalue()
            fn = f"{Path(name).stem}.png"
            zf.writestr(fn, data)
            digest.update(fn.encode() + b"\0" + data)
    return {"zip_path": str(zp), "zip_sha256": sha(zp),
            "entries": len(names), "zip_content_sha256": digest.hexdigest()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--base", default=str(CHAMPION))
    ap.add_argument("--student", help="ATDA checkpoint to compose")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--tag", help="package tag; default derived from alpha")
    ap.add_argument("--eval-only", help="evaluate this .pt directly, no composition")
    ap.add_argument("--verify-champion", action="store_true",
                    help="also evaluate theta237 itself and diff against its package")
    ap.add_argument("--champion-share-json",
                    default="/home/jhy/rare_diag_20260915/champ237_test1_cache_20260918/FORWARD_META.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    _phys = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if _phys not in ("2", "3"):
        raise SystemExit(f"require CUDA_VISIBLE_DEVICES in (2,3), got {_phys!r}")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise SystemExit("require CUDA_DEVICE_ORDER=PCI_BUS_ID")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for p in ("/home/jhy/gpu2_day_queue", str(SNAP)):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch
    from tta_anatomy_v2 import build_model

    names = sorted((p.name for p in TEST.glob("*.png")),
                   key=lambda n: int(Path(n).stem.split("_")[-1]))
    if names != [f"test1_{i}.png" for i in range(1, 501)]:
        raise SystemExit("test1 names must be exactly test1_1..test1_500")

    if sha(Path(args.base)) != CHAMPION_SHA256:
        raise SystemExit("base is not the verified champion")

    torch.manual_seed(42)
    torch.set_num_threads(4)
    results = {}

    targets = []
    if args.verify_champion:
        targets.append(("theta237_ctrl", Path(args.base)))
    target_path = Path(args.eval_only) if args.eval_only else None
    if args.student:
        alpha = float(args.alpha)
        tag = args.tag or f"atda_a{alpha:g}".replace(".", "p")
        composed = outdir / f"composed_{tag}.pt"
        info = compose(Path(args.base), Path(args.student), alpha, composed)
        print(f"composed alpha={alpha} -> {composed} sha={info['sha256'][:16]}", flush=True)
        results["compose_" + tag] = info
        target_path = composed
        targets.append((tag, composed))
    elif target_path is not None:
        tag = args.tag or target_path.stem
        targets.append((tag, target_path))
    if not targets:
        raise SystemExit("nothing to do: pass --student or --eval-only")

    for tag, ckpt in targets:
        ck_sha = sha(ckpt)
        print(f"START {tag} sha={ck_sha[:16]}", flush=True)
        t0 = time.time()
        model = build_model(CONFIG, ckpt, args.device)
        scores = forward_all(model, names, args.device)
        del model
        torch.cuda.empty_cache()
        if sha(ckpt) != ck_sha:
            raise SystemExit(f"checkpoint changed during inference: {ckpt}")
        packed = pack(scores, names, outdir, tag)
        vec = np.asarray(BSTAR, dtype=np.float32)[:, None, None]
        hist = np.zeros(NUM_CLASSES, dtype=np.int64)
        for name in names:
            hist += np.bincount((scores[name] + vec).argmax(0).ravel(), minlength=NUM_CLASSES)
        share = 100.0 * hist / hist.sum()
        results[tag] = {"checkpoint": str(ckpt), "checkpoint_sha256": ck_sha,
                        "packages": packed, "elapsed_sec": round(time.time() - t0, 1),
                        "bstar_class_share_pct": [round(float(x), 4) for x in share]}
        print(f"DONE {tag} in {time.time()-t0:.0f}s -> {packed['zip_path']}", flush=True)

    if args.verify_champion:
        meta = json.loads(Path(args.champion_share_json).read_text(encoding="utf-8"))
        want = meta["bstar_class_share_pct"]
        got = results["theta237_ctrl"]["bstar_class_share_pct"]
        classes = list(want)
        worst = max(abs(got[i] - float(want[c])) for i, c in enumerate(classes))
        results["champion_reproduction"] = {
            "expected_pct": want,
            "got_pct": dict(zip(classes, got)),
            "max_abs_share_diff_pp": round(worst, 4),
            "pass": worst <= 0.01,
        }
        print(f"champion reproduction max class-share diff = {worst:.4f}pp", flush=True)
        if worst > 0.01:
            raise SystemExit("champion reproduction FAILED; the packer is not trustworthy")

    (outdir / "EXPORT_META.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(),
        "scope": "ATDA71_EXPORT_BSTAR_ONLY",
        "base": str(args.base), "base_sha256": CHAMPION_SHA256,
        "base_platform_score": CHAMPION_SCORE,
        "operator": "Bstar", "bstar": BSTAR,
        "protocol": "D4 TTA8, FP32, autocast off, FP32 view mean, cwfa disabled, strict load",
        "class_axis": "index i = train class i = official class i+1",
        "variants_generated": "Bstar only",
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("export meta written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
