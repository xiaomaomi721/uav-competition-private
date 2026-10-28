#!/usr/bin/env python3
"""M2F LR-reopen arms: one D4-TTA8 per tail checkpoint, then pack from cache.

Same validated inference contract as the Run1b trajectory work (M2F native
semantic mixture, autocast off, FP32 mean over 8 D4 views). One forward per
checkpoint, all bias variants derived on CPU from the cached score, so no bias
can influence another.

Variants: ZERO and FG020 (uniform foreground +0.20) only. C/#13 are excluded on
purpose: measured on the M2F score space they flip ~90% of all pixels, because
the CHAMPION vector was tuned for Stage23's raw-logit range. They are not
meaningful calibration operators here.

Also reports flip% per variant, which is the local geometry readout: on the
Run1b trajectory the FG020 flip fell 1.3749 -> 1.1646 -> 1.1409% as training
progressed, and the question for these arms is whether reopening the LR moves it.

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
from types import SimpleNamespace

import numpy as np

NUM_CLASSES = 8
IGNORE = 255
PHYSICAL_GPU_INDEX = 2
ROOT = Path("/home/jhy/rare_diag_20260915/m2f_gpu_20260915")
BUNDLE = Path("/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1")
TEST = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/images")
MODEL_DIR = "/home/jhy/models/segformer-b3-ade512"
ZERO = [0.0] * NUM_CLASSES
FG020 = [0.0] + [0.20] * 7
VARIANTS = [("ZERO", ZERO), ("FG020", FG020)]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pack_variants(cdir: Path, names, outdir: Path, tag: str):
    """Pack ZERO/FG020 (+flip geometry) from a scores dir. Pure CPU."""
    import zipfile as _zip
    from PIL import Image as _Image
    handles = {}
    for vname, vec in VARIANTS:
        zp = outdir / f"{tag}_{vname}.zip"
        handles[vname] = [vec, zp,
                          _zip.ZipFile(zp, "x", compression=_zip.ZIP_STORED),
                          hashlib.sha256()]
    total_px = 0
    flips = {n: 0 for n, _ in VARIANTS}
    bg2fg = {n: 0 for n, _ in VARIANTS}
    for name in names:
        arr = np.load(cdir / f"{Path(name).stem}.npy")
        base = arr.argmax(0)
        total_px += base.size
        for vname, (vec, _, zf, digest) in handles.items():
            # ZERO is the plain argmax; every other variant adds its bias to the
            # class axis first. (Handing the 3-D score array to Image.fromarray
            # for ZERO is a TypeError -- that cost one TTA8 re-run on 09-17.)
            pred = base if vname == "ZERO" else (
                arr + np.asarray(vec, dtype=np.float32)[:, None, None]).argmax(0)
            if vname != "ZERO":
                moved = base != pred
                flips[vname] += int(moved.sum())
                bg2fg[vname] += int(((base == 0) & moved).sum())
            png = np.ascontiguousarray(pred.astype(np.uint8) + 1)
            payload = io.BytesIO()
            _Image.fromarray(png, mode="L").save(payload, format="PNG")
            data = payload.getvalue()
            fn = f"{Path(name).stem}.png"
            zf.writestr(fn, data)
            digest.update(fn.encode() + b"\0" + data)

    pack = {}
    for vname, (vec, zp, zf, digest) in handles.items():
        zf.close()
        pack[vname] = {"vector": vec, "zip_path": str(zp),
                       "zip_sha256": sha(zp),
                       "png_content_sha256": digest.hexdigest(),
                       "entries": len(names)}
        if vname != "ZERO":
            f = flips[vname]
            pack[vname]["flip_pct"] = 100.0 * f / total_px
            pack[vname]["bg_to_fg_pct_of_flips"] = 100.0 * bg2fg[vname] / f if f else 0.0
            print(f"  [{tag} {vname:5s}] flip={pack[vname]['flip_pct']:.4f}%  "
                  f"BG->FG {pack[vname]['bg_to_fg_pct_of_flips']:.1f}%", flush=True)
    return pack


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--checkpoint", action="append", default=[],
                    help="repeatable; each is a path to a score checkpoint")
    ap.add_argument("--pack-only", default=None,
                    help="skip inference and re-pack from an existing scores_* dir "
                         "(recovery path: the forward is expensive, packing is not)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(PHYSICAL_GPU_INDEX):
        raise SystemExit("require CUDA_VISIBLE_DEVICES=2")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise SystemExit("require CUDA_DEVICE_ORDER=PCI_BUS_ID")

    outdir = Path(args.outdir)
    if outdir.exists() and not args.pack_only:
        raise SystemExit(f"outdir exists: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    for p in (str(ROOT / "py"), str(BUNDLE), "/home/jhy/gpu2_day_queue"):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch
    from PIL import Image
    from transformers import AutoModelForSemanticSegmentation
    from b3_m2f_head.head import B3M2FStyleHead
    from b3_m2f_head.semantic import semantic_scores
    from src.transforms import normalize_to_tensor
    from tta_anatomy_v2 import forward_views

    class M2FScores(torch.nn.Module):
        def __init__(self, ck):
            super().__init__()
            self.encoder = AutoModelForSemanticSegmentation.from_pretrained(
                MODEL_DIR, num_labels=8, ignore_mismatched_sizes=True,
                local_files_only=True).segformer
            self.head = B3M2FStyleHead(in_channels=(64, 128, 320, 512), num_classes=8,
                                       hidden_dim=256, num_queries=100, nhead=8,
                                       num_layers=6, dim_feedforward=1024,
                                       ignore_index=IGNORE)
            self.encoder.load_state_dict(ck["model"], strict=True)
            self.head.load_state_dict(ck["head"], strict=True)

        def forward(self, pixel_values):
            feats = self.encoder(pixel_values=pixel_values,
                                 output_hidden_states=True).hidden_states
            o = self.head(feats)
            return SimpleNamespace(logits=semantic_scores(o["class_logits"],
                                                          o["mask_logits"]))

    names = sorted((p.name for p in TEST.glob("*.png")),
                   key=lambda n: int(Path(n).stem.split("_")[-1]))
    if len(names) != 500:
        raise SystemExit(f"expected 500 test images, got {len(names)}")

    torch.manual_seed(42)
    torch.set_num_threads(4)
    summary = {}

    # ---- pack-only recovery: reuse existing scores dirs, never re-run TTA8.
    # NOTE: this must NOT return -- it falls through so that any --checkpoint
    # given on the same command line still gets its forward run.
    if args.pack_only:
        pod = Path(args.pack_only)
        for cdir in sorted(pod.glob("scores_*")):
            tag = cdir.name[len("scores_"):]
            if (outdir / f"{tag}_ZERO.zip").is_file() and \
               (outdir / f"{tag}_ZERO.zip").stat().st_size > 1000:
                print(f"SKIP {tag}: packages already exist", flush=True)
                continue
            arr0 = np.load(cdir / f"{names[0][:-4]}.npy")
            if arr0.shape != (NUM_CLASSES, 1024, 1024):
                raise SystemExit(f"{cdir}: unexpected score shape {arr0.shape}")
            t2 = time.time()
            print(f"PACK {tag} from {cdir}", flush=True)
            pack = pack_variants(cdir, names, outdir, tag)
            summary[tag] = {"checkpoint": None, "checkpoint_sha256": None,
                            "data_iteration": None, "packages": pack,
                            "elapsed_sec": round(time.time() - t2, 1),
                            "source": str(cdir)}
            print(f"PACKED {tag}", flush=True)

    for ck_path in args.checkpoint:
        ckpt = Path(ck_path)
        if not ckpt.is_file():
            raise SystemExit(f"missing checkpoint: {ckpt}")
        ck_sha = sha(ckpt)
        tag = f"iter{ckpt.stem.split('iter')[-1]}" if "iter" in ckpt.stem else ckpt.stem
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        it = ck.get("data_iteration")
        model = M2FScores(ck).to(args.device).eval()
        del ck
        cdir = outdir / f"scores_{tag}"
        cdir.mkdir()
        t1 = time.time()
        print(f"START {tag} iter={it} sha={ck_sha[:16]}", flush=True)

        argmax_zero = np.empty((len(names), 1024, 1024), dtype=np.uint8)
        with torch.inference_mode():
            for i, name in enumerate(names, 1):
                with Image.open(TEST / name) as im:
                    img = np.asarray(im.convert("RGB")).copy()
                px = normalize_to_tensor(img).unsqueeze(0).to(args.device)
                with torch.amp.autocast("cuda", enabled=False):
                    sc = forward_views(model, px, 1).mean(dim=0, dtype=torch.float32)
                if not torch.isfinite(sc).all():
                    raise SystemExit(f"nonfinite scores: {name}")
                arr = sc.cpu().numpy().astype(np.float32)
                np.save(cdir / f"{Path(name).stem}.npy", arr)
                argmax_zero[i - 1] = arr.argmax(0).astype(np.uint8)
                del px, sc
                if i % 100 == 0:
                    el = time.time() - t1
                    print(f"  {tag} {i}/500 {el:.0f}s ({el/i:.2f}s/img)", flush=True)
        del model
        torch.cuda.empty_cache()
        if sha(ckpt) != ck_sha:
            raise SystemExit(f"checkpoint changed: {ckpt}")

        # ---- pack from the cache (CPU) ----
        pack = pack_variants(cdir, names, outdir, tag)
        summary[tag] = {"checkpoint": str(ckpt), "checkpoint_sha256": ck_sha,
                        "data_iteration": it, "packages": pack,
                        "elapsed_sec": round(time.time() - t1, 1)}
        print(f"DONE {tag} in {time.time()-t1:.0f}s", flush=True)

    (outdir / "EVAL_META.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(),
        "scope": "M2F_LRREOPEN_TAIL_EVAL",
        "mode": "pack_only+forward" if args.pack_only else "forward",
        "tta": "D4 TTA8, FP32, autocast off, FP32 view mean",
        "class_axis": "index i = train class i = official class i+1",
        "excluded": {"C": "flips ~90% on this score space",
                     "13": "same as C"},
        "results": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ndone -> {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
