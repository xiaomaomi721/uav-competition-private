#!/usr/bin/env python3
"""M2F Run1b calibration-response trajectory: pack ZERO / FG020 / C / #13 from
the cached TTA8 scores. Pure CPU, no GPU, no second forward.

Existing packages already cover part of the grid, so only the genuinely NEW
cells are produced:

  ZERO  : 99 / 100 / 101 already submitted -> produced here ONLY as a check
  FG020 : 56k already submitted (89, 0.685100) -> 20k / 40k are new
  C     : all three new
  #13   : all three new

The ZERO packages this writes are compared PIXEL-BY-PIXEL against the
already-submitted zips. Any mismatch means the new pipeline diverged from the
one that produced the platform scores, and everything else here is void.

Bias semantics: bias is added to the class axis of the mean score AFTER the
FP32 TTA8 view mean, then argmax + 1. index i = train class i = official class i+1.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

NUM_CLASSES = 8
ZERO = [0.0] * NUM_CLASSES
FG020 = [0.0] + [0.20] * 7          # index 0 = background, 1..7 = foreground
C = [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5]
V13 = [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 2.9215, -0.5]
VARIANTS = [("ZERO", ZERO), ("FG020", FG020), ("C", C), ("13", V13)]

# already-submitted packages, keyed by step -> path on this machine
EXISTING_ZERO = {
    20000: "/home/jhy/rare_diag_20260915/m2f_gpu_20260915/test1_zero_run1b_v1/"
           "03_M2F_Run1b_20k_TTA8_ZERO.zip",
    40000: "/home/jhy/rare_diag_20260915/m2f_gpu_20260915/test1_zero_run1b_v1/"
           "02_M2F_Run1b_40k_TTA8_ZERO.zip",
    56000: "/home/jhy/rare_diag_20260915/m2f_gpu_20260915/test1_zero_run1b_v1/"
           "01_M2F_Run1b_56k_TTA8_ZERO.zip",
}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--steps", type=int, nargs="+", default=[20000, 40000, 56000])
    ap.add_argument("--verify-zero", action="store_true", default=True)
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    outdir = Path(args.outdir)
    if outdir.exists():
        raise SystemExit(f"outdir exists: {outdir}")
    if not cache.is_dir():
        raise SystemExit(f"missing cache dir: {cache}")
    outdir.mkdir(parents=True)

    results = {}
    for step in args.steps:
        cdir = cache / f"score_{step}"
        if not cdir.is_dir():
            raise SystemExit(f"missing score dir: {cdir}")
        stems = sorted(p.stem for p in cdir.glob("*.npy"))
        if len(stems) != 500:
            raise SystemExit(f"{cdir}: expected 500 score files, got {len(stems)}")

        handles = {}
        for name, vec in VARIANTS:
            zpath = outdir / f"M2F_Run1b_{step//1000}k_{name}.zip"
            handles[name] = [vec, zpath,
                             zipfile.ZipFile(zpath, "x", compression=zipfile.ZIP_STORED),
                             hashlib.sha256()]
        preds_zero = {}

        for stem in stems:
            mean = np.load(cdir / f"{stem}.npy")
            if mean.shape != (NUM_CLASSES, 1024, 1024):
                raise SystemExit(f"{stem}: unexpected shape {mean.shape}")
            for name, (vec, _, zf, digest) in handles.items():
                pred = (mean + np.asarray(vec, dtype=np.float32)[:, None, None]).argmax(0)
                png = np.ascontiguousarray(pred.astype(np.uint8) + 1)
                if name == "ZERO":
                    preds_zero[stem] = png
                payload = io.BytesIO()
                Image.fromarray(png, mode="L").save(payload, format="PNG")
                data = payload.getvalue()
                fname = f"{stem}.png"
                zf.writestr(fname, data)
                digest.update(fname.encode() + b"\0" + data)

        pack = {}
        for name, (vec, zpath, zf, digest) in handles.items():
            zf.close()
            pack[name] = {"vector": vec, "zip_path": str(zpath),
                          "zip_sha256": sha_file(zpath),
                          "png_content_sha256": digest.hexdigest(),
                          "entries": len(stems)}

        # ---- pipeline check: ZERO must equal the already-submitted package ----
        verify = {"checked": False}
        if args.verify_zero and step in EXISTING_ZERO:
            ref = Path(EXISTING_ZERO[step])
            if ref.is_file():
                mismatch = 0
                total = 0
                with zipfile.ZipFile(ref) as rz:
                    ref_names = sorted(n for n in rz.namelist() if n.endswith(".png"))
                    if len(ref_names) != 500:
                        raise SystemExit(f"reference {ref.name} has {len(ref_names)} png")
                    for rn in ref_names:
                        stem = Path(rn).stem
                        a = np.asarray(Image.open(io.BytesIO(rz.read(rn))))
                        b = preds_zero.get(stem)
                        if b is None:
                            raise SystemExit(f"reference name not in cache: {rn}")
                        mismatch += int((a != b).sum())
                        total += a.size
                verify = {"checked": True, "reference": str(ref),
                          "reference_sha256": sha_file(ref),
                          "pixels_compared": total, "mismatched_pixels": mismatch,
                          "identical": mismatch == 0}
                print(f"  [{step}] ZERO vs submitted package: "
                      f"{'IDENTICAL' if mismatch == 0 else str(mismatch)+' PIXELS DIFFER'}",
                      flush=True)
            else:
                verify = {"checked": False, "reason": f"reference missing: {ref}"}

        results[step] = {"packages": pack, "zero_pipeline_check": verify}
        print(f"[{step}] {', '.join(sorted(pack))} done", flush=True)

    (outdir / "PACK_META.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(),
        "scope": "M2F_RUN1B_CALIBRATION_RESPONSE_TRAJECTORY",
        "cache_dir": str(cache),
        "class_axis": "index i = train class i = official class i+1",
        "png_encoding": "argmax(mean_score + bias, class) + 1, mode L",
        "note": "ZERO here duplicates already-submitted 99/100/101 and exists "
                "only as a pipeline check; FG020 at 56k duplicates submitted 89.",
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\ndone -> {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
