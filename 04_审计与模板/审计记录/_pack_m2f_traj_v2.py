#!/usr/bin/env python3
"""Pack the NEW cells of the M2F Run1b (Z, R) trajectory. CPU only.

Scope, decided from the flip-geometry readout (see FLIP_GEOMETRY.json):

  FG020  (foreground uniform +0.20)  -- the operator M2F actually responds to.
          20k / 40k are NEW. 56k is already submitted as #89 (0.685100).
          Included at 56k too, but ONLY as a pipeline check.

  ZERO   -- 20k / 40k / 56k all already submitted as #101 / #100 / #99.
          Produced here ONLY to verify this pipeline reproduces the submitted
          packages pixel-for-pixel. A mismatch voids everything else.

  C / #13 -- DELIBERATELY EXCLUDED. Measured on this M2F score space they flip
          ~90% of all pixels (E-table: a normal calibration operator flips
          ~1%). The CHAMPION vector was tuned for Stage23's raw-logit space,
          whose range differs by orders of magnitude; on M2F it destroys the
          prediction rather than nudging it. Spending submission slots on it at
          20k/40k would be waste. (The 56k C/#13 numbers were a useful lesson
          and are already on record.)

Reads the score cache written by _m2f_calib_traj_cache.py.
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
FG020 = [0.0] + [0.20] * 7
VARIANTS = [("ZERO", ZERO), ("FG020", FG020)]

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
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    outdir = Path(args.outdir)
    if outdir.exists():
        raise SystemExit(f"outdir exists: {outdir}")
    outdir.mkdir(parents=True)

    results = {}
    for step in args.steps:
        cdir = cache / f"score_{step}"
        stems = sorted(p.stem for p in cdir.glob("*.npy"))
        if len(stems) != 500:
            raise SystemExit(f"{cdir}: expected 500, got {len(stems)}")

        handles = {}
        for name, vec in VARIANTS:
            zpath = outdir / f"M2F_Run1b_{step//1000}k_{name}.zip"
            handles[name] = [vec, zpath,
                             zipfile.ZipFile(zpath, "x", compression=zipfile.ZIP_STORED),
                             hashlib.sha256()]
        preds_zero = {}

        for si, stem in enumerate(stems, 1):
            mean = np.load(cdir / f"{stem}.npy")
            if mean.shape != (NUM_CLASSES, 1024, 1024):
                raise SystemExit(f"{stem}: shape {mean.shape}")
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
            if si % 250 == 0:
                print(f"  {step} {si}/500", flush=True)

        pack = {}
        for name, (vec, zpath, zf, digest) in handles.items():
            zf.close()
            pack[name] = {"vector": vec, "zip_path": str(zpath),
                          "zip_sha256": sha_file(zpath),
                          "png_content_sha256": digest.hexdigest(),
                          "entries": len(stems)}

        verify = {"checked": False}
        ref = Path(EXISTING_ZERO[step])
        if ref.is_file():
            mismatch = 0
            total = 0
            with zipfile.ZipFile(ref) as rz:
                rn_all = sorted(n for n in rz.namelist() if n.endswith(".png"))
                if len(rn_all) != 500:
                    raise SystemExit(f"{ref.name}: {len(rn_all)} png")
                for rn in rn_all:
                    a = np.asarray(Image.open(io.BytesIO(rz.read(rn))))
                    b = preds_zero.get(Path(rn).stem)
                    if b is None:
                        raise SystemExit(f"name not in cache: {rn}")
                    mismatch += int((a != b).sum())
                    total += a.size
            verify = {"checked": True, "reference": str(ref),
                      "reference_sha256": sha_file(ref), "pixels_compared": total,
                      "mismatched_pixels": mismatch, "identical": mismatch == 0}
            verdict = "IDENTICAL" if mismatch == 0 else f"{mismatch} PIXELS DIFFER"
            print(f"  [{step}] ZERO vs submitted package: {verdict}", flush=True)

        results[step] = {"packages": pack, "zero_pipeline_check": verify}
        print(f"[{step}] done", flush=True)

    (outdir / "PACK_META.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(),
        "scope": "M2F_RUN1B_CALIBRATION_RESPONSE_TRAJECTORY",
        "cache_dir": str(cache),
        "class_axis": "index i = train class i = official class i+1",
        "png_encoding": "argmax(mean_score + bias, class) + 1, mode L",
        "excluded": {
            "C": "flips ~90% of pixels on this score space; CHAMPION was tuned "
                 "for Stage23 raw logits. Not a meaningful operator for M2F.",
            "13": "same as C, farmland axis variant.",
        },
        "note": "ZERO duplicates submitted #99/#100/#101 and exists only as a "
                "pipeline check. FG020 at 56k duplicates submitted #89.",
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\ndone -> {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
