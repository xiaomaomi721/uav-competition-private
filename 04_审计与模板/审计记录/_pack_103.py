#!/usr/bin/env python3
"""#103 (Stage66 SEMEDA+CWFA best@51800): pack from the cached TTA8 scores. CPU only.

Produces:
  ZERO  -- duplicates scored package #103 (0.680750); kept as a pipeline check
  FG020 -- uniform foreground +0.20, the M2F-family operator for comparison
  C     -- CHAMPION vector
  13    -- CHAMPION with farmland 3.3215 -> 2.9215

Every variant also reports its flip% against ZERO, because a bias that flips a
huge fraction of pixels is a destructive operator, not a calibration one. (On
the M2F score space C flips ~90% and was excluded for that reason; Stage66 is a
plain SegFormer whose raw-logit space is what CHAMPION was tuned for, so a
normal ~1-5% is expected here.)

The ZERO package is compared pixel-for-pixel against the already-scored #103 zip.
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
C = [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5]
V13 = [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 2.9215, -0.5]
VARIANTS = [("ZERO", ZERO), ("FG020", FG020), ("C", C), ("13", V13)]

REF_ZERO = Path("/home/jhy/rare_diag_20260915/zero_backfill_20260916/"
                "stage66_best51800/predictions")


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
    ap.add_argument("--ref-pack", default=None,
                    help="zip whose ZERO prediction must be reproduced")
    args = ap.parse_args()

    cdir = Path(args.cache_dir) / "scores"
    outdir = Path(args.outdir)
    if outdir.exists():
        raise SystemExit(f"outdir exists: {outdir}")
    stems = sorted(p.stem for p in cdir.glob("*.npy"))
    if len(stems) != 500:
        raise SystemExit(f"expected 500 score files, got {len(stems)}")
    outdir.mkdir(parents=True)

    handles = {}
    for name, vec in VARIANTS:
        zpath = outdir / f"stage66_51800_{name}.zip"
        handles[name] = [vec, zpath,
                         zipfile.ZipFile(zpath, "x", compression=zipfile.ZIP_STORED),
                         hashlib.sha256()]
    preds_zero = {}
    total_px = 0
    flips = {n: 0 for n, _ in VARIANTS}
    bg2fg = {n: 0 for n, _ in VARIANTS}

    for si, stem in enumerate(stems, 1):
        mean = np.load(cdir / f"{stem}.npy")
        if mean.shape != (NUM_CLASSES, 1024, 1024):
            raise SystemExit(f"{stem}: shape {mean.shape}")
        base = mean.argmax(0)
        total_px += base.size
        for name, (vec, _, zf, digest) in handles.items():
            pred = (mean + np.asarray(vec, dtype=np.float32)[:, None, None]).argmax(0)
            if name == "ZERO":
                preds_zero[stem] = pred.astype(np.int64)
            else:
                moved = base != pred
                flips[name] += int(moved.sum())
                bg2fg[name] += int(((base == 0) & moved).sum())
            png = np.ascontiguousarray(pred.astype(np.uint8) + 1)
            payload = io.BytesIO()
            Image.fromarray(png, mode="L").save(payload, format="PNG")
            data = payload.getvalue()
            fname = f"{stem}.png"
            zf.writestr(fname, data)
            digest.update(fname.encode() + b"\0" + data)
        if si % 250 == 0:
            print(f"  {si}/500", flush=True)

    pack = {}
    for name, (vec, zpath, zf, digest) in handles.items():
        zf.close()
        pack[name] = {"vector": vec, "zip_path": str(zpath),
                      "zip_sha256": sha_file(zpath),
                      "png_content_sha256": digest.hexdigest(),
                      "entries": len(stems)}
        if name != "ZERO":
            f = flips[name]
            pack[name]["flip_pct"] = 100.0 * f / total_px
            pack[name]["bg_to_fg_pct_of_flips"] = 100.0 * bg2fg[name] / f if f else 0.0
            print(f"  [{name:5s}] flip={pack[name]['flip_pct']:.4f}%  "
                  f"BG->FG {pack[name]['bg_to_fg_pct_of_flips']:.1f}%", flush=True)

    verify = {"checked": False}
    if args.ref_pack and Path(args.ref_pack).is_file():
        mismatch = 0
        total = 0
        with zipfile.ZipFile(args.ref_pack) as rz:
            rn_all = sorted(n for n in rz.namelist() if n.endswith(".png"))
            if len(rn_all) != 500:
                raise SystemExit(f"reference has {len(rn_all)} png")
            for rn in rn_all:
                a = np.asarray(Image.open(io.BytesIO(rz.read(rn)))).astype(np.int64)
                b = preds_zero.get(Path(rn).stem)
                if b is None:
                    raise SystemExit(f"name not in cache: {rn}")
                mismatch += int((a != b).sum())
                total += a.size
        verify = {"checked": True, "reference": args.ref_pack,
                  "pixels_compared": total, "mismatched_pixels": mismatch,
                  "identical": mismatch == 0}
        print(f"  ZERO vs scored #103 package: "
              f"{'IDENTICAL' if mismatch == 0 else str(mismatch)+' DIFFER'}", flush=True)

    (outdir / "PACK_META.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(),
        "scope": "STAGE66_51800_CALIBRATION",
        "cache_dir": str(cdir), "class_axis": "index i = train class i = official class i+1",
        "png_encoding": "argmax(mean_score + bias, class) + 1, mode L",
        "submitted_reference": "#103 ZERO = 0.680750",
        "packages": pack, "zero_pipeline_check": verify,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ndone -> {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
