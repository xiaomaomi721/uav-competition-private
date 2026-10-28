#!/usr/bin/env python3
"""Champion affine-calibration packages from the kept test1 score cache.

Operator: s'_c = a_c * s_c + b_c, applied to the final TTA8 mean score, exactly
where b* already operates. This adds ONE variable class (per-class scale); the
score domain and the TTA protocol are unchanged.

Candidates come from the B2 val audit, which found exactly three classes with a
live single-class scale signal on the champion, and all three point the way the
audit's precision/recall diagnosis predicts:

    Barren        a=0.80   val +0.3741pp   P 0.555 / R 0.878 -> over-predicts
    Agricultural  a=0.80   val +0.1961pp   P 0.779 / R 0.965 -> over-predicts
    Background    a=1.20   val +0.1714pp   under-predicts

The other five classes moved <= 0.007pp at every scale, so they are left at 1.0.

Val only screens; the platform decides. Nothing here is a claim that +0.37pp val
becomes +0.37pp test1.

Also emits A0 == ZERO-of-this-champion so the pipeline can be checked against
the submitted 168/173 numbers.

CPU only. Reads the cache written by _champ_test1_cache.py.
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
BSTAR = np.array([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], dtype=np.float32)
IDX = {"bg": 0, "bldg": 1, "road": 2, "water": 3,
       "barren": 4, "veg": 5, "agri": 6, "veh": 7}

# name -> {class index: scale}
CANDIDATES = {
    "A0_bstar_ctrl": {},
    "A1_barren080": {"barren": 0.80},
    "A2_agri080": {"agri": 0.80},
    "A3_bg120": {"bg": 1.20},
    "A4_barren080_agri080": {"barren": 0.80, "agri": 0.80},
    "A5_A4_bg120": {"barren": 0.80, "agri": 0.80, "bg": 1.20},
}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    cache = Path(args.cache) / "scores"
    outdir = Path(args.outdir)
    if outdir.exists():
        raise SystemExit(f"outdir exists: {outdir}")
    stems = sorted(p.stem for p in cache.glob("*.npy"))
    if len(stems) != 500:
        raise SystemExit(f"expected 500 scores, got {len(stems)}")
    outdir.mkdir(parents=True)

    rec = {"created": datetime.now(timezone.utc).isoformat(),
           "scope": "CHAMPION_AFFINE_CALIBRATION",
           "cache": str(cache), "bstar": BSTAR.tolist(),
           "operator": "s'_c = a_c * s_c + b*_c",
           "class_axis": "index i = train class i = official class i+1",
           "packages": {}}

    for name, scales in CANDIDATES.items():
        a = np.ones(NUM_CLASSES, dtype=np.float32)
        for cls, av in scales.items():
            a[IDX[cls]] = av
        zpath = outdir / f"{name}.zip"
        dig = hashlib.sha256()
        with zipfile.ZipFile(zpath, "x", compression=zipfile.ZIP_STORED) as zf:
            for stem in stems:
                s = np.load(cache / f"{stem}.npy").astype(np.float32)
                if s.shape != (NUM_CLASSES, 1024, 1024):
                    raise SystemExit(f"{stem}: {s.shape}")
                pred = (s * a[:, None, None] + BSTAR[:, None, None]).argmax(0)
                png = np.ascontiguousarray(pred.astype(np.uint8) + 1)
                buf = io.BytesIO()
                Image.fromarray(png, mode="L").save(buf, format="PNG")
                data = buf.getvalue()
                fn = f"{stem}.png"
                zf.writestr(fn, data)
                dig.update(fn.encode() + b"\0" + data)
        rec["packages"][name] = {"scales": scales, "a_vector": a.tolist(),
                                 "zip_path": str(zpath), "zip_sha256": sha(zpath),
                                 "entries": len(stems),
                                 "png_content_sha256": dig.hexdigest()}
        print(f"  {name:24s} {scales}  {zpath.stat().st_size} B", flush=True)

    (outdir / "AFFINE_META.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ndone -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
