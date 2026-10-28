#!/usr/bin/env python3
"""Continue the KD6400 farmland axis below the last measured point.

The measured curve is still monotone at the bottom of the previous range:

    3.3215 -> 0.692202
    3.1215 -> 0.692557
    2.9215 -> 0.692851
    2.5215 -> 0.693291
    2.1215 -> 0.693554
    1.7215 -> 0.693595   <- still rising, no vertex reached

Stage23's farmland vertex sat near 2.0-2.4, so KD6400's geometry is not the same
shape and cannot be assumed to turn there. This extends the same 0.4-step scan
two more points down to 0.9215, plus two half-steps to resolve the turning region
once it appears.

Pure CPU from the existing score cache. Writes only under --outdir.
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
CHAMPION = [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5]
AGRI_IDX = 6
CACHE = ("/home/jhy/rare_diag_20260915/kd6400_calib_20260917/"
         "scores_checkpoint_step6400")

# absolute farmland values to render (champion is 3.3215)
TARGETS = [1.3215, 0.9215, 1.5215, 1.1215]
MEASURED = {"3.3215": 0.692202, "3.1215": 0.692557, "2.9215": 0.692851,
            "2.5215": 0.693291, "2.1215": 0.693554, "1.7215": 0.693595}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if outdir.exists():
        raise SystemExit(f"outdir exists: {outdir}")
    outdir.mkdir(parents=True)

    cdir = Path(CACHE)
    stems = sorted(p.stem for p in cdir.glob("*.npy"))
    if len(stems) != 500:
        raise SystemExit(f"{cdir}: {len(stems)} score files")

    record = {"created": datetime.now(timezone.utc).isoformat(),
              "scope": "KD6400_FARMLAND_AXIS_EXTENSION",
              "cache": str(cdir), "axis_index": AGRI_IDX,
              "champion_farmland": CHAMPION[AGRI_IDX],
              "measured_curve": MEASURED, "packages": {}}

    for agri in TARGETS:
        vec = list(CHAMPION)
        vec[AGRI_IDX] = agri
        tag = f"KD6400_agri{str(agri).replace('.', 'p')}"
        zpath = outdir / f"{tag}.zip"
        dig = hashlib.sha256()
        with zipfile.ZipFile(zpath, "x", compression=zipfile.ZIP_STORED) as zf:
            for stem in stems:
                arr = np.load(cdir / f"{stem}.npy")
                if arr.shape != (NUM_CLASSES, 1024, 1024):
                    raise SystemExit(f"{stem}: {arr.shape}")
                pred = (arr + np.asarray(vec, dtype=np.float32)[:, None, None]).argmax(0)
                png = np.ascontiguousarray(pred.astype(np.uint8) + 1)
                buf = io.BytesIO()
                Image.fromarray(png, mode="L").save(buf, format="PNG")
                data = buf.getvalue()
                fn = f"{stem}.png"
                zf.writestr(fn, data)
                dig.update(fn.encode() + b"\0" + data)
        record["packages"][tag] = {"vector": vec, "agricultural": agri,
                                   "zip_path": str(zpath), "zip_sha256": sha(zpath),
                                   "entries": len(stems),
                                   "png_content_sha256": dig.hexdigest()}
        print(f"  {tag:28s} agri={agri:<8} {zpath.stat().st_size} B", flush=True)

    (outdir / "KD6400_FARMLAND_EXT_META.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ndone -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
