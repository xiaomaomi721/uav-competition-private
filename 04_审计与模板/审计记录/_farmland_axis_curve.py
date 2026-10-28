#!/usr/bin/env python3
"""Farmland-axis 1D curve for two models, prepared from existing TTA8 caches.

Context
-------
Both models already have two measured points on the Agricultural axis, and both
show the score still rising as that axis is pressed down:

    soupC   C (3.3215) 0.695792  ->  #13 (2.9215) 0.696373   (+0.0581pp)
    KD6400  C (3.3215) 0.692202  ->  #13 (2.9215) 0.692851   (+0.0649pp)

One point below C tells us the direction but not where the vertex is. Stage23's
historical farmland scan put its vertex near 2.0-2.4, so the curve is bracketed
by stepping further down at the SAME 0.4 step that produced the measured gain,
plus one step back up to confirm the sign.

This is a 1D scan on a single axis, not an 8D sweep, and it reuses the step size
already validated on this model family rather than inventing one.

Pure CPU: both score caches already exist, so no GPU forward is run.
Writes only under --outdir.
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
AGRI_IDX = 6          # index i = train class i = official class i+1; 6 = Agricultural
STEP = 0.4            # the step size the measured C -> #13 gain was produced with

# steps relative to CHAMPION's 3.3215
OFFSETS = [-0.2, -0.4, -0.8, -1.2, -1.6]   # 3.1215 / 2.9215 / 2.5215 / 2.1215 / 1.7215

MODELS = {
    "soupC": "/home/jhy/rare_diag_20260915/soup_family_eval_20260917/scores_soupC",
    "KD6400": "/home/jhy/rare_diag_20260915/kd6400_calib_20260917/"
              "scores_checkpoint_step6400",
}
# platform-measured anchors, for the record
ANCHORS = {
    "soupC": {"3.3215": 0.695792, "2.9215": 0.696373, "zero": 0.681892},
    "KD6400": {"3.3215": 0.692202, "2.9215": 0.692851, "zero": 0.680602},
}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def vec_at(offset: float):
    v = list(CHAMPION)
    v[AGRI_IDX] = round(v[AGRI_IDX] + offset, 6)
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if outdir.exists():
        raise SystemExit(f"outdir exists: {outdir}")
    outdir.mkdir(parents=True)

    record = {"created": datetime.now(timezone.utc).isoformat(),
              "scope": "FARMLAND_AXIS_1D_CURVE", "axis_index": AGRI_IDX,
              "axis_name": "Agricultural", "step": STEP,
              "champion": CHAMPION, "anchors": ANCHORS,
              "note": "prepared but NOT submitted; awaiting the official ruling on "
                      "whether post-training weight averaging is compliant",
              "packages": {}}

    for model, cache in MODELS.items():
        cdir = Path(cache)
        if not cdir.is_dir():
            raise SystemExit(f"missing score cache: {cdir}")
        stems = sorted(p.stem for p in cdir.glob("*.npy"))
        if len(stems) != 500:
            raise SystemExit(f"{cdir}: {len(stems)} score files, expected 500")

        for off in OFFSETS:
            vec = vec_at(off)
            agri = vec[AGRI_IDX]
            tag = f"{model}_agri{str(agri).replace('.', 'p')}"
            zpath = outdir / f"{tag}.zip"
            digest = hashlib.sha256()
            with zipfile.ZipFile(zpath, "x", compression=zipfile.ZIP_STORED) as zf:
                for stem in stems:
                    arr = np.load(cdir / f"{stem}.npy")
                    if arr.shape != (NUM_CLASSES, 1024, 1024):
                        raise SystemExit(f"{stem}: shape {arr.shape}")
                    pred = (arr + np.asarray(vec, dtype=np.float32)[:, None, None]
                            ).argmax(0)
                    png = np.ascontiguousarray(pred.astype(np.uint8) + 1)
                    buf = io.BytesIO()
                    Image.fromarray(png, mode="L").save(buf, format="PNG")
                    data = buf.getvalue()
                    fn = f"{stem}.png"
                    zf.writestr(fn, data)
                    digest.update(fn.encode() + b"\0" + data)
            record["packages"][tag] = {
                "model": model, "vector": vec, "agricultural": agri,
                "offset_from_champion": off, "zip_path": str(zpath),
                "zip_sha256": sha(zpath), "entries": len(stems),
                "png_content_sha256": digest.hexdigest(),
            }
            print(f"  {tag:34s} agri={agri:<8} {zpath.stat().st_size} B", flush=True)
        print(f"{model}: {len(OFFSETS)} packages", flush=True)

    (outdir / "FARMLAND_CURVE_META.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ndone -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
