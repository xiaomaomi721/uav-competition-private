#!/usr/bin/env python3
"""soupB KD Run1 @3200: build ZERO / C / #13 submission packages from the
single cached D4-TTA8 score forward. Pure CPU, no GPU, no second TTA8.

Each package is a zip of 500 PNGs (mode L, official class ids 1..8), produced
as argmax(mean_logits + bias, axis=class) + 1. The three packages read the SAME
per-image mean logits, so they differ only by the constant bias vector.

Read-only with respect to training. Writes only under --outdir (must not exist).
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
C = [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5]
V13 = [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 2.9215, -0.5]
VARIANTS = [("ZERO", ZERO), ("C", C), ("13", V13)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-dir", required=True,
                    help="dir written by _score_cache_step3200.py (probs/*.npy)")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    score_dir = Path(args.score_dir)
    outdir = Path(args.outdir)
    if outdir.exists():
        raise SystemExit(f"outdir exists: {outdir}")
    if not score_dir.is_dir():
        raise SystemExit(f"missing score dir: {score_dir}")

    # score dir also holds ARGMAX_ZERO.npy, a derived file written by the cache
    # script; only per-image score files count as inputs.
    derived = {"ARGMAX_ZERO", "SCORE_META"}
    stems = sorted(p.stem for p in score_dir.glob("*.npy") if p.stem not in derived)
    if len(stems) != 500:
        raise SystemExit(f"expected 500 score files, got {len(stems)}")

    outdir.mkdir(parents=True)

    handles = {}
    for name, vec in VARIANTS:
        p = outdir / f"soupB_kd3200_{name}.zip"
        handles[name] = [vec, p, zipfile.ZipFile(p, "x", compression=zipfile.ZIP_STORED),
                         hashlib.sha256()]

    for i, stem in enumerate(stems):
        mean = np.load(score_dir / f"{stem}.npy")
        if mean.shape != (NUM_CLASSES, 1024, 1024):
            raise SystemExit(f"{stem}: unexpected shape {mean.shape}")
        for name, (vec, _, zf, digest) in handles.items():
            pred = (mean + np.asarray(vec, dtype=np.float32)[:, None, None]).argmax(0)
            png = np.ascontiguousarray(pred.astype(np.uint8) + 1)   # official ids 1..8
            payload = io.BytesIO()
            Image.fromarray(png, mode="L").save(payload, format="PNG")
            data = payload.getvalue()
            fname = f"{stem}.png"
            zf.writestr(fname, data)
            digest.update(fname.encode() + b"\0" + data)
        if (i + 1) % 255 == 0:
            print(f"  {i+1}/{len(stems)}", flush=True)

    results = {}
    for name, (vec, p, zf, digest) in handles.items():
        zf.close()
        results[name] = {"vector": vec, "zip_path": str(p),
                         "zip_sha256": sha256_file(p),
                         "png_content_sha256": digest.hexdigest(),
                         "entries": len(stems)}
        print(f"[{name:5s}] {p.name}  {p.stat().st_size} bytes  "
              f"zip_sha256={results[name]['zip_sha256'][:16]}")

    (outdir / "PACK_META.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(),
        "scope": "SOUPB_KD_RUN1_STEP3200_THREE_PACKAGES",
        "score_dir": str(score_dir),
        "score_meta": str(score_dir / "SCORE_META.json"),
        "class_axis": "index i = train class i = official class i+1",
        "png_encoding": "argmax(mean_logits + bias, class) + 1, mode L",
        "packages": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"done 3 packages -> {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
