#!/usr/bin/env python3
"""soupC axis scans: background (historically the #1 lever) and water (#3).

Base point is the current best measured soupC configuration:
    CHAMPION with farmland 3.3215 -> 2.1215   (= 0.696818, the compliant record)

Background
----------
The ledger's first-priority axis is background, scanned 0.000 -> -0.200
monotonically improving and stopping at -0.200. The only points below -0.200 in
that ledger (-0.25 / -0.30) were run WITH barren raised to 3.5, so background
below -0.2 has never been measured cleanly. This scan isolates it.

Water
-----
Third-priority axis, also recorded as still improving as it was pressed down
(1.069 -> 0.70, +0.012). CHAMPION carries water at 0.7; two points below it.

Barren is left at its CHAMPION value of 0.1: measured today it is monotone
decreasing upward (0.1 -> 0.6 costs only 0.0008pp, 0.6 -> 3.6 costs 1.29pp), so
the vertex sits at or just below 0.1 and the axis holds no budget.

Pure CPU from the existing soupC score cache. Writes only under --outdir.
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
BASE = [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 2.1215, -0.5]   # farmland already optimal
CACHE = "/home/jhy/rare_diag_20260915/soup_family_eval_20260917/scores_soupC"

BACKGROUND_IDX = 0
WATER_IDX = 3

BACKGROUND_TARGETS = [-0.3, -0.4, -0.5, -0.6, -0.8]
WATER_TARGETS = [0.4, 0.1]

MEASURED = {"zero": 0.681892, "C": 0.695792, "farmland_2.1215": 0.696818,
            "barren_0.6": 0.696006, "barren_3.6": 0.683895}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def render(cdir, stems, vec, zpath):
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
    return dig.hexdigest()


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
              "scope": "SOUPC_BACKGROUND_AND_WATER", "cache": str(cdir),
              "base_vector": BASE, "measured": MEASURED, "packages": {}}

    jobs = ([("bg", BACKGROUND_IDX, t) for t in BACKGROUND_TARGETS] +
            [("water", WATER_IDX, t) for t in WATER_TARGETS])

    for name, idx, val in jobs:
        vec = list(BASE)
        vec[idx] = val
        tag = f"soupC_{name}{str(val).replace('-', 'm').replace('.', 'p')}"
        zpath = outdir / f"{tag}.zip"
        png_sha = render(cdir, stems, vec, zpath)
        record["packages"][tag] = {"vector": vec, "axis_index": idx,
                                   "axis_name": name, "value": val,
                                   "zip_path": str(zpath), "zip_sha256": sha(zpath),
                                   "entries": len(stems),
                                   "png_content_sha256": png_sha}
        print(f"  {tag:26s} = {val:<7} {zpath.stat().st_size} B", flush=True)

    (outdir / "SOUPC_BG_WATER_META.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ndone -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
