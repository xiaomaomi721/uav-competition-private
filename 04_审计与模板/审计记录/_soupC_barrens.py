#!/usr/bin/env python3
"""soupC barrens-axis scan, on top of the already-optimised farmland vertex.

Why barrens
-----------
The CHAMPION vector carries Barren at 0.1, but the project's own platform ledger
records the barrens axis as the SECOND-STRONGEST lever ever measured:

    "荒地偏置：抬升单调到 3.60 见顶 (+0.41 大杠杆) → 复赛第二优先"
    1.40 -> 68.807208 (+0.085) ... 3.60 -> 69.257555 (precise vertex) ... 3.80 -> falls

soupC has never had this axis moved. Farmland alone already returned +0.1026pp
on this model, and the barrens lever is historically the larger of the two.

Farmland is pinned at 2.1215, the vertex just measured on soupC:
    2.5215 -> 0.696741, 2.1215 -> 0.696818 (best), 1.7215 -> 0.696478
so the remaining axes are searched against the current best point, not against
the raw CHAMPION.

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
# CHAMPION with farmland already optimised to its measured vertex
BASE = [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 2.1215, -0.5]
BARREN_IDX = 1
CACHE = "/home/jhy/rare_diag_20260915/soup_family_eval_20260917/scores_soupC"

# barrens values to render; 0.1 is the CHAMPION anchor, 3.6 the historical vertex
TARGETS = [0.6, 1.1, 1.6, 2.1, 2.6, 3.1, 3.6]
MEASURED_SOUPC = {"zero": 0.681892, "C": 0.695792, "farmland_2.1215": 0.696818}


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
              "scope": "SOUPC_BARRENS_AXIS", "cache": str(cdir),
              "base_vector": BASE, "barren_index": BARREN_IDX,
              "measured": MEASURED_SOUPC, "packages": {}}

    for barren in TARGETS:
        vec = list(BASE)
        vec[BARREN_IDX] = barren
        tag = f"soupC_barren{str(barren).replace('.', 'p')}"
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
        record["packages"][tag] = {"vector": vec, "barren": barren,
                                   "zip_path": str(zpath), "zip_sha256": sha(zpath),
                                   "entries": len(stems),
                                   "png_content_sha256": dig.hexdigest()}
        print(f"  {tag:28s} barren={barren:<6} {zpath.stat().st_size} B", flush=True)

    (outdir / "SOUPC_BARRENS_META.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ndone -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
