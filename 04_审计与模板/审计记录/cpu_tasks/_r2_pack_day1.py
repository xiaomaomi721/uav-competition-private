#!/usr/bin/env python3
"""Build frozen Round2 Day1 packages from existing native-logit caches only.

Protocol matches prelim deployment ordering:
  native 256 logits -> add old Bstar -> bilinear logit upsample to 1024
  (align_corners=False) -> argmax -> official uint8 class IDs 1..8 -> PNG.

No model forward and no platform submission. The forbidden shortcut
argmax@256 -> nearest label upsample is never used.
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/home/jhy/rare_diag_20260915/round2_20260921")
OUT = ROOT / "day1_formal_packages"
BSTAR = np.asarray([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], dtype=np.float32)
N = 1300
SIZE = (1024, 1024)
VARIANTS = {
    "A_theta278_oldBstar": {
        "role": "Exploit",
        "cache": ROOT / "cache_278_test2_raw",
        "model_sha256": "5f8a092b92f1e5c68e7173131c946f50b1663ade1fd43fd71a2ce2d394f001dc",
    },
    "B_theta261_oldBstar": {
        "role": "Explore",
        "cache": ROOT / "cache_261_test2_raw",
        "model_sha256": "9ffdd0beb8480a1efd361503d637d3d53cf716c808537c551fefae09f7fe2786",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cache_manifest(cache: Path) -> dict:
    scores = cache / "scores"
    names = [f"test2_{i}.npy" for i in range(1, N + 1)]
    actual = sorted((p.name for p in scores.glob("*.npy")), key=lambda n: int(Path(n).stem.split("_")[-1]))
    if actual != names:
        raise SystemExit(f"bad cache filenames/count in {scores}: {len(actual)}")
    h = hashlib.sha256()
    files = {}
    for i, name in enumerate(names, 1):
        path = scores / name
        digest = sha256_file(path)
        files[name] = digest
        h.update(name.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
        if i % 200 == 0:
            print(f"  cache hash {cache.name}: {i}/{N}", flush=True)
    meta = cache / "META.json"
    return {
        "cache_dir": str(cache),
        "score_count": N,
        "cache_manifest_sha256": h.hexdigest(),
        "cache_meta_path": str(meta),
        "cache_meta_sha256": sha256_file(meta),
        "cache_meta": json.loads(meta.read_text()),
        "score_file_sha256": files,
    }


def upsample_logits(scores: np.ndarray) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    if scores.shape != (8, 256, 256):
        raise SystemExit(f"bad native logits shape: {scores.shape}")
    x = torch.from_numpy(np.ascontiguousarray(scores.astype(np.float32, copy=False)))
    x = x + torch.from_numpy(BSTAR)[:, None, None]
    y = F.interpolate(x[None], size=SIZE, mode="bilinear", align_corners=False)[0]
    if tuple(y.shape) != (8, 1024, 1024) or not bool(torch.isfinite(y).all()):
        raise SystemExit("bad upsampled logits")
    return (y.argmax(0).numpy().astype(np.uint8) + 1)


def pack_variant(tag: str, cfg: dict, script_sha: str) -> tuple[dict, list[np.ndarray]]:
    cache = cfg["cache"]
    manifest = cache_manifest(cache)
    meta_model_sha = manifest["cache_meta"].get("checkpoint_sha256")
    if meta_model_sha != cfg["model_sha256"]:
        raise SystemExit(f"model SHA mismatch for {tag}: {meta_model_sha}")

    png_dir = OUT / tag
    if png_dir.exists():
        shutil.rmtree(png_dir)
    png_dir.mkdir(parents=True)
    preds: list[np.ndarray] = []
    hist = np.zeros(8, dtype=np.int64)
    png_manifest_h = hashlib.sha256()

    for i in range(1, N + 1):
        name = f"test2_{i}.png"
        scores = np.load(cache / "scores" / f"test2_{i}.npy", allow_pickle=False)
        pred = upsample_logits(scores)
        if pred.shape != SIZE or pred.dtype != np.uint8:
            raise SystemExit(f"bad prediction format: {name}")
        vals = np.unique(pred)
        if vals.min() < 1 or vals.max() > 8:
            raise SystemExit(f"illegal class ID in {name}: {vals}")
        path = png_dir / name
        Image.fromarray(pred, mode="L").save(path, format="PNG")
        png_sha = sha256_file(path)
        png_manifest_h.update(name.encode("utf-8") + b"\0" + png_sha.encode("ascii") + b"\n")
        hist += np.bincount(pred.ravel() - 1, minlength=8)
        preds.append(pred)
        if i % 100 == 0:
            print(f"  {tag}: {i}/{N}", flush=True)

    zip_path = OUT / f"{tag}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_STORED) as zf:
        for i in range(1, N + 1):
            path = png_dir / f"test2_{i}.png"
            zf.write(path, arcname=path.name)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        expected = [f"test2_{i}.png" for i in range(1, N + 1)]
        if names != expected or any("/" in name or "\\" in name for name in names):
            raise SystemExit(f"zip root/name failure for {tag}")
        for i, name in enumerate(names, 1):
            arr = np.asarray(Image.open(io.BytesIO(zf.read(name))))
            if arr.shape != SIZE or arr.dtype != np.uint8 or arr.ndim != 2:
                raise SystemExit(f"zip format failure {tag}/{name}: {arr.shape}, {arr.dtype}")
            lo, hi = int(arr.min()), int(arr.max())
            if lo < 1 or hi > 8:
                raise SystemExit(f"zip class ID failure {tag}/{name}: {lo}..{hi}")

    total = int(hist.sum())
    report = {
        "tag": tag,
        "role": cfg["role"],
        "zip_path": str(zip_path),
        "zip_sha256": sha256_file(zip_path),
        "png_count": N,
        "format_check": {
            "names_exact_test2_1_to_1300": True,
            "zip_root_only": True,
            "single_channel": True,
            "size_1024x1024": True,
            "dtype_uint8": True,
            "class_ids_legal_1_to_8": True,
        },
        "model_sha256": cfg["model_sha256"],
        "bstar": BSTAR.tolist(),
        "protocol": "native fp16 cache -> float32 -> +Bstar@256 -> torch bilinear logit upsample 1024 align_corners=False -> argmax -> +1 -> uint8 L PNG",
        "forbidden_protocol_not_used": "argmax@256 -> nearest-neighbor label upsample",
        "cache": manifest,
        "packing_script_path": str(Path(__file__).resolve()),
        "packing_script_sha256": script_sha,
        "png_manifest_sha256": png_manifest_h.hexdigest(),
        "predicted_share_pct_class_ids_1_to_8": [float(x) for x in (100.0 * hist / total)],
    }
    return report, preds


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    script_sha = sha256_file(Path(__file__).resolve())
    reports = {}
    pred_sets = {}
    for tag, cfg in VARIANTS.items():
        print(f"START {tag}", flush=True)
        reports[tag], pred_sets[tag] = pack_variant(tag, cfg, script_sha)

    a = pred_sets["A_theta278_oldBstar"]
    b = pred_sets["B_theta261_oldBstar"]
    diff = sum(int(np.count_nonzero(x != y)) for x, y in zip(a, b))
    total = N * 1024 * 1024
    final = {
        "created": datetime.now(timezone.utc).isoformat(),
        "scope": "ROUND2_DAY1_FORMAL_PACKAGES_NO_SUBMIT",
        "variants": reports,
        "A_vs_B": {
            "different_pixels": diff,
            "pixels": total,
            "prediction_flip_frac": diff / total,
            "prediction_flip_pct": 100.0 * diff / total,
            "single_difference": "model checkpoint only: A=theta278, B=theta261; same old Bstar, same caches protocol, same logit upsample/argmax/PNG packer",
        },
        "platform_submission_performed": False,
    }
    (OUT / "PACKAGE_REPORT.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps({
        "A": {k: reports["A_theta278_oldBstar"][k] for k in ("zip_path", "zip_sha256", "predicted_share_pct_class_ids_1_to_8")},
        "B": {k: reports["B_theta261_oldBstar"][k] for k in ("zip_path", "zip_sha256", "predicted_share_pct_class_ids_1_to_8")},
        "A_vs_B": final["A_vs_B"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
