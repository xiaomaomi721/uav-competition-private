"""Module H: Pseudo500-v2 CPU preparation.

Uses the 278 predictions (submitted Bstar zip) as the new pseudo teacher labels,
and stratifies them against the 261 predictions and the 261 confidence cache:

  P0  all-hard manifest          : all 500 test1 images, 278 argmax as hard label
  P1  agreement manifest         : pixels where 261 == 278 (high-confidence)
  P2  disagreement manifest      : pixels where 261 != 278, stratified by 261
                                   confidence band

Produces labels/masks/manifests/statistics only. No training, no forward.
The hard pseudo masks are written as PNG (official class 1..8), one per image,
to be used as the pseudo teacher when GPU returns.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ATDA = Path("/home/jhy/rare_diag_20260915/atda_20260918")
RARE = Path("/home/jhy/rare_diag_20260915")
ZIP261 = RARE / "phase0_eval_20260918/A_cleanSEM_M_k100_th209_Bstar.zip"
ZIP278 = ATDA / "export/atda_8k_a1p25_Bstar.zip"
REF_CACHE = ATDA / "ref261_cache/scores"
OUT = RARE / "pseudo500_v2_cpu"

N = 8
BSTAR = np.array([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], dtype=np.float32)
CONF_BANDS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]


def read_zip(zpath):
    with zipfile.ZipFile(zpath) as zf:
        names = sorted(zf.namelist(), key=lambda n: int(Path(n).stem.split("_")[-1]))
        return {n: np.asarray(Image.open(io.BytesIO(zf.read(n)))).astype(np.int64)
                for n in names}


def conf261_1024(index: int):
    import torch
    import torch.nn.functional as F
    arr = np.load(REF_CACHE / f"test1_{index + 1}.npy").astype(np.float32)
    z = torch.from_numpy(arr).unsqueeze(0)
    p = torch.softmax(z + torch.tensor(BSTAR).view(1, N, 1, 1), dim=1)[0]
    c = p.max(0).values
    return F.interpolate(c.unsqueeze(0).unsqueeze(0), size=(1024, 1024),
                         mode="nearest")[0, 0].numpy().astype(np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    masks_dir = OUT / "masks"
    masks_dir.mkdir(exist_ok=True)

    pred261 = read_zip(ZIP261)
    pred278 = read_zip(ZIP278)
    assert set(pred261) == set(pred278) and len(pred261) == 500

    p0 = []
    p1 = []
    p2 = []
    per_class_hard = np.zeros(N, dtype=np.int64)
    band_pixel = {f"{a}-{b}": 0 for a, b in CONF_BANDS}

    for idx in range(500):
        name = f"test1_{idx + 1}.png"
        a261 = pred261[name]
        a278 = pred278[name]
        conf = conf261_1024(idx)
        agree = a261 == a278
        disagree = ~agree

        # P0: write hard pseudo mask (278 argmax, official 1..8)
        Image.fromarray(np.ascontiguousarray(a278.astype(np.uint8)), mode="L").save(
            masks_dir / name)
        per_class_hard += np.bincount((a278 - 1).ravel(), minlength=N)
        p0.append({"image": name, "mask": str(masks_dir / name),
                   "source": "278_prediction"})

        # P1 agreement fraction
        p1.append({"image": name,
                   "agree_frac": round(float(agree.mean()), 6),
                   "n_agree": int(agree.sum()),
                   "n_disagree": int(disagree.sum())})

        # P2 disagreement stratified by 261 confidence
        strata = []
        for a, b in CONF_BANDS:
            m = disagree & (conf >= a) & (conf < b)
            band_pixel[f"{a}-{b}"] += int(m.sum())
            if m.any():
                strata.append({"band": f"{a}-{b}", "n_pixels": int(m.sum()),
                               "mean_conf": round(float(conf[m].mean()), 6)})
        p2.append({"image": name, "strata": strata})

    total = 1024 * 1024 * 500
    report = {
        "scope": "pseudo500_v2_cpu_prep",
        "teacher": "278 predictions (atda_8k_a1p25, mIoU 0.706154)",
        "teacher_zip": str(ZIP278),
        "hard_label_class_share_pct": [round(float(x), 4)
                                       for x in 100.0 * per_class_hard / per_class_hard.sum()],
        "total_pixels": total,
        "overall_agreement_pct": round(100.0 * float(
            sum(p["n_agree"] for p in p1)) / total, 4),
        "overall_disagreement_pct": round(100.0 * float(
            sum(p["n_disagree"] for p in p1)) / total, 4),
        "disagreement_by_261_confidence_band": band_pixel,
        "manifests": {
            "P0_all_hard": str(OUT / "manifest_P0.json"),
            "P1_agreement": str(OUT / "manifest_P1.json"),
            "P2_disagreement": str(OUT / "manifest_P2.json"),
        },
    }
    (OUT / "manifest_P0.json").write_text(json.dumps(p0), encoding="utf-8")
    (OUT / "manifest_P1.json").write_text(json.dumps(p1), encoding="utf-8")
    (OUT / "manifest_P2.json").write_text(json.dumps(p2), encoding="utf-8")
    (OUT / "PREP_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
