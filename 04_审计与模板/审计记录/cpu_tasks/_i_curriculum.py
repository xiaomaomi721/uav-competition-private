"""Module I: target curriculum statistics (CPU only).

Per test1 image: 261->278 flip fraction, 261 mean confidence, prediction
entropy, class shares, and major-transition shares. Then a simple k-means
stratification into stable / medium / high-shift strata for a future ATDA
curriculum (which images to weight more heavily).

Reuses the ref261 native-256 cache for confidence AND entropy. No forward.
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
OUT = RARE / "cpu_curriculum"

N = 8
CLASS_NAMES = ["Background", "Building", "Road", "Water",
               "Barren", "Vegetation", "Agricultural", "Vehicle"]
BSTAR = np.array([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], dtype=np.float32)


def read_zip(zpath):
    with zipfile.ZipFile(zpath) as zf:
        names = sorted(zf.namelist(), key=lambda n: int(Path(n).stem.split("_")[-1]))
        return {n: np.asarray(Image.open(io.BytesIO(zf.read(n)))).astype(np.int64)
                for n in names}


def prob_and_entropy(index: int):
    import torch
    import torch.nn.functional as F
    arr = np.load(REF_CACHE / f"test1_{index + 1}.npy").astype(np.float32)
    z = torch.from_numpy(arr).unsqueeze(0)
    p = torch.softmax(z + torch.tensor(BSTAR).view(1, N, 1, 1), dim=1)[0]  # [8,256,256]
    conf = p.max(0).values
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(0)
    # upsample to 1024
    conf1024 = F.interpolate(conf.unsqueeze(0).unsqueeze(0), size=(1024, 1024),
                             mode="nearest")[0, 0].numpy().astype(np.float32)
    ent1024 = F.interpolate(ent.unsqueeze(0).unsqueeze(0), size=(1024, 1024),
                            mode="nearest")[0, 0].numpy().astype(np.float32)
    return p, conf1024, ent1024


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pred261 = read_zip(ZIP261)
    pred278 = read_zip(ZIP278)

    rows = []
    for idx in range(500):
        name = f"test1_{idx + 1}.png"
        a261 = pred261[name] - 1
        a278 = pred278[name] - 1
        p, conf, ent = prob_and_entropy(idx)
        moved = a261 != a278

        # class shares of 278 prediction
        share278 = np.bincount(a278.ravel(), minlength=N) / a278.size

        # major-transition shares (BG<->Agri/Veg/Barren as fraction of pixels)
        major = {
            "BG->Agri": float(((a261 == 0) & (a278 == 6)).mean()),
            "BG->Veg": float(((a261 == 0) & (a278 == 5)).mean()),
            "BG->Barren": float(((a261 == 0) & (a278 == 4)).mean()),
            "Agri->BG": float(((a261 == 6) & (a278 == 0)).mean()),
            "Veg->BG": float(((a261 == 5) & (a278 == 0)).mean()),
            "Barren->BG": float(((a261 == 4) & (a278 == 0)).mean()),
        }
        rows.append({
            "image": name,
            "flip_pct": round(float(moved.mean()) * 100, 5),
            "mean_conf261": round(float(conf.mean()), 6),
            "mean_entropy": round(float(ent.mean()), 6),
            "class_share_278": [round(float(x), 5) for x in share278],
            "major_transition_share": {k: round(v, 7) for k, v in major.items()},
        })

    # k-means stratification (3 strata) on flip_pct + mean_entropy
    from sklearn.cluster import KMeans
    X = np.array([[r["flip_pct"], r["mean_entropy"]] for r in rows], dtype=np.float32)
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    # order clusters by mean flip_pct (stable < medium < high-shift)
    order = np.argsort([X[km.labels_ == c, 0].mean() for c in range(3)])
    label_map = {order[0]: "stable", order[1]: "medium", order[2]: "high_shift"}
    for i, r in enumerate(rows):
        r["stratum"] = label_map[int(km.labels_[i])]

    strata = {}
    for name in ("stable", "medium", "high_shift"):
        sub = [r for r in rows if r["stratum"] == name]
        strata[name] = {
            "n": len(sub),
            "mean_flip_pct": round(float(np.mean([r["flip_pct"] for r in sub])), 5),
            "mean_entropy": round(float(np.mean([r["mean_entropy"] for r in sub])), 6),
            "mean_conf261": round(float(np.mean([r["mean_conf261"] for r in sub])), 6),
            "images": [r["image"] for r in sub],
        }

    report = {
        "scope": "target_curriculum_statistics",
        "n_images": 500,
        "strata": {k: {kk: vv for kk, vv in v.items() if kk != "images"}
                   for k, v in strata.items()},
        "strata_image_lists": {k: v["images"] for k, v in strata.items()},
        "per_image": rows,
    }
    (OUT / "curriculum.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    # CSV
    with open(OUT / "curriculum.csv", "w") as f:
        f.write("image,stratum,flip_pct,mean_conf261,mean_entropy\n")
        for r in rows:
            f.write(f"{r['image']},{r['stratum']},{r['flip_pct']},"
                    f"{r['mean_conf261']},{r['mean_entropy']}\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'images'}
                      for k, v in strata.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
