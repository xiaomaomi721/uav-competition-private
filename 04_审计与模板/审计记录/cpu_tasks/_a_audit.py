"""Module A: 261 -> 278 prediction audit (CPU only, PNGs + confidence cache).

Reads the 261 Bstar zip, the 278 Bstar zip, and the 261 native-256 confidence
derived from the ref261 cache. Emits the full transition audit. Everything is a
prediction-level comparison of two submitted packages; NOTHING is called GT
error (the competition has no test1 GT).
"""
from __future__ import annotations

import hashlib
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
OUT = RARE / "cpu_audit_261_vs_278"

CLASS_NAMES = ["Background", "Building", "Road", "Water",
               "Barren", "Vegetation", "Agricultural", "Vehicle"]
N = 8
BSTAR = np.array([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], dtype=np.float32)

# major transitions to report on explicitly (prediction change 261->278)
MAJOR_PAIRS = [(0, 6), (0, 5), (0, 4), (0, 1), (0, 2),  # BG -> Agri/Veg/Barren/Building/Road
               (6, 0), (5, 0), (4, 0), (1, 0), (2, 0)]


def read_zip(zpath: Path):
    """yield (name, int64 pred array in official 1..8)"""
    with zipfile.ZipFile(zpath) as zf:
        names = sorted(zf.namelist(), key=lambda n: int(Path(n).stem.split("_")[-1]))
        for name in names:
            arr = np.asarray(Image.open(io.BytesIO(zf.read(name))))
            yield name, arr.astype(np.int64)


def conf_from_cache(index: int):
    """261 native-256 max-class confidence (post b_star), upsampled to 1024."""
    import torch
    import torch.nn.functional as F
    arr = np.load(REF_CACHE / f"test1_{index + 1}.npy").astype(np.float32)
    z = torch.from_numpy(arr).unsqueeze(0)
    p = torch.softmax(z + torch.tensor(BSTAR).view(1, N, 1, 1), dim=1)[0]
    conf = p.max(0).values  # [256,256]
    conf1024 = F.interpolate(conf.unsqueeze(0).unsqueeze(0), size=(1024, 1024),
                             mode="nearest")[0, 0]
    return conf1024.numpy().astype(np.float32)


def connected_component_sizes(mask: np.ndarray):
    """Label 4-connectivity components and return their sizes (int array)."""
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.array([], dtype=np.int64)
    return np.bincount(lab.ravel(), minlength=n + 1)[1:]


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    trans = np.zeros((N, N), dtype=np.int64)      # trans[from261, to278]
    per_class_261 = np.zeros(N, dtype=np.int64)
    per_class_278 = np.zeros(N, dtype=np.int64)
    changed_conf = []
    unchanged_conf = []
    flip_pct = {}
    per_image = {}

    # sample a cap for the (large) confidence histograms to keep memory sane
    ch_conf = []
    un_conf = []

    names261 = {n: a for n, a in read_zip(ZIP261)}
    names278 = {n: a for n, a in read_zip(ZIP278)}
    assert set(names261) == set(names278) and len(names261) == 500

    # component-size accumulation per major pair
    cc_bins = {pair: [] for pair in MAJOR_PAIRS}

    for idx in range(500):
        name = f"test1_{idx + 1}.png"
        a261 = names261[name]  # 1..8
        a278 = names278[name]
        conf = conf_from_cache(idx)

        moved = a261 != a278
        n_moved = int(moved.sum())
        flip_pct[name] = 100.0 * n_moved / a261.size

        # 0-based class index for transition matrix
        f = a261 - 1
        t = a278 - 1
        np.add.at(trans, (f, t), 1)

        per_class_261 += np.bincount(f.ravel(), minlength=N)
        per_class_278 += np.bincount(t.ravel(), minlength=N)

        ch_conf.append(conf[moved].astype(np.float32))
        un_conf.append(conf[~moved].astype(np.float32))

        # per-image metadata
        per_image[name] = {
            "flip_pct": round(flip_pct[name], 5),
            "n_pixels": int(a261.size),
            "n_changed": n_moved,
            "mean_conf_changed": round(float(conf[moved].mean()), 6) if n_moved else None,
            "mean_conf_unchanged": round(float(conf[~moved].mean()), 6),
        }

        # major transition components
        for (c0, c1) in MAJOR_PAIRS:
            m = (f == c0) & (t == c1)
            if m.any():
                cc_bins[(c0, c1)].append(connected_component_sizes(m))

    changed_conf = np.concatenate(ch_conf)
    unchanged_conf = np.concatenate(un_conf)

    def hist(a):
        h, e = np.histogram(a, bins=20, range=(0.0, 1.0))
        return {"counts": h.tolist(), "edges": [round(float(x), 4) for x in e]}

    # component size bins
    cc_summary = {}
    for pair, sizes in cc_bins.items():
        if not sizes:
            cc_summary[f"{CLASS_NAMES[pair[0]]}->{CLASS_NAMES[pair[1]]}"] = {
                "components": 0, "total_pixels": 0, "size_bins": {}}
            continue
        all_s = np.concatenate(sizes)
        bins = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096]
        counts, edges = np.histogram(all_s, bins=bins)
        cc_summary[f"{CLASS_NAMES[pair[0]]}->{CLASS_NAMES[pair[1]]}"] = {
            "components": int(len(all_s)),
            "total_pixels": int(all_s.sum()),
            "mean_size": round(float(all_s.mean()), 2),
            "max_size": int(all_s.max()),
            "size_bins": {f"{edges[i]}-{edges[i+1]}": int(c)
                          for i, c in enumerate(counts)},
        }

    # net area change per class (share of all pixels, in pp)
    total_px = int(per_class_261.sum())
    net_area_pp = 100.0 * (per_class_278 - per_class_261) / total_px

    # top/bottom 50 flip images
    ordered = sorted(per_image.items(), key=lambda kv: -kv[1]["flip_pct"])
    top50 = [{"image": k, **v} for k, v in ordered[:50]]
    bottom50 = [{"image": k, **v} for k, v in ordered[-50:]]

    # transition matrix with row-normalized percentages
    trans_pct = 100.0 * trans / total_px

    report = {
        "scope": "261_vs_278_prediction_audit",
        "disclaimer": "prediction-level comparison of two submitted packages; "
                      "no test1 GT exists, so NOTHING here is an error rate",
        "zip261": str(ZIP261), "zip278": str(ZIP278),
        "total_pixels": total_px,
        "transition_matrix_count": trans.tolist(),
        "transition_matrix_pct_of_all_pixels": np.round(trans_pct, 6).tolist(),
        "class_names": CLASS_NAMES,
        "net_area_change_pp": {CLASS_NAMES[i]: round(float(net_area_pp[i]), 5)
                               for i in range(N)},
        "class_share_261_pct": [round(float(x), 4) for x in 100.0 * per_class_261 / total_px],
        "class_share_278_pct": [round(float(x), 4) for x in 100.0 * per_class_278 / total_px],
        "changed_confidence_hist": hist(changed_conf),
        "unchanged_confidence_hist": hist(unchanged_conf),
        "changed_confidence_mean": round(float(changed_conf.mean()), 6),
        "unchanged_confidence_mean": round(float(unchanged_conf.mean()), 6),
        "changed_confidence_median": round(float(np.median(changed_conf)), 6),
        "unchanged_confidence_median": round(float(np.median(unchanged_conf)), 6),
        "overall_flip_pct_mean": round(float(np.mean(list(flip_pct.values()))), 5),
        "overall_flip_pct_min": round(float(min(flip_pct.values())), 5),
        "overall_flip_pct_max": round(float(max(flip_pct.values())), 5),
        "top50_flip_images": top50,
        "bottom50_flip_images": bottom50,
        "major_transitions_component_sizes": cc_summary,
    }
    (OUT / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    # compact CSV of per-image flips
    with open(OUT / "per_image_flip.csv", "w") as f:
        f.write("image,flip_pct,n_changed,mean_conf_changed,mean_conf_unchanged\n")
        for name in sorted(per_image, key=lambda n: int(n.split("_")[-1].split(".")[0])):
            r = per_image[name]
            f.write(f"{name},{r['flip_pct']},{r['n_changed']},"
                    f"{r['mean_conf_changed']},{r['mean_conf_unchanged']}\n")
    print(json.dumps({
        "total_pixels": total_px,
        "overall_flip_pct_mean": report["overall_flip_pct_mean"],
        "changed_conf_mean": report["changed_confidence_mean"],
        "unchanged_conf_mean": report["unchanged_confidence_mean"],
        "net_area_change_pp": report["net_area_change_pp"],
        "out": str(OUT / "audit.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
