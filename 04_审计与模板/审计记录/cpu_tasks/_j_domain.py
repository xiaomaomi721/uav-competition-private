"""Module J: cheap train/test domain analysis (CPU only, no deep encoder).

Computes traditional CPU image features for all 6996 train images and the 500
test1 images, then does PCA + k-means + kNN distance to answer:
  - which source sub-domain is each test1 image closest to
  - does the 5597-train vs 1399-holdout split cover test1 differently
  - target-similarity source weights for a future sampler

No nearest-neighbour label prediction is performed (that would leak test1 into
a label); the output is only a source-weight manifest.

Feature vector per image (computed on a 256x256 downsample for speed):
  RGB mean/std (6), HSV mean/std (6), color histogram (RGB 3x16=48),
  Sobel edge density (1), Laplacian variance (1), grayscale entropy (1),
  FFT ring power (6 bins). Total 69 dims.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

DATA = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集")
TRAIN_IMG = DATA / "train" / "images"
TEST_IMG = DATA / "images"
ATDA = Path("/home/jhy/rare_diag_20260915/atda_20260918")
SRC_IDS = ATDA / "source_ids_5597.txt"
OUT = Path("/home/jhy/rare_diag_20260915/cpu_domain")

RS = 256


def features(img_path: Path):
    from scipy import ndimage
    im = np.asarray(Image.open(img_path).convert("RGB").resize((RS, RS))).astype(np.float32)
    rgb = im / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(-1)
    mn = rgb.min(-1)
    delta = mx - mn
    sat = np.where(delta > 0, delta / (mx + 1e-8), 0)
    val = mx
    hsv_h = np.zeros_like(sat)
    # hue
    h = np.zeros_like(r)
    m = np.where(delta <= 0)
    h = np.where(mx == r, ((g - b) / (delta + 1e-8)) % 6, h)
    h = np.where(mx == g, (b - r) / (delta + 1e-8) + 2, h)
    h = np.where(mx == b, (r - g) / (delta + 1e-8) + 4, h)
    h = h / 6.0

    gray = (0.299 * r + 0.587 * g + 0.114 * b)
    sobel = np.hypot(ndimage.sobel(gray, 0), ndimage.sobel(gray, 1))
    lap = ndimage.laplace(gray)

    hist_r, _ = np.histogram(r, bins=16, range=(0, 1))
    hist_g, _ = np.histogram(g, bins=16, range=(0, 1))
    hist_b, _ = np.histogram(b, bins=16, range=(0, 1))
    hist = np.concatenate([hist_r, hist_g, hist_b]) / (RS * RS)

    ent = 0.0
    gh, _ = np.histogram(gray, bins=64, range=(0, 1))
    gh = gh / gh.sum()
    ent = float(-(gh[gh > 0] * np.log(gh[gh > 0])).sum())

    # FFT ring power
    f = np.fft.fftshift(np.fft.fft2(gray - gray.mean()))
    mag = np.abs(f)
    cy, cx = RS // 2, RS // 2
    yy, xx = np.mgrid[0:RS, 0:RS]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).ravel()
    mm = mag.ravel()
    ring = []
    edges = np.linspace(0, RS // 2, 7)
    for i in range(6):
        mask = (rr >= edges[i]) & (rr < edges[i + 1])
        ring.append(float(np.mean(mm[mask])) if mask.any() else 0.0)
    ring = np.array(ring)
    ring = ring / (ring.sum() + 1e-8)

    return np.concatenate([
        [r.mean(), r.std(), g.mean(), g.std(), b.mean(), b.std()],
        [h.mean(), h.std(), sat.mean(), sat.std(), val.mean(), val.std()],
        hist,
        [float(sobel.mean()), float(lap.var()), ent],
        ring,
    ]).astype(np.float32)


def worker(args):
    stem, path = args
    return stem, features(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import torch  # only for nothing; use numpy

    source_ids = set(l.strip() for l in SRC_IDS.read_text().splitlines() if l.strip())
    all_train = sorted(p.stem for p in TRAIN_IMG.glob("*.png"))
    val_ids = [s for s in all_train if s not in source_ids]
    test_names = sorted((p.stem for p in TEST_IMG.glob("*.png")),
                        key=lambda n: int(n.split("_")[-1]))

    # build task list: (label, path)
    tasks = []
    for s in sorted(source_ids):
        tasks.append((f"src_{s}", TRAIN_IMG / f"{s}.png"))
    for v in val_ids:
        tasks.append((f"val_{v}", TRAIN_IMG / f"{v}.png"))
    for t in test_names:
        tasks.append((f"tst_{t}", TEST_IMG / f"{t}.png"))

    print(f"computing features for {len(tasks)} images ...", flush=True)
    feats = {}
    with ProcessPoolExecutor(max_workers=48) as ex:
        for i, (key, vec) in enumerate(ex.map(worker, tasks, chunksize=16)):
            feats[key] = vec
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(tasks)}", flush=True)

    src_keys = [f"src_{s}" for s in sorted(source_ids)]
    val_keys = [f"val_{v}" for v in val_ids]
    tst_keys = [f"tst_{t}" for t in test_names]

    X_src = np.stack([feats[k] for k in src_keys])
    X_val = np.stack([feats[k] for k in val_keys])
    X_tst = np.stack([feats[k] for k in tst_keys])
    X_all = np.vstack([X_src, X_val, X_tst])

    # standardize
    mu = X_all.mean(0)
    sd = X_all.std(0) + 1e-8
    Z = (X_all - mu) / sd
    Z_src, Z_val, Z_tst = Z[:len(src_keys)], Z[len(src_keys):len(src_keys)+len(val_keys)], \
                           Z[-len(tst_keys):]

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=20)
    P = pca.fit_transform(Z)
    P_src = P[:len(src_keys)]
    P_val = P[len(src_keys):len(src_keys)+len(val_keys)]
    P_tst = P[-len(tst_keys):]

    # kNN distance: each test image -> nearest train image (src or val)
    from sklearn.neighbors import NearestNeighbors
    nn_src = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(P_src)
    nn_val = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(P_val)
    d_src, _ = nn_src.kneighbors(P_tst)
    d_val, _ = nn_val.kneighbors(P_tst)

    # per-test: which subset is closer on average (top-1 distance)
    closer_to_src = int((d_src[:, 0] < d_val[:, 0]).sum())
    closer_to_val = int((d_val[:, 0] <= d_src[:, 0]).sum())

    # k-means cluster of ALL images in PCA space, then test distribution over clusters
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=12, n_init=10, random_state=0).fit(P)
    cluster_of = lambda Xp: km.predict(Xp)
    src_cl = cluster_of(P_src)
    val_cl = cluster_of(P_val)
    tst_cl = cluster_of(P_tst)
    src_share = np.bincount(src_cl, minlength=12) / len(src_cl)
    val_share = np.bincount(val_cl, minlength=12) / len(val_cl)
    tst_share = np.bincount(tst_cl, minlength=12) / len(tst_cl)

    # target-similarity source weights: for each source image, count how many test
    # images share its cluster; weight = soft count
    src_weight = np.zeros(len(src_keys))
    for t in tst_cl:
        src_weight[src_cl == t] += 1
    src_weight = src_weight / (src_weight.sum() + 1e-8)

    weight_manifest = [
        {"stem": src_keys[i][4:], "weight": round(float(src_weight[i]), 8)}
        for i in np.argsort(-src_weight)[:200]
    ]

    report = {
        "scope": "train_test_domain_analysis",
        "n_src": len(src_keys), "n_val": len(val_keys), "n_tst": len(tst_keys),
        "pca_explained_variance_ratio": [round(float(x), 6)
                                         for x in pca.explained_variance_ratio_[:10]],
        "test_closer_to_src": closer_to_src,
        "test_closer_to_val": closer_to_val,
        "cluster_share_src": [round(float(x), 4) for x in src_share],
        "cluster_share_val": [round(float(x), 4) for x in val_share],
        "cluster_share_tst": [round(float(x), 4) for x in tst_share],
        "top200_target_similar_sources": weight_manifest[:200],
        "note": "weights are coverage counts (how many test images share the PCA cluster); "
                "NOT nearest-neighbour label prediction",
    }
    (OUT / "domain_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    (OUT / "target_similarity_weights.json").write_text(
        json.dumps(weight_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k != "top200_target_similar_sources"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
