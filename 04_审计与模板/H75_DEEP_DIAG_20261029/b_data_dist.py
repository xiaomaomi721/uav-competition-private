"""Task B: Source vs Target data distribution analysis (CPU only, sampled).

Groups:
  source_train  (5597) sample 600
  source_val    (1399) sample 600
  target_test1  (500)  sample 300
  target_test2  (1300) sample 300
  target_test3_new (1794) sample 300

Per group: RGB channel mean/std, brightness, RMS contrast, Laplacian-variance sharpness,
colorfulness; plus 32x32 grayscale thumbnail hash for near-duplicate detection
between source_train and source_val.

GT class distribution for source groups (real labels).
Predicted class distribution for target groups (from H75 logits cache; NOT ground truth).
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path("/home/jhy/aic2026_h76_20261026")
SRCIMG = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/train/images")
SRCMASK = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/train/masks")
TCACHE = Path("/home/jhy/aic2026_ziponly_20261028/01_h75_cache/target3594_fp32/scores")
OUT = Path("/home/jhy/aic2026_h75_deep_diag_20261029")
CLS = ["Background", "Building", "Road", "Water", "Barren", "Vegetation", "Agricultural", "Vehicle"]
BSTAR = np.array([-.2, .1, .0709, .4, 3.6, .4, 2.1215, -.5], dtype=np.float32)
N_SRC, N_TGT = 600, 300
rng = np.random.default_rng(20261029)


def img_stats(rgb):
    a = rgb.astype(np.float32)
    g = a.mean(2)
    lap = (np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])).mean()
    rg = a[:, :, 0] - a[:, :, 1]; yb = 0.5 * (a[:, :, 0] + a[:, :, 1]) - a[:, :, 2]
    colorful = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return {"r_mean": a[:, :, 0].mean(), "g_mean": a[:, :, 1].mean(), "b_mean": a[:, :, 2].mean(),
            "r_std": a[:, :, 0].std(), "g_std": a[:, :, 1].std(), "b_std": a[:, :, 2].std(),
            "brightness": g.mean(), "contrast_rms": g.std(),
            "sharpness_lapvar": float(lap), "colorfulness": float(colorful)}


def thumb(rgb):
    g = Image.fromarray(rgb).convert("L").resize((32, 32), Image.BILINEAR)
    v = np.asarray(g, dtype=np.float32)
    v = v - v.mean()
    n = np.linalg.norm(v)
    return (v / n) if n > 0 else v


def agg(rows):
    keys = rows[0].keys()
    return {k: {"mean": round(float(np.mean([r[k] for r in rows])), 3),
                "std": round(float(np.std([r[k] for r in rows])), 3)} for k in keys}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "02_DATA_DISTRIBUTION").mkdir(exist_ok=True)
    sp = json.loads((ROOT / "code" / "h76_3594" / "splits" / "SOURCE_AUDIT_SPLIT.json").read_text())
    train_ids = [x.strip() for x in (ROOT / "code" / "h76_3594" / "splits" / "train.txt").read_text().split() if x.strip()]
    val_ids = sp["H300"] + sp["R1099"]
    import csv
    trows = list(csv.DictReader(open(ROOT / "manifests" / "semifinal_3594.csv")))
    by = defaultdict(list)
    for r in trows:
        by[r["subset"]].append(r)

    groups = {
        "source_train": ("src", train_ids),
        "source_val": ("src", val_ids),
        "target_test1": ("tgt", [r["input_name"] for r in by["test1"]]),
        "target_test2": ("tgt", [r["input_name"] for r in by["test2"]]),
        "target_test3_new": ("tgt", [r["input_name"] for r in by["test3_new"]]),
    }

    stats_out, gt_shares, pred_shares, thumbs = {}, {}, {}, {}
    for g, (kind, ids) in groups.items():
        n = N_SRC if kind == "src" else N_TGT
        sel = [ids[i] for i in rng.choice(len(ids), size=min(n, len(ids)), replace=False)]
        rows = []
        cnt_gt = np.zeros(8, np.int64)
        cnt_pred = np.zeros(8, np.int64)
        tot_gt = tot_pred = 0
        for name in sel:
            if kind == "src":
                rgb = np.asarray(Image.open(SRCIMG / f"{name}.png").convert("RGB")).copy()
                rows.append(img_stats(rgb))
                mk = np.asarray(Image.open(SRCMASK / f"{name}.png"))
                v = mk > 0
                if v.any():
                    cnt_gt += np.bincount(mk[v].astype(np.int64) - 1, minlength=8)
                    tot_gt += int(v.sum())
            else:
                p = ROOT / "data" / "test3" / "images" / name
                rgb = np.asarray(Image.open(p).convert("RGB")).copy()
                rows.append(img_stats(rgb))
                stem = Path(name).stem
                f = TCACHE / f"{stem}.npy"
                if f.exists():
                    z = torch.from_numpy(np.load(f))[None]
                    up = F.interpolate(z + torch.from_numpy(BSTAR)[None, :, None, None],
                                       size=(1024, 1024), mode="bilinear", align_corners=False)
                    pr = up.argmax(1).numpy().ravel()
                    cnt_pred += np.bincount(pr, minlength=8)
                    tot_pred += pr.size
            if g == "source_train" or g == "source_val":
                thumbs.setdefault(g, []).append(thumb(rgb))
        stats_out[g] = {"n_sampled": len(sel), **agg(rows)}
        if tot_gt:
            gt_shares[g] = [round(float(x) / tot_gt * 100, 3) for x in cnt_gt]
        if tot_pred:
            pred_shares[g] = [round(float(x) / tot_pred * 100, 3) for x in cnt_pred]
        print(g, len(sel), flush=True)

    # near-duplicate: source_train vs source_val thumbnails
    near = []
    if "source_train" in thumbs and "source_val" in thumbs:
        A = np.stack(thumbs["source_train"]); Bv = np.stack(thumbs["source_val"])
        sim = Bv @ A.T                      # (nv, nt) cosine
        mx = sim.max(1)
        near = {"n_val_sampled": int(len(Bv)), "n_train_sampled": int(len(A)),
                "cos_gt_0.99": int((mx > 0.99).sum()), "cos_gt_0.95": int((mx > 0.95).sum()),
                "cos_gt_0.90": int((mx > 0.90).sum()),
                "cos_mean": round(float(mx.mean()), 4), "cos_p95": round(float(np.percentile(mx, 95)), 4),
                "note": "32x32 grayscale normalized-thumbnail cosine; near-duplicate proxy, not exact-byte identity"}

    res = {"class_order": CLS, "image_stats": stats_out,
           "source_gt_class_share_pct": gt_shares,
           "target_pred_class_share_pct": pred_shares,
           "target_pred_note": "PREDICTED distribution only; Target has no ground truth, never read as class prior",
           "source_train_val_thumbnail_similarity": near}
    (OUT / "02_DATA_DISTRIBUTION" / "B_DATA_DIST.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    with open(OUT / "02_DATA_DISTRIBUTION" / "Source_Target分布差异.csv", "w", encoding="utf-8") as f:
        keys = list(stats_out["source_train"].keys())
        keys = [k for k in keys if k != "n_sampled"]
        f.write("metric," + ",".join(groups.keys()) + "\n")
        for k in keys:
            f.write(k + "," + ",".join(f'{stats_out[g][k]["mean"]:.3f}' for g in groups) + "\n")
        f.write("\n# GT class share (source only, from real labels)\n")
        f.write("class," + ",".join(gt_shares.keys()) + "\n")
        for i, c in enumerate(CLS):
            f.write(c + "," + ",".join(f"{gt_shares[g][i]:.3f}" for g in gt_shares) + "\n")
        f.write("\n# PREDICTED class share (target only, NOT ground truth)\n")
        f.write("class," + ",".join(pred_shares.keys()) + "\n")
        for i, c in enumerate(CLS):
            f.write(c + "," + ",".join(f"{pred_shares[g][i]:.3f}" for g in pred_shares) + "\n")

    print(json.dumps(near, indent=2))
    print(json.dumps({"gt_shares": gt_shares, "pred_shares": pred_shares}, indent=1))


if __name__ == "__main__":
    main()
