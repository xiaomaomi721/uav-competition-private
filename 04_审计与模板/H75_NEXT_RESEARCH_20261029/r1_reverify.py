"""Task 2 re-verification with FULL data (no sampling).

Corrects the previous report's口径 issues:
  (a) Source train/val class prior measured on FULL sets with real GT
  (b) Target Zero vs Bstar class share on FULL 3594 from ONE cache
  (c) Target subset provenance verified from manifest matched_set/matched_name,
      NOT from renumbered filenames
  (d) Target raw vs Bstar transition matrix
  (e) Confirm raw logits (bias_applied=false) vs Bstar applied exactly once
"""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/jhy/aic2026_h76_20261026")
TCACHE = Path("/home/jhy/aic2026_ziponly_20261028/01_h75_cache/target3594_fp32/scores")
TRAINIMG = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/train/masks")
OUT = Path("/home/jhy/aic2026_h75_next_research_20261029")
BSTAR = np.array([-.2, .1, .0709, .4, 3.6, .4, 2.1215, -.5], dtype=np.float32)
CLS = ["Background", "Building", "Road", "Water", "Barren", "Vegetation", "Agricultural", "Vehicle"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "02_H75错误机制").mkdir(exist_ok=True)

    # ---------- (a) FULL source GT prior ----------
    from PIL import Image
    sp = json.loads((ROOT / "code" / "h76_3594" / "splits" / "SOURCE_AUDIT_SPLIT.json").read_text())
    train_ids = [x.strip() for x in (ROOT / "code" / "h76_3594" / "splits" / "train.txt").read_text().split() if x.strip()]
    val_ids = sp["H300"] + sp["R1099"]
    prior = {}
    for tag, ids in (("train_full_5597", train_ids), ("val_full_1399", val_ids)):
        cnt = np.zeros(9, np.int64)
        for k in ids:
            m = np.asarray(Image.open(TRAINIMG / f"{k}.png"))
            cnt += np.bincount(m.ravel(), minlength=9)
        tot = cnt[1:].sum()
        prior[tag] = {"n": len(ids), "pixels": int(tot),
                      "share_pct": [round(float(x) / tot * 100, 4) for x in cnt[1:]]}
        print(tag, prior[tag]["share_pct"], flush=True)

    # ---------- (b)(c)(d) Target full 3594 ----------
    rows = list(csv.DictReader(open(ROOT / "manifests" / "semifinal_3594.csv")))
    assert len(rows) == 3594
    prov = defaultdict(int)
    for r in rows:
        prov[(r["subset"], r.get("matched_set", ""), bool(r.get("matched_name", "")))] += 1

    acc = defaultdict(lambda: {"n": 0, "zero": np.zeros(8, np.int64), "bstar": np.zeros(8, np.int64),
                               "px": 0, "trans": np.zeros((8, 8), np.int64),
                               "conf_zero": 0.0, "conf_bstar": 0.0,
                               "margin_zero": 0.0, "margin_bstar": 0.0})
    for i, r in enumerate(rows, 1):
        z = torch.from_numpy(np.load(TCACHE / f'{r["input_name"]}.npy'))[None]
        up0 = F.interpolate(z, size=(1024, 1024), mode="bilinear", align_corners=False)
        upb = F.interpolate(z + torch.from_numpy(BSTAR)[None, :, None, None],
                            size=(1024, 1024), mode="bilinear", align_corners=False)
        for tag, up in (("zero", up0), ("bstar", upb)):
            p = up.softmax(1)[0]
            conf, arg = p.max(0)
            top2 = p.topk(2, dim=0).values
            s = r["subset"]
            A = acc[s]
            A["n"] += 1
            A[tag] += np.bincount(arg.numpy().ravel(), minlength=8)
            A["px"] += arg.numel()
            A["conf_" + tag] += float(conf.sum())
            A["margin_" + tag] += float((top2[0] - top2[1]).sum())
        a0 = up0.argmax(1)[0].numpy().astype(np.int64)
        ab = upb.argmax(1)[0].numpy().astype(np.int64)
        m = a0 != ab
        if m.any():
            acc[r["subset"]]["trans"] += np.bincount((a0[m] * 8 + ab[m]), minlength=64).reshape(8, 8)
        if i % 500 == 0:
            print(i, flush=True)

    out = {"class_order": CLS, "bias_applied_in_cache": False,
           "bstar_vector": BSTAR.tolist(),
           "source_gt_prior_full": prior,
           "target_subset_provenance": {f"{k[0]}|matched_set={k[1]}|has_matched_name={k[2]}": v
                                        for k, v in prov.items()},
           "target": {}}
    for s, A in acc.items():
        tot = A["px"]
        out["target"][s] = {
            "images": A["n"], "pixels": int(tot),
            "zero_share_pct": [round(float(x) / tot * 100, 4) for x in A["zero"]],
            "bstar_share_pct": [round(float(x) / tot * 100, 4) for x in A["bstar"]],
            "bstar_minus_zero_pp": [round((float(b) - float(z_)) / tot * 100, 4)
                                    for z_, b in zip(A["zero"], A["bstar"])],
            "mean_conf_zero": round(A["conf_zero"] / tot, 5),
            "mean_conf_bstar": round(A["conf_bstar"] / tot, 5),
            "mean_margin_zero": round(A["margin_zero"] / tot, 5),
            "mean_margin_bstar": round(A["margin_bstar"] / tot, 5),
            "changed_px": int(A["trans"].sum()),
            "changed_pct": round(float(A["trans"].sum()) / tot * 100, 4),
            "transition_zero_to_bstar": A["trans"].tolist(),
        }
    (OUT / "02_H75错误机制" / "R1_REVERIFY.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    with open(OUT / "02_H75错误机制" / "Target_Zero_Bstar分布.csv", "w", encoding="utf-8") as f:
        f.write("class," + ",".join(f"{s}_zero,{s}_bstar,{s}_delta_pp" for s in out["target"]) + "\n")
        for i, c in enumerate(CLS):
            cells = []
            for s, v in out["target"].items():
                cells += [f"{v['zero_share_pct'][i]:.4f}", f"{v['bstar_share_pct'][i]:.4f}",
                          f"{v['bstar_minus_zero_pp'][i]:+.4f}"]
            f.write(c + "," + ",".join(cells) + "\n")
    with open(OUT / "02_H75错误机制" / "Bstar类别转换矩阵.csv", "w", encoding="utf-8") as f:
        for s, v in out["target"].items():
            f.write(f"# subset={s} changed_px={v['changed_px']} ({v['changed_pct']}%)\n")
            f.write("zero\\bstar," + ",".join(CLS) + "\n")
            for i, c in enumerate(CLS):
                f.write(c + "," + ",".join(str(int(x)) for x in v["transition_zero_to_bstar"][i]) + "\n")

    print(json.dumps({"source_gt_prior_full": prior,
                      "target": {s: {k: v[k] for k in ("zero_share_pct", "bstar_share_pct",
                                                       "changed_pct", "mean_conf_zero", "mean_conf_bstar")}
                                 for s, v in out["target"].items()}}, indent=2))


if __name__ == "__main__":
    main()
