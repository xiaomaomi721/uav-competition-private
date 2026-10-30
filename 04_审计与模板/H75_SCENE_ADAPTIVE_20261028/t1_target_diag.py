"""Stage-1 diagnostic A: Target3594 behaviour of H75 raw vs Bstar, per subset.

Pure CPU, reads the existing H75 D4 teacher cache (no GPU, no new inference).
No target GT exists -> only prediction-side statistics. Never report target IoU.
"""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path("/home/jhy/aic2026_h76_20261026")
CACHE = ROOT / "code" / "reliable_kd_20261028" / "teacher_d4"
OUT = Path("/home/jhy/aic2026_h75_scene_adaptive_20261028")
BSTAR = np.array([-.2, .1, .0709, .4, 3.6, .4, 2.1215, -.5], dtype=np.float32)
CLS = ["Background", "Building", "Road", "Water", "Barren", "Vegetation", "Agricultural", "Vehicle"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(ROOT / "manifests" / "semifinal_3594.csv")))
    acc = defaultdict(lambda: {
        "n": 0, "raw": np.zeros(8, np.int64), "cal": np.zeros(8, np.int64),
        "flip_px": 0, "px": 0,
        "conf_raw_sum": 0.0, "conf_cal_sum": 0.0,
        "hn_sum": 0.0, "votes_sum": 0.0,
        "degen_raw": 0, "degen_cal": 0,
        # per-class flip counters: how many raw-class-X pixels were moved to each class by Bstar
        "flip_matrix": np.zeros((8, 8), np.int64),
    })

    for r in rows:
        s = r["subset"]
        stem = Path(r["input_name"]).stem
        d = np.load(CACHE / f"{stem}.npz")
        z = d["z_d4"].astype(np.float32)          # (8,256,256) raw logits, no Bstar
        votes = d["votes"].astype(np.float32)
        hn = d["hn"].astype(np.float32)
        flipf = d["flip"].astype(np.uint8)

        raw_arg = z.argmax(0)
        cal = z + BSTAR[:, None, None]
        cal_arg = cal.argmax(0)
        p_cal = torch_softmax(cal)
        conf_cal = p_cal.max(0)

        A = acc[s]
        A["n"] += 1
        A["px"] += raw_arg.size
        A["raw"] += np.bincount(raw_arg.ravel(), minlength=8)
        A["cal"] += np.bincount(cal_arg.ravel(), minlength=8)
        A["flip_px"] += int((raw_arg != cal_arg).sum())
        A["conf_cal_sum"] += float(conf_cal.sum())
        A["hn_sum"] += float(hn.sum())
        A["votes_sum"] += float(votes.sum())
        if np.bincount(raw_arg.ravel(), minlength=8).max() / raw_arg.size > 0.99:
            A["degen_raw"] += 1
        if np.bincount(cal_arg.ravel(), minlength=8).max() / cal_arg.size > 0.99:
            A["degen_cal"] += 1
        m = raw_arg != cal_arg
        if m.any():
            enc = (raw_arg[m] * 8 + cal_arg[m])
            A["flip_matrix"] += np.bincount(enc, minlength=64).reshape(8, 8)
        del d, z, votes, hn, flipf

    out = {}
    for s, A in acc.items():
        tot = A["px"]
        out[s] = {
            "images": A["n"],
            "pixels": int(tot),
            "raw_share_pct": [round(float(x) / tot * 100, 3) for x in A["raw"]],
            "cal_share_pct": [round(float(x) / tot * 100, 3) for x in A["cal"]],
            "bstar_flip_pct": round(A["flip_px"] / tot * 100, 4),
            "mean_conf_calibrated": round(A["conf_cal_sum"] / tot, 5),
            "mean_normalized_entropy": round(A["hn_sum"] / tot, 5),
            "mean_d4_votes": round(A["votes_sum"] / tot, 3),
            "degenerate_raw_images": A["degen_raw"],
            "degenerate_cal_images": A["degen_cal"],
            "flip_matrix_raw_to_cal": A["flip_matrix"].tolist(),
        }
    (OUT / "T1_TARGET_DIAG.json").write_text(json.dumps({"class_order": CLS, "per_subset": out},
                                                        indent=2), encoding="utf-8")

    with open(OUT / "T1_TARGET_SUBSET_TABLE.csv", "w", encoding="utf-8") as f:
        f.write("subset,images,bstar_flip_pct,mean_conf_cal,mean_norm_entropy,mean_d4_votes,"
                "degen_raw,degen_cal," + ",".join("raw_" + c for c in CLS) + ","
                + ",".join("cal_" + c for c in CLS) + "\n")
        for s, v in out.items():
            f.write(f'{s},{v["images"]},{v["bstar_flip_pct"]},{v["mean_conf_calibrated"]},'
                    f'{v["mean_normalized_entropy"]},{v["mean_d4_votes"]},'
                    f'{v["degenerate_raw_images"]},{v["degenerate_cal_images"]},'
                    + ",".join(str(x) for x in v["raw_share_pct"]) + ","
                    + ",".join(str(x) for x in v["cal_share_pct"]) + "\n")

    for s, v in out.items():
        print(s, json.dumps({k: v[k] for k in ("images", "bstar_flip_pct", "mean_conf_calibrated",
                                               "mean_normalized_entropy", "mean_d4_votes",
                                               "degenerate_cal_images")}, ensure_ascii=False))
        print("   raw:", v["raw_share_pct"])
        print("   cal:", v["cal_share_pct"])


def torch_softmax(x):
    x = x - x.max(0, keepdims=True)
    e = np.exp(x)
    return e / e.sum(0, keepdims=True)


if __name__ == "__main__":
    main()
