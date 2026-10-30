"""Target-domain statistics for H75 / C0 / C1 on the 3594 unlabeled test images.

NO target GT exists. This only reports prediction-side statistics:
  - per-subset class pixel shares
  - per-image change rate vs H75
  - per-class transition matrix H75 -> candidate
  - D4 view agreement distribution
  - degenerate image counts (single class covering >99%)
"""
import argparse
from rkd_common import *

CLS = ["Background", "Building", "Road", "Water", "Barren", "Vegetation", "Agricultural", "Vehicle"]


def evaluate_model(ckpt, tag, rows, limit=0):
    model = model_from(ckpt).eval()
    if limit:
        rows = rows[:limit]
    shares = {g: np.zeros(8, dtype=np.int64) for g in ("test1", "test2", "test3_new")}
    per_image_pred = {}
    degenerate = {g: 0 for g in shares}
    agree_hist = np.zeros(9, dtype=np.int64)
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        if i % 50 == 1:
            guard()
        x = normalize_to_tensor(read_rgb(TARGET_IMAGES / r["input_name"])).unsqueeze(0).cuda()
        zT, votes, hn, flip = d4_stack_with_votes(model, x)
        pred = logits_to_png(zT, bias=True)[0].cpu().numpy()
        cnt = np.bincount(pred.reshape(-1), minlength=8)
        shares[r["subset"]] += cnt
        per_image_pred[r["input_name"]] = pred
        if (cnt.max() / cnt.sum()) > 0.99:
            degenerate[r["subset"]] += 1
        agree_hist += np.bincount(votes[0].cpu().numpy().reshape(-1), minlength=9)
        if i % 300 == 0:
            emit(stage="stats", tag=tag, done=i, total=len(rows), elapsed=round(time.time() - t0, 1))
    out = {
        "tag": tag, "images": len(rows),
        "class_order": CLS,
        "shares_pct": {g: [round(float(v) / max(1, s.sum()) * 100, 4) for v in s]
                       for g, s in shares.items()},
        "degenerate_single_class_images": degenerate,
        "d4_votes_histogram_0to8": agree_hist.tolist(),
        "runtime_sec": round(time.time() - t0, 1),
    }
    return out, per_image_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="tag=path pairs, e.g. H75=/path/H75.pt")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    guard(); seed_all()
    rows = target_rows()
    results = {}
    preds = {}
    for spec in a.checkpoints:
        tag, path = spec.split("=", 1)
        res, p = evaluate_model(path, tag, rows, a.limit)
        results[tag] = res
        preds[tag] = p
    # transitions vs the first (reference) tag
    ref = a.checkpoints[0].split("=", 1)[0]
    for tag, p in preds.items():
        if tag == ref:
            continue
        trans = np.zeros((8, 8), dtype=np.int64)
        changed = 0
        total = 0
        for name, pr in p.items():
            if name not in preds[ref]:
                continue
            q = preds[ref][name]
            enc = (q * 8 + pr).reshape(-1)
            trans += np.bincount(enc, minlength=64).reshape(8, 8)
            changed += int((q != pr).sum())
            total += q.size
        results[tag]["change_vs_" + ref] = {
            "pixel_change_pct": round(changed / max(1, total) * 100, 4),
            "transition_matrix_rows_pred_" + ref: trans.tolist(),
        }
    dump(WORK / "target_stats" / "TARGET_STATS.json", results)
    print(json.dumps(results, indent=2, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
