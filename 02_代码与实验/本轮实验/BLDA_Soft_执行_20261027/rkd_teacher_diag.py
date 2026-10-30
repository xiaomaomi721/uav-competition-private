"""Teacher reliability diagnosis on source1399 (has GT).

Compares, for the SAME H75 weights and SAME official GT mapping:
  weak  = single-view raw logits
  d4    = D4-TTA8 aligned raw logits mean
  cal   = d4 + Bstar
Reports per-class IoU, raw->cal flips (corrected vs harmed), and accuracy
bucketed by agreement `a`, normalized entropy `H`, and raw/cal disagreement.

This measures whether the teacher upgrade adds RELIABLE information.
It does NOT measure target-domain accuracy (no GT there).
"""
import argparse
from rkd_common import *

OUT = WORK / "teacher_diag"
SRC_IMG = SOURCE_DATA / "train" / "images"
SRC_MSK = SOURCE_DATA / "train" / "masks"
CLS = ["Background", "Building", "Road", "Water", "Barren", "Vegetation", "Agricultural", "Vehicle"]


def val_ids():
    d = json.loads((SRC / "splits" / "SOURCE_AUDIT_SPLIT.json").read_text())
    ids = d["H300"] + d["R1099"]
    assert len(ids) == 1399
    return ids


def upsample_pred(z, bias):
    if bias is not None:
        z = z + torch.tensor(bias, device=z.device)[None, :, None, None]
    return F.interpolate(z, size=(1024, 1024), mode="bilinear", align_corners=False).argmax(1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    guard(); seed_all()
    OUT.mkdir(parents=True, exist_ok=True)
    ids = val_ids()
    if a.limit:
        ids = ids[: a.limit]

    from src.label_mapping import official_to_train
    model = model_from()

    cm = {k: np.zeros((8, 8), dtype=np.int64) for k in ("weak_raw", "weak_cal", "d4_raw", "d4_cal")}
    flip = {"raw2cal_d4": 0, "corrected": 0, "harmed": 0}
    bucket = {k: {"n": 0, "correct": 0} for k in
              ("a_lo", "a_hi", "h_lo", "h_hi", "flip_px", "noflip_px")}
    t0 = time.time()
    for i, stem in enumerate(ids, 1):
        if i % 50 == 1:
            guard()
        with Image.open(SRC_IMG / f"{stem}.png") as f:
            rgb = np.asarray(f.convert("RGB")).copy()
        with Image.open(SRC_MSK / f"{stem}.png") as f:
            gt = official_to_train(np.asarray(f).copy())
        gt_t = torch.from_numpy(gt.astype(np.int64)).cuda()[None]
        gt1024 = F.interpolate(gt_t[:, None].float(), size=(1024, 1024), mode="nearest")[:, 0].long()
        valid = gt1024 != 255

        x = normalize_to_tensor(rgb).unsqueeze(0).cuda()
        with torch.inference_mode():
            z_single = model(pixel_values=x).logits.float()
        zT, votes, hn, flipmap = d4_stack_with_votes(model, x)
        zT = zT.float()
        votes = votes.float()
        hn = hn.float()
        flipmap = flipmap.float()

        pw_raw = upsample_pred(z_single, None)
        pw_cal = upsample_pred(z_single, BSTAR)
        pd_raw = upsample_pred(zT, None)
        pd_cal = upsample_pred(zT, BSTAR)

        for name, pred in (("weak_raw", pw_raw), ("weak_cal", pw_cal),
                           ("d4_raw", pd_raw), ("d4_cal", pd_cal)):
            t = gt1024[0]
            p = pred
            v = valid[0]
            enc = (t[v] * 8 + p[v]).cpu().numpy()
            cm[name] += np.bincount(enc, minlength=64).reshape(8, 8)

        # flips on d4 raw -> cal
        t = gt1024[0]; v = valid[0]
        ch = (pd_raw != pd_cal) & v
        n_ch = int(ch.sum())
        flip["raw2cal_d4"] += n_ch
        flip["corrected"] += int(((pd_raw != t) & (pd_cal == t) & v).sum())
        flip["harmed"] += int(((pd_raw == t) & (pd_cal != t) & v).sum())

        # bucket accuracy of d4_cal
        def up(z3):
            return F.interpolate(z3.unsqueeze(0).unsqueeze(0), size=(1024, 1024),
                                 mode="bilinear", align_corners=False)[0, 0]
        a_up = up(votes[0])
        h_up = up(hn[0])
        f_up = up(flipmap[0])
        correct = (pd_cal == t) & v
        a_hi = a_up > 0.99
        if bool(v.any()):
            h_hi = h_up < h_up[v].median()
        else:
            h_hi = torch.zeros_like(h_up, dtype=torch.bool)
        for key, mask in (("a_lo", v & ~a_hi), ("a_hi", v & a_hi),
                          ("h_lo", v & ~h_hi), ("h_hi", v & h_hi),
                          ("flip_px", v & (f_up > 0.5)), ("noflip_px", v & (f_up <= 0.5))):
            bucket[key]["n"] += int(mask.sum())
            bucket[key]["correct"] += int((correct & mask).sum())

        if i % 100 == 0 or i == len(ids):
            emit(stage="diag", done=i, total=len(ids), elapsed=round(time.time() - t0, 1))

    res = {}
    for name, m in cm.items():
        inter = np.diag(m).astype(np.float64)
        union = m.sum(1) + m.sum(0) - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
        res[name] = {"mIoU": float(np.nanmean(iou) * 100),
                     "per_class": [round(float(v) * 100, 4) if np.isfinite(v) else None for v in iou]}
    res["flips"] = flip
    res["buckets"] = {k: {"n": v["n"], "acc": round(v["correct"] / max(1, v["n"]) * 100, 4)}
                      for k, v in bucket.items()}
    res["n_images"] = len(ids)
    res["classes"] = CLS
    dump(OUT / "TEACHER_DIAG.json", res)
    emit(stage="diag_complete", **{k: res[k]["mIoU"] for k in ("weak_raw", "weak_cal", "d4_raw", "d4_cal")})
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
