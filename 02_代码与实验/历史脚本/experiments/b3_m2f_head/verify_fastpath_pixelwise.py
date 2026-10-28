"""True pixelwise fast-path vs full-8 verification over ALL pixels.

The C2 report's `FASTPATH_VS_FULL8_DISAGREEMENTS` counts CANDIDATES whose mIoU
differs. That is only a candidate-level mIoU diagnostic and does NOT establish
implementation equivalence: two different per-pixel predictions can coincidentally
produce the same mIoU. It is not an exactness gate.

This script computes the real quantities, over every candidate and every pixel:

    M_all   = sum 1[ pred_fast != pred_full ]                     (all pixels)
    M_valid = sum 1[ GT != 0 and pred_fast != pred_full ]         (valid pixels)

M_all is the correct gate for the statement "the fast path is identical to the
full 8-channel argmax": the prediction algorithm does not know whether GT is
ignore, and the cache is already known to contain a genuine tie inside an ignore
region. M_valid is reported alongside because it bounds the effect on any mIoU.

INDEPENDENCE (why this is a real check and not a tautology)
  - The candidate (class, bias) values are READ from C2's frozen result file.
    They are never regenerated from a quantile or a tag, so the nextafter-based
    thresholds are bit-identical to the ones C2 actually swept.
  - The full path uses torch.argmax directly on (S + bias) with no shared code.
  - The fast path is re-derived here from the raw 8-channel scores: take the max
    and the lowest-index argmax over the seven non-target channels, then apply
    s_c + b > o_v, or (s_c + b == o_v and c < o_a).
  - Nothing is imported from c2_search.py, so a bug in C2's helper cannot make
    both sides wrong together.

On the first disagreement the raw record is printed with no interpretation.

Read-only. No training, no packaging.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

SHM = Path("/dev/shm/tta8_56k_v2")
ROOT = Path("/home/jhy/rare_diag_20260915/m2f_gpu_20260915")
C2 = ROOT / "c2_56k" / "C2_RESULT.json"
OUT = ROOT / "c2_56k"
NAMES = ["Background", "Building", "Road", "Water", "Barren", "Vegetation",
         "Agricultural", "Vehicle"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-images", type=int, default=0,
                    help="0 = all images (the real run); >0 for a smoke test")
    ap.add_argument("--synthetic-candidates", action="store_true",
                    help="SMOKE TEST ONLY: build a synthetic candidate grid "
                         "instead of reading C2_RESULT.json, so the mechanics can "
                         "be exercised before C2 has written its result. The real "
                         "run must never pass this flag.")
    ap.add_argument("--out", default="C2_FASTPATH_PIXELWISE.json")
    args = ap.parse_args()
    torch.set_num_threads(int(__import__("os").environ.get("THREADS", "32")))

    stems = sorted(p.stem for p in SHM.glob("*.npy")
                   if not p.stem.startswith("gt_") and p.stem != "_DONE")
    if args.max_images:
        stems = stems[:args.max_images]
    n = len(stems)
    S = [torch.from_numpy(np.load(SHM / f"{s}.npy")) for s in stems]
    GT = [torch.from_numpy(np.load(SHM / f"gt_{s}.npy")).long().reshape(-1)
          for s in stems]
    print(f"loaded {n} images", flush=True)

    if args.synthetic_candidates:
        print("SMOKE MODE: synthetic candidate grid, NOT the C2 result", flush=True)
        cands = []
        for k in range(1, 8):
            for b in (0.0, 0.05, 0.3, -0.05, -0.3):
                cands.append({"class": NAMES[k], "bias": b, "tag": "synthetic"})
    else:
        if not C2.exists():
            raise SystemExit(f"{C2} not found; C2 must finish first (or pass "
                             f"--synthetic-candidates for a smoke test)")
        res = json.loads(C2.read_text())
        cands = res.get("all_candidates", [])
        if not cands:
            raise SystemExit("C2_RESULT.json has no all_candidates")

    # exact (class, bias) pairs read from the FROZEN C2 result -- never
    # regenerated from a quantile or a tag, so nextafter-derived thresholds are
    # bit-identical to the ones C2 actually swept
    raw = []
    for r in cands:
        k = NAMES.index(r["class"])
        b = float(r["bias"])
        raw.append((k, b, r.get("tag")))

    raw_count = len(raw)
    pair_map = {}
    for k, b, tag in raw:
        pair_map.setdefault(k, {})
        if b not in pair_map[k]:
            pair_map[k][b] = {"tags": [], "bias_hex": float(b).hex()}
        pair_map[k][b]["tags"].append(tag)
    n_unique = sum(len(v) for v in pair_map.values())
    n_dup = raw_count - n_unique

    # candidate class coverage must be EXACTLY the seven adjustable classes:
    # a subset test would silently pass if a class were missing entirely
    expected_classes = set(range(1, 8))
    actual_classes = set(k for k, _, _ in raw)
    missing = expected_classes - actual_classes
    extra = actual_classes - expected_classes
    if missing or extra:
        raise SystemExit(
            f"candidate class coverage wrong: "
            f"missing={[NAMES[k] for k in sorted(missing)]}, "
            f"extra={sorted(extra)}")
    if not raw:
        raise SystemExit("no candidates found in the frozen C2 result")

    print(f"raw candidate count (C2 all_candidates): {raw_count}", flush=True)
    print(f"unique (class,bias) pairs: {n_unique}  duplicates collapsed: {n_dup}",
          flush=True)
    print("candidate class coverage: exactly the 7 adjustable classes", flush=True)
    for k in sorted(pair_map):
        nb = len(pair_map[k])
        assert nb > 0, f"{NAMES[k]} has zero unique biases"
        print(f"   {NAMES[k]:13s} {nb} unique biases", flush=True)

    pairs = pair_map
    keep_cache = {k: [j for j in range(8) if j != k] for k in range(1, 8)}
    kt_cache = {k: torch.tensor(keep_cache[k], dtype=torch.int64)
                for k in range(1, 8)}

    M_all = 0
    M_valid = 0
    px_all = 0
    px_valid = 0
    first_bad = None
    per_cand = {(NAMES[k], b): {"class": NAMES[k], "bias": b,
                                "bias_hex": float(b).hex(),
                                "tags": pairs[k][b]["tags"],
                                "mismatch_all_px": 0, "mismatch_valid_px": 0}
                for k in pairs for b in pairs[k]}

    t0 = time.time()
    for i in range(n):
        tf = S[i].reshape(-1, 8)
        g = GT[i]
        valid = g != 0
        n_px = tf.shape[0]
        px_all += n_px
        px_valid += int(valid.sum())
        for k in range(1, 8):
            if k not in pairs:
                continue
            keep = keep_cache[k]
            kt = kt_cache[k]
            r7 = tf[:, keep]
            a = r7.argmax(1)
            ov = r7.gather(1, a.unsqueeze(1)).squeeze(1)
            oa = kt[a]
            for b in sorted(pairs[k]):
                # ONE FP32 scalar used by BOTH paths: no double-vs-fp32 ambiguity
                b32 = torch.tensor(float(b), dtype=torch.float32)
                bb32 = float(b32.item())
                sc = tf[:, k] + b32
                win = (sc > ov) | ((sc == ov) & (k < oa))
                fast = torch.where(win, torch.full_like(oa, k + 1, dtype=torch.int64),
                                   oa.to(torch.int64) + 1)
                bias_vec = torch.zeros(8, dtype=torch.float32)
                bias_vec[k] = b32
                full = (tf + bias_vec).argmax(1).to(torch.int64) + 1
                mism = fast != full
                nb = int(mism.sum())
                nbv = int((mism & valid).sum())
                M_all += nb
                M_valid += nbv
                e = per_cand[(NAMES[k], float(b))]
                e["mismatch_all_px"] += nb
                e["mismatch_valid_px"] += nbv
                if nb and first_bad is None:
                    p = int(torch.nonzero(mism)[0])
                    row = tf[p]
                    # the ACTUAL FP32 biased row, the same object the full path
                    # compares. Everything below derives from it so that a
                    # tie/rounding diagnosis is not polluted by Python doubles.
                    biased_row = (tf + bias_vec)[p]
                    t2 = torch.topk(biased_row, 2).values
                    first_bad = {
                        "stem": stems[i],
                        "image_index": i,
                        "flat_pixel_index": p,
                        "gt_official": int(g[p]),
                        "gt_is_ignore": bool(g[p] == 0),
                        "target_class": NAMES[k],
                        "target_official_id": k + 1,
                        "bias_float": float(b),
                        "bias_hex": float(b).hex(),
                        "bias_as_fp32": bb32,
                        "bias_fp32_hex": b32.item().hex(),
                        "tags": pairs[k][float(b)]["tags"],
                        "score_vector_8d_raw_fp32": [float(x) for x in row],
                        "score_plus_bias_8d": [float(x) for x in biased_row],
                        "fast_winner_official": int(fast[p]),
                        "full_winner_official": int(full[p]),
                        "other_value": float(ov[p]),
                        "other_class_official": int(oa[p]) + 1,
                        "target_biased_score": float(sc[p]),
                        "comparison_is_exact_tie": bool(float(sc[p]) == float(ov[p])),
                        # top2 gap of the BIASED row, i.e. of what full argmax saw
                        "biased_top2_gap": float(t2[0] - t2[1]),
                        "biased_row_argmax_official": int(
                            biased_row.argmax().item()) + 1,
                    }
                    del biased_row, t2
                del b32, sc, win, fast, bias_vec, full, mism
            del r7, a, ov, oa
        del tf, g, valid
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{n}  {el:.0f}s  M_all={M_all} M_valid={M_valid} "
                  f"(eta {el/(i+1)*(n-i-1):.0f}s)", flush=True)

    verdict = "PASS" if (M_all == 0 and M_valid == 0) else "FAIL"
    max_all = max(e["mismatch_all_px"] for e in per_cand.values())
    max_valid = max(e["mismatch_valid_px"] for e in per_cand.values())
    print(f"IMAGES_VERIFIED {n}", flush=True)
    print(f"PIXELS_COMPARED all={px_all} valid={px_valid}", flush=True)
    print(f"TOTAL_ALL_PIXEL_MISMATCH {M_all}", flush=True)
    print(f"TOTAL_VALID_PIXEL_MISMATCH {M_valid}", flush=True)
    print(f"MAX_MISMATCH_PER_CANDIDATE all={max_all} valid={max_valid}", flush=True)
    print(f"VERDICT {verdict}", flush=True)
    if first_bad:
        print("FIRST_DISAGREEMENT_RAW (no interpretation):", flush=True)
        print(json.dumps(first_bad, indent=2), flush=True)

    out = {
        "schema": "M2F_56K_C2_FASTPATH_VS_FULL8_PIXELWISE_V2",
        "criterion": "TOTAL_ALL_PIXEL_MISMATCH == 0 and TOTAL_VALID_PIXEL_MISMATCH == 0",
        "candidate_source": "exact (class,bias) values read from C2_RESULT.json "
                            "all_candidates; never regenerated from quantile/tag",
        "independence": "full = torch.argmax(S+bias); fast re-derived from raw "
                        "scores; nothing imported from c2_search.py",
        "images_verified": n,
        "pixels_compared_all": px_all,
        "pixels_compared_valid": px_valid,
        "total_all_pixel_mismatch": M_all,
        "total_valid_pixel_mismatch": M_valid,
        "max_mismatch_per_candidate_all": max_all,
        "max_mismatch_per_candidate_valid": max_valid,
        "verdict": verdict,
        "n_candidates": n_unique,
        "per_candidate": list(per_cand.values()),
        "first_disagreement_raw": first_bad,
        "note": "the C2 report field FASTPATH_VS_FULL8_DISAGREEMENTS is a "
                "candidate-level mIoU diagnostic only and is NOT an exactness gate",
    }
    (OUT / args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False,
                                           allow_nan=False))
    print("PIXELWISE_CHECK_DONE", flush=True)


if __name__ == "__main__":
    main()
