"""Stage C2: event-count-driven 1D bias scan for M2F 56k, TTA8 FP32 space.

WHY THIS REPLACES STAGE C
  Stage C used a fixed 9-point grid on half[k] = (score q99 - q01)/2 ~= 0.4997,
  i.e. step 0.125. Measured FN tau quantiles show q01 ~= 0.011 while q50 ~= 0.63,
  a ~58x spread, so a single fixed step cannot serve both ends: 0.125 flips a
  handful of pixels at the low end and never reaches the median flip point.
  Result was ~0 gain on all seven classes, which is a statement about the grid,
  not about calibration.

MEASURED SWEEP DESIGN (union of four generators, deduped per class)
  pos  b = nextafter(FN tau quantile for rate r, +inf)   r in FN_RATES
  neg  b = nextafter(FP tau quantile for rate r, -inf)   r in FP_RATES
  both b such that the flipped fraction of ALL valid px reaches a target
  near b in NEAR_ZERO fine grid around 0
  The flip condition is the STRICT inequality, so nextafter moves one ULP past
  the boundary instead of sitting exactly on it.

EVENT COUNTS are derived analytically, not by re-scanning images:
  corrected_FN(b>0) = #{FN tau in [0, b)}       exact, from the sorted FN array
  created_FP(b>0)   = #{TN tau in [0, b)}       from the fine TN histogram
  removed_FP(b<0)   = #{FP tau in (b, 0]}       exact, from the sorted FP array
  destroyed_TP(b<0) = #{TP tau in (b, 0]}       from the fine TP histogram
  The TN/TP histograms are 1e-5 wide over [-1.5, 1.5], so for |b| >= 1e-3 the
  count error is <= 1%. FN/FP counts are exact.

SCORING
  The sweep uses the exact single-class fast path, which was verified pixel-exact
  against full 8-channel argmax (three self-checks, 0 mismatch in 67M pixels).
  EVERY reported candidate (top-20 overall + all per-class bests + the whole
  near-zero grid) is then RE-SCORED with full 8-channel FP32 exact argmax, and
  both numbers are reported so any disagreement is visible.

No training, no weight change, no packaging, no coordinate descent, no joint search.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

SHM = Path("/dev/shm/tta8_56k_v2")
ARRS = Path("/dev/shm/boundary_56k_v2")
ROOT = Path("/home/jhy/rare_diag_20260915/m2f_gpu_20260915")
OUT = ROOT / "c2_56k"
NAMES = ["Background", "Building", "Road", "Water", "Barren", "Vegetation",
         "Agricultural", "Vehicle"]
FN_RATES = [0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.25]
FP_RATES = [0.005, 0.01, 0.025, 0.05, 0.10, 0.25]
GLOBAL_RATES = [0.0001, 0.00025, 0.0005, 0.001, 0.002, 0.005]
NEAR_ZERO = [0.0, 0.001, -0.001, 0.002, -0.002, 0.005, -0.005,
             0.01, -0.01, 0.02, -0.02, 0.05, -0.05]
HLO, HHI, HBIN = -1.5, 1.5, 1e-5
NEDGES = int(round((HHI - HLO) / HBIN)) + 1
torch.set_num_threads(32)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    stems = sorted(p.stem for p in SHM.glob("*.npy")
                   if not p.stem.startswith("gt_") and p.stem != "_DONE")
    n = len(stems)
    assert n == 1399, n
    S = [torch.from_numpy(np.load(SHM / f"{s}.npy")) for s in stems]
    GT = [torch.from_numpy(np.load(SHM / f"gt_{s}.npy")).long() for s in stems]
    H, W = S[0].shape[:2]
    print(f"loaded {n} maps", flush=True)

    # ---------- phase 1: fine TN/TP histograms ----------
    edges = np.linspace(HLO, HHI, NEDGES)
    hist = {k: {"TN": np.zeros(NEDGES - 1, dtype=np.int64),
                "TP": np.zeros(NEDGES - 1, dtype=np.int64)} for k in range(1, 8)}
    t0 = time.time()
    for i in range(n):
        s = S[i].reshape(-1, 8)
        g = GT[i].reshape(-1)
        v = g != 0
        s, g = s[v], g[v]
        a1 = s.argmax(1)
        m1 = s.gather(1, a1.unsqueeze(1)).squeeze(1)
        sm = s.clone()
        sm.scatter_(1, a1.unsqueeze(1), float("-inf"))
        m2 = sm.gather(1, sm.argmax(1).unsqueeze(1)).squeeze(1)
        for k in range(1, 8):
            tau = torch.where(a1 == k, m2, m1).double() - s[:, k].double()
            is_c = (a1 == k)
            gt_c = (g == (k + 1))
            for grp, mask in (("TN", ~gt_c & ~is_c), ("TP", gt_c & is_c)):
                if bool(mask.any()):
                    hist[k][grp] += np.histogram(tau[mask].numpy(), bins=edges)[0]
        if (i + 1) % 200 == 0:
            print(f"  hist {i+1}/{n} {time.time()-t0:.0f}s", flush=True)
    print(f"histograms done in {time.time()-t0:.0f}s", flush=True)

    def hist_count(k, grp, lo, hi):
        """#{tau in [lo, hi)} for the TN or TP group of class k (bin-limited)."""
        a = max(int(np.floor((lo - HLO) / HBIN)), 0)
        b = min(int(np.ceil((hi - HLO) / HBIN)), NEDGES - 1)
        return int(hist[k][grp][a:b].sum()) if b > a else 0

    TAU_FN = {k: np.load(ARRS / f"tau_FN_{NAMES[k]}.npy") for k in range(1, 8)}
    TAU_FP = {k: np.load(ARRS / f"tau_FP_{NAMES[k]}.npy") for k in range(1, 8)}
    VALID = int(sum(int((g.reshape(-1) != 0).sum()) for g in GT))

    def n_below(arr, v):
        """#{x in arr : x < v} (arr sorted ascending)."""
        return int(np.searchsorted(arr, v, side="left"))

    def events(k, b):
        """Analytic event counts for the single-class move b_k = b."""
        fnA, fpA = TAU_FN[k], TAU_FP[k]
        if b > 0:
            return {"corrected_FN": n_below(fnA, b),
                    "created_FP": hist_count(k, "TN", 0.0, b),
                    "removed_FP": n_below(fpA, 0.0),
                    "destroyed_TP": 0,
                    "created_FP_note": "from 1e-5 TN histogram",
                    "removed_FP_note": "b>0 removes no FP; value shown is FP with tau<0"}
        return {"corrected_FN": 0,
                "created_FP": 0,
                "removed_FP": fpA.size - int(np.searchsorted(fpA, b, side="left")),
                "destroyed_TP": hist_count(k, "TP", b, 0.0),
                "destroyed_TP_note": "from 1e-5 TP histogram"}

    # ---------- phase 2: reference prediction and fast-path machinery ----------
    def build_ref(k):
        keep = [j for j in range(8) if j != k]
        kt = torch.tensor(keep, dtype=torch.int64)
        ov = torch.empty(n, H * W, dtype=torch.float32)
        oa = torch.empty(n, H * W, dtype=torch.uint8)
        for i, t in enumerate(S):
            r = t.reshape(-1, 8)[:, keep]
            a = r.argmax(1)
            ov[i] = r.gather(1, a.unsqueeze(1)).squeeze(1)
            oa[i] = kt[a]
            del r, a
        return ov, oa

    def fast_pred(k, b, ov, oa):
        out = []
        for i, t in enumerate(S):
            sc = t.reshape(-1, 8)[:, k] + float(b)
            win = (sc > ov[i]) | ((sc == ov[i]) & (k < oa[i]))
            out.append(torch.where(win, torch.full_like(oa[i], k + 1, dtype=torch.int64),
                                   oa[i].to(torch.int64) + 1).to(torch.uint8))
            del sc, win
        return out

    def full_pred(vec):
        b = torch.tensor(vec, dtype=torch.float32).view(1, 1, 8)
        return [((t + b).argmax(-1).reshape(-1) + 1).to(torch.uint8) for t in S]

    def conf_and_miou(pred_list, idx=None):
        idx = range(n) if idx is None else idx
        tot = np.zeros((9, 9), dtype=np.int64)
        for i in idx:
            g = GT[i].reshape(-1)
            p = pred_list[i].reshape(-1)
            m = g != 0
            tot += np.bincount((g[m] * 9 + p[m]).numpy(), minlength=81).reshape(9, 9)
        sub = tot[1:, 1:].astype(np.float64)
        inter = np.diag(sub)
        union = sub.sum(1) + sub.sum(0) - inter
        pp = union > 0
        iou = np.where(pp, inter / np.maximum(union, 1e-12), np.nan)
        return tot, float(np.nanmean(iou[pp])), iou

    t1 = time.time()
    p_zero = full_pred([0.0] * 8)
    cm0, base, base_iou = conf_and_miou(p_zero)
    print(f"ZERO_TTA8_VAL_MIOU {base:.10f}  ({time.time()-t1:.1f}s)", flush=True)
    print("ZERO_PER_CLASS " + json.dumps({NAMES[i]: round(float(base_iou[i]), 6)
                                          for i in range(8)}), flush=True)

    # ---------- phase 3: candidate generation ----------
    def q_at(arr, rate):
        i = int(np.searchsorted(arr, np.quantile(arr, rate), side="left"))
        return float(arr[min(max(i, 0), arr.size - 1)])

    cand = {k: set() for k in range(1, 8)}
    for k in range(1, 8):
        fnA, fpA = TAU_FN[k], TAU_FP[k]
        for r in FN_RATES:
            cand[k].add(("pos", float(np.nextafter(q_at(fnA, r), np.inf)), f"FN_r{r}"))
        for r in FP_RATES:
            cand[k].add(("neg", float(np.nextafter(q_at(fpA, 1 - r), -np.inf)), f"FP_r{r}"))
        for r in GLOBAL_RATES:
            tgt = max(int(r * VALID), 1)
            if fnA.size >= tgt:
                cand[k].add(("pos", float(np.nextafter(fnA[tgt - 1], np.inf)), f"glob_r{r}"))
            if fpA.size >= tgt:
                cand[k].add(("neg", float(np.nextafter(-fpA[tgt - 1], -np.inf)), f"glob_r{r}"))
        for b in NEAR_ZERO:
            if b != 0.0:
                cand[k].add(("pos" if b > 0 else "neg", float(b), "near_zero"))
            else:
                cand[k].add(("zero", 0.0, "near_zero"))
    n_cand = sum(len(v) for v in cand.values())
    print(f"candidates: {n_cand}", flush=True)

    # ---------- phase 4: sweep with the exact fast path ----------
    report = []
    t2 = time.time()
    done = 0
    for k in range(1, 8):
        ov, oa = build_ref(k)
        for sgn, b, tag in sorted(cand[k], key=lambda x: x[1]):
            if tag == "near_zero" and b == 0.0:
                continue                       # zero baseline already scored
            p = fast_pred(k, b, ov, oa)
            _, m, iou = conf_and_miou(p)
            report.append({"class": NAMES[k], "official_id": k + 1, "bias": b,
                           "sign": sgn, "tag": tag, "miou": m,
                           "gain_pp": (m - base) * 100,
                           "scored_by": "exact single-class fast path",
                           "class_iou": float(iou[k]),
                           "class_iou_gain_pp": (float(iou[k]) - float(base_iou[k])) * 100,
                           "events": events(k, b)})
            del p
            done += 1
        del ov, oa
        print(f"  swept {NAMES[k]}: {done}/{n_cand}  {time.time()-t2:.0f}s", flush=True)

    # ---------- phase 5: re-score the reported set with full 8-channel argmax ----------
    report.sort(key=lambda r: -r["miou"])
    per_class_best = {}
    for k in range(1, 8):
        per_class_best[NAMES[k]] = max((r for r in report if r["class"] == NAMES[k]),
                                       key=lambda r: r["miou"])
    to_rescore = {id(r) for r in report[:20]}
    to_rescore |= {id(r) for r in per_class_best.values()}
    to_rescore |= {id(r) for r in report if r["tag"] == "near_zero"}
    print(f"re-scoring {len(to_rescore)} candidates with full 8-channel argmax", flush=True)
    t3 = time.time()
    for r in report:
        if id(r) in to_rescore:
            k = NAMES.index(r["class"])
            vec = [0.0] * 8
            vec[k] = r["bias"]
            _, m, iou = conf_and_miou(full_pred(vec))
            r["miou_full8"] = m
            r["gain_pp_full8"] = (m - base) * 100
            r["class_iou_full8"] = float(iou[k])
            r["fastpath_full8_agree"] = abs(m - r["miou"]) < 1e-12
    print(f"  re-scored in {time.time()-t3:.0f}s", flush=True)
    disagree = [r for r in report if "fastpath_full8_agree" in r and not r["fastpath_full8_agree"]]
    print(f"FASTPATH_VS_FULL8_DISAGREEMENTS {len(disagree)}", flush=True)
    for r in disagree[:10]:
        print(f"   {r['class']} b={r['bias']:+.6f} fast={r['miou']:.10f} full8={r['miou_full8']:.10f}", flush=True)

    res = {
        "schema": "M2F_56K_TTA8_C2_EVENT_DRIVEN_V1",
        "baseline_zero_miou": base,
        "baseline_per_class_iou": {NAMES[i]: float(base_iou[i]) for i in range(8)},
        "valid_px_total": VALID,
        "n_candidates_swept": len(report),
        "candidate_design": {
            "FN_rates": FN_RATES, "FP_rates": FP_RATES,
            "global_rates": GLOBAL_RATES, "near_zero": NEAR_ZERO,
            "quantile_convention": "smallest tau with P(x < tau) >= rate, then nextafter one ULP past the boundary",
            "rationale": "FN/FP tau q01 ~= 0.011 vs q50 ~= 0.63 (58x); a fixed grid cannot serve both"},
        "histogram": {"lo": HLO, "hi": HHI, "bin": HBIN,
                      "note": "TN/TP event counts carry <=1% error for |b| >= 1e-3; FN/FP counts are exact"},
        "fastpath_vs_full8_disagreements": len(disagree),
        "top20": report[:20],
        "per_class_best": per_class_best,
        "near_zero_grid": [r for r in report if r["tag"] == "near_zero"],
        "all_candidates": report,
        "source_shas": {
            "c2_script": sha256(Path(__file__)),
            "boundary_script": sha256(ROOT / "boundary_56k/boundary_tau_v2.py"),
            "boundary_json": sha256(ROOT / "boundary_56k/BOUNDARY_TAU.json"),
            "stage_c_json": sha256(ROOT / "calib56k_v5_tieexact/stage_scale_sensitivity.json")},
        "constraints": "1D single-class scans only; no coordinate descent, no joint search, no packaging, no training",
    }
    (OUT / "C2_RESULT.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, allow_nan=False))
    print("C2_DONE", flush=True)
    for r in report[:10]:
        print(f"  TOP {r['class']:13s} b={r['bias']:+.6f} ({r['tag']}) "
              f"mIoU={r['miou']:.8f} ({r['gain_pp']:+.4f}pp) "
              f"cls={r['class_iou_gain_pp']:+.4f}pp", flush=True)
    for k in range(1, 8):
        b_ = per_class_best[NAMES[k]]
        print(f"  BEST {NAMES[k]:13s} b={b_['bias']:+.6f} ({b_['tag']}) "
              f"mIoU={b_['miou']:.8f} ({b_['gain_pp']:+.4f}pp)", flush=True)


if __name__ == "__main__":
    main()
