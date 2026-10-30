"""Marginal matching: fit an 8-dim bias so a model's Target3594 argmax area per class equals the
champion's (H75 raw cache + Bstar), on the native 256 grid. CPU only.

Update: b_k += eta * (log A*_k - log A_k(b)), Background fixed at Bstar[0] (argmax is shift-invariant).
Usage: python mm_fit.py --cache DIR [--init zero|bstar] --out JSON
"""
import argparse, csv, json, time
from multiprocessing.pool import ThreadPool
from pathlib import Path
import numpy as np
import torch

MAN = "/home/jhy/aic2026_ziponly_20261028/manifests/semifinal_3594.csv"
REF = "/home/jhy/aic2026_ziponly_20261028/01_h75_cache/target3594_fp32/scores"
BSTAR = np.array([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], np.float32)
torch.set_num_threads(64)
names = [r["output_name"] for r in csv.DictReader(open(MAN))]


DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def load(d):
    Z = torch.empty((len(names), 8, 256 * 256), dtype=torch.float16, device=DEV)
    with ThreadPool(16) as p:
        for i, a in enumerate(p.imap(lambda n: np.load(f"{d}/{n}.npy").astype(np.float16).reshape(8, -1), names)):
            Z[i] = torch.from_numpy(a).to(DEV)
    return Z  # (N,8,HW) fp16 on DEV


def areas(Z, b):
    tot = torch.zeros(8, dtype=torch.float64, device=DEV)
    bt = torch.from_numpy(b).to(DEV)[None, :, None]
    for i in range(0, Z.shape[0], 200):
        tot += torch.bincount((Z[i:i + 200].float() + bt).argmax(1).reshape(-1), minlength=8).double()
    return (tot / tot.sum()).cpu().numpy()


def fit(Z, target, b0, iters=600, eta=0.5, tol=5e-4):
    b, log = b0.copy(), []
    for it in range(iters):
        A = areas(Z, b)
        rel = A / target - 1
        log.append({"it": it, "bias": b.round(4).tolist(), "max_rel_err": float(np.abs(rel).max())})
        if np.abs(rel).max() < tol:
            break
        step = eta * (np.log(target) - np.log(np.maximum(A, 1e-9)))
        step[0] = 0.0
        b = (b + step).astype(np.float32)
    return b, A, log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--init", default="bstar", choices=["bstar", "zero"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    target_file = Path(a.out).parent / "CHAMP_AREA_256.npy"
    if target_file.exists():
        target = np.load(target_file)
    else:
        target = areas(load(REF), BSTAR); np.save(target_file, target)
    Z = load(a.cache)
    b0 = BSTAR.copy() if a.init == "bstar" else np.array([BSTAR[0]] + [0.0] * 7, np.float32)
    A_bstar = areas(Z, BSTAR)
    b, A, log = fit(Z, target, b0)
    res = {"cache": a.cache, "init": a.init, "target_area": target.round(6).tolist(),
           "area_with_bstar": A_bstar.round(6).tolist(), "fitted_bias": b.round(5).tolist(),
           "fitted_area": A.round(6).tolist(), "delta_vs_bstar": (b - BSTAR).round(4).tolist(),
           "iterations": len(log), "final_max_rel_err": log[-1]["max_rel_err"], "sec": round(time.time() - t0, 1)}
    json.dump({**res, "log": log}, open(a.out, "w"), indent=1)
    print(json.dumps(res))
