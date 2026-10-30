"""H75_BARREN_GATE4096: champion H75+Bstar with region-gated Barren flips. CPU only.

Barren flip = champion pixel is Barren while the Zero prediction (no bias) is not Barren.
Flipped-to-Barren pixels whose 8-connected component (1024 grid) has < TAU pixels revert to the
Zero prediction. Base = the scored champion ZIP, so every other pixel is byte-identical to it.
Zero prediction = H75 FP32 D4 raw cache -> bilinear 1024 (align_corners=False) -> argmax,
the same protocol as package_semifinal_bias.py without the bias.
TAU=4096 (64x64): smallest tested threshold at which Target Barren presence matches the official
count (879 -> 722 vs 721); on Source1399 such small Barren flips are ~90% wrong.
"""
import csv, hashlib, io, json, zipfile
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

torch.set_num_threads(1)
TAU, BARREN = 4096, 4
CHAMP = "/home/jhy/aic2026_ctrl3594_teacher_20261027/reference/H75_BSTAR_20261026/H75_BSTAR_semifinal.zip"
CHAMP_SHA = "c012bae1237160a0c0d5fbccd168f58facf62c03bf60ee2cc82877cb21454d30"
TC = "/home/jhy/aic2026_ziponly_20261028/01_h75_cache/target3594_fp32/scores/"
MAN = "/home/jhy/aic2026_ctrl3594_teacher_20261027/manifests/semifinal_3594.csv"
MAN_SHA = "76463ff99a495afe82085bb59530032f13f9efbff2cc1936f09178285e6cdd19"
TAG = "H75_BARREN_GATE4096"
OUT = Path("/home/jhy/aic2026_claude_diag_20261029/pack") / TAG
_zf = None


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def job(name):
    global _zf
    if _zf is None:
        _zf = zipfile.ZipFile(CHAMP)
    a = np.asarray(Image.open(io.BytesIO(_zf.read(name)))).astype(np.int64) - 1
    assert a.shape == (1024, 1024) and a.min() >= 0 and a.max() <= 7, name
    z = torch.from_numpy(np.load(TC + name + ".npy").astype(np.float32))
    pz = F.interpolate(z[None], size=(1024, 1024), mode="bilinear", align_corners=False)[0].argmax(0).numpy()
    m = (a == BARREN) & (pz != BARREN)
    out, n_rev = a.copy(), 0
    if m.any():
        lab, _ = ndimage.label(m, structure=np.ones((3, 3)))
        s = np.bincount(lab.ravel()); small = s < TAU; small[0] = False
        rev = small[lab] & m
        out[rev] = pz[rev]; n_rev = int(rev.sum())
    Image.fromarray((out + 1).astype(np.uint8), mode="L").save(OUT / "predictions" / name)
    return name, n_rev, int(m.sum())


if __name__ == "__main__":
    assert sha(CHAMP) == CHAMP_SHA and sha(MAN) == MAN_SHA
    rows = list(csv.DictReader(open(MAN)))
    names = [r["output_name"] for r in rows]
    assert len(names) == 3594 and len(set(names)) == 3594
    (OUT / "predictions").mkdir(parents=True, exist_ok=False)
    with Pool(48) as p:
        res = p.map(job, names, chunksize=20)
    zp = OUT / f"{TAG}_semifinal.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        for n in names:
            zf.write(OUT / "predictions" / n, n)
    with zipfile.ZipFile(zp) as zf:
        assert zf.testzip() is None and zf.namelist() == names
    rev, flip = sum(r[1] for r in res), sum(r[2] for r in res)
    rep = {"tag": TAG, "tau_px_1024": TAU, "connectivity": 8, "gated_class": "Barren",
           "base_zip": CHAMP, "base_zip_sha256": CHAMP_SHA, "zero_cache": TC, "manifest_sha256": MAN_SHA,
           "script_sha256": sha(__file__), "images": len(names),
           "barren_flip_px": flip, "reverted_px": rev, "reverted_pct_of_all": round(100 * rev / (3594 * 1024 * 1024), 5),
           "images_changed": sum(1 for r in res if r[1]), "zip": str(zp), "zip_sha256": sha(zp),
           "single_model": True, "ground_truth_used": False, "platform_submission_performed": False}
    json.dump(rep, open(OUT / "PACKAGE_REPORT.json", "w"), indent=1)
    print(json.dumps(rep))
