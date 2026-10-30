"""Build T_d4_raw (+ T_weak_raw) teacher caches on 3594 target images with H75.

Per image .npz:
  z_d4   (8,256,256) fp16 : D4-TTA8 aligned raw logits mean, NO Bstar
  z_single (8,256,256) fp16: identity-view raw logits, NO Bstar   (T_weak_raw)
  votes  (256,256) uint8  : #views whose calibrated argmax == calibrated argmax of z_d4
  hn     (256,256) fp16   : normalized entropy of softmax(z_d4 + Bstar), in [0,1]
  flip   (256,256) uint8  : 1 if argmax(z_d4) != argmax(z_d4 + Bstar)
"""
import argparse
from rkd_common import *

CACHE = WORK / "teacher_d4"


def build(limit=0):
    guard(); seed_all()
    rows = target_rows()
    if limit:
        rows = rows[:limit]
    CACHE.mkdir(parents=True, exist_ok=True)
    identity = {
        "model": str(H75), "model_sha256": H75_SHA,
        "target_manifest": str(TARGET_MANIFEST), "target_manifest_sha256": sha256(TARGET_MANIFEST),
        "d4_views": "[(r,hf) for r in range(4) for hf in (False,True)]; fwd rot90 then flip; inv flip then rot90(-r)",
        "grid": "native 256x256", "bstar_in_cache": False, "bstar": BSTAR,
        "dtype": "fp16 for logits/hn; uint8 for votes/flip",
    }
    ip = CACHE / "IDENTITY.json"
    if ip.exists():
        assert json.loads(ip.read_text()) == identity, "cache identity mismatch"
    else:
        dump(ip, identity)

    done = {}
    fp = CACHE / "FILES.json"
    if fp.exists():
        done = json.loads(fp.read_text())

    pending = []
    for r in rows:
        stem = Path(r["input_name"]).stem
        out = CACHE / f"{stem}.npz"
        if out.exists() and stem in done:
            with np.load(out) as d:
                assert d["z_d4"].shape == (8, 256, 256) and np.isfinite(d["z_d4"]).all()
            continue
        pending.append((r, stem, out))

    emit(stage="cache_start", pending=len(pending), exist=len(done), total=len(rows))
    if not pending:
        return

    model = model_from()
    t0 = time.time()
    n = 0
    for r, stem, out in pending:
        n += 1
        if n % 20 == 1:
            guard()
        x = normalize_to_tensor(read_rgb(TARGET_IMAGES / r["input_name"])).unsqueeze(0).cuda()
        zT, votes, hn, flip = d4_stack_with_votes(model, x)
        z_single = inv_d4(model(pixel_values=d4(x, 0)).logits.float(), 0)[0]
        assert zT.shape == (8, 256, 256) and torch.isfinite(zT).all()
        np.savez_compressed(
            out,
            z_d4=zT[0].cpu().numpy().astype(np.float16),
            z_single=z_single.cpu().numpy().astype(np.float16),
            votes=votes[0].cpu().numpy().astype(np.uint8),
            hn=hn[0].cpu().numpy().astype(np.float16),
            flip=flip[0].cpu().numpy().astype(np.uint8),
        )
        done[stem] = {"sha256": sha256(out), "bytes": out.stat().st_size}
        if n % 100 == 0 or n == len(pending):
            dump(fp, done)
            emit(stage="cache", done=len(done), total=len(rows),
                 elapsed=round(time.time() - t0, 1),
                 eta=round((time.time() - t0) / n * (len(pending) - n), 1))
    dump(fp, done)
    dump(CACHE / "COMPLETE.json", {**identity, "count": len(done),
                                   "runtime_sec": round(time.time() - t0, 1)})
    emit(stage="cache_complete", count=len(done), runtime_sec=round(time.time() - t0, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    build(a.limit)
