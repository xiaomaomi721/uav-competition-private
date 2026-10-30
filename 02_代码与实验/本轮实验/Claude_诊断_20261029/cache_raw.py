"""Cache D4-TTA8 raw native-256 logits (no bias) for one checkpoint on Target3594.

Same protocol as package_semifinal_bias / rkd_common.predict_raw_tta; saved as fp16 to fit disk.
Usage: EXPECTED_GPU_INDEX=0 CUDA_VISIBLE_DEVICES=0 python cache_raw.py --ckpt X --sha Y --out DIR
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, "/home/jhy/aic2026_h76_20261026/code/reliable_kd_20261028")
from rkd_common import *  # noqa: E402,F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert sha256(a.ckpt) == a.sha, "checkpoint SHA mismatch"
    out = Path(a.out); (out / "scores").mkdir(parents=True, exist_ok=True)
    guard(); seed_all()
    model = model_from(a.ckpt); model.eval()
    rows = target_rows(); t0 = time.time()
    for i, r in enumerate(rows, 1):
        p = out / "scores" / (r["output_name"] + ".npy")
        if p.exists():
            continue
        x = normalize_to_tensor(read_rgb(TARGET_IMAGES / r["input_name"])).unsqueeze(0).cuda()
        z = predict_raw_tta(model, x)[0]
        np.save(p, z.cpu().numpy().astype(np.float16))
        if i % 200 == 0:
            guard(); emit(stage="cache", done=i, sec=round(time.time() - t0, 1))
    dump(out / "CACHE_META.json", {"ckpt": a.ckpt, "ckpt_sha256": a.sha, "n": len(rows),
                                   "protocol": "D4-TTA8 raw native256 FP32 mean, saved fp16, no bias",
                                   "manifest": str(TARGET_MANIFEST), "sec": round(time.time() - t0, 1)})
    emit(stage="cache_done", n=len(rows))


if __name__ == "__main__":
    main()
