"""G2.2 canary: 40 fixed images stratified over test1 / test2 / test3_new.

For H75, T3, Deep25 under identical D4-TTA8 + Bstar:
  - per-image pixel change rate vs H75
  - class area shares
  - degenerate images (single class > 99%)
  - Barren / Agricultural / Vehicle deltas
No target GT exists; these are catastrophe screens only.
"""
import collections, csv, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path("/home/jhy/aic2026_h76_20261026")
SRC = ROOT / "code" / "h76_3594"
import sys
sys.path.insert(0, str(SRC))
import os
os.environ.setdefault("UAVSEG_MODEL_SOURCE", str(ROOT / "models" / "segformer-b3-ade512"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from src.models import build_model
from src.utils import load_config
from src.transforms import normalize_to_tensor

H75 = ROOT / "models" / "H75.pt"
T3 = ROOT / "experiments" / "T3_FULL_ST_20261027_import" / "checkpoints" / "checkpoint_update3594.pt"
DEEP = Path("/home/jhy/aic2026_h75_t3_deep25_20261028/H75_T3_DEEP25.pt")
OUT = Path("/home/jhy/aic2026_h75_t3_deep25_20261028")
BSTAR = np.array([-.2, .1, .0709, .4, 3.6, .4, 2.1215, -.5], dtype=np.float32)
CLS = ["Background", "Building", "Road", "Water", "Barren", "Vegetation", "Agricultural", "Vehicle"]
PER_SUB = 14


def d4(x, v):
    k, f = divmod(v, 2)
    if k: x = torch.rot90(x, k, (-2, -1))
    if f: x = x.flip(-1)
    return x


def inv_d4(x, v):
    k, f = divmod(v, 2)
    if f: x = x.flip(-1)
    if k: x = torch.rot90(x, -k, (-2, -1))
    return x.contiguous()


def load(ckpt):
    cfg = load_config(str(SRC / "configs" / "base.yaml"))
    cfg["cwfa"] = {"enabled": False}; cfg["mixstyle"] = {"enabled": False}
    cfg["pretrained_model_path"] = str(ROOT / "models" / "segformer-b3-ade512")
    m = build_model(cfg)
    m.load_state_dict(torch.load(str(ckpt), map_location="cpu", weights_only=False).get("model"), strict=True)
    return m.to("cuda:0").eval()


def predict(model, img):
    tx = normalize_to_tensor(img).unsqueeze(0).cuda()
    b = torch.from_numpy(BSTAR).cuda()[None, :, None, None]
    acc = None
    with torch.inference_mode():
        for v in range(8):
            z = inv_d4(model(pixel_values=d4(tx, v)).logits.float(), v)
            acc = z if acc is None else acc + z
        logits = acc / 8.0 + b
        return F.interpolate(logits, size=(1024, 1024), mode="bilinear",
                             align_corners=False).argmax(1).add(1).to(torch.uint8)[0].cpu().numpy()


def main():
    rows = list(csv.DictReader(open(ROOT / "manifests" / "semifinal_3594.csv")))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["subset"]].append(r)
    chosen = []
    for s in ("test1", "test2", "test3_new"):
        chosen += by[s][:PER_SUB]
    print(f"canary images: {len(chosen)}  " + str({s: len([c for c in chosen if c['subset']==s]) for s in by}), flush=True)

    models = {"H75": load(H75), "T3": load(T3), "DEEP25": load(DEEP)}
    preds = {k: {} for k in models}
    for r in chosen:
        img = np.asarray(Image.open(ROOT / "data" / "test3" / "images" / r["input_name"]).convert("RGB")).copy()
        for k, m in models.items():
            preds[k][r["input_name"]] = predict(m, img)

    out = {}
    for k in models:
        shares = collections.defaultdict(lambda: np.zeros(8, dtype=np.int64))
        degen = collections.defaultdict(int)
        changed = collections.Counter()
        tot = collections.Counter()
        for r in chosen:
            p = preds[k][r["input_name"]]
            s = r["subset"]
            c = np.bincount(p.reshape(-1), minlength=9)[1:]
            shares[s] += c
            if c.max() / c.sum() > 0.99:
                degen[s] += 1
            if k != "H75":
                q = preds["H75"][r["input_name"]]
                changed[s] += int((p != q).sum())
                tot[s] += p.size
        out[k] = {
            "shares_pct": {s: [round(float(x) / max(1, shares[s].sum()) * 100, 3) for x in shares[s]]
                           for s in shares},
            "degenerate": dict(degen),
            "change_vs_H75_pct": {s: round(changed[s] / max(1, tot[s]) * 100, 4) for s in changed},
            "n_per_subset": {s: len([c for c in chosen if c["subset"] == s]) for s in by},
        }
        print(k, json.dumps(out[k], ensure_ascii=False), flush=True)

    (OUT / "G2_CANARY.json").write_text(json.dumps({"class_order": CLS, "results": out}, indent=2), encoding="utf-8")
    with open(OUT / "TARGET_SUBSET_DIAGNOSTICS.csv", "w", encoding="utf-8") as f:
        f.write("model,subset,change_vs_H75_pct,degenerate_images," + ",".join(CLS) + "\n")
        for k in out:
            for s in out[k]["shares_pct"]:
                f.write(f"{k},{s},{out[k]['change_vs_H75_pct'].get(s,0)},"
                        f"{out[k]['degenerate'].get(s,0)},"
                        + ",".join(str(x) for x in out[k]["shares_pct"][s]) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
