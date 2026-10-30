"""P0 identity / consistency audit for H75 TargetSSL pilot. Zero training.

Checks:
  1. H75 real file sha256
  2. encoder feature shapes at 1024 input (stage0..3), forward-shape assertion
  3. D4-TTA8 + Bstar prediction reproduction vs an existing H75 prediction artifact
  4. train/eval mode, BN running stats immutability across a forward
  5. manifest counts and identity
"""
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path("/home/jhy/aic2026_h76_20261026")
SRC = ROOT / "code" / "h76_3594"
sys.path.insert(0, str(SRC))
import os
os.environ.setdefault("UAVSEG_MODEL_SOURCE", str(ROOT / "models" / "segformer-b3-ade512"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.models import build_model
from src.utils import load_config
from src.transforms import normalize_to_tensor

H75 = ROOT / "models" / "H75.pt"
BSTAR = np.array([-.2, .1, .0709, .4, 3.6, .4, 2.1215, -.5], dtype=np.float32)
OUT = Path(__file__).resolve().parent


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--ref-zip", default=str(ROOT / "deliver" / "h75_bias" / "H75_BSTAR.zip"))
    a = ap.parse_args()

    rep = {"h75": str(H75), "h75_sha256": sha256(H75), "h75_bytes": H75.stat().st_size}

    cfg = load_config(str(SRC / "configs" / "base.yaml"))
    cfg["cwfa"] = {"enabled": False}; cfg["mixstyle"] = {"enabled": False}
    cfg["pretrained_model_path"] = str(ROOT / "models" / "segformer-b3-ade512")
    model = build_model(cfg)
    obj = torch.load(str(H75), map_location="cpu", weights_only=False)
    state = obj.get("model", obj) if isinstance(obj, dict) else obj
    model.load_state_dict(state, strict=True)
    model = model.to("cuda:0").eval()
    rep["state_keys"] = len(state)
    rep["n_params"] = int(sum(p.numel() for p in model.parameters()))

    # ---- feature shapes ----
    x = torch.randn(1, 3, 1024, 1024, device="cuda:0")
    with torch.inference_mode():
        out = model(pixel_values=x, output_hidden_states=True)
    hs = out.hidden_states
    rep["hidden_state_shapes"] = [list(h.shape) for h in hs]
    rep["logits_shape"] = list(out.logits.shape)
    dec = model.decode_head
    rep["decode_head_type"] = type(dec).__name__

    # ---- BN buffer immutability across forward ----
    bn_before = {n: b.clone() for n, b in model.named_buffers()}
    with torch.inference_mode():
        model(pixel_values=x)
    bn_changed = [n for n, b in model.named_buffers() if not torch.equal(b, bn_before[n])]
    rep["buffers_changed_after_forward"] = bn_changed
    rep["model_training_flag"] = model.training

    # ---- prediction reproduction ----
    rows = list(csv.DictReader(open(ROOT / "manifests" / "semifinal_3594.csv")))
    rows = rows[: a.n]
    b = torch.from_numpy(BSTAR).cuda()[None, :, None, None]
    refzip = Path(a.ref_zip)
    import zipfile
    have_ref = refzip.is_file()
    rep["ref_zip"] = str(refzip)
    rep["ref_zip_sha256"] = sha256(refzip) if have_ref else None

    diffs = []
    zf = zipfile.ZipFile(refzip) if have_ref else None
    for r in rows:
        img = np.asarray(Image.open(ROOT / "data" / "test3" / "images" / r["input_name"]).convert("RGB")).copy()
        tx = normalize_to_tensor(img).unsqueeze(0).cuda()
        acc = None
        with torch.inference_mode():
            for v in range(8):
                z = inv_d4(model(pixel_values=d4(tx, v)).logits.float(), v)
                acc = z if acc is None else acc + z
            logits = acc / 8.0 + b
            pred = F.interpolate(logits, size=(1024, 1024), mode="bilinear",
                                 align_corners=False).argmax(1).add(1).to(torch.uint8)[0].cpu().numpy()
        if have_ref:
            with zf.open(r["output_name"]) as f:
                ref = np.array(Image.open(f))
            diffs.append(int((pred != ref).sum()))
    rep["ref_compared_images"] = len(diffs)
    rep["ref_diff_pixels_total"] = int(sum(diffs)) if diffs else None
    rep["ref_diff_pixels_max_per_image"] = int(max(diffs)) if diffs else None
    rep["ref_diff_histogram"] = diffs[:20]

    # ---- manifests ----
    rep["target_manifest"] = str(ROOT / "manifests" / "semifinal_3594.csv")
    rep["target_count"] = len(list(csv.DictReader(open(ROOT / "manifests" / "semifinal_3594.csv"))))
    rep["target_images_on_disk"] = len(list((ROOT / "data" / "test3" / "images").glob("*.png")))
    subs = {}
    for r in csv.DictReader(open(ROOT / "manifests" / "semifinal_3594.csv")):
        subs[r["subset"]] = subs.get(r["subset"], 0) + 1
    rep["subsets"] = subs
    rep["source_train_ids"] = len((SRC / "splits" / "train.txt").read_text().split())

    (OUT / "baseline_manifest.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
