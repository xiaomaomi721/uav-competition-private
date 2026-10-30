"""G0: structural compatibility + per-module layer delta between H75 and T3."""
import hashlib, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

H75 = Path("/home/jhy/aic2026_h76_20261026/models/H75.pt")
T3 = Path("/home/jhy/aic2026_h76_20261026/experiments/T3_FULL_ST_20261027_import/checkpoints/checkpoint_update3594.pt")
H75_SHA = "b5eea2795d8a6d90c803a15b2ca66cf0abcf62322ef1aacd0d4d0ec82098a28a"
T3_SHA = "cb958870ce102b47eed8fa822b5ff9bfc017db7bb7e61b4e73105e46a472e8e8"
OUT = Path("/home/jhy/aic2026_h75_t3_deep25_20261028")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def get_state(obj):
    if isinstance(obj, dict):
        for k in ("model", "student_model", "model_state_dict", "state_dict"):
            v = obj.get(k)
            if isinstance(v, dict) and v and all(torch.is_tensor(t) for t in v.values()):
                return v, k
    if isinstance(obj, dict) and obj and all(torch.is_tensor(t) for t in obj.values()):
        return obj, "<root>"
    raise RuntimeError("no state dict")


def group(name):
    for p in ("segformer.stages.0.", "segformer.stages.1.", "segformer.stages.2.",
              "segformer.stages.3.", "decode_head."):
        if name.startswith(p):
            return p.rstrip(".")
    return "other:" + name.split(".")[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s75, s3 = sha(H75), sha(T3)
    rep = {"h75": {"path": str(H75), "sha256": s75, "sha_ok": s75 == H75_SHA},
           "t3": {"path": str(T3), "sha256": s3, "sha_ok": s3 == T3_SHA}}
    if not (rep["h75"]["sha_ok"] and rep["t3"]["sha_ok"]):
        print(json.dumps(rep, indent=2)); raise SystemExit("SHA mismatch -> FAIL")

    o75 = torch.load(str(H75), map_location="cpu", weights_only=False)
    o3 = torch.load(str(T3), map_location="cpu", weights_only=False)
    st75, k75 = get_state(o75)
    st3, k3 = get_state(o3)
    rep["wrapper_key_h75"], rep["wrapper_key_t3"] = k75, k3
    rep["t3_top_level_keys"] = sorted([k for k in o3.keys()]) if isinstance(o3, dict) else None

    missing = sorted(set(st75) - set(st3))
    unexpected = sorted(set(st3) - set(st75))
    shape_mismatch = [k for k in st75 if k in st3 and tuple(st75[k].shape) != tuple(st3[k].shape)]
    dtype_mismatch = [k for k in st75 if k in st3 and st75[k].dtype != st3[k].dtype]
    rep["n_keys_h75"], rep["n_keys_t3"] = len(st75), len(st3)
    rep["missing_in_t3"] = missing[:50]
    rep["unexpected_in_t3"] = unexpected[:50]
    rep["shape_mismatch"] = shape_mismatch[:50]
    rep["dtype_mismatch"] = dtype_mismatch[:50]
    rep["compatible"] = not (missing or unexpected or shape_mismatch)

    rows = []
    agg = defaultdict(lambda: {"n": 0, "l2d": 0.0, "l2h": 0.0, "maxabs": 0.0,
                               "cos_num": 0.0, "identical": 0})
    for k in sorted(set(st75) & set(st3)):
        a, b = st75[k], st3[k]
        if tuple(a.shape) != tuple(b.shape):
            continue
        af = a.detach().float().reshape(-1)
        bf = b.detach().float().reshape(-1)
        d = bf - af
        l2d = float(d.norm())
        l2h = float(af.norm())
        maxabs = float(d.abs().max()) if d.numel() else 0.0
        cosn = float((af * bf).sum())
        ident = int(torch.equal(a, b))
        g = group(k)
        rows.append({"key": k, "group": g, "shape": list(a.shape), "dtype": str(a.dtype),
                     "l2_h75": l2h, "l2_delta": l2d,
                     "rel_l2": (l2d / l2h) if l2h > 0 else None,
                     "max_abs_delta": maxabs, "identical": ident})
        A = agg[g]
        A["n"] += 1; A["l2d"] += l2d ** 2; A["l2h"] += l2h ** 2
        A["maxabs"] = max(A["maxabs"], maxabs); A["cos_num"] += cosn
        A["identical"] += ident

    summary = {}
    for g, A in sorted(agg.items()):
        summary[g] = {"n_tensors": A["n"],
                      "identical_tensors": A["identical"],
                      "l2_delta": A["l2d"] ** 0.5,
                      "l2_h75": A["l2h"] ** 0.5,
                      "rel_l2": (A["l2d"] ** 0.5) / A["l2h"] ** 0.5 if A["l2h"] > 0 else None,
                      "max_abs_delta": A["maxabs"]}
    rep["group_summary"] = summary

    with open(OUT / "ASSET_SHA_AND_LAYER_DELTA.csv", "w", encoding="utf-8") as f:
        f.write("key,group,shape,dtype,l2_h75,l2_delta,rel_l2,max_abs_delta,identical\n")
        for r in rows:
            f.write(f'{r["key"]},{r["group"]},"{r["shape"]}",{r["dtype"]},{r["l2_h75"]:.6e},'
                    f'{r["l2_delta"]:.6e},{r["rel_l2"] if r["rel_l2"] is None else round(r["rel_l2"],6)},'
                    f'{r["max_abs_delta"]:.6e},{r["identical"]}\n')

    (OUT / "G0_STRUCTURE.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n_keys_h75", "n_keys_t3", "compatible",
                                          "missing_in_t3", "unexpected_in_t3",
                                          "shape_mismatch", "wrapper_key_t75" if False else "wrapper_key_h75")},
                     indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
