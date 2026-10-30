"""Structural compatibility + layer delta for the BlindRecall CONTROL/TREAT pair,
and check whether either appears in the H75 ancestry chain."""
import hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

B = Path("/home/jhy/rare_diag_20260915/a05_blindrecall_pair_20260927")
CTRL = B / "05_formal_control/checkpoint_step2000.pt"
TREAT = B / "06_formal_treat/checkpoint_step2000.pt"
A05 = Path("/home/jhy/rare_diag_20260915/breakthrough_v2_20260926/03_weight_projection/A05_WEIGHT_alpha05.pt")
H75 = Path("/home/jhy/aic2026_h76_20261026/models/H75.pt")
OUT = Path("/home/jhy/aic2026_h75_next_research_20261029/01_复赛两模型融合")
CLS = ["Background", "Building", "Road", "Water", "Barren", "Vegetation", "Agricultural", "Vehicle"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def state(p):
    o = torch.load(str(p), map_location="cpu", weights_only=False)
    for k in ("model", "student_model", "model_state_dict", "state_dict"):
        v = o.get(k) if isinstance(o, dict) else None
        if isinstance(v, dict) and v:
            return v, k
    return o, "<root>"


def group(n):
    for p in ("segformer.stages.0.", "segformer.stages.1.", "segformer.stages.2.",
              "segformer.stages.3.", "decode_head."):
        if n.startswith(p):
            return p.rstrip(".")
    return "other"


def cmp(a, b, na, nb):
    ka, kb = set(a), set(b)
    rep = {"n_a": len(ka), "n_b": len(kb),
           "missing": sorted(ka - kb)[:20], "unexpected": sorted(kb - ka)[:20],
           "shape_mismatch": [k for k in ka & kb if tuple(a[k].shape) != tuple(b[k].shape)][:20]}
    rep["compatible"] = not (rep["missing"] or rep["unexpected"] or rep["shape_mismatch"])
    agg = defaultdict(lambda: {"n": 0, "l2d": 0.0, "l2a": 0.0, "ident": 0, "maxabs": 0.0})
    for k in sorted(ka & kb):
        if tuple(a[k].shape) != tuple(b[k].shape):
            continue
        x = a[k].detach().float().reshape(-1); y = b[k].detach().float().reshape(-1)
        d = y - x
        g = group(k)
        A = agg[g]
        A["n"] += 1; A["l2d"] += float(d.norm()) ** 2; A["l2a"] += float(x.norm()) ** 2
        A["ident"] += int(torch.equal(a[k], b[k])); A["maxabs"] = max(A["maxabs"], float(d.abs().max()))
    rep["groups"] = {g: {"n": v["n"], "identical": v["ident"],
                         "rel_l2": round((v["l2d"] ** .5) / (v["l2a"] ** .5), 6) if v["l2a"] > 0 else None,
                         "max_abs": v["maxabs"]} for g, v in agg.items()}
    rep["compare"] = f"{na} vs {nb}"
    return rep


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = {"CTRL_70.272": CTRL, "TREAT_70.867": TREAT, "A05": A05, "H75": H75}
    meta = {}
    for k, p in ids.items():
        if not p.is_file():
            meta[k] = {"path": str(p), "status": "NOT_FOUND"}; continue
        st, wk = state(p)
        meta[k] = {"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p),
                   "wrapper_key": wk, "n_tensors": len(st), "status": "VERIFIED_NOW"}

    res = {"checkpoints": meta}
    sc, st = state(CTRL), state(TREAT)
    res["CTRL_vs_TREAT"] = cmp(sc, st, "CTRL", "TREAT")
    if A05.is_file():
        sa, _ = state(A05)
        res["CTRL_vs_A05"] = cmp(sa, sc, "A05", "CTRL")
        res["TREAT_vs_A05"] = cmp(sa, st, "A05", "TREAT")
    if H75.is_file():
        sh, _ = state(H75)
        res["CTRL_vs_H75"] = cmp(sc, sh, "CTRL", "H75")
        res["TREAT_vs_H75"] = cmp(st, sh, "TREAT", "H75")

    (OUT / "PAIR_COMPAT.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items()}, indent=2, ensure_ascii=False))
    for key in ("CTRL_vs_TREAT", "CTRL_vs_A05", "TREAT_vs_A05", "CTRL_vs_H75", "TREAT_vs_H75"):
        if key in res:
            r = res[key]
            print(key, "compatible=", r["compatible"], "groups=",
                  json.dumps(r["groups"], ensure_ascii=False))


if __name__ == "__main__":
    main()
