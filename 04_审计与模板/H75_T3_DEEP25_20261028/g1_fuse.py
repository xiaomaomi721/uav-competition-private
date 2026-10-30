"""G1: single locked deep-delta fusion, alpha=0.25.

theta_new = theta_H + alpha*(theta_T - theta_H)  in FP32, for floating *parameters*
of segformer.stages.2/3 and decode_head.
stages.0/1 copied bitwise from H75. All buffers (BN running stats, ints) kept from H75.
"""
import hashlib, json
from pathlib import Path
import torch

H75 = Path("/home/jhy/aic2026_h76_20261026/models/H75.pt")
T3 = Path("/home/jhy/aic2026_h76_20261026/experiments/T3_FULL_ST_20261027_import/checkpoints/checkpoint_update3594.pt")
OUTDIR = Path("/home/jhy/aic2026_h75_t3_deep25_20261028")
ALPHA = 0.25
DEEP = ("segformer.stages.2.", "segformer.stages.3.", "decode_head.")
BUFFERY = ("running_mean", "running_var", "num_batches_tracked")


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
            if isinstance(v, dict) and v:
                return v
    return obj


def fuse(st75, st3, alpha):
    out = {}
    stats = {"copied_h75": 0, "interpolated": 0, "kept_buffer": 0, "max_delta": 0.0}
    for k, a in st75.items():
        b = st3[k]
        is_deep = k.startswith(DEEP)
        is_param_like = a.is_floating_point() and not any(k.endswith(s) for s in BUFFERY)
        if is_deep and is_param_like:
            new = (a.double() + alpha * (b.double() - a.double())).to(a.dtype)
            out[k] = new.clone()
            stats["interpolated"] += 1
            d = float((new.float() - a.float()).abs().max())
            stats["max_delta"] = max(stats["max_delta"], d)
        else:
            out[k] = a.clone()
            stats["copied_h75"] += 1
            if a.is_floating_point() and any(k.endswith(s) for s in BUFFERY):
                stats["kept_buffer"] += 1
    return out, stats


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    o75 = torch.load(str(H75), map_location="cpu", weights_only=False)
    o3 = torch.load(str(T3), map_location="cpu", weights_only=False)
    st75, st3 = get_state(o75), get_state(o3)
    assert set(st75) == set(st3), "key set mismatch"

    new, stats = fuse(st75, st3, ALPHA)

    ck = {"model": new,
          "fusion": {"alpha": ALPHA, "source_h75": str(H75), "source_h75_sha256": sha(H75),
                     "source_t3": str(T3), "source_t3_sha256": sha(T3),
                     "interpolated_prefixes": list(DEEP),
                     "rule": "theta_H + alpha*(theta_T-theta_H) in FP64->target dtype for floating params; "
                             "stages.0/1 and all buffers copied from H75",
                     "stats": stats}}
    outp = OUTDIR / "H75_T3_DEEP25.pt"
    torch.save(ck, outp)

    # ---------- self checks ----------
    checks = {}
    a0, _ = fuse(st75, st3, 0.0)
    checks["alpha0_equals_h75"] = all(torch.equal(a0[k], st75[k]) for k in st75)

    # alpha=1 on deep layers must reproduce T3 there; shallow must equal H75
    a1, _ = fuse(st75, st3, 1.0)
    deep_ok = True
    shallow_ok = True
    for k in st75:
        if k.startswith(DEEP) and st75[k].is_floating_point() and not any(k.endswith(s) for s in BUFFERY):
            if not torch.equal(a1[k], st3[k]):
                deep_ok = False
        else:
            if not torch.equal(a1[k], st75[k]):
                shallow_ok = False
    checks["alpha1_recovers_t3_on_deep"] = deep_ok
    checks["alpha1_keeps_h75_elsewhere"] = shallow_ok
    checks["shallow_zero_diff"] = all(torch.equal(new[k], st75[k])
                                      for k in st75 if k.startswith(("segformer.stages.0.", "segformer.stages.1.")))
    checks["all_finite"] = all(torch.isfinite(v).all().item() for v in new.values() if v.is_floating_point())
    checks["same_keys"] = set(new) == set(st75)
    checks["alpha_value"] = ALPHA

    # reload through the production loader path
    reloaded = get_state(torch.load(str(outp), map_location="cpu", weights_only=False))
    checks["reload_identical"] = all(torch.equal(reloaded[k], new[k]) for k in new)
    checks["new_sha256"] = sha(outp)
    checks["new_bytes"] = outp.stat().st_size

    (OUTDIR / "G1_SELFCHECK.json").write_text(json.dumps({"stats": stats, "checks": checks}, indent=2),
                                              encoding="utf-8")
    print(json.dumps({"stats": stats, "checks": checks}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
