"""Uniform parameter average of same-recipe checkpoints into ONE state_dict (offline, compliant).

Frozen tensors (Stage0/1, embeddings, BN buffers in eval) must be bit-identical across members;
otherwise abort. Usage: python soup.py OUT.pt A.pt B.pt [...]
"""
import hashlib, json, sys
import torch


def sd(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj["model"] if isinstance(obj, dict) and "model" in obj else obj


out, members = sys.argv[1], sys.argv[2:]
S = [sd(p) for p in members]
keys = list(S[0])
assert all(list(s) == keys for s in S), "key mismatch"
res, n_avg, n_same = {}, 0, 0
for k in keys:
    ts = [s[k] for s in S]
    if all(torch.equal(ts[0], t) for t in ts[1:]):
        res[k] = ts[0].clone(); n_same += 1
    elif ts[0].is_floating_point():
        res[k] = torch.stack([t.double() for t in ts]).mean(0).to(ts[0].dtype); n_avg += 1
    else:
        raise SystemExit(f"non-float tensor differs: {k}")
diff = [k for k in keys if not all(torch.equal(S[0][k], s[k]) for s in S[1:])]
bad = [k for k in diff if not k.startswith(("segformer.stages.2.", "segformer.stages.3.", "decode_head."))]
assert not bad, f"frozen tensors differ: {bad[:5]}"
avg_prefixes = sorted({".".join(k.split(".")[:3]) for k in diff})
torch.save({"model": res, "soup_members": members, "rule": "uniform mean of differing float tensors"}, out)
h = hashlib.sha256(open(out, "rb").read()).hexdigest()
print(json.dumps({"out": out, "sha256": h, "averaged_tensors": n_avg, "identical_tensors": n_same,
                  "averaged_prefixes": avg_prefixes}))
