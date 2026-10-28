#!/usr/bin/env python3
"""Phase 0: Champion identity audit."""
import hashlib, json, os, sys, zipfile
import numpy as np

ROOT = "/home/jhy/rare_diag_20260915"
OUT = os.path.join(ROOT, "d1_breakthrough_20260926", "00_identity")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

# Key files
files = {
    "theta261": os.path.join(ROOT, "phase0_candidates_20260918/A_cleanSEM_M_k100_th209.pt"),
    "theta278_f_s2p100": os.path.join(ROOT, "atda_20260918/cpu_candidates/f_s2p100_s3p125.pt"),
    "D1_step2000": os.path.join(ROOT, "round2_multiprobe_20260922/03_dart_fixed/D1/checkpoint_step2000.pt"),
}

# ZIP files from local submission copies on server
zip_candidates = [
    "/home/jhy/submit_packages/D1_2k_theta278_test2.zip",
    "/home/jhy/submit_packages/D1_alpha050_test2.zip",
]
# Also check known paths
for p in [
    os.path.join(ROOT, "round2_20260923/D1_2k_probe/package/D1_2k_theta278_test2.zip"),
]:
    if os.path.exists(p):
        zip_candidates.append(p)

ledger = {}
for name, path in files.items():
    if os.path.exists(path):
        ledger[name] = {"path": path, "sha256": sha256_file(path)}
        print(f"{name}: {ledger[name]['sha256'][:16]}...  {path}")
    else:
        ledger[name] = {"path": path, "missing": True}
        print(f"MISSING: {path}")

# Compare two theta278 checkpoints
import torch
t261 = torch.load(files["theta261"], map_location="cpu", weights_only=False)
t278 = torch.load(files["theta278_f_s2p100"], map_location="cpu", weights_only=False)
d1 = torch.load(files["D1_step2000"], map_location="cpu", weights_only=False)

# They may have different wrapper keys
def get_state_dict(obj):
    if isinstance(obj, dict):
        for k in ["model", "state_dict", "student", "module"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj

t261_sd = get_state_dict(t261)
t278_sd = get_state_dict(t278)
d1_sd = get_state_dict(d1)

# Compare keys
print(f"\ntheta261 keys: {len(t261_sd)}")
print(f"theta278 keys: {len(t278_sd)}")
print(f"D1 keys: {len(d1_sd)}")

# Check if theta278_f_s2p100 matches expected SHA
expected_shas = {
    "theta261": "9ffdd0beb8480a1efd361503d637d3d53cf716c808537c551fefae09f7fe2786",
    "theta278_f_s2p100": "c3c3df488c7ded69f6f2444708e39348fb6cf05e6e7ce4abdaa1c805b147e32f",
    "D1_step2000": "70f45afcddfb1711a0b8576f2e335ce5fa3f88df56c48724b761716e10f34b14",
}
for name, expected in expected_shas.items():
    actual = ledger.get(name, {}).get("sha256", "MISSING")
    match = actual == expected
    ledger[name]["expected_sha256"] = expected
    ledger[name]["sha_match"] = match
    print(f"{name}: SHA match={match}")

# Parameter delta analysis
def group_delta(sd_a, sd_b, keyword):
    total = 0.0; count = 0; tensors = 0
    for k in sd_a:
        if keyword not in k: continue
        if k not in sd_b: continue
        if "num_batches_tracked" in k: continue
        a = sd_a[k].float(); b = sd_b[k].float()
        d = (a - b).reshape(-1)
        total += (d @ d).item()
        count += d.numel()
        tensors += 1
    return {"tensors": tensors, "numel": count, "l2_norm": total**0.5, "rms": (total/max(count,1))**0.5}

delta_report = {}
for kw in ["stages.0", "stages.1", "stages.2", "stages.3", "decode_head"]:
    delta_report[kw] = group_delta(d1_sd, t278_sd, kw)
    print(f"D1 vs theta278 {kw}: l2={delta_report[kw]['l2_norm']:.6f} rms={delta_report[kw]['rms']:.8f}")

# ZIP audit
zip_results = {}
for zp in zip_candidates:
    if not os.path.exists(zp): continue
    zsha = sha256_file(zp)
    with zipfile.ZipFile(zp) as zf:
        names = zf.namelist()
        pngs = [n for n in names if n.endswith(".png")]
        testzip = zf.testzip()
    zip_results[zp] = {
        "sha256": zsha,
        "num_entries": len(names),
        "num_png": len(pngs),
        "testzip_ok": testzip is None,
        "first_three": pngs[:3],
        "last_three": pngs[-3:],
    }
    print(f"\nZIP: {os.path.basename(zp)}")
    print(f"  SHA: {zsha[:16]}...")
    print(f"  PNGs: {len(pngs)}, testzip: {'OK' if testzip is None else 'FAIL'}")

# Write ledger
os.makedirs(OUT, exist_ok=True)
result = {
    "checkpoints": ledger,
    "parameter_delta_d1_vs_theta278": delta_report,
    "zips": zip_results,
    "expected_scores": {
        "D1_2k_single_model": 70.736134,
        "A05_logit_teacher": 70.763755,
        "theta278_f_s2p100": 70.586921,
    },
    "compliance_notes": {
        "A05_LOGIT_TEACHER": "UNRESOLVED - uses 0.5*theta278_logits + 0.5*D1_logits at inference = multi-checkpoint fusion",
        "D1_2k": "PASS - single checkpoint",
    }
}
with open(os.path.join(OUT, "CHAMPION_LEDGER.json"), "w") as f:
    json.dump(result, f, indent=2, default=str)
print(f"\nWrote {OUT}/CHAMPION_LEDGER.json")
