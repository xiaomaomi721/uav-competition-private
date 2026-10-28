#!/usr/bin/env python3
"""Flip audit: alpha050 vs theta278 and vs D1."""
import zipfile, io, json
import numpy as np
from PIL import Image

THETA278_ZIP = "/home/jhy/rare_diag_20260915/round2_20260921/day1_formal_packages/A_theta278_oldBstar.zip"
D1_ZIP = "/home/jhy/rare_diag_20260915/round2_20260923/D1_2k_probe/package/D1_2k_theta278_test2.zip"
ALPHA_ZIP = "/home/jhy/rare_diag_20260915/d1_interp_20260925/D1_alpha050_test2.zip"
CLASS_NAMES = ["Background","Building","Road","Water","Barren","Vegetation","Agricultural","Vehicle"]

def load_zip(zp):
    z = zipfile.ZipFile(zp)
    names = sorted([n for n in z.namelist() if n.endswith(".png")],
                   key=lambda n: int(n.split("_")[-1].replace(".png","")))
    return [np.array(Image.open(io.BytesIO(z.read(n)))) for n in names]

print("Loading...")
p278 = load_zip(THETA278_ZIP)
pD1 = load_zip(D1_ZIP)
pA = load_zip(ALPHA_ZIP)
print(f"  {len(p278)}, {len(pD1)}, {len(pA)} images")

# alpha050 vs theta278
flip_a_278 = np.mean([(a != b).mean() for a, b in zip(pA, p278)])
# D1 vs theta278
flip_d1_278 = np.mean([(a != b).mean() for a, b in zip(pD1, p278)])
# alpha050 vs D1
flip_a_d1 = np.mean([(a != b).mean() for a, b in zip(pA, pD1)])

print(f"\n=== Flip rates ===")
print(f"  alpha0.5 vs theta278: {flip_a_278*100:.3f}%")
print(f"  D1(1.0) vs theta278:  {flip_d1_278*100:.3f}%")
print(f"  alpha0.5 vs D1:       {flip_a_d1*100:.3f}%")

# Per-image stats
flips_a = np.array([(a != b).mean() for a, b in zip(pA, p278)])
print(f"\nalpha0.5 per-image flip:")
print(f"  mean={flips_a.mean()*100:.3f}% median={np.median(flips_a)*100:.3f}% p90={np.percentile(flips_a,90)*100:.3f}% max={flips_a.max()*100:.3f}%")

# Class shares
def shares(preds):
    s = np.zeros(8)
    for p in preds:
        for c in range(1,9): s[c-1] += int((p==c).sum())
    return (s/s.sum()*100)

print(f"\n=== Class shares ===")
s278 = shares(p278); sD1 = shares(pD1); sA = shares(pA)
print(f"{'Class':<15} {'theta278':>10} {'D1':>10} {'alpha0.5':>10} {'Δ vs 278':>10}")
for i, name in enumerate(CLASS_NAMES):
    print(f"{name:<15} {s278[i]:>10.3f} {sD1[i]:>10.3f} {sA[i]:>10.3f} {sA[i]-s278[i]:>+10.3f}")

# Transition matrix alpha050 vs theta278
trans = np.zeros((8,8), dtype=np.int64)
for a, b in zip(pA, p278):
    for gc in range(1,9):
        for pc in range(1,9):
            trans[gc-1,pc-1] += int(((b==gc)&(a==pc)).sum())

print(f"\n=== Top transitions (theta278 -> alpha0.5) ===")
items = []
for i in range(8):
    for j in range(8):
        if i != j and trans[i,j] > 0:
            items.append((trans[i,j], CLASS_NAMES[i], CLASS_NAMES[j]))
items.sort(reverse=True)
total_changed = sum(x[0] for x in items)
for cnt, frm, to in items[:15]:
    print(f"  {frm:<15} -> {to:<15} {cnt:>10,} ({cnt/total_changed*100:.1f}%)")
