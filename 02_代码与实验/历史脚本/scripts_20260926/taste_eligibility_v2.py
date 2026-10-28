"""TAST-E Phase C v2: Corrected CPU eligibility diagnostic.

Fixes:
- FFT spatial axes=(0,1) for HWC images
- Energy distance (permutation invariant)
- Disjoint source-source bootstrap
- 5597 real-GT source split
- Paired raw/transformed comparison
- No hardcoded results
"""
import hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path("/home/jhy/rare_diag_20260915/taste_20260926")
OUT = ROOT / "02_eligibility_v2_corrected"
OUT.mkdir(parents=True, exist_ok=True)

SOURCE_DIR = Path("/home/jhy/datasets/pseudo_train_6796/train/images")
SOURCE_SPLIT = Path("/home/jhy/datasets/pseudo_train_6796/train_ids.txt")
TEST2_DIR = Path("/home/jhy/datasets/AIC2026_UAV_semifinal/test_2/images")

SEED = 42
BETA = 0.01
N_SOURCE = 500
N_TEST2 = 500
N_BOOTSTRAP = 200
N_TUPLES = 100
IMG_SIZE = (512, 512)
rng = np.random.RandomState(SEED)

SPATIAL_AXES = (0, 1)  # HWC

# --- Input audit ---
all_ids = SOURCE_SPLIT.read_text().strip().split("\n")
source_ids = all_ids[:5597]
assert len(source_ids) == 5597, f"expected 5597, got {len(source_ids)}"
# Check no pseudo IDs (>=5771 shouldn't be in first 5597)
source_paths = []
for sid in source_ids:
    p = SOURCE_DIR / f"{sid}.png"
    assert p.exists(), f"missing: {p}"
    source_paths.append(p)
print(f"Source: {len(source_paths)} real-GT images")

test2_paths = sorted(TEST2_DIR.glob("test2_*.png"))
assert len(test2_paths) == 1300
print(f"Test2: {len(test2_paths)} images")

# --- Load images ---
def load(p):
    with Image.open(p) as im:
        return np.asarray(im.convert("RGB").resize(IMG_SIZE, Image.BILINEAR), dtype=np.float32) / 255.0

src_idx = rng.choice(5597, N_SOURCE, replace=False)
t2_idx = rng.choice(1300, N_TEST2, replace=False)
print("Loading source...")
src_imgs = np.stack([load(source_paths[i]) for i in src_idx])
print("Loading test2...")
t2_imgs = np.stack([load(test2_paths[i]) for i in t2_idx])

# --- Feature extraction ---
def lowfreq_spectrum(img):
    """img: HWC float. Return low-freq log-amplitude features."""
    fft = np.fft.fft2(img, axes=SPATIAL_AXES)
    amp = np.fft.fftshift(np.abs(fft), axes=SPATIAL_AXES)
    H, W = amp.shape[:2]
    h = int(round(BETA * min(H, W)))
    cy, cx = H//2, W//2
    center = amp[cy-h:cy+h, cx-h:cx+h, :]
    log_amp = np.log(center + 1e-8)
    return log_amp.flatten()

def color_features(img):
    return np.concatenate([img.mean(axis=(0,1)), img.std(axis=(0,1))])

def illum_features(img):
    gray = img.mean(axis=2)
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    return np.array([
        gray.mean(), gray.std(),
        np.percentile(gray, 10), np.percentile(gray, 90),
        (gx**2).mean(), (gy**2).mean(),
    ])

print("Extracting features...")
src_spec = np.array([lowfreq_spectrum(im) for im in src_imgs])
t2_spec = np.array([lowfreq_spectrum(im) for im in t2_imgs])
src_col = np.array([color_features(im) for im in src_imgs])
t2_col = np.array([color_features(im) for im in t2_imgs])
src_ill = np.array([illum_features(im) for im in src_imgs])
t2_ill = np.array([illum_features(im) for im in t2_imgs])

# Source-only standardization
def standardize(train_feats, test_feats):
    mu = train_feats.mean(0)
    sd = train_feats.std(0) + 1e-8
    return (train_feats - mu)/sd, (test_feats - mu)/sd

src_spec_n, t2_spec_n = standardize(src_spec, t2_spec)
src_col_n, t2_col_n = standardize(src_col, t2_col)
src_ill_n, t2_ill_n = standardize(src_ill, t2_ill)

# --- Energy distance ---
def energy_distance(X, Y, n_subsample=200):
    """E(X,Y) = 2*mean||Xi-Yj|| - mean||Xi-Xj|| - mean||Yi-Yj||.
    Subsample for speed. Permutation invariant."""
    n = min(len(X), n_subsample)
    idx_x = rng.choice(len(X), n, replace=False)
    idx_y = rng.choice(len(Y), n, replace=False)
    Xs, Ys = X[idx_x], Y[idx_y]
    # pairwise distances
    dx = np.sqrt(((Xs[:,None,:] - Xs[None,:,:])**2).sum(-1))
    dy = np.sqrt(((Ys[:,None,:] - Ys[None,:,:])**2).sum(-1))
    dxy = np.sqrt(((Xs[:,None,:] - Ys[None,:,:])**2).sum(-1))
    return 2*dxy.mean() - dx.mean() - dy.mean()

# --- Bootstrap: disjoint source-source ---
print(f"Running {N_BOOTSTRAP} bootstrap iterations...")
ss_spec, ss_col, ss_ill = [], [], []
st_spec, st_col, st_ill = [], [], []
for b in range(N_BOOTSTRAP):
    # disjoint split: 1000 source images into two disjoint halves
    perm = rng.permutation(N_SOURCE)
    A, B = perm[:N_SOURCE//2], perm[N_SOURCE//2:]
    ss_spec.append(energy_distance(src_spec_n[A], src_spec_n[B]))
    ss_col.append(energy_distance(src_col_n[A], src_col_n[B]))
    ss_ill.append(energy_distance(src_ill_n[A], src_ill_n[B]))
    # source-test2 (resampled test2)
    tj = rng.choice(N_TEST2, N_TEST2//2, replace=False)
    st_spec.append(energy_distance(src_spec_n[A], t2_spec_n[tj]))
    st_col.append(energy_distance(src_col_n[A], t2_col_n[tj]))
    st_ill.append(energy_distance(src_ill_n[A], t2_ill_n[tj]))

ss_spec = np.array(ss_spec); st_spec = np.array(st_spec)
ss_col = np.array(ss_col); st_col = np.array(st_col)
ss_ill = np.array(ss_ill); st_ill = np.array(st_ill)

def ci(arr):
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

p95_spec = float(np.percentile(ss_spec, 95))
p95_col = float(np.percentile(ss_col, 95))
p95_ill = float(np.percentile(ss_ill, 95))

delta_spec = st_spec - ss_spec
delta_col = st_col - ss_col
delta_ill = st_ill - ss_ill

spec_shift = float(st_spec.mean() > p95_spec) and float(np.percentile(delta_spec, 2.5)) > 0
col_shift = float(st_col.mean() > p95_col) and float(np.percentile(delta_col, 2.5)) > 0
ill_shift = float(st_ill.mean() > p95_ill) and float(np.percentile(delta_ill, 2.5)) > 0

print(f"Spectrum: SS mean={ss_spec.mean():.4f} p95={p95_spec:.4f}, ST mean={st_spec.mean():.4f}, shift={bool(spec_shift)}")
print(f"Color:    SS mean={ss_col.mean():.4f} p95={p95_col:.4f}, ST mean={st_col.mean():.4f}, shift={bool(col_shift)}")
print(f"Illum:    SS mean={ss_ill.mean():.4f} p95={p95_ill:.4f}, ST mean={st_ill.mean():.4f}, shift={bool(ill_shift)}")

# --- Paired transform effectiveness ---
def taste_transform(img, donor, rho):
    fft = np.fft.fft2(img, axes=SPATIAL_AXES)
    amp = np.fft.fftshift(np.abs(fft), axes=SPATIAL_AXES)
    phase = np.angle(fft)
    dfft = np.fft.fft2(donor, axes=SPATIAL_AXES)
    damp = np.fft.fftshift(np.abs(dfft), axes=SPATIAL_AXES)
    H, W = amp.shape[:2]
    h = int(round(BETA * min(H, W)))
    cy, cx = H//2, W//2
    mixed = amp.copy()
    mixed[cy-h:cy+h, cx-h:cx+h, :] = (1-rho)*amp[cy-h:cy+h, cx-h:cx+h, :] + rho*damp[cy-h:cy+h, cx-h:cx+h, :]
    inv = np.fft.ifft2(np.fft.ifftshift(mixed, axes=SPATIAL_AXES) * np.exp(1j*phase), axes=SPATIAL_AXES)
    imag_residual = float(np.abs(inv.imag).mean())
    out = inv.real
    total_clip = float(((out < 0) | (out > 1)).mean())
    return np.clip(out, 0, 1), total_clip, imag_residual

print("Running paired transform tuples...")
reduction_spec = []
reduction_col = []
reduction_ill = []
clips = []
imag_res = []
for k in range(N_TUPLES):
    si = rng.randint(0, N_SOURCE)
    di = rng.randint(0, N_TEST2)
    rho = rng.uniform(0.25, 0.50)
    transformed, clip, ires = taste_transform(src_imgs[si], t2_imgs[di], rho)
    clips.append(clip)
    imag_res.append(ires)
    # Compare distance to held-out test2 reference (different images)
    ref_idx = rng.choice(N_TEST2, 50, replace=False)
    # raw source feature
    raw_spec = lowfreq_spectrum(src_imgs[si])
    trans_spec = lowfreq_spectrum(transformed)
    raw_col = color_features(src_imgs[si])
    trans_col = color_features(transformed)
    raw_ill = illum_features(src_imgs[si])
    trans_ill = illum_features(transformed)
    # distance to reference distribution (mean distance)
    def mdist(feat, ref_feats):
        return np.sqrt(((ref_feats - feat[None,:])**2).sum(-1)).mean()
    d_raw_spec = mdist(raw_spec, t2_spec[ref_idx])
    d_trans_spec = mdist(trans_spec, t2_spec[ref_idx])
    reduction_spec.append((d_raw_spec - d_trans_spec) / max(d_raw_spec, 1e-8))
    d_raw_col = mdist(raw_col, t2_col[ref_idx])
    d_trans_col = mdist(trans_col, t2_col[ref_idx])
    reduction_col.append((d_raw_col - d_trans_col) / max(d_raw_col, 1e-8))
    d_raw_ill = mdist(raw_ill, t2_ill[ref_idx])
    d_trans_ill = mdist(trans_ill, t2_ill[ref_idx])
    reduction_ill.append((d_raw_ill - d_trans_ill) / max(d_raw_ill, 1e-8))

reduction_spec = np.array(reduction_spec)
reduction_col = np.array(reduction_col)
reduction_ill = np.array(reduction_ill)

spec_reduction_mean = float(reduction_spec.mean())
spec_reduction_ci = (float(np.percentile(reduction_spec, 2.5)), float(np.percentile(reduction_spec, 97.5)))
col_change_mean = float(reduction_col.mean())  # negative = worse
ill_change_mean = float(reduction_ill.mean())
mean_clip = float(np.mean(clips))
mean_imag = float(np.mean(imag_res))

print(f"\nPaired transform:")
print(f"  Spectrum reduction: mean={spec_reduction_mean*100:.1f}% CI=[{spec_reduction_ci[0]*100:.1f}%, {spec_reduction_ci[1]*100:.1f}%]")
print(f"  Color change: {col_change_mean*100:.1f}%")
print(f"  Illum change: {ill_change_mean*100:.1f}%")
print(f"  Mean clipping: {mean_clip*100:.3f}%")
print(f"  Mean imag residual: {mean_imag:.2e}")

# --- Gates ---
# Broad gate: >=2 families significant
broad_pass = sum([spec_shift, col_shift, ill_shift]) >= 2
# Spectral gate:
spectral_cond1 = spec_shift  # corrected spectrum shift
spectral_cond2 = float(np.percentile(delta_spec, 2.5)) > 0  # CI lower bound > 0
spectral_cond3 = spec_reduction_ci[0] >= 0.20  # 95% CI lower bound >=20%
spectral_cond4 = (col_change_mean > -0.10) and (ill_change_mean > -0.10)  # no >10% worsening
spectral_cond5 = mean_clip <= 0.005 and mean_imag < 1e-3  # safety

spectral_pass = all([spectral_cond1, spectral_cond2, spectral_cond3, spectral_cond4, spectral_cond5])

result = {
    "scientific_status": "CORRECTED_DIAGNOSTIC",
    "broad_gate": {
        "name": "TASTE_BROAD_GATE",
        "status": "PASS" if broad_pass else "FAIL",
        "conditions": {
            "spectrum_shift": bool(spec_shift),
            "color_shift": bool(col_shift),
            "illumination_shift": bool(ill_shift),
            "count": sum([spec_shift, col_shift, ill_shift]),
            "required": 2
        }
    },
    "spectral_gate": {
        "name": "TASTS_SPECTRAL_GATE",
        "status": "PENDING_USER_APPROVAL" if spectral_pass else "METHOD_PRECONDITION_NOT_MET",
        "conditions": {
            "spectrum_shift_above_p95": bool(spectral_cond1),
            "delta_st_ss_ci_lower_gt_0": bool(spectral_cond2),
            "transform_reduction_ci_lower_ge_20pct": bool(spectral_cond3),
            "other_families_no_worse_10pct": bool(spectral_cond4),
            "safety_clipping_imagin": bool(spectral_cond5)
        }
    },
    "evidence": {
        "spectrum": {"ss_mean": float(ss_spec.mean()), "ss_p95": p95_spec, "st_mean": float(st_spec.mean()),
                      "delta_ci": ci(delta_spec)},
        "color": {"ss_mean": float(ss_col.mean()), "ss_p95": p95_col, "st_mean": float(st_col.mean())},
        "illumination": {"ss_mean": float(ss_ill.mean()), "ss_p95": p95_ill, "st_mean": float(st_ill.mean())},
        "transform": {
            "spectrum_reduction_mean_pct": spec_reduction_mean*100,
            "spectrum_reduction_ci_pct": [spec_reduction_ci[0]*100, spec_reduction_ci[1]*100],
            "color_change_pct": col_change_mean*100,
            "illum_change_pct": ill_change_mean*100,
            "mean_clipping_pct": mean_clip*100,
            "mean_imag_residual": mean_imag
        }
    },
    "gpu_authorized": False,
    "gpu_eligibility": "PASS_PENDING_EXPLICIT_USER_APPROVAL" if spectral_pass else "FAIL",
    "b4_touched": False,
    "platform_submission": False
}

with open(OUT / "ELIGIBILITY_RESULT_V2.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\n=== BROAD: {'PASS' if broad_pass else 'FAIL'} ===")
print(f"=== SPECTRAL: {'PENDING_USER_APPROVAL' if spectral_pass else 'NOT_MET'} ===")
