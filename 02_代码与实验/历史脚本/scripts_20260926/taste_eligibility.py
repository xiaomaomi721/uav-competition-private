"""TAST-E eligibility diagnostic (CPU only).

Checks whether low-frequency amplitude shift between source and test2
is large enough to justify Fourier amplitude adaptation.
"""
import json, time
from pathlib import Path
import numpy as np
from PIL import Image

SOURCE_DIR = Path("/home/jhy/datasets/pseudo_train_6796/train/images")
TEST2_DIR = Path("/home/jhy/datasets/AIC2026_UAV_semifinal/test_2/images")
OUT_DIR = Path("/home/jhy/rare_diag_20260915/taste_20260926/02_eligibility")
BETA = 0.01
RHO_RANGE = (0.25, 0.50)
SEED = 42
N_BOOTSTRAP = 100
N_SOURCE_SAMPLE = 500
N_TEST2_SAMPLE = 500

OUT_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.RandomState(SEED)

def load_image(path, size=(512,512)):
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB").resize(size, Image.BILINEAR), dtype=np.float32) / 255.0
    return arr

def fft_amp_phase(img):
    fft = np.fft.fft2(img, axes=(-2,-1))
    amp = np.abs(fft)
    phase = np.angle(fft)
    amp_shifted = np.fft.fftshift(amp, axes=(-2,-1))
    return amp_shifted, phase

def lowfreq_half(H, W, beta=BETA):
    h = int(round(beta * min(H, W)))
    return h

def radial_band_stats(amp_shifted, h):
    """Extract low-freq log-amplitude stats from center window."""
    H, W = amp_shifted.shape[:2]
    cy, cx = H//2, W//2
    center = amp_shifted[cy-h:cy+h, cx-h:cx+h, :]
    log_amp = np.log(center + 1e-8)
    return log_amp.mean(axis=(0,1)), log_amp.std(axis=(0,1))

def color_stats(img):
    return img.mean(axis=(0,1)), img.std(axis=(0,1))

def illum_stats(img):
    gray = img.mean(axis=2)
    return gray.mean(), gray.std(), np.percentile(gray, 10), np.percentile(gray, 90)

def taste_transform(src_img, donor_img, rho):
    """Apply Fourier amplitude swap in low-freq window."""
    H, W = src_img.shape[:2]
    h = lowfreq_half(H, W)
    src_fft = np.fft.fft2(src_img, axes=(-2,-1))
    src_amp = np.abs(src_fft)
    src_phase = np.angle(src_fft)
    src_amp_s = np.fft.fftshift(src_amp, axes=(-2,-1))

    donor_fft = np.fft.fft2(donor_img, axes=(-2,-1))
    donor_amp = np.abs(donor_fft)
    donor_amp_s = np.fft.fftshift(donor_amp, axes=(-2,-1))

    cy, cx = H//2, W//2
    mixed = src_amp_s.copy()
    mixed[cy-h:cy+h, cx-h:cx+h, :] = (1-rho)*src_amp_s[cy-h:cy+h, cx-h:cx+h, :] + rho*donor_amp_s[cy-h:cy+h, cx-h:cx+h, :]

    mixed_unshifted = np.fft.ifftshift(mixed, axes=(-2,-1))
    out = np.fft.ifft2(mixed_unshifted * np.exp(1j*src_phase), axes=(-2,-1)).real

    pre_clamp_low = float((out < 0).mean())
    pre_clamp_high = float((out > 1).mean())
    out_clamped = np.clip(out, 0, 1)
    return out_clamped, pre_clamp_low, pre_clamp_high

print("Scanning source images...")
source_ids = open("/home/jhy/datasets/pseudo_train_6796/train_ids_5597pseudo.txt").read().strip().split("\n")
source_paths = [SOURCE_DIR / f"{sid}.png" for sid in source_ids if (SOURCE_DIR / f"{sid}.png").exists()]
print(f"Source images found: {len(source_paths)}")

test2_paths = sorted(TEST2_DIR.glob("test2_*.png"))
print(f"Test2 images found: {len(test2_paths)}")

# Sample
src_idx = rng.choice(len(source_paths), N_SOURCE_SAMPLE, replace=False)
t2_idx = rng.choice(len(test2_paths), N_TEST2_SAMPLE, replace=False)

print("Loading source images...")
src_imgs = np.stack([load_image(source_paths[i]) for i in src_idx])
print("Loading test2 images...")
t2_imgs = np.stack([load_image(test2_paths[i]) for i in t2_idx])

# Compute stats per image
print("Computing spectrum stats...")
H, W = 512, 512
h = lowfreq_half(H, W)

src_spec = np.array([radial_band_stats(fft_amp_phase(img)[0], h)[0] for img in src_imgs])
t2_spec = np.array([radial_band_stats(fft_amp_phase(img)[0], h)[0] for img in t2_imgs])

src_color = np.array([color_stats(img)[0] for img in src_imgs])
t2_color = np.array([color_stats(img)[0] for img in t2_imgs])

src_illum = np.array([illum_stats(img)[:2] for img in src_imgs])
t2_illum = np.array([illum_stats(img)[:2] for img in t2_imgs])

# Distances
def mmd2(a, b):
    d = a - b
    return float(np.mean(d**2))

def mdd(a, b):
    return float(np.mean((a-b)**2))

# Source-source bootstrap
print("Running source-source bootstrap...")
ss_distances = []
for _ in range(N_BOOTSTRAP):
    i = rng.choice(N_SOURCE_SAMPLE, N_SOURCE_SAMPLE//2, replace=False)
    j = rng.choice(N_SOURCE_SAMPLE, N_SOURCE_SAMPLE//2, replace=False)
    ss_distances.append({
        "spectrum": mmd2(src_spec[i], src_spec[j]),
        "color": mmd2(src_color[i], src_color[j]),
        "illumination": mmd2(src_illum[i], src_illum[j]),
    })

ss_spec_95 = np.percentile([d["spectrum"] for d in ss_distances], 95)
ss_color_95 = np.percentile([d["color"] for d in ss_distances], 95)
ss_illum_95 = np.percentile([d["illumination"] for d in ss_distances], 95)

# Source-test2 distance
st_spec = mmd2(src_spec, t2_spec)
st_color = mmd2(src_color, t2_color)
st_illum = mmd2(src_illum, t2_illum)

print(f"\nSource-Source 95th percentile: spec={ss_spec_95:.6f}, color={ss_color_95:.6f}, illum={ss_illum_95:.6f}")
print(f"Source-Test2 distance: spec={st_spec:.6f}, color={st_color:.6f}, illum={st_illum:.6f}")

above_spec = st_spec > ss_spec_95
above_color = st_color > ss_color_95
above_illum = st_illum > ss_illum_95
n_above = sum([above_spec, above_color, above_illum])
print(f"Above 95th: spectrum={above_spec}, color={above_color}, illumination={above_illum} ({n_above}/3)")

# TAST transform effectiveness
print("\nTesting TAST transform...")
rho = (RHO_RANGE[0] + RHO_RANGE[1]) / 2
clip_rates = []
transformed_spec_dists = []
transformed_color_dists = []

for k in range(50):
    si = rng.randint(0, N_SOURCE_SAMPLE)
    di = rng.randint(0, N_TEST2_SAMPLE)
    transformed, cl, ch = taste_transform(src_imgs[si], t2_imgs[di], rho)
    clip_rates.append(max(cl, ch))
    t_spec = radial_band_stats(fft_amp_phase(transformed)[0], h)[0]
    t_color = color_stats(transformed)[0]
    transformed_spec_dists.append(mmd2(t_spec[None,:], t2_spec[di:di+1]))
    transformed_color_dists.append(mmd2(t_color[None,:], t2_color[di:di+1]))

mean_clip = np.mean(clip_rates)
# Compare: distance from transformed source to test2 vs distance from raw source to test2
# Per-image: raw source spec vs test2 spec
raw_src_to_t2_spec = np.mean([mdd(src_spec[i], t2_spec[j]) for i,j in zip(range(50), rng.choice(N_TEST2_SAMPLE, 50, replace=False))])
taste_src_to_t2_spec = np.mean(transformed_spec_dists)
raw_src_to_t2_color = np.mean([mdd(src_color[i], t2_color[j]) for i,j in zip(range(50), rng.choice(N_TEST2_SAMPLE, 50, replace=False))])
taste_src_to_t2_color = np.mean(transformed_color_dists)

spec_reduction = (raw_src_to_t2_spec - taste_src_to_t2_spec) / max(raw_src_to_t2_spec, 1e-12)
color_reduction = (raw_src_to_t2_color - taste_src_to_t2_color) / max(raw_src_to_t2_color, 1e-12)

print(f"\nTAST effect (rho={rho}):")
print(f"  Spectrum distance: raw={raw_src_to_t2_spec:.6f} -> transformed={taste_src_to_t2_spec:.6f} (reduction={spec_reduction*100:.1f}%)")
print(f"  Color distance: raw={raw_src_to_t2_color:.6f} -> transformed={taste_src_to_t2_color:.6f} (reduction={color_reduction*100:.1f}%)")
print(f"  Mean clipping: {mean_clip*100:.3f}%")

# PASS conditions
condition1 = n_above >= 2
condition3_spec = spec_reduction >= 0.20
condition3_color = color_reduction >= 0.20
condition3 = condition3_spec or condition3_color
condition5 = mean_clip <= 0.005

status = "PASS" if (condition1 and condition3 and condition5) else "METHOD_PRECONDITION_NOT_MET"

report = {
    "status": status,
    "gpu_training_authorized_by_diagnostic": False,
    "source_source_95th": {"spectrum": ss_spec_95, "color": ss_color_95, "illumination": ss_illum_95},
    "source_test2_distance": {"spectrum": st_spec, "color": st_color, "illumination": st_illum},
    "above_95th": {"spectrum": above_spec, "color": above_color, "illumination": above_illum, "count": n_above},
    "tast_effect": {
        "rho": rho,
        "spectrum_distance_reduction_pct": spec_reduction*100,
        "color_distance_reduction_pct": color_reduction*100,
        "mean_clipping_pct": mean_clip*100,
    },
    "conditions": {
        "two_families_above_95th": condition1,
        "tast_reduces_distance_20pct": condition3,
        "clipping_under_0.5pct": condition5,
    },
    "note": "PASS only makes method eligible; user approval still required"
}

with open(OUT_DIR / "ELIGIBILITY_RESULT.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"\n=== RESULT: {status} ===")
print(json.dumps(report, indent=2))
