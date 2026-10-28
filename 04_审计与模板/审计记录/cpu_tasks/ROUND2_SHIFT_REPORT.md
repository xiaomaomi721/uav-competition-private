# Round2 Domain-Shift Report (test1 → test2)

Date: 2026-09-21
Scope: semifinal `test_2` (1300 images) vs prelim `test_1` (500 images) vs source (6996 official labeled).
Constraint honored: NO training, NO auto-submit, NO GT reading for test1/test2. All quantities below are prediction-behavior / distribution statistics, never accuracy or error.

## Verdict: SHIFT = MEDIUM

test2 is a real but modest domain shift from test1. It is *not* LOW (every measurement direction agrees there is a gap) and *not* HIGH (test2 sits closer to test1 than either is to source, and the model's own confidence/agreement does not collapse on test2). The safest operational read: **test2 should be scored with the same champion line (261/278 + Bstar), no emergency re-ranking, but Day1 must include a 278-based submission to hedge the ~2× larger 261↔278 disagreement on test2.**

## Evidence

### 1. Data integrity — PASS
- test2 archive SHA256 `c8784dba28f19b86e7043bced92e42f98b346ae3f3ea13326f51ef6af295b65a`, 2140500706 bytes, 1300 PNGs, names continuous `test2_1..1300`, all 1024×1024 RGB. Manifest `test2_files_sha256.json` (1300 entries).
- No missing/duplicate/corrupt files. Clean.

### 2. Source-only probe identity — PASS
- Exp18 source-only SegFormer-B3, checkpoint SHA `66667268…a88fd`, source val mIoU 0.8105, step 47600. Negative provenance checks all clean (no test1/pseudo/UDA/teacher/resume). This probe is genuinely source-only; deep features inherit that provenance.

### 3. Image-space shift (69-D traditional, source6996-only fit, matched-size bootstrap)
| metric | test1 (500) | source→test2_sub500 (100×) | source internal 500v500 |
|---|---|---|---|
| Fréchet | 4.175 | 5.375 ± 0.758 | 2.685 ± 0.690 |
| centroid | 1.021 | 1.226 ± 0.147 | 0.477 ± 0.106 |
| MMD² (unbiased) | 0.00401 | 0.00740 ± 0.00222 | −0.00016 ± 0.00056 |

- test1 Fréchet sits at the **6th percentile** of the test2-sub500 distribution → test2 is typically *further* from source than test1 was.
- source→test2 MMD² 0.0074 ≫ source-internal 0.0000: a real, nonzero shift. But its magnitude is image-level, not catastrophic.

### 4. Feature-space shift (Exp18 source-only pooled stage2/stage3, source6996-only fit, matched-size bootstrap)
| stage | source→test1 Fréchet | source→test2_full Fréchet | source→test2_sub500 Fréchet | source internal 500v500 | test1 pct within test2_sub500 |
|---|---|---|---|---|---|
| stage2 | 32.67 | 34.95 | 37.58 ± 2.63 | 9.90 ± 1.49 | 0.04 |
| stage3 | 109.23 | 123.63 | 127.87 ± 13.19 | 16.59 ± 3.11 | 0.13 |

- PCA 64 keeps 91% (stage2) / 94% (stage3) variance — the distance signal is real, not noise.
- Consistent monotone ordering **source < test1 < test2** in Fréchet, centroid, MMD², kNN-to-source at both depths. test1 Fréchet lands at 4–13th percentile of test2_sub500 → test2 is a *stronger* source-shift than test1.
- test1↔test2 direct Fréchet (7.39 / 14.28) is ~1/5 of source↔test2. test2 therefore仍属于与test1相近的广义 target family，但在多种测量空间中表现出更大的 source distance。未计算 source→test1 与 source→test2 centroid-shift vector cosine，因此不宣称二者沿同一 source→target 梯度。
- Cluster occupancy confirms shared support (occupancies overlap; test2 slightly more concentrated on a few clusters than test1's near-uniform 12-way spread).

### 5. Model-space (theta278 raw caches; 278-test1 & 278-test2, native-256 D4 TTA8)
| quantity | test1 | test2 | Δ (t2−t1) |
|---|---|---|---|
| confidence mean | 0.9569 | 0.9487 | −0.008 |
| entropy mean | 0.1099 | 0.1287 | +0.019 |
| D4 disagreement mean | 0.0150 | 0.0215 | +0.0064 |
| 261→278 flip % | 1.535 | 3.281 | +1.75 pp |
| high-conf (≥0.9) flip / changed | 0.015 | 0.215 | +0.20 |
| old-Bstar flip % on 278 | 2.782 | 3.337 | +0.56 pp |

- Every direction agrees: test2 is mildly less confident, higher entropy, higher internal D4 disagreement, and the two champion checkpoints (261 vs 278) diverge ~2.1× more on test2.
- The 261↔278 disagreement is concentrated in **high-confidence** pixels on test2 (21.5% of changed pixels carry ≥0.9 conf, vs 1.5% on test1): **278在test2更频繁地推翻261原本高置信的决策；无test2 GT，因此不判断这些flip正确与否。** This is the single strongest reason to include both 278 and 261 on Day1.
- Class share shifts are small (e.g. Background +4.4pp, Vegetation −2.4pp test1→test2). **旧Bstar在test2未表现gross mismatch，作用方向与test1相似，因此Day1把它作为两个模型的固定共同算子；是否为test2最优calibration仍待平台验证。**

## Basis for MEDIUM (not LOW, not HIGH)
- **Not LOW**: image-space MMD² 0.0074 vs source-internal 0.0000; deep Fréchet source→test2 is 3–8× source-internal; model confidence/entropy/D4 all move the same way; 261↔278 disagreement doubles. A real gap exists.
- **Not HIGH**: test2 is ~5× closer to test1 than to source in deep space; confidence stays high (median 0.999), entropy low (median 0.009); no class-share collapse; old Bstar shows no gross mismatch on test2. Nothing resembles a catastrophic domain break.

## Operational implications (no action taken yet)
1. Do not re-rank the champion line based on domain alone — 261/278 remain the correct candidates.
2. test2's larger 261↔278 divergence argues for **submitting 278 first**, not 261.
3. No new training before Day1; the measured shift is small enough that adaptation (if any) should be decided only after Day1 scores land.
