# Round2 Day1 Submission Plan

Date: 2026-09-21
Verdict from ROUND2_SHIFT_REPORT: **SHIFT = MEDIUM** → resolves to **CASE B**.
Constraint honored: no auto-submit; manual submission only; semifinal allows 2 platform submissions/day.

## Two recommended submissions

| # | Model | Prelim test1 score | Checkpoint SHA256 (stable anchor) | Role |
|---|---|---|---|---|
| **1 (submit first)** | **theta278** `atda_8k_a1p25` | **0.706154** | `c3e87f6c51bd1fb32e6249783618d9bec7f1fc431a4a36d4ef3066d54e480eac` | best-known score; domain-adaptive |
| **2 (submit second)** | **theta261** champion `A_cleanSEM_M_k100_th209` | 0.702836 | `9ffdd0beb8480a1efd361503d637d3d53cf716c808537c551fefae09f7fe2786` | conservative anchor baseline |

Both use the deployment operator **Bstar** = `[-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5]`.

## Why 278 first

1. It is the highest prelim score (0.706154 vs 261's 0.702836, +0.0033).
2. On test2 the 261↔278 models diverge ~2.1× more than on test1 (flip 3.28% vs 1.54%), and the extra disagreement is **high-confidence** (21.5% of changed pixels carry ≥0.9 conf, vs 1.5% on test1). **278在test2更频繁地推翻261原本高置信的决策；无test2 GT，因此不判断这些flip正确与否。** This makes the pair informative for testing whether the ATDA 261→278 gain transfers across test domains.
3. If the medium shift instead favors conservatism, submission #2 (261) catches that fall-back.

## Why 261 second (not another 278 variant)

The three 8k candidates scored 0.706126 / 0.706147 / 0.706154 — effectively tied. Re-submitting a sibling of 278 spends a submission for ~0.00003 expected gain. 261 is the *orthogonal* choice: the source-champion that never saw ATDA adaptation, giving the pair maximum coverage across the "does test2 reward adaptation or not" axis. This is the highest-value use of the 2nd slot.

## Required pre-submit step (inference/export, NOT training)

The scored test1 zips are 500-mask packages and cannot be reused: test2 has **1300** images, so both submission packages must be re-generated. This is forward-only inference and is permitted (the standing constraint blocks *training*, not inference).

For each candidate, from the already-computed raw native-256 D4 caches:
- **278**: `cache_278_test2_raw/scores` (1300; old run has no full-run FP16 quantization metric; its fixed 8-image sanity and loader-parity checks PASS)
- **261**: `cache_261_test2_raw/scores` (1300)
1. `native logits + Bstar` at 256×256.
2. **Bilinearly upsample logits** exactly to 1024×1024 with `align_corners=False`.
3. Argmax only after logit upsampling, then encode official class IDs 1..8 as single-channel uint8 PNG.
4. Name as `test2_<i>.png` (1..1300), zip, SHA256-anchor the zip.
5. Explicitly forbidden: `argmax@256 → nearest-neighbor label upsample`.

The previously reported **0.00261% full-run quantization argmax disagreement belongs only to the new 278-test1 cache**, not the old 278-test2 cache.

旧Bstar在test2未表现gross mismatch，作用方向与test1相似，因此Day1把它作为两个模型的固定共同算子；是否为test2最优calibration仍待平台验证。

Do NOT trust the zip filename at submission time — verify the zip SHA256 against the anchor recorded at pack time before uploading.

## Sequence
1. Export + pack **278** test2 zip → anchor SHA → user manually submits as Day1 #1.
2. Export + pack **261** test2 zip → anchor SHA → user manually submits as Day1 #2.
3. Stop. Wait for both Day1 scores before any adaptation decision (no new training until then).

## Explicit non-actions
- No auto-submit (I will never push to the platform).
- No training of any kind before Day1 scores land.
- No nearest-neighbor label prediction for test2; no GT access; no filename-guessed mapping.
