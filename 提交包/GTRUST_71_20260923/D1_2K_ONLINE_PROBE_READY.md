# D1-2K ONLINE PROBE READY

## 1. Checkpoint
- **Path**: `/home/jhy/rare_diag_20260915/round2_multiprobe_20260922/03_dart_fixed/D1/checkpoint_step2000.pt`
- **SHA256**: `70f45afcddfb1711a0b8576f2e335ce5fa3f88df56c48724b761716e10f34b14`

## 2. Inference Script
- **Path**: `/home/jhy/rare_diag_20260915/round2_20260921/scripts/_r2_test2_cache.py`
- **SHA256**: `e519152b5cbe19579c4a4af15503c69166644d3983751e6b6685d1d70e951eb9`
- **Note**: 完全复用 theta278 正式包推理代码，仅替换 checkpoint 参数。

## 3. Pack/Audit Script
- **Path**: `/home/jhy/rare_diag_20260915/round2_20260923/D1_2k_probe/d1_pack_audit_fast.py`
- **SHA256**: `cc7284497a5a291a8503709cf208248977fba13eee145852609b17bb2c2b95a9`

## 4. Bstar
- **Values**: `[-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5]`
- **Applied once**: YES（在256×256 raw logits上加一次，然后bilinear upsample到1024）
- **Checkpoint未folded Bstar**: YES（D1 checkpoint是纯model state dict，644 tensors，无Bstar bias）

## 5. Test2 Data
- **Count**: 1300
- **Names**: test2_1.png ... test2_1300.png
- **Archive SHA256**: `c8784dba28f19b86e7043bced92e42f98b346ae3f3ea13326f51ef6af295b65a`

## 6. Global Flip vs theta278
- **Global flip rate**: 1.933%
- **Per-image flip**: mean=1.93%, median=0.99%, p90=4.32%, p95=6.26%, max=43.49%

## 7. Class Shares
| Class | theta278 (%) | D1 (%) | Δ (pp) | Rel. change (%) |
|-------|-------------|--------|--------|-----------------|
| Background | 32.14 | 31.84 | -0.29 | -0.91 |
| Building | 17.80 | 17.87 | +0.07 | +0.42 |
| Road | 6.87 | 6.90 | +0.03 | +0.38 |
| Water | 7.87 | 8.00 | +0.13 | +1.71 |
| Barren | 1.56 | 1.58 | +0.03 | +1.66 |
| Vegetation | 22.14 | 22.27 | +0.13 | +0.59 |
| Agricultural | 11.09 | 10.99 | -0.11 | -0.96 |
| Vehicle | 0.54 | 0.55 | +0.01 | +1.48 |

## 8. Top Transitions (theta278 → D1)
1. Agricultural → Background: 3.58M pixels
2. Background → Vegetation: 3.44M pixels
3. Background → Building: 2.75M pixels
4. Background → Agricultural: 2.53M pixels
5. Background → Road: 1.93M pixels
6. Background → Water: 1.91M pixels
7. Building → Background: 1.84M pixels
8. Road → Background: 1.55M pixels
9. Vegetation → Background: 1.38M pixels
10. Barren → Background: 0.96M pixels

## 9. Changed-Pixel theta278 Confidence
- **Mean**: 0.571
- **Median**: 0.547
- **P(conf ≥ 0.90 | changed)**: 1.06%
- **P(conf ≥ 0.95 | changed)**: 0.29%
- **P(conf ≥ 0.99 | changed)**: 0.02%

## 10. Confidence & Entropy
| Metric | theta278 | D1-2k |
|--------|----------|-------|
| Mean max confidence | 0.9498 | 0.9521 |
| Mean entropy | 0.1269 | 0.1211 |

## 11. Red Flags
NONE.

## 12. ZIP
- **Path**: `/home/jhy/rare_diag_20260915/round2_20260923/D1_2k_probe/package/D1_2k_theta278_test2.zip`
- **Local**: `桌面\无人机\提交包\GTRUST_71_20260923\D1_2k_theta278_test2.zip`
- **SHA256**: `fa34f98fe119650d8eda865ae2ead33bcfd45023f02b67f3e68fbdfa3bd5308c`
- **Size**: 12.5 MB

## 13. ZIP Format Validation
- 1300 files: YES
- Names test2_1.png ... test2_1300.png: YES
- Flat ZIP root: YES
- 1024×1024: YES
- PIL mode L: YES
- uint8: YES
- Class IDs 1..8: YES

READY_FOR_MANUAL_REVIEW
