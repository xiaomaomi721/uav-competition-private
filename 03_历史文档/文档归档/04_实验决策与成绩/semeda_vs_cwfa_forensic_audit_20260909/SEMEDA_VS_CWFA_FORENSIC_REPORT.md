# SEMEDA vs CWFA Forensic Audit Report

> 2026-09-09。Exp23(B3+CWFA) vs Exp47(B3+SEMEDA) 在1399固定val的公平对比。无TTA无校准单视图。只报事实。
> 推理: Exp23用Stage1已有1399预测; Exp47新推理(seme da纯loss,标准B3结构,与Exp23同口径可比)。

---

## 0. 模型身份（严格可比性确认）
| 项 | Exp23 | Exp47 |
|---|---|---|
| experiment_id | stage23_w1_exp23_b3_1024_cwfa | stage47_w1_exp47_b3_semeda |
| checkpoint sha | d3740b4ef73...e228 | 27e1441c008...13e |
| backbone | SegFormer-B3 | SegFormer-B3 (SAME) |
| resolution | 1024 | 1024 (SAME) |
| loss | CE | CE+SEMEDA(λ0.5) — **唯一loss差异** |
| 训练增广 | RareCrop+CWFA | RareCrop (CWFA OFF) — **注意: 除SEMEDA外CWFA也不同** |
| lr/wd/step/seed | 6e-05/0.01/56000/42 | 同 (SAME) |
| 推理 | 单视图无TTA无校准 | 单视图无TTA无校准 (SAME) |

⚠️ **NOT_STRICT_SINGLE_FACTOR**: Exp23含CWFA而Exp47无——两者差异=SEMEDA(-CWFA+SEMEDA)。故对比反映的是"用SEMEDA替换CWFA"的整体效果，非SEMEDA单独add-on效果。严格SEMEDA单独效应需 Exp23(noCWFA) vs Exp47，但Exp23本无no-CWFA变体作为同seed对照。此限制如实记录。

## 1. Exp47校准真相
- **exp47_calibration_vector** = `[-0.2000, -0.0384, 0.0709, 0.7000, 3.6000, 0.4000, 3.3215, -0.5000]`(来源 launch_exp47_package.sh实际--prior-bias)
- **vs冠军build_p100**: 仅Building维度不同(Exp47 -0.0384 vs 冠军 +0.10)。其余7维全同。
- **无TTA8-only结果**: calibv2_exp47是TTA8+校准(非TTA8-only)。→ NO_CONFIRMED_EXP47_TTA8_ONLY_RESULT

## 2. 总体
- Exp23 mIoU = **0.8110** / Exp47 = **0.8096** → **delta -0.0014**
- source-val上SEMEDA(CWFA替换版)不优于CWFA

## 3. 各类IoU delta (Exp23→Exp47)
| 类 | ΔIoU |
|---|---|
| Road | +0.0049 (最大改善) |
| Vehicle | +0.0024 |
| Water | +0.0008 |
| Building | -0.0018 |
| Background | -0.0023 |
| Vegetation | -0.0029 |
| Barren | -0.0033 |
| Agricultural | **-0.0091** (最大退步) |

## 4. Domain (修正口径, present-only)
| Domain | Exp23 | Exp47 | Δ |
|---|---|---|---|
| D0 城郊/荒地 | 0.685 | 0.682 | -0.003 |
| D1 农田/水域 | 0.699 | 0.695 | -0.003 |
| D2 城市 | 0.888 | 0.889 | +0.001 |
→ 无domain显著改善; D0/D1(困难域)均略降

## 5. Confusion关键变化
**改善**: Agricultural→Background -430K(-6.6%) / Background→Road -1.33M(-15.4%) / Building→Background -997K(-6.3%)
**恶化**: Background→Agricultural +1.77M(+18.4%) / Background→Barren +363K(+8.1%) / Road→Background +519K(+5.6%)
→ SEMEDA减少"正类误入背景"但新增"背景误报正类(尤其Agri)"

## 6. Error transition (1399 val逐像素)
| GT | FIXED | NEW_ERROR | 净 |
|---|---|---|---|
| Background | 11.84M | 12.93M | -1.09M |
| Vegetation | 3.94M | 4.82M | -0.88M |
| Road | 1.81M | 2.33M | -0.53M |
| Building | 3.82M | 3.40M | +0.42M |
| Agri | 2.22M | 1.88M | +0.34M |
| Barren | 1.45M | 1.41M | +0.03M |
→ 净修复集中在Building/Agri; Background/Veg净恶化

## 7. Boundary/Interior
- Background: >10px深内部 fn_delta **+1.13M**(Exp47深内部恶化); ≤2px -0.12M(边界微改善)
- Building: ≤2px -152K / 2-5px -157K / 5-10px -221K (边界带改善); >10px +112K
→ **SEMEDA改善集中在中短距离边界带(Building)，Background深内部恶化**。不能称boundary-only，亦非deep-only；类间模式异。

## 8. Barren专项
- Barren IoU 0.6113→0.6079 (Δ-0.0033); Barren→Background +34K(微恶化)
- FIXED 1.45M vs NEW_ERROR 1.41M → 净≈0
- **source-val上SEMEDA未改善Barren**(P0目标未达)

## 9. Agricultural专项
- Agri IoU 0.8013→0.7921 (Δ-0.0091)
- Agri→Background -430K(改善), 但Background→Agri +1.77M(恶化) → 净负
- source-val上Agri净退步

## 10. 主要修复 vs 新增错误
**修复最明显**: Building误入Background(-997K), Background误入Road(-1.33M→实际是Road召回↑)
**新增最明显**: Background→Agricultural(+1.77M), Background→Barren(+363K), Background→Veg(+760K)

## 11. SEMEDA vs CWFA行为关系
基于预测行为: **二者解决不同问题**(CWFA=域泛化正则提升整体；SEMEDA替换后净≈0但错误结构改变——减少正类漏检进背景、新增背景误报)。source-val上SEMEDA不优于CWFA; online上Exp47标准+1.056(67.796 vs 66.739)但此增益无法在source-val复现→指向test域特性(域差异),非source-val可测能力。
**结论: 从source-val行为看, SEMEDA与CWFA解决不同问题(UNKNOWN机制, 行为证据=错误结构不同)**。

## 12. Exp16可比性
Exp14(B2/1024/CE/RareCrop/SegFormer原生decoder, local 0.804564) vs Exp16(+UPerNet, local 0.806580) = **STRICT单因子**(唯一差异decoder_type:upernet; lr/wd/step/seed/crop全同)
