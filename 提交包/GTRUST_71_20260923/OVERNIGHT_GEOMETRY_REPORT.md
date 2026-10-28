# OVERNIGHT GEOMETRY REPORT
## G-TRUST Geometry Pipeline — 2026-09-23

**线上基准**: theta278 + old Bstar + D4 = **70.569757**
**状态**: 全部完成，未提交平台，未自动进入4k/8k

---

## 1. H300 Parity Gate

| 指标 | 预期值 | 实测值 | 绝对偏差 | 容差 | 判定 |
|------|--------|--------|----------|------|------|
| G0 (BASE) | 0.7301584631 | 0.7301584631 | 0.000000 | — | EXACT MATCH |
| G1 (PLAIN) | 0.7387726099 | 0.7391135363 | 0.000341 | <0.001 | PASS |

**Parity: PASS** — 实现与历史锚点一致，未进入R1099/test2前已验证。

---

## 2. H300 冻结结果 (300张)

| 候选 | mIoU | Δ vs base (pp) | 95% CI (pp) | flip vs base |
|------|------|----------------|-------------|--------------|
| BASE (G0) | 0.730158 | — | — | — |
| PLAIN_G1 | 0.739114 | +0.8955 | [+0.480, +1.314] | 4.23% |
| GTRUST_T | 0.738903 | +0.8745 | [+0.625, +1.100] | 2.07% |
| GTRUST_TE | 0.734617 | +0.4458 | [+0.332, +0.550] | 1.00% |

---

## 3. R1099 冻结验证 (1099张)

| 候选 | mIoU | Δ vs base (pp) | 95% CI (pp) | HARD_FAIL判定 |
|------|------|----------------|-------------|---------------|
| BASE (G0) | 0.810461 | — | — | — |
| PLAIN_G1 | 0.809978 | **-0.048** | [-0.408, +0.297] | **UNCERTAIN** (CI跨0, Δ>-0.30pp, 无大类下降) |
| GTRUST_T | 0.816724 | **+0.626** | [+0.393, +0.869] | **QUALIFIED** |
| GTRUST_TE | 0.813805 | **+0.334** | [+0.218, +0.462] | **QUALIFIED** |

### HARD_FAIL 逐项检查 (PLAIN_G1)
- CI上界 < 0? → +0.297 > 0 → **否**
- ΔmIoU ≤ -0.30pp? → -0.048 > -0.30 → **否**
- 任一类降 ≥ 2.0pp? → 最大降幅 class2=-0.896pp → **否**
- 两类各降 ≥ 1.0pp? → 无类别降幅≥1pp → **否**

**结论**: PLAIN_G1 不是 HARD_FAIL，标记 UNCERTAIN，允许生成 test2。

---

## 4. test2 无GT审计 (1300张)

### 4.1 Flip rate vs theta278 whole

| 候选 | flip rate (pixel) |
|------|-------------------|
| PLAIN_G1 | ~3.01% |
| GTRUST_T | ~1.30% |
| GTRUST_TE | ~0.61% |

### 4.2 Sanity checks
- GTRUST_T flip ≤ PLAIN_G1 flip? → 1.30% ≤ 3.01% → **PASS**
- GTRUST_TE flip ≤ PLAIN_G1 flip? → 0.61% ≤ 3.01% → **PASS**
- GTRUST accepted transitions 来自 H300 冻结允许表? → **PASS** (代码逻辑强制)
- 输出PNG: 1024×1024, uint8, IDs 1-8, 1300张 → **PASS** (pack脚本验证)

### 4.3 Class share (test2 1300张)

| Class | PLAIN_G1 | GTRUST_T | GTRUST_TE |
|-------|----------|----------|-----------|
| 0 | 33.23% | 33.33% | 32.70% |
| 1 | 17.56% | 17.41% | 17.61% |
| 2 | 6.70% | 6.87% | 6.87% |
| 3 | 7.86% | 7.90% | 7.88% |
| 4 | 1.29% | 1.28% | 1.44% |
| 5 | 22.21% | 22.22% | 22.17% |
| 6 | 10.62% | 10.46% | 10.79% |
| 7 | 0.53% | 0.54% | 0.54% |

无类别share漂移>3pp或相对变化>50%。无RED_FLAG。

### 4.4 Mask disagreement

| 对比 | mask disagreement | 判定 |
|------|-------------------|------|
| GTRUST_T vs PLAIN_G1 | ~1.71% | >0.20%, 信息价值充足 |
| GTRUST_TE vs PLAIN_G1 | ~2.40% | >0.20%, 信息价值充足 |

无 LOW_INFORMATION_VALUE。

---

## 5. 打包产物

| 文件 | SHA256 | 大小 |
|------|--------|------|
| PLAIN_G1_theta278_G1.zip | `2641e6023b78ac91...` | 12.8 MB |
| GTRUST_T_theta278_G1.zip | `e8035cafd2e8d1fb...` | 12.8 MB |
| GTRUST_TE_theta278_G1.zip | `97db9bfaa681adca...` | 12.7 MB |

每个ZIP: 1300 PNG, flat root, test2_1.png ~ test2_1300.png, 1024×1024, uint8 L模式。

---

## 6. 明天提交槽推荐

### Submission A: PLAIN_G1
**推荐**: 是
**理由**:
- H300 parity PASS
- R1099 UNCERTAIN (非HARD_FAIL)
- test2 无输出异常
- 最直接的Geometry方案，H300证据最强 (+0.90pp)
- R1099略负但CI跨0，不构成拒绝理由

### Submission B: GTRUST_TE
**推荐**: 是
**理由**:
- R1099 QUALIFIED (+0.33pp, CI正)
- 多一层entropy rejection，风险更低
- 与PLAIN_G1 mask disagreement ~2.40%，不是重复提交
- 测试的承重假设与PLAIN互补: PLAIN信任全部局部证据，TE只在熵改善时信任
- 备选: 若TE线上表现差，GTRUST_T (+0.63pp R1099) 可作为第三槽

### 不推荐自动提交
- 以上仅为推荐，**未自动提交平台**
- 需要人工在比赛平台上传ZIP
- 建议A槽先交PLAIN_G1，B槽交GTRUST_TE

---

## 7. 本轮未做的事
- 未修改任何算法/阈值/超参
- 未回H300调gate
- 未新开DART/FDA
- 未重新训练theta278
- 未自动提交平台
- 未生成test2 GT对比
