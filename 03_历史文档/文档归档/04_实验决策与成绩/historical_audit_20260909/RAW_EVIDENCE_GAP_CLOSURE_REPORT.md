# RAW EVIDENCE GAP CLOSURE REPORT

> 2026-09-09。纯事实收集，从服务器实际运行文件读取。不分析。
> 状态：RAW_EVIDENCE_GAP_CLOSURE_PASS（Task A全锁死；Task B全盘点；Exp16缺线上分=已知gap如实记录）

---

## TASK A: 69.281290 冠军最终8维校准锁死（代码级证据）

### A1-A2. 提交身份与最终向量（来源：run_calib_road_build_scan.sh + calib_build_p100/run_config.yaml）
| 字段 | 值 |
|---|---|
| online_score | 69.281290（成绩表§2.11 build_p100行）|
| submission | **calib_build_p100** |
| submission_zip | /home/jhy/calib_build_p100.zip (4,375,801 B) |
| zip_sha256 | 6be4e8b0b15aaa4a002bb99d1b1fe0677ec9fb67fe1ee32ac4e0fda5d763ee04 |
| prediction_dir | /home/jhy/calib_build_p100/predictions (500张) |
| checkpoint | stage23_w1_exp23_b3_1024_cwfa_ce_formal_seed42/best_miou.pt |
| checkpoint_sha256 | d3740b4ef73fbb3ac57ddeedc3547ce4ce1ba31442dc7c70e1fe96b1fab3e228 |
| class_order(train_id) | Background, Building, Road, Water, Barren, Vegetation, Agricultural, Vehicle |
| **final_8d_vector** | **[-0.2000, +0.1000, 0.0709, 0.7000, 3.6000, 0.4000, 3.3215, -0.5000]** |
| vector来源 | run_calib_road_build_scan.sh build_p100 行实际参数 |

### A3. Calibration公式（来源：rot_tta.py 源码 line 138-188）
- **calibration_space = LOGITS**
- **formula = `logits = logits + bias_tensor`**（bias shape (1,8,1,1)，逐通道加法）
- **scale**：给 --prior-bias 时不应用 prior_scale（prior_scale=3.0 只在默认 PRIOR_LOG_BIAS 分支用）
- bias_applied_before_tta_fusion = **NO**
- bias_applied_after_tta_fusion = **YES**（accum/n_views 之后）
- bias_applied_before_softmax = **YES**（无softmax，直接argmax）
- bias_applied_after_softmax = **NO**

### A4. TTA→Calibration精确顺序（rot_tta.py）
```
STEP1: 每视图(8个: 4旋转×2镜像) forward → logits
STEP2: inverse transform (undo flip → undo rot90)
STEP3: accum += logits (逐视图累加)
STEP4: logits = accum / 8  (TTA融合=平均logits)
STEP5: logits += bias_tensor  (8维校准，唯一一次，在融合后)
STEP6: argmax(dim=1).add(1)  (train_id 0-7 → official 1-8)
STEP7: 保存 official PNG
```

### A5. build_p100真实含义（来源：run_calib_road_build_scan.sh）
- **which_class = Building(建筑, train_id=1)**
- **old_value = -0.0384 → new_value = +0.10**
- p100 = building bias +0.10 的提交；run_config 确认 8视图D4 + prior_shift scale3.0

### A6. 证据冲突解决
- **-0.0384版向量 = 6维锁定版**（建筑未动，run_calibv2.sh那版）
- **+0.10版 = 8维最终版**（build_p100，建筑是第8个挤出维度）
- 两者都是 stage23+rotta8+校准，差在第2维建筑bias
- 见 champion_69_281290_final_calibration_truth.json/.md

---

## TASK B: SEMEDA Exp47 + UPerNet Exp16 资产盘点

### Exp47 SEMEDA（来源：服务器+成绩表§4.11）
| 项 | 值 |
|---|---|
| 身份 | stage47 B3 1024, loss=CE+SEMEDA(λ0.5,fs256,noise0.1), rare_crop, 无CWFA |
| local best | 0.809569@56000 |
| checkpoint sha | 27e1441c008ab1b90eaa18bb45f44d6e9163f635ea544f1b6469a39cbd63d13e |
| **online 67.796 归属** | ✅ 成绩表§4.11 "Exp47_SEMEDA_标准版=67.796(无TTA口径)"，确认属Exp47 |
| raw预测 | AVAILABLE /home/jhy/submit_exp47/predictions (500张, 单视图无TTA) |
| TTA8预测 | AVAILABLE /home/jhy/calibv2_exp47/predictions |
| 校准预测 | AVAILABLE calibv2_exp47(含锁定bias) = online 69.0776 |
| 提交zip | submit_exp47.zip(标准)/calibv2_exp47.zip(TTA8+校准) |

### Exp16 UPerNet（来源：服务器run_config）
| 项 | 值 |
|---|---|
| 身份 | stage15 B2 1024, decoder=UPerNet, CE, rare_crop |
| local best | 0.806580@54600 |
| checkpoint sha | f3cd64e5d7435540ed75f6d631212633090d7880c626ddd6f7436dfd8787bc20 |
| **online score** | **NO_CONFIRMED_ONLINE_SCORE**（成绩表+phase2 csv均无exp16）|
| raw test预测 | AVAILABLE submission_exp16_best_f3cd64e5 (500张, run_config:TTA false) |
| 提交zip | exp16_best_f3cd64e5.zip (5,080,052B, sha 9a13be0d...b701) |
| 可比性 | 仅报事实: Exp06(B2/768 CE)=0.787114, Exp15(B2/mixed1024)=0.798495 — 是否可比由ChatGPT判断 |

---

## 仍然UNKNOWN
1. Exp16(及Exp15/17/21/22) 线上绝对分：**无正式记录**（成绩表未收录、phase2 csv不含）
2. 冠军8维校准的"总提交/评估次数"逐次日志：无独立文件（成绩表逐行可数≥111校准行）
3. 早期zip(Exp15-22)与分数严格一一对应：无回填
4. 部分实验 best_miou.pt 完整 sha256（有 config_sha 无 pt 完整 sha）

## 来源文件索引
| 数据 | 文件 |
|---|---|
| 冠军向量/脚本 | /home/jhy/run_calib_road_build_scan.sh, /home/jhy/calib_build_p100/run_config.yaml |
| 校准公式/TTA顺序 | .../scripts/rot_tta.py line 138-188 |
| 冠军checkpoint | stage23 best_miou.pt sha d3740b4e |
| Exp47 | /home/jhy/submit_exp47*, calibv2_exp47*, launch_exp47_package.sh |
| Exp16 | /home/jhy/deploy_exp16/.../stage15_exp16_submissions/, run_config.yaml |
| 线上分 | 成绩表.md §2.11/§4.11; phase2_online_scores_exact.csv |
