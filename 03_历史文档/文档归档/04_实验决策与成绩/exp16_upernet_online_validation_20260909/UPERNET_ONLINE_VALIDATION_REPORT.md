# Exp16 UPerNet Online Validation Report

> 2026-09-09。零训练。Exp16(B2+UPerNet)目标线上裁决准备。

## 状态: UPERNET_ONLINE_VALIDATION_PASS（2026-09-09用户提交后回填）

**Exp16 UPerNet 线上裁决完成**
- Exp16 online = **0.659078（= 65.9078百分制）**
- Exp14 baseline = 66.443
- **delta = -0.5352**（UPerNet线上低于baseline）

## 1. Exp16身份（已锁定）
| 项 | 值 |
|---|---|
| experiment_id | stage15_w1_exp16_b2_1024_upernet_ce_formal_seed42 |
| checkpoint sha | f3cd64e5d743...8787bc20 |
| backbone/decoder | SegFormer-B2 / **UPerNet** |
| resolution | 1024 |
| loss | CE |
| lr/wd/step/seed | 6e-05/0.01/56000/42 |
| **local best mIoU** | **0.806580** @54600 |

## 2. Exp16提交包（已验证合格）
- zip: exp16_best_f3cd64e5.zip (5,080,052B, sha 9a13be0d...b701)
- **500张** test1_1~500, 1024×1024, PNG L模式
- **值域 1-8**（无0/无255/无>8）实测通过
- TTA:false / ensemble:false / single-view / 无校准 / 无后处理
- 与历史成功提交格式一致

## 3. Baseline参照（Exp14）
- **baseline online = 66.443**（来源: 本地决策文档多处一致引用"Exp14（B2基线）66.443"，精确尾数.162未能在正式ledger验证→记录为文档精度66.443）
- ⚠️ 注意: phase2_online_scores只有Exp02-11(768分辨率)，无B2/1024的Exp14/16线上条目；66.443来自决策文档非主成绩表

## 4. 待用户执行的提交
把桌面 zip `exp16_upernet_online_validation_20260909.zip` 内的 `exp16_best_f3cd64e5.zip`（或直接从服务器拉该zip）提交到平台，**只提交一次**，读取线上分后回填。

## 5. Target裁决公式（用户提交后）
```
baseline_online = 66.443 (Exp14)
upernet_online = [待提交回填]
delta = upernet_online - 66.443
```

## 产物
- exp16_identity.json / exp16_submission_identity.json / baseline_online_evidence.json / online_result.json(待回填) / 本报告
