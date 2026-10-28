# AIC2026 UAV 语义分割 — 项目交接（2026-09-26）

> 当前状态入口。`PROJECT_HANDOVER_20260925.md` 等旧文件按原样保留为当日快照。

## 1. 当前结论

- **当前 test2 合规冠军：A05_WEIGHT，70.775011。**
- A05_WEIGHT 是 `parent + 0.5 × (D1 - parent)` 形成的单 merged checkpoint；推理只加载一个 checkpoint。
- checkpoint SHA256：`023b65800d434069f0fb0ad881083fd439b89011f731feebc88d6e9217727134`。
- 提交 ZIP：`A05_WEIGHT_alpha05_test2.zip`。
- ZIP SHA256：`9069b2475cd7b0c8060d4f9105f7a9f9a54cb4488d0632ce2aafacb6bb3d2277`。
- 相对旧合规冠军 D1_2k（70.736134）：**+0.038877pp**。
- 相对旧非合规 logit teacher（70.763755）：**+0.011256pp**。

按项目自 2026-09-18 起采用的规则口径，参数空间合成后形成单一 merged checkpoint 可用于推理；推理期多 checkpoint logits/probability 融合禁止。相关一手官方截图仍待补档，现有规则 provenance 见 `文档归档/01_赛题与规则/官方答疑_方案边界_补充_20260918.md`。

## 2. 2026-09-26 最新线上结果

| 候选 | test2 mIoU（%） | 推理构成 | 合规状态 | 当前定位 |
|---|---:|---|---|---|
| **A05_WEIGHT** | **70.775011** | 单 merged checkpoint | 按项目当前规则口径合规 | **当前 test2 合规冠军** |
| D1_alpha050 | 70.763755 | parent/D1 双 checkpoint logits alpha=0.5 | 非合规 | 历史 teacher 行为参考 |
| D1_2k | 70.736134 | 单 checkpoint | 合规 | 旧合规冠军 |
| D1_NO_BARREN_POS_alpha050 | 70.506962 | parent/new 双 checkpoint raw-logits alpha=0.5 | **非合规** | 实验记录，不参与冠军排序 |
| theta278 | 70.586921 | 单 checkpoint | 合规 | D1 基线 |

D1_NO_BARREN_POS_alpha050 相对：

- A05_WEIGHT：**−0.268049pp**；
- D1_2k：**−0.229172pp**；
- 旧 D1_alpha050 teacher：**−0.256793pp**；
- theta278：**−0.079959pp**。

该分数属于生成 ZIP 的双 checkpoint logits 融合结果，**不能记作单个 D1_NO_BARREN_POS checkpoint 的线上分数**。它证明本地 Barren 指标改善没有迁移为 test2 总分提升，但没有 test2 GT，不能据此断言线上下降完全由 Barren 类造成。

## 3. 证据边界

两项新分数均由用户在当前会话直接回报，结构化记录标记为：

```text
evidence_level = USER_REPORTED
platform_submission_id = null
platform_submitted_at = null
platform_screenshot = null
```

当前没有平台提交 ID、提交时间或截图，不得写成“平台回执已独立核验”。文件名、SHA、包结构和生成方法已在本地材料中核对。

## 4. 文件身份

### A05_WEIGHT

- 项目报告：`A05_WEIGHT_执行报告_20260926.md`
- checkpoint：`/home/jhy/rare_diag_20260915/breakthrough_v2_20260926/03_weight_projection/A05_WEIGHT_alpha05.pt`
- checkpoint SHA256：`023b65800d434069f0fb0ad881083fd439b89011f731feebc88d6e9217727134`
- ZIP：`提交包/批次_20260926_A05_D1Barren/A05_WEIGHT_alpha05_test2.zip`
- ZIP SHA256：`9069b2475cd7b0c8060d4f9105f7a9f9a54cb4488d0632ce2aafacb6bb3d2277`
- 行为审计：teacher KL 相对 D1 降低 92.1%，candidate-vs-teacher flip 0.269%，H300 相对 D1 +0.286pp。

### D1 Barren 消融

- 归档报告：`文档归档/04_实验决策与成绩/D1_Barren_Ablation_Handover_20260926.md`
- 实验组 checkpoint SHA256：`9f55284339fe7a11de69922866ecaea553f0125a696bb9cdec2702a7b17ece5e`
- scored ZIP：`提交包/批次_20260926_A05_D1Barren/D1_NO_BARREN_POS_alpha050_test2.zip`
- ZIP SHA256：`c494d46a8db85e25f244c2a177deab00810da9c44af8ebcc55a1573a86e03d20`
- 生成方式：`0.5 × parent raw logits + 0.5 × new-model raw logits`；双 checkpoint 推理融合，非合规。

## 5. 本批次归档

```text
提交包/批次_20260926_A05_D1Barren/
├── A05_WEIGHT_alpha05_test2.zip
├── D1_NO_BARREN_POS_alpha050_test2.zip
└── BATCH_MANIFEST.json
```

本批次**有意不分配全局编号**，因为 2026-09-18 以后批次尚未统一并回旧编号账本。不要自行推算 279/280。

机器可读成绩记录：

```text
提交包/清单与记录/USER_REPORTED_SCORES_20260926.json
```

## 6. 后续决策含义

1. A05_WEIGHT 已证明：沿 D1 的正确参数方向做单 checkpoint 权重中点，比高维 ETKD 模仿更有效，并在线上超过旧 logit teacher。
2. ETKD-v1/v2 P1 保持失败结论，不因 A05 成功而重新开放。
3. D1 Barren 消融显示 source holdout 的逐类正收益仍不能预测 test2；不应继续沿“更强抑制 Barren”方向追加训练。
4. 后续若继续突破，应以 A05_WEIGHT 为当前合规基线，优先研究可归因的低维参数投影或新的 target-domain signal；不得自动提交。

## 7. 历史文件关系

- `PROJECT_HANDOVER_20260925.md`：2026-09-25 状态快照，其中“当前冠军”只代表当时记录。
- `README_从这里开始.md`、`提交包/README_先看这个.md`：正文主体整理于 2026-09-16，已在顶部增加当前入口提示。
- `_包登记表_20260916.*`、`_包分数总表_20260916.json`：旧日期机器账本，本次不修改。
