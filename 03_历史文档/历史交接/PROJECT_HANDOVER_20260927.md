# AIC2026 UAV 语义分割 — 项目交接（2026-09-27）

> 当前唯一入口。2026-09-25/26 handover 保留为历史快照。

## 1. 当前权威状态

| 候选 | test2 分数 | 工件状态 | 当前定位 |
|---|---:|---|---|
| **A05_WEIGHT** | **70.775011** | 有效；单 merged checkpoint | **当前合规冠军** |
| B4 best53200 ZERO | 68.194181 | 有效评分包 | 低于冠军，关闭部署主线 |
| MIC TREAT 旧包 | 6.976285 | **无效推理工件** | 不得作为模型/方法证据 |
| MIC TREAT 修复包 | 未评分 | **本地/远端全链审核通过；未自动提交** | 等待用户人工提交决定 |

成绩均为用户回报；没有平台 ID、提交时间或截图时不得表述为独立核验的平台回执。结构化账本：

```text
提交包/清单与记录/USER_REPORTED_SCORES_20260927.json
```

## 2. 当前冠军 A05

```text
score: 70.775011
checkpoint SHA256:
023b65800d434069f0fb0ad881083fd439b89011f731feebc88d6e9217727134
ZIP SHA256:
9069b2475cd7b0c8060d4f9105f7a9f9a54cb4488d0632ce2aafacb6bb3d2277
canonical ZIP:
提交包/批次_20260926_A05_D1Barren/A05_WEIGHT_alpha05_test2.zip
```

A05 继续作为唯一部署冠军。它的历史执行报告已归档：

```text
文档归档/04_实验决策与成绩/20260926_20260927_实验报告与执行手册/A05_WEIGHT_执行报告_20260926.md
```

## 3. B4 结论

```text
score: 68.194181
delta vs A05: -2.580830pp
checkpoint SHA256:
d7a63b0c1124f5ab213b027125322ace8ce7cf834f48fcfb45e34b20fdef6000
ZIP SHA256:
0e3da5d1259c8991dbd10d4135fdd0d754aea69dec5fb5aaa161ca4c8560b65f
canonical ZIP:
提交包/批次_20260927_B4/B4_STAGE56_6097_BEST53200_D4_ZERO_test2.zip
```

B4 与历史 B3 并非纯 matched capacity 对照，但线上差距足以关闭该 checkpoint 的部署主线。禁止 last.pt、B3 Bstar 迁移和开放式 bias 扫描。

## 4. MIC 旧提交为何无效

旧 TREAT 包：

```text
score: 6.976285
checkpoint SHA256:
e17fa56fbd809e9b8362f9d6ea3cf8723b9da3e1024ec3d8c1580825d1d455d0
ZIP SHA256:
16aba189b3e3e9eb25198d17a19dae06d547b34eb1b673b1c636010be281fae5
```

已确认两个实现缺陷：

1. 输入 `sorted(TEST2.iterdir())` 使用字符串字典序，但输出用循环数字命名，导致 1299/1300 张预测绑定到错误 basename；
2. 第 8 个 D4 view 重复且 inverse 不正确。

同名 MIC/A05 像素一致率只有 `20.8959%`，按旧字典序身份重排后升至 `94.6596%`，因此：

```text
TRAINING_COLLAPSE = NOT_SUPPORTED
MIC_METHOD_RESULT = NOT_DETERMINED
```

旧包只保存在：

```text
提交包/对照与诊断_不提交/MIC_INVALID_20260927/
```

禁止再次提交或作为方法分数使用。

## 5. MIC 修复状态

不重训，使用现有 TREAT step8000 checkpoint。新实现位于：

```text
scripts_20260927/mic_repair/
```

历史缺陷脚本原样保存在：

```text
scripts_20260927/desktop_originals/_mic_infer.py
```

修复要求：输入/输出 basename 逐项相同、完整 D4 round-trip、A05 8 图 byte-exact 金丝雀、物理 GPU2、单 checkpoint、新输出目录、新 `FIXED` ZIP、服务器/本地 SHA 一致、禁止自动提交。

修复已完成且全链审核通过：

```text
local ZIP:
提交包/待提交/批次_20260927_MIC_FIXED/MIC_TREAT_S20260927_step8000_D4_TTA8_Bstar_FIXED_test2.zip

ZIP SHA256:
c9b67be903090722834c8d4e4f3f9d82e886559320202180fe919e0bfc667e41
```

验收结果：

- CPU 单测 5/5 PASS；
- CUDA-hidden fail-closed 且未创建输出；
- 只使用物理 GPU2；
- A05 8 图金丝雀 8/8 byte-exact；
- 1300/1300 input/output basename 逐项一致；
- ZIP 顺序、CRC、flat root、PNG mode/shape/dtype/labels 全通过；
- server/local ZIP SHA 一致；
- 未重训、未读取 test2 GT、未自动提交。

该包已具备人工提交的工程资格，但仍未评分；任何无 GT behavior 统计不得称准确率。

## 6. GPU 与执行治理

```text
GPU0/1 属于他人
GPU3 禁用
只允许 physical GPU2
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=2
logical cuda:0
```

无人机任务由主助手端到端负责，不再委托其他 AI 实现、审核或给出提交结论。任何平台提交都必须由用户人工完成。

## 7. 整理记录

两个桌面的 UAV 文件已按显式 allowlist 归拢；个人、学校、课程与快捷方式文件未触碰。移动前后 SHA 和未触碰项检查位于：

```text
审计记录/桌面归拢_20260927/
```

旧执行手册均为历史快照，不是当前授权命令；集中归档在：

```text
文档归档/04_实验决策与成绩/20260926_20260927_实验报告与执行手册/
```
