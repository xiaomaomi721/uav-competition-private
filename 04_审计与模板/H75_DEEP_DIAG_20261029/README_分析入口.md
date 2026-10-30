# H75 深度诊断包｜分析入口

日期：2026-10-29　主机：`teacher-4090`（`jhy`）
性质：**只读诊断**。未训练、未提交、未覆盖冠军资产。

---

## 一、先读哪一份

**只想看结论 → `05_FINAL_RESEARCH_REPORT/深度诊断总报告.md`**

它回答了任务书第五节的 8 个问题，并给出唯一推荐的下一项实验。

---

## 二、核心结论（30 秒版）

1. **Bstar 是"目标域先验补偿项"，不是"模型能力改进项"。**
   在源域它纠正 2177 万、致错 4376 万（比值 0.497），几乎纯亏；在 test3 上却是 **+3.02pp**。

2. **反转的原因是先验轴不匹配：**
   - Source val 的 Barren GT 先验 = **2.10%**
   - H75 在 Target 上 raw 只预测 **0.74%**
   - Bstar 的 Barren +3.6 在 Target 上补真实亏缺，在 Source 上无病吃药（把 precision 从 79.4 打到 48.4）

3. **Source1399 是按"稀有类富集"构造的**（split JSON 原文：`rarity/coverage greedy`），
   Barren 先验是 train 的 **1.19~1.84 倍**。所以凡是"压 Barren bias"的改动在 val 上都显得对，在 Target 上都错。

4. **这解释了全部历史反转**：Source1399 分数越高（越接近 Zero 行为），线上越低。

5. **推荐的下一个实验**：把 Source1399 按类别重加权到 Target 先验后再算 mIoU，检验排序是否反转。**零训练、纯 CPU、可证伪。**

---

## 三、目录说明

```
H75_DEEP_DIAG_20261029/
├── README_分析入口.md                     ← 本文件
├── SHA256SUMS.txt                          ← 全部交付文件哈希
│
├── 00_研究问题与证据清单/
│   ├── 资产身份与SHA256.csv
│   ├── 已有实验与线上成绩对照.csv
│   └── 缺失证据与分析限制.md
│
├── 01_H75_CODE/
│   ├── H75_完整生成链分析.md
│   ├── 模型训练与推理实现核查.md
│   └── 源码/                               ← 推理与模型代码（不含权重）
│
├── 02_DATA_DISTRIBUTION/
│   ├── B_DATA_DIST.json                    ← 原始统计
│   ├── Source_Target分布差异.csv
│   └── 数据分布诊断报告.md
│
├── 03_ERROR_ANALYSIS/
│   ├── A3_ERROR_ANALYSIS.json              ← 完整混淆矩阵/逐类/尺寸/边界
│   ├── H75逐类混淆矩阵.csv
│   ├── H75逐图错误统计.csv
│   ├── Bstar纠错与致错分析.csv
│   └── 分割错误机制诊断.md
│
├── 04_PAPER_AND_IMPLEMENTATION/
│   ├── 论文机制与本地实现对照.md
│   ├── 已验证方法与未验证机制.md
│   └── 文献来源与对应实现位置.md
│
└── 05_FINAL_RESEARCH_REPORT/
    ├── 深度诊断总报告.md                   ← 主报告
    ├── 关键发现与证据等级.md
    ├── 待验证假设与最小验证方法.md
    └── 下一阶段研究建议.md
```

---

## 四、关键数字速查

| 指标 | 值 | 出处 |
|---|---:|---|
| H75 raw（源域） | 80.4418 | 本次重算，= 历史 80.4417713 |
| H75 + Bstar（源域） | 77.5311 | 本次重算，= 历史 77.5311488 |
| H75 + Bstar（线上） | **70.998665** | 用户报分 |
| H75 + Zero（线上） | 67.979564 | 用户报分 |
| Bstar 源域纠正/致错 | 2177万 / 4376万 = **0.497** | 本次 |
| Barren precision：raw → bstar | 79.42 → **48.38**（−31.04） | 本次 |
| Barren recall：raw → bstar | 70.41 → 90.90（+20.49） | 本次 |
| Source val Barren 先验 | 2.10% | 本次（600 抽样） |
| Source train Barren 先验 | 1.77%（全量）/ 1.14%（抽样） | 早前全量 / 本次抽样 |
| Target raw Barren 预测占比 | 0.74% | 本次（test3_new 300 抽样） |

---

## 五、复现方式

```bash
W=/home/jhy/aic2026_h75_deep_diag_20261029
/home/jhy/.venvs/uavseg/bin/python $W/a3_error_analysis.py   # Task C
/home/jhy/.venvs/uavseg/bin/python $W/b_data_dist.py         # Task B
```

依赖的既有缓存（只读）：
`/home/jhy/aic2026_ziponly_20261028/01_h75_cache/source1399_fp32/scores/*.npy`
`/home/jhy/aic2026_ziponly_20261028/01_h75_cache/target3594_fp32/scores/*.npy`

---

## 六、本包不含什么

按任务书要求，**不含**官方原始 RGB、GT 标签、测试图像、任何完整预测 PNG、任何模型权重。
所有分析均为统计与代码。原始数据保留在授权服务器。

## 七、本次未做

- 未启动任何训练
- 未提交任何平台结果
- 未恢复 M2F 长训 / FDA / SAM3 等已关闭任务
- 未重新推理（全部复用既有 logits 缓存）
