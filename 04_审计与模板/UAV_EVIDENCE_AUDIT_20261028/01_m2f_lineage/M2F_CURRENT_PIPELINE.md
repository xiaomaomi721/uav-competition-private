# M2F 当前管线与"是否值得长训"判定

证据时间：2026-10-28 20:11–20:35 CST。全部来自服务器原始文件读取。

---

## 1. 已存在的产物（VERIFIED_NOW）

`/home/jhy/aic2026_secondgpu_m2f_20261028/run_m2f_clean_secondgpu/`

| 子目录 | 内容 | 判定 |
|---|---|---|
| `long_m2f_clean_train/` | `state.json`（TRAINING, step 4796/60000）、`run_manifest.json`、`resolved_config.py` | **正在训练** |
| `review_20k/` | `SCIENTIFIC_REVIEW_2000.md`、`SOURCE1399_METRICS_2000_UNIFIED768.json`、`EMA_VERIFY_2000.json` | 2000 步复核，**塌缩** |
| `bench200/` | `benchmark.json`、`M2F_BENCHMARK_200.json`、`RESTART_TEST.json`、`microiter_profile.json` | 工程基准，非效果 |
| `smoke/` | 同上 | 工程冒烟 |

**未发现**：40 图 canary 输出、`TARGET3594_DIAGNOSTICS.json`、3594 张预测 ZIP、全量 QA。
→ **当前没有任何 M2F 版本可供队长人工上传。**

---

## 2. 2000 步复核的实际结果（这是最关键的证据）

`review_20k/SCIENTIFIC_REVIEW_2000.md` 原文摘要：

```
统一 768 协议, FP32, no TTA
ema mIoU = 0.7380%   （SCIENTIFIC_REVIEW_WARNING；塌缩：Background, Building, Road, Vegetation, Agricultural, Vehicle）
raw mIoU = 22.8262%  （警告；塌缩：Barren）
训练有效步：4623 / 60000
```

逐类（%）：

| 类 | ema | raw |
|---|---:|---:|
| Background | 0.0 | 34.57 |
| Building | 0.0 | 23.60 |
| Road | 0.0 | 20.29 |
| Water | 4.07 | 22.93 |
| Barren | 1.83 | **0.0** |
| Vegetation | 0.0 | 63.06 |
| Agricultural | 0.0 | 11.88 |
| Vehicle | 0.0 | 6.28 |

对照参考（报告自述，协议未互核）：Exp00R 79.381034 / E02 B1 79.565390。

---

## 3. 配置对照：新长训 vs 历史

| 维度 | 新长训 `m2f_clean_60k` | 历史 Exp00R / E02 B1 | 是否单变量 |
|---|---|---|---|
| 总步数 | 60,000 effective（max_iters 480,000, accum 8） | 30,000 | ❌ 变量 1 |
| schedule | LinearLR(1e-6→) 1500×8 warmup + PolyLR power 1.0 | 未核 | ❌ 变量 2 |
| EMA | `EffectiveStepManagerHook(momentum=1e-4)` | 未核 | ❌ 变量 3 |
| 节点 | 2000 / 10000 / 20000 / 40000 / 60000 | 未核 | — |
| backbone | Swin-L（来源未在本次核验） | Swin-L | 待核 |

---

## 4. 科学性判定

**`SCIENTIFIC_JUSTIFICATION_NOT_ESTABLISHED`**

理由：
1. 相对历史模型，新训练同时改变了**步数（30k→60k）、lr schedule、EMA 开关**，属多变量，无法归因。
2. config 头部注释只写 "60k effective steps (48k/64k not authorised)"，这是**工程预算冻结**，不是科研假设。
3. 没有任何文档说明"为什么 60k 会比 30k 更好"，也没有预注册的对照（如 30k 同配方重跑）。
4. **且 2000 步复核已显示严重塌缩**（EMA 0.74%，raw 22.83%，6 类归零）。

**综合判定：`NEW_LONGTRAIN_NOT_JUSTIFIED_YET`，且当前已出现 `SCIENTIFIC_REVIEW_WARNING` 级失败信号。**

---

## 5. 建议（不构成授权）

1. **优先保留 Exp00R（donor_iter_30000）作为唯一已验证 M2F 资产**（`M2F_REUSE_FIRST`），其权重 SHA 已闭合。
2. 新长训若继续，至少需要在 **10000 步节点**再做一次统一协议复核；若仍塌缩，应停止而不是跑到 60k。
3. 任何 M2F 候选若要上线，必须先补齐：40 图 canary → 3594 全量推理 → 格式 QA → SHA 链，**当前一步都没有**。
4. **不要**把 H75 的 Bstar 套用到 M2F 概率空间（两者 score 语义不同）。

---

## 6. 本次未做

- 未停止/暂停该长训（PID 2431347 仍在跑）
- 未重跑任何 M2F 推理
- 未核验 Swin-L 预训练权重来源与许可证（标 `UNKNOWN`）
