# H75 × T3-FULL-ST 深层 25% 融合｜执行报告

执行时间：2026-10-28 20:55–21:35 CST。主机：`teacher-4090`（`ubuntu-Super-Server`，`jhy`）。
**未提交平台、未启动新训练、未覆盖冠军权重、未中断任何他人任务。**

## 状态：`READY_FOR_MANUAL_SUBMISSION`（附一条 parity 说明，见 §4）

---

## 1. 输入身份（VERIFIED_NOW）

| 角色 | 路径 | 字节 | SHA256 | 校验 |
|---|---|---:|---|---|
| H75 冠军 | `/home/jhy/aic2026_h76_20261026/models/H75.pt` | 189,091,023 | `b5eea2795d8a6d90c803a15b2ca66cf0abcf62322ef1aacd0d4d0ec82098a28a` | ✅ 与期望一致 |
| T3-FULL-ST | `/home/jhy/aic2026_h76_20261026/experiments/T3_FULL_ST_20261027_import/checkpoints/checkpoint_update3594.pt` | 544,323,501 | `cb958870ce102b47eed8fa822b5ff9bfc017db7bb7e61b4e73105e46a472e8e8` | ✅ 与期望一致 |
| 已评分冠军 ZIP（只读） | `/home/jhy/aic2026_ctrl3594_teacher_20261027/reference/H75_BSTAR_20261026/H75_BSTAR_semifinal.zip` | 31,099,544 | `c012bae1237160a0c0d5fbccd168f58facf62c03bf60ee2cc82877cb21454d30` | ✅ |

T3 的 `CHECKPOINT_VERIFY.json` 记录：`strict_load=true`、`model_keys=644`、`successful_updates=3594`、`source_val_mIoU_historical=0.7762672`。

---

## 2. G0 结构与血统（PASS）

- 两侧 state_dict 均为 **644 键**，wrapper key 均为 `model`
- `missing / unexpected / shape_mismatch` 全部为空 → **结构完全兼容**
- **T3 确实起始于 H75**：`segformer.stages.0`（72 张量）与 `stages.1`（94 张量）与 H75 **逐位相同**，`max_abs_delta = 0.0`

### 分层 delta（H75 → T3）

| 组 | 张量数 | 逐位相同 | rel_L2 | max_abs_delta |
|---|---:|---:|---:|---:|
| segformer.stages.0 | 72 | **72** | 0.0 | 0.0 |
| segformer.stages.1 | 94 | **94** | 0.0 | 0.0 |
| segformer.stages.2 | 402 | 0 | 0.5966% | 0.00535 |
| segformer.stages.3 | 60 | 0 | 0.6516% | 0.00901 |
| decode_head | 16 | 3 | **3.01e-07** | 0.000358 |

**关键观察：`decode_head` 几乎未变（rel_L2 = 3e-7）**，T3 的实际改动集中在 stage2/stage3。因此本次融合实质是"把 T3 的 stage2/3 增量以 25% 注入"。逐张量明细见 `ASSET_SHA_AND_LAYER_DELTA.csv`。

---

## 3. G1 融合与自检（PASS）

固定 **α = 0.25**，无网格搜索：
```
θ_new = θ_H75 + 0.25 × (θ_T3 − θ_H75)   （FP64 计算，再转回原 dtype）
适用范围：segformer.stages.2.* / segformer.stages.3.* / decode_head.* 的浮点参数
stages.0/1 及所有 buffer（BN running stats、整数张量）→ 从 H75 原字节复制
```

| 项 | 结果 |
|---|---|
| 插值张量数 | **475** |
| 从 H75 复制张量数 | **169**（含 2 个 BN buffer） |
| 最大单元素变化 | 2.2517e-03 |
| 输出 checkpoint | `/home/jhy/aic2026_h75_t3_deep25_20261028/H75_T3_DEEP25.pt` |
| 新权重 SHA256 | **`1f46b247bacb720ff9099defbf3fa38764ebf1b42a1e066f0bccc7147549b003`** |
| 字节数 | 189,137,843 |

**自检全部 PASS**：
- `alpha0_equals_h75` = true（α=0 精确还原 H75）
- `alpha1_recovers_t3_on_deep` = true（α=1 深层精确还原 T3）
- `alpha1_keeps_h75_elsewhere` = true
- `shallow_zero_diff` = true（stage0/1 bitwise 不变）
- `all_finite` = true、`same_keys` = true、`reload_identical` = true

---

## 4. G2 推理门槛

### 4.1 Parity（**部分通过，需注意**）

对官方已评分 ZIP 的 10 张固定图逐像素重放：

| 数值模式 | 差异像素（10 张合计，共 10,470,600 px） |
|---|---:|
| **带 autocast（FP16）** | **783** |
| **不带 autocast（FP32，官方实际模式）** | **9** |

- 官方打包器源码 `package_semifinal_bias.py`（SHA `206cc80d…d7723a1`）中**只有 `torch.inference_mode()`，没有 autocast**。任务书里"包括autocast"的说明与该源码不符。
- 剩余 9 像素（0.000086%）**本机两次重跑完全一致**（确定性），且 `work/src/models.py` SHA 与官方迁移副本完全相同（`7c003956…`），bias JSON、manifest、协议字符串均一致。
- **判定：9 像素差异来自跨机器 GPU/驱动数值差（官方 ZIP 产自 uavmain，本次在 teacher-4090），不是协议或代码错误。** 但仍未达到"逐像素完全相同"的严格门槛，如实报告。

### 4.2 Canary（42 张，三子集各 14）

| 模型 | test1 变化 | test2 变化 | test3_new 变化 | 退化图 |
|---|---:|---:|---:|---:|
| T3 | 0.9834% | 1.5075% | 1.2339% | 0/14 |
| **DEEP25** | **0.2425%** | **0.3982%** | **0.3369%** | **0/14** |

- **零退化图**（无单类覆盖 >99% 的异常）
- Deep25 改动幅度约为 T3 的 26%，与 α=0.25 高度一致 → 融合确实生效
- **不是 `NO_EFFECT`**

### 4.3 Source1399 安全门槛（PASS）

| 模型 | mIoU | vs H75 |
|---|---:|---:|
| H75 | 77.5320 | — |
| T3 | 77.6266 | +0.0946 |
| **DEEP25** | **77.5936** | **+0.0616** |

预注册硬停检查：

| 条件 | 实际 | 触发? |
|---|---|---|
| NaN / Inf | 无 | ❌ |
| 全类塌缩 | 无 | ❌ |
| 任一类下降 > 3.0pp | 最大降幅 0.49pp（T3 的 Barren 46.15→45.66） | ❌ |
| 整体 mIoU 下降 > 0.5pp | 三模型均高于 H75 | ❌ |

逐类（%）：H75 `[69.18, 87.61, 83.57, 88.70, 46.15, 89.33, 74.00, 81.73]`；Deep25 `[69.23, 87.61, 83.70, 88.79, 46.04, 89.36, 74.22, 81.79]`。

**未因 Source 微增而宣称线上会涨。** Source1399 与训练图存在空间重叠，只作安全门槛。

---

## 5. G3 出包

使用**官方生产打包器** `package_semifinal_bias.py`（未修改，SHA `206cc80d…d7723a1`）+ 迁移的 `work/` 环境 + `bias_bstar.json`（SHA `5848eb3c…d860cbe0`）。

- 单模型、单 checkpoint、D4-TTA8、FP32 native 均值、Bstar 恰好一次、`bilinear 1024 align_corners=False`、`argmax+1`、uint8 mode L、ZIP 根目录平铺
- **注意：官方打包器不使用 autocast**（源码只有 `torch.inference_mode()`），本次严格照此执行
- 输出：`/home/jhy/aic2026_h75_t3_deep25_20261028/deliver_deep25/H75_T3_DEEP25_semifinal.zip`
- **ZIP SHA256：`e3fa44dd4f00512e4f12fd2e1aaa9fb4294776ddd2c316ece2af0f0cdbbff2f9`**
- ZIP 字节数：31,068,652

### 格式 QA（全部 PASS）

```
count = 3594            （期望 3594）
name_set_equal = true   （与 manifest 输出名集合完全相等）
duplicates = 0
nested_folder = false   （根目录平铺）
crc_error = null        （ZIP 内 CRC 全部通过）
modes = ["L"]           （单通道灰度）
sizes = [[1024, 1024]]
value_min = 1, value_max = 8
```

**提交状态：`NOT_SUBMITTED`。** 未登录平台、未上传、未宣称任何线上提分。

---

## 6. 候选与冠军的关系（请决策者注意）

| 候选 | Source1399 | 与冠军关系 |
|---|---:|---|
| H75 + Bstar（冠军） | 77.5320 | 线上已评分 70.998665 |
| T3-FULL-ST + Bstar | 77.6266 | 线上已评分 70.990138（**低于冠军 0.0085**） |
| **Deep25 + Bstar** | **77.5936** | 未提交 |

⚠️ **必须指出**：T3 在 Source1399 上比 H75 高 +0.0946pp，但**其线上分数 70.990138 反而比冠军低 0.0085pp**。这构成一个**直接的历史反例**：Source1399 的增益没有转化为线上增益。

Deep25 位于两者之间（Source +0.0616），因此**不能因为 Source 高就预期线上更高**。是否提交由用户判断。

---

## 7. 复现命令

```bash
W=/home/jhy/aic2026_h75_t3_deep25_20261028
T=/home/jhy/aic2026_ctrl3594_teacher_20261027
R=/home/jhy/aic2026_h76_20261026

# G0
python $W/g0_structure.py
# G1
python $W/g1_fuse.py
# G2.1 parity
python $W/g2_parity.py
# G2.2 canary
python $W/g2_canary.py
# G2.3 source1399
python /tmp/tta8_eval.py --ckpt $W/H75_T3_DEEP25.pt --tag DEEP25
# G3 package
python $T/code/package_semifinal_bias.py --work $T/work \
  --checkpoint $W/H75_T3_DEEP25.pt \
  --expected-checkpoint-sha 1f46b247bacb720ff9099defbf3fa38764ebf1b42a1e066f0bccc7147549b003 \
  --images-dir $R/data/test3/images --manifest $R/manifests/semifinal_3594.csv \
  --output-dir $W/deliver_deep25 --tag H75_T3_DEEP25 --expected-count 3594 \
  --bias-config $T/configs/bias_bstar.json
```
