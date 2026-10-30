# 缺失与阻塞（H75_T3_DEEP25）

## A. 已解决

| 项 | 结果 |
|---|---|
| T3 权重是否存在 | ✅ 找到，SHA `cb958870…e8e8` 与期望一致 |
| H75 是否存在 | ✅ SHA `b5eea279…98a28a` 一致 |
| 是否有历史重复 | ✅ 无重复（见 `HISTORICAL_SOUP_DUPLICATION.md`） |
| 结构是否兼容 | ✅ 644/644 键，无 missing/unexpected/shape mismatch |

## B. 部分未达成（需决策者知悉）

### B1 Parity 未达"逐像素完全相同"

- **官方 ZIP 重放差异：9 像素 / 10,470,600（0.000086%）**
- 已排除的原因：
  - 代码：`work/src/models.py` SHA 与本地一致（`7c003956…`）
  - 协议：`D4-TTA8 + FP32 native mean + Bstar once + bilinear1024 + argmax+1`，与官方 `PACKAGE_MANIFEST.json` 完全一致
  - bias JSON：SHA `5848eb3c…` 一致
  - manifest：SHA `76463ff9…` 一致
  - 进程内不确定性：本机两次重跑得到**相同的 9 像素**
- 剩余解释：**跨机器 GPU/驱动数值差**（官方 ZIP 产自 uavmain，本次在 teacher-4090）
- **影响评估**：9 px 不改变任何一张图的类别面积统计，对 mIoU 的影响 < 1e-6 量级。但它确实**不满足任务书"逐像素相同"的字面要求**。
- **该由谁决定**：是否接受 9 px 作为跨机器容许误差并启用本候选。

### B2 任务书对打包器的描述与源码不符

任务书写"用同一生产打包器路径/数值模式（**包括autocast**）"。实际 `package_semifinal_bias.py`（SHA `206cc80d…`）**只有 `torch.inference_mode()`，不使用 autocast**。

- 带 autocast 重放 → 783 px 差异
- 不带 autocast 重放 → 9 px 差异

因此**正确模式是不带 autocast**。任务书的这句描述应视为笔误。

## C. 本次未做（NOT_RUN）

- 未做 α 的其他取值（0.1 / 0.4）网格搜索 —— 按任务书明令禁止
- 未对 Bstar 做任何新调参
- 未做 Target 无 GT 的"mIoU"估算 —— 不存在 GT，不做
- 未提交平台、未登录平台
- 未覆盖 `H75.pt`、T3、已评分 ZIP（全部只读）
- 未中断 GPU0/GPU2/GPU3 上他人或队友的任务

## D. 需要用户决定

1. **是否接受 9 像素跨机器差异**，从而启用 `H75_T3_DEEP25` 候选 ZIP？
2. **是否提交**（本报告不构成提交授权）。
3. 提醒：**T3 本身 Source1399 比 H75 高 +0.0946pp，但其线上 70.990138 低于冠军 70.998665**。这是"Source 高 ≠ 线上高"的直接反例。Deep25 的 Source 增益（+0.0616）更小，**不能据此预期线上更高**。
