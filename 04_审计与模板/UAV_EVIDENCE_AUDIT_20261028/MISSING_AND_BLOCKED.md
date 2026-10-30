# 缺失与阻塞清单（MISSING_AND_BLOCKED）

日期：2026-10-28 20:35 CST。主机：`teacher-4090`。

---

## A. 本次未找到（NOT_FOUND）

| 项 | 说明 | 影响 |
|---|---|---|
| **平台原始回执**（H75+Bstar 70.998665 / Zero 67.979564 / CTRL@3594 70.318575 / T3-FULL-ST 70.990138） | 全盘授权范围内未见截图、JSON 回执或平台响应文件 | 所有线上分数只能是 `REPORTED_ONLY`，不能升级为平台已核验 |
| **E02 B1 / E02 C0 权重文件** | 仅在历史报告中出现；本次在 `/home/jhy` 下未定位到 `.pth` | M2F_IDENTITY_TABLE 中对应行标 NOT_FOUND |
| **Exp03 / Exp04 权重** | 同上 | 同上 |
| **V1 原始 813 像素 / 3 图 的原始 CSV** | 只见到 V2 目录 | V1 说法保持 REPORTED_ONLY |
| **DINO-B 教师 2500+2500 步训练日志与权重 SHA** | 未定位 | Source1399=60.0826 为 REPORTED_ONLY |
| **11-01 预测 ZIP / 11-02 Docker+技术报告 的负责人与产物** | 未见 | `OWNER_UNASSIGNED_OR_EVIDENCE_MISSING` |

---

## B. 存在但本次无授权/不可复核（BLOCKED / NEEDS_AUTHORIZED_COMPUTE）

| 项 | 原因 | 最小解除方式 |
|---|---|---|
| 逐视图 logits 分歧重算 | `teacher_d4/*.npz` 只存均值，未持久化 8 视图 logits | 授权后对 20 张图重跑 8 视图前向，约 5 分钟 GPU |
| votes 直方图（native 256 口径） | 需重算而非读缓存 | CPU + 缓存即可，约 2 分钟 |
| λ=0 逐参数梯度一致性 | 需一次前向+反传对比 | 空闲 GPU，约 2 分钟 |
| M2F E02/Exp00R 的 source1399 评测复现 | 需 GPU 推理 | 单模型 1399×8 视图 ≈ 20 分钟 |
| `uavmain`（10.8.131.68）上的 p0b 原始链路 | 本次按要求仅访问 teacher-4090 | 需另行授权 |

---

## C. 本次明确未执行（NOT_RUN，按任务书要求）

- 启动任何新训练、长时推理、Docker 构建
- 向比赛平台提交任何结果
- 抢占/终止任何进程（包括 GPU0 的 STPP 实验与 GPU2 的 M2F 长训）
- 修改 `H75.pt`、已评分 ZIP、任何 checkpoint / 日志 / 配置
- 全量 3594 目标域统计（仅用已有 60 张结果）

---

## D. 需要用户决策的三件事

1. **M2F 60k 长训是否立即停止？**
   2000 步复核 EMA mIoU 0.74%、raw 22.83%、6 类塌缩。当前仍在跑（ETA 10-30 18:36）。本任务未干预。

2. **GPU0 上的 `H75_STPP_LITE_V2_CLASSAWARE` 是什么、由谁发起？**
   不在本任务范围，仅记录其在运行（约 2.7 GB）。若需要审计其科学性，需另外授权。

3. **最终交付材料由谁负责？**
   11-01 20:00 预测截止、11-02 20:00 材料截止。当前无责任人证据。

---

## E. 证据包自身的局限

- 打包不含模型权重、官方 RGB、完整预测 PNG、`.npz` 缓存 —— 因此**任何需要重算的结论都不在本包内**。
- 部分 `.json` 为服务器原始文件直接复制，未做二次编辑（保持原件字节），其 SHA 记录在 `SHA256SUMS.txt`。
- 本包的 Markdown 分析文档（README_FIRST / FINAL_AUDIT_HANDOFF / D4_VOTES_AUDIT / C0_C1_MATCHED_CHECK / SSL_STATUS_CORRECTION）为**本次撰写**，其中的判断均标注了证据等级。
