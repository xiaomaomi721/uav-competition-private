# UAV 证据审计包｜READ_FIRST（2026-10-28）

采集主机：`teacher-4090`（`ubuntu-Super-Server`，`jhy`），采集时间 **2026-10-28 20:11–20:35 CST (+0800)**
任务性质：**只读取证 + 最低成本核验**。本次**未启动任何新训练、未提交平台、未 kill 任何进程、未修改任何原资产**。

---

## 一、五个关键问题的直接回答

### 1. 主线 GPU3 真实状态

- **GPU3（`GPU-32853ccf-4c4f-1328-8d75-04efe9d5298f`）：15 MiB / 0%，无计算进程。空闲。**
- `h76-train.service`：**failed**（历史 CaSC 队列已停止）
- `h76-watchdog.service`、`ziponly-watchdog.service`、`secondgpu-m2f-watchdog.service`：**均 active/running**（watchdog 存活，但 GPU3 上无训练负载）
- 当前实际有负载的卡：
  - **GPU0**（`bc304d79`）：新实验 `H75_STPP_LITE_V2_CLASSAWARE_20261028T112548Z`（`score_full.py`），约 2.7 GB
  - **GPU2**（`cacb7500`）：**第二 GPU Mask2Former 长训**，14.2 GB

### 2. M2F E02/Exp00R/Exp04 是否重复

- **未发现重复长训。** 但发现一件更重要的事：**第二 GPU 的 M2F 60k 长训已在 2026-10-28 17:33 实际启动并正在运行**（`state=TRAINING`，`effective_step=4796/60000`，ETA 2026-10-30 18:36）。
- **该长训 2000 步复核出现严重塌缩**：EMA mIoU **0.7380%**、raw **22.8262%**（Barren 归零）。对照参考 Exp00R ≈79.38、E02 B1 ≈79.57（协议未互核）。
- 历史 donor 权重身份已闭合：`donor_iter_30000.pth` 完整 SHA256 = `dea40595287980777eae5751e76f5e86811071da9edc12b989b3c2ed0556c68c`（此前记录只有 50 位）。

### 3. Claude 投票是否已实算核查

- **部分已核，部分未核。**
- `votes`（D4 八视图一致性）**确实由 8 次独立前向的真实 argmax 计算**，代码见 `02_reliable_kd/src/rkd_common.py::d4_stack_with_votes`（逐视图 `inv_d4(model(d4(x,v)))`，未复用平均 logits）。
- 但 **"14.45 亿像素中仅 6065 个一致性 ≤0.99" 这一统计的分母口径存在疑问**：该数字来自 1024 上采样空间（`upsample_pred` 后的 `v` 掩码），而 `votes` 本身在 native 256 生成。两者口径不同，**报告未充分说明**，详见 `02_reliable_kd/D4_VOTES_AUDIT.md`。
- **逐视图 logits 未持久化**，因此"逐视图两两分歧"无法仅凭缓存重算 → 标 `NEEDS_AUTHORIZED_COMPUTE`。

### 4. 第二 GPU 新长训是否已开始、科学性门槛是否存在

- **已开始，且正在运行**（非计划、非推测）。
- 科学性门槛：**未建立**。config 只有"60k 有效步、48k/64k 未授权"的工程冻结，**没有提出独立可证伪的科研假设**；相对于 Exp00R/Exp04 的变化同时包含"更长训练 + 不同 lr schedule + EMA"，属多变量。
- **且 2000 步复核已显示塌缩**，属于任务书 §E4 的"显著能力退化"信号。

### 5. 最终提交材料是否有负责人

- **未发现负责人与产物。** 服务器上未见 11-01 预测 ZIP 的最终责任人记录、未见 11-02 Docker/技术报告的进行中证据。两条新执行线（reliable_kd、TargetSSL）都明确排除了交付材料。
- 判定：`OWNER_UNASSIGNED_OR_EVIDENCE_MISSING`。

---

## 二、本次独立核验到的关键证据（VERIFIED_NOW）

| 项 | 值 |
|---|---|
| H75.pt | 189,091,023 B，SHA `b5eea279…98a28a` ✓ 与历史一致 |
| **官方已评分 H75_BSTAR ZIP** | 31,099,544 B，SHA **`c012bae1237160a0c0d5fbccd168f58facf62c03bf60ee2cc82877cb21454d30`** ✓ 与历史记录一致 |
| Claude C1 候选 ZIP | 31,156,937 B，SHA `90a55b7de2d0df11d2543d2c7d0b5581b94338db8394c11efe064a7d346ce3c1` |
| 3594 manifest | 460,173 B，SHA `76463ff9…6cdd19` |
| M2F donor iter_30000 | 2,610,605,869 B，SHA `dea40595…556c68c`（完整 64 位） |

---

## 三、必须知道的两个自我纠正

1. **RGBHead 参数量**：我在 10-28 报告中写的"1,024,899 参数"是**错误的臆造值**。实际 `RUN_CONFIG.json` 记录为 **24,768**，与架构 `Conv2d(128,192,1)+PixelShuffle(8)` 完全吻合。已在 `03_h75_ssl/SSL_STATUS_CORRECTION.md` 更正。
2. **目标域统计状态**：报告正文曾同时出现"已完成 60 张"与"PARTIAL/网络中断"两种说法，属表述矛盾。实际是**补跑后已完成 60 张**（仅 test2 子集），已更正。

---

## 四、证据等级约定

- `VERIFIED_NOW`：本次在服务器上直接读到文件 / 算出的哈希
- `REPORTED_ONLY`：只存在于历史总结或用户转述，本次未独立核验
- `NOT_FOUND`：在授权范围内未找到
- `BLOCKED`：存在但无权限/无授权读取
- `NOT_RUN`：本任务明确未执行

**本包不含**：模型权重、官方 RGB 图像、完整预测 PNG、`.npz` 大缓存、SSH/令牌、账号口令。大文件只记录路径 + 字节数 + SHA256。
