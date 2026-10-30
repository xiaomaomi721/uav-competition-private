# Claude 无人机提分实验反馈｜2026-10-28

## 0. 一页结论

- **执行状态：PARTIAL**（训练与评估全部完成；候选 ZIP 已生成并通过格式 QA；**未提交平台**）
- **本轮方案身份**：C0（均匀权软 KL 自蒸馏）与 C1（可靠性加权软 KL）。最终推荐 **C1**，但**理由有限**，见下。
- **历史冠军：70.998665（H75 + Bstar）**
- **新线上成绩：NOT_SUBMITTED**
- **新方案是否在真实线上超越冠军：UNKNOWN**（无平台回执）
- **桌面实际路径**：
  - `C:\Users\Lenovo\OneDrive\Documents\桌面\Claude_无人机提分实验反馈_20261028.md`
  - `C:\Users\Lenovo\OneDrive\Documents\桌面\Claude_无人机提分实验反馈_最新.md`

### 一句话结论

C0/C1 在 source1399 上比 H75 高 **+0.66/+0.70pp**，但 **C1 相对 C0 的可靠性加权机制只贡献了 +0.038pp（噪声级）**。按任务书 §5 的 NO_GO 判据中"没有相对于 C0 的明确非平凡机制影响"这一条**成立**。同时，10-27 已有直接反证：CTRL 在 source1399 上 +1.59pp，线上却是 **−0.68pp**。因此**不能把 +0.70pp 源域增益写成线上提分预期**。

---

## 1. 资产/历史审计

| 资产 | 路径 | SHA256 | 状态 |
|---|---|---|---|
| H75 checkpoint | `/home/jhy/aic2026_h76_20261026/models/H75.pt` | `b5eea2795d8a6d90c803a15b2ca66cf0abcf62322ef1aacd0d4d0ec82098a28a` | **FOUND，SHA 校验通过** |
| test3 图像 | `/home/jhy/aic2026_h76_20261026/data/test3/images/` | — | 3594 张，FOUND |
| manifest | `/home/jhy/aic2026_h76_20261026/manifests/semifinal_3594.csv` | — | 3594 行（test1 500 / test2 1300 / test3_new 1794）|
| source 数据 | `/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/` | — | 6996 图（5597/1399），FOUND |
| 多尺度 TTA 历史负结果 | 桌面历史文档 | — | 已复核，本轮**未把多尺度 TTA 作为路线** |
| BLDA-Soft 既有资产 | 新服务器 `/data/uav/runs/CODEX_BLDA_SOFT_20261027/` | — | FOUND（单弱视图缓存 3594 + 50 步 smoke）；**未伪称 D4 教师**；本轮为新机制，未复用其数学实现 |

**未发现项**：无。所有本轮依赖资产均在授权主机 teacher-4090 上找到并核验。

---

## 2. 完整实现清单

工作根：`/home/jhy/aic2026_h76_20261026/code/reliable_kd_20261028/`

| 文件 | 作用 |
|---|---|
| `rkd_common.py` | D4 变换、H75 加载（SHA 断言）、strong_photo、bias、`d4_stack_with_votes` |
| `rkd_cache_teacher.py` | 构建 3594 张 D4 教师缓存（z_d4 / z_single / votes / hn / flip），支持 resume |
| `rkd_teacher_diag.py` | source1399 教师可靠性诊断 |
| `rkd_train.py` | C0/C1 匹配训练 |
| `rkd_target_stats.py` | 目标域预测统计（无 GT） |
| `chain_train.sh` | 缓存完成 → smoke → 并行开两臂 |
| `package_ctrl3594.py`（复用，已加 `--expected-sha256`） | 3594 张 D4-TTA8 出包 |

### 教师缓存的精确实现

对目标图 `x`：
```
z_i = inv_D4_i( model(D4_i(x)) )       # i=0..7，native 256
z_T = mean_i(z_i)                       # FP32 均值，缓存为 FP16，不含 Bstar
q   = softmax(z_T + Bstar)
votes = #{i : argmax(z_i+Bstar) == argmax(z_T+Bstar)}   ∈ [0,8]
H_norm = -Σ q_c log q_c / log(8)
flip = 1[ argmax(z_T) != argmax(z_T+Bstar) ]
```
D4 顺序与 `package_semifinal_bias.py` 完全一致（正向 rot90 后 flip；逆向 flip 后 rot90(-r)）。

### C0 与 C1 的唯一差别

```
L_total = CE_src + 0.1 * KL(H75_src || student_src) + λ_t(step) * L_target
λ_t = 0.2 * min(1, step / (0.2*3594))
kl_pix = Σ_c q_c (log q_c - log p_c)      # p = softmax(z_student + Bstar)

C0: L_target = mean(kl_pix)
C1: w = a^2 * c,  a = votes/8, c = 1 - H_norm
    w *= 0.5 where flip == 1
    L_target = Σ w*kl_pix / Σ w
```

**两臂在起点、样本顺序、增强、优化器、冻结层、步数上完全一致，唯一差别是目标像素权重。**

- 可训练：`segformer.stages.2.*` / `segformer.stages.3.*` / `decode_head.*`（共 44,336,200 参数）
- 冻结：`segformer.stages.0/1`；教师 `eval()` + `no_grad()`
- 优化器：AdamW，base lr 1e-5、head lr 5e-7、wd 0.01、poly (1-(step-1)/10000)
- 步数：3594（一次完整目标遍历）

---

## 3. 运行命令与原始日志

```bash
# 1) 教师缓存
CUDA_VISIBLE_DEVICES=3 EXPECTED_GPU_INDEX=3 python rkd_cache_teacher.py
# 2) 教师诊断
CUDA_VISIBLE_DEVICES=2 EXPECTED_GPU_INDEX=2 python rkd_teacher_diag.py
# 3) 训练（chain_train.sh 自动：等缓存 → smoke → 并行两臂）
bash chain_train.sh
# 4) 评估
CUDA_VISIBLE_DEVICES=0 python /tmp/tta8_eval.py --ckpt models/H75.pt --tag H75
CUDA_VISIBLE_DEVICES=1 python /tmp/tta8_eval.py --ckpt train/C0/checkpoint_step3594.pt --tag C0_3594
CUDA_VISIBLE_DEVICES=2 python /tmp/tta8_eval.py --ckpt train/C1/checkpoint_step3594.pt --tag C1_3594
# 5) 出包
python code/h76_3594/package_ctrl3594.py \
  --checkpoint .../train/C1/checkpoint_step3594.pt \
  --expected-sha256 c37cd0138c008ea98fe35eb7f8d2d93d22b850d2747b0e8af454216e77503c36 \
  --outdir deliver/C1_3594 --tag C1_3594 --bias-variants bstar
```

| 阶段 | 结果 |
|---|---|
| 教师缓存 | 3594/3594，`COMPLETE.json` 已写 |
| 冒烟（C1 5 步） | rc=0 |
| C0 训练 | 3594 步，1650.4s，ESS=1.000，峰值 6020.1 MiB |
| C1 训练 | 3594 步，1339.4s，ESS 均值 0.94957（最小 0.73499），峰值 6020.6 MiB |
| AMP skip / 非有限梯度 | 无 |

日志：`logs/cache.log`、`logs/diag.log`、`logs/train_c0.log`、`logs/train_c1.log`、`logs/package_c1.log`

---

## 4. 定量结果

### 4.1 source1399 全量 D4-TTA8（与部署同协议）

| 模型 | mIoU | Background | Building | Road | Water | Barren | Vegetation | Agricultural | Vehicle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H75 | **77.5320** | 69.18 | 87.61 | 83.57 | 88.70 | 46.15 | 89.33 | 74.00 | 81.73 |
| C0 | **78.1955** | 70.34 | 87.73 | 84.22 | 89.23 | 47.29 | 89.53 | 75.39 | 81.83 |
| C1 | **78.2339** | 70.43 | 87.68 | 84.24 | 89.24 | 47.25 | 89.50 | 75.65 | 81.88 |

- C0 − H75 = **+0.6635pp**
- C1 − H75 = **+0.7019pp**
- **C1 − C0 = +0.0384pp**（机制的净贡献，噪声级）
- 逐类：无任一类别退化，均为小幅上升；Barren +1.1pp 是最大单类增益

### 4.2 教师可靠性诊断（source1399，1399 张）

| 教师 | mIoU | 与历史对照 |
|---|---:|---|
| 单视图 raw | 80.194 | — |
| 单视图 +Bstar | 77.431 | — |
| **D4-TTA8 raw** | **80.442** | 历史 H75+Zero = 80.4417713 ✅ |
| **D4-TTA8 +Bstar** | **77.531** | 历史 H75+Bstar = 77.5311488 ✅ |

**这组对照证明本轮 D4/评估实现与历史口径完全一致（对齐到小数点后三位）。**

关键机制事实：
1. **D4 教师仅比单视图强 +0.25pp**（80.442 vs 80.194），不是大杠杆。
2. **`a`（八视图一致性）几乎无方差**：1,445,188,329 个有效像素中，仅 **6,065 个**一致性 ≤0.99。→ **C1 的 `a²` 项 ≈ 1，实际生效的机制是熵加权 + 翻转降权，不是一致性加权。**
3. **翻转像素准确率 31.02%，未翻转 92.02%** → 对翻转像素降权有实测依据。
4. 熵分桶有效：低熵 96.41% vs 高熵 81.56%。
5. Bstar 在源域净有害：改动 7321 万像素，纠正 2177 万、致错 4376 万（2 倍）。

### 4.3 目标域（无 GT，只报预测侧统计）

**取样缺陷**：`--limit 800` 取的是 manifest 前 800 行，全部属于 `test2` 子集，因此 test1 / test3_new 无数据。以下仅代表 test2 子集。

| 模型 | test2 类别占比 | 相对 H75 像素变化 | 单类覆盖 >99% 的图数 |
|---|---|---:|---:|
| H75 | [31.15, 18.39, 6.57, 8.19, 1.89, 21.42, 11.81, 0.58] | — | 13/800 |
| C0 | [31.29, 18.37, 6.49, 8.19, 1.94, 21.59, 11.57, 0.57] | 1.62% | 14/800 |
| C1 | [31.43, 18.40, 6.50, 8.13, 1.96, 21.59, 11.44, 0.57] | 1.71% | 14/800 |

**判定：无类别塌缩、无退化图数量激增（13→14 与基线同级）、预测仅变化 1.6~1.7%。** 目标域无 GT，**不声称任何目标域 IoU**。

---

## 5. 过拟合和确认偏差风险

1. **来源是否真的新增知识？存疑。** 教师 = H75 自身冻结副本，学生从同一 H75 初始化。这是**自蒸馏**，本质上没有引入外部新信息；+0.66~0.70pp 更可能来自"在目标域分布上再做若干步梯度更新"本身，而不是可靠性机制。
2. **C1 机制增量仅 +0.038pp**，且其中一致性项 `a` 已退化（见 §4.2），**不能声称"可靠性加权有效"**。
3. **源域增益不迁移的直接反证**：10-27 CTRL 在 source1399 上 +1.59pp，线上 **−0.68pp**。本轮增益（+0.70pp）更小，没有理由更可靠。
4. **目标域统计只覆盖 test2 子集**，未覆盖 test1/new1794，不能代表完整 3594 的分布行为。
5. 本轮**没有读取任何目标 GT**（不存在），也没有用无标签统计冒充 mIoU。

---

## 6. 出包、线上提交及回执

| 项 | 值 |
|---|---|
| 候选 | C1（`train/C1/checkpoint_step3594.pt`） |
| checkpoint SHA256 | `c37cd0138c008ea98fe35eb7f8d2d93d22b850d2747b0e8af454216e77503c36` |
| ZIP | `/home/jhy/aic2026_h76_20261026/deliver/C1_3594/C1_3594_BSTAR.zip` |
| **ZIP SHA256** | **`90a55b7de2d0df11d2543d2c7d0b5581b94338db8394c11efe064a7d346ce3c1`** |
| ZIP 大小 | 31,156,937 B |
| 格式 QA | 3594 张，`bad_count = 0`（mode L / 1024×1024 / 值域 1-8 / 根目录平铺）|
| 配置 | 单模型、D4-TTA8、Bstar 施加一次、bilinear 1024、argmax+1、mode L、根目录平铺 |
| 线上提交 | **NOT_SUBMITTED** |
| 当天剩余额度 | **UNKNOWN**（需用户核实） |

**未提交。** 提交需用户确认额度与授权。

---

## 7. GO/NO-GO 裁决与下一轮最值得尝试的单一方案

### 硬门槛（§5 HARD_STOP）——全部通过
- 数据名/标签映射正确 ✅
- teacher/champion parity 正确（77.531 / 80.442 与历史一致）✅
- 无非有限 loss ✅
- 单模型推理、无 ensemble ✅
- 输出格式待 ZIP QA 结果（见 §9）

### 机制判据（§5 NO_GO）
- **C1 相对 C0 无明确非平凡机制影响（+0.038pp）→ 该条成立**
- 增益仅在源域可见（目标域无 GT 可验证）
- 一致性项 `a` 退化，实际机制与设计意图不符

### 我的裁决：**REVIEW_FOR_ONLINE（弱）**

- 不建议把本轮结论表述为"提分方案"。
- 若团队决定用掉一个名额，C1 是本轮唯一合格候选（无异常、格式合规、单模型）；理由仅是"它是 H75 之外的唯一无异常单模型候选"，**不是"预期会涨"**。
- 若团队决定不交，**保留 H75 + Bstar（70.998665）** 是完全合理的结论。

### 下一轮最值得尝试的单一方案

**停止在 H75 能力范围内做自蒸馏/加权类改动。** 三次实测（CTRL −0.68、T3-FULL-ST −0.009、本轮仅源域涨）一致表明模型侧已饱和。
唯一被平台验证过的杠杆是 **bias（Bstar 值 3.02pp）**，且必须**用平台反馈而非 source1399** 来调。可证伪假设：在 H75 基础上单维改农田 bias（test3 预测农田 12.02% vs source 真实 7.14%，过预测 4.9pp；source 消融 D 亦为 +0.34pp），是当前唯一两条独立证据同向的维度。

---

## 8. 我给 ChatGPT 的审核问题

1. C1−C0 = +0.038pp、且 `a` 项退化（仅 6065/1.445B 像素有区分度），是否应当直接判定"可靠性加权无效"，而不是继续调 `w` 的形式？
2. 自蒸馏（教师=自身冻结副本）在目标域上 +0.70pp 源域增益，结合 10-27 CTRL 的 +1.59→−0.68 反转，是否存在任何理由预期它会在 test3 上正向？请独立判断。
3. 我的目标域统计因 `--limit 800` 只覆盖 test2，是否需要重跑覆盖 test1/new1794 才能作为放行证据？
4. 若采纳"唯一杠杆是 bias"，每天 2 次额度下，单维扫描 3 个有效维度（背景/荒地/农田）是否比继续做训练方法更优？请给出排序理由。

---

## 9. 完整工件索引与可复现说明

| 工件 | 绝对路径 | 说明 |
|---|---|---|
| 教师缓存 | `/home/jhy/aic2026_h76_20261026/code/reliable_kd_20261028/teacher_d4/` | 3594 个 `.npz` + `IDENTITY.json` + `FILES.json` + `COMPLETE.json` |
| 教师诊断 | `.../teacher_diag/TEACHER_DIAG.json` | 四配置 mIoU、翻转统计、分桶准确率 |
| C0 训练 | `.../train/C0/` | `RUN_CONFIG.json`、`PAIR_ORDER.json`、`history.jsonl`、`SUMMARY.json`、checkpoint 1000/2000/3594 |
| C1 训练 | `.../train/C1/` | 同上 |
| C0 checkpoint SHA | — | `c82c8f23fdacb30bd022a51d24dabbfc9221d56f90e20c0c6b3d57b0159a072f` |
| C1 checkpoint SHA | — | `c37cd0138c008ea98fe35eb7f8d2d93d22b850d2747b0e8af454216e77503c36` |
| 候选 ZIP | `.../deliver/C1_3594/C1_3594_BSTAR.zip` | SHA 与 QA 见 `PACKAGE_REPORT.json` |
| 本地代码副本 | `桌面/无人机/02_代码与实验/本轮实验/BLDA_Soft_执行_20261027/rkd_*.py` | 与服务器同源 |

**复现命令**：见 §3。**缺失项**：完整 3594 目标域统计（当前仅 test2 前 800 张）标记为 `PARTIAL`。