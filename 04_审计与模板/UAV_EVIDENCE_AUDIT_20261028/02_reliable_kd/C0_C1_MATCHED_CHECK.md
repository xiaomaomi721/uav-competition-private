# C0 / C1 匹配性核查（reliable_kd）

依据源码：`rkd_train.py`（本地副本 `02_reliable_kd/src/rkd_train.py`）。

---

## 1. 逐项匹配检查表

| 项 | C0 | C1 | 一致？ | 代码依据 |
|---|---|---|---|---|
| 初始权重 | H75（SHA 断言） | 同 | ✅ | `student = model_from()`（内部 `assert sha256(path)==H75_SHA`） |
| 随机种子 | `SEED=20261028` | 同 | ✅ | `seed_all()` |
| 配对列表 | `build_pairs(steps)`：source perm(SEED+10001)、target perm(SEED+10002) | 同 | ✅ | 同一函数，同参数 |
| 增强 | `strong_photo` + `d4(tx, randrange(8))`，RNG 由 `SEED+200000+step` 决定 | 同 | ✅ | 逐 step 确定性 |
| 可训练范围 | `segformer.stages.2/3` + `decode_head` | 同 | ✅ | `p.requires_grad_(n.startswith(TRAIN_PREFIX) or ...)` |
| 学习率 | base 1e-5 / head 5e-7，poly `1-(step-1)/10000` | 同 | ✅ | 同一 `opt` 构造与循环内更新 |
| step 数 | 3594 | 3594 | ✅ | 同命令行 |
| AMP | `GradScaler` enabled | 同 | ✅ | — |
| 目标 GT | **未使用**（test3 无 GT） | 同 | ✅ | 只读 `teacher_d4/*.npz` 与图像 |
| **唯一差别** | `w = ones` | `w = (votes/8)^2 * (1-hn)`，`flip` 处置 ×0.5 | ✅ | 见下 |

**结论：除目标像素权重外，其余变量完全一致。C0/C1 是合法的配对实验。**

---

## 2. Bstar 施加位置追踪（重点排查"训练目标与部署解耦"）

| 位置 | 代码 | Bstar 次数 |
|---|---|---|
| 教师概率 | `q = (zT + b).softmax(1).detach()` | **1 次**（zT 本身是不含 b 的 raw 均值） |
| 学生概率 | `p = (ts + b).log_softmax(1)`，`ts` 为学生 raw logits | **1 次** |
| KL 计算 | `kl = (q * (q.log() - p)).sum(1)` | 在**同一 b 空间**比较 ✅ |
| 部署推理 | `package_ctrl3594.py`：`biased = logits + bt` 后 upsample | **1 次** |

**结论：不存在"教师先加 Bstar、学生又加一次"的双重施加。训练目标与部署在同一 bias 空间，无解耦风险。**

⚠️ 但需注意一个**语义层面的强假设**：训练时直接强迫学生去拟合"加了 Bstar 的教师分布"，等价于把 `+Bstar` 这个部署后处理**内化进权重**。若将来换用不同 bias 部署，该权重与该 bias 绑定。这不是 bug，但**限制了候选的可复用性**，应在报告中如实说明。

---

## 3. 目标像素空间对齐

- 教师缓存 `zT` 为 **native 256×256**，生成自**未增强**的原图。
- 学生输入 `aug = strong_photo(d4(tx, view))`，输出经 `inv_d4` **回到 canonical 256×256**。
- 因此 `q` 与 `p` 都在 canonical native 256 空间逐像素对齐 ✅。
- 目标图未做随机 crop / resize，不存在几何错位风险 ✅。

---

## 4. 未核查项（NOT RUN）

- **λ=0 逐参数梯度一致性**：未做（见 `03_h75_ssl/SSL_STATUS_CORRECTION.md` 同类说明）。
- **教师缓存 `z_d4` 与在线重算的一致性**：未复核（缓存生成脚本 `rkd_cache_teacher.py` 与推理脚本 `package_ctrl3594.py` 的 D4 顺序经人工比对一致，但未做数值级 canary）。建议补：固定 3 张图，缓存值 vs 在线重算的 `max|Δlogit|`。
