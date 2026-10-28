# soupB-init → KD continuation Run1：3200 Gate 结果与 STOP 记录（2026-09-16）

**结论：G4 STOP。承重假设被平台实验直接否掉，不是"差一点"。**

训练已干净退出（PID 331495，verdict=STOP，`KD_TRAIN_DONE`，0 报错）。
未续 6400，未改 λ/T，未换 soup，未做 4800 probe，未做额外 calibration sweep。
所有产物保留，未删除任何东西。

---

## 一、平台分数（用户回报，2026-09-16）

提交顺序 = 本地编号顺序：

| 编号 | 包 | bias | mIoU |
|---:|---|---|---:|
| 127 | SOUPKD_ZERO | `[0,0,0,0,0,0,0,0]` | **0.679242** |
| 128 | SOUPKD_C | `[-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5]` | **0.686964** |
| 129 | SOUPKD_13 | `[-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 2.9215, -0.5]` | **0.688002** |

```
Z            = 0.679242
C            = 0.686964
#13          = 0.688002
R_C  = C − Z = +0.007722 = +0.7722pp
R_13 = 13 − Z = +0.008760 = +0.8760pp
best_final   = 0.688002  (#13)
```

## 二、相对参照

| 对比 | Δ |
|---|---:|
| ZERO 0.683042 → 0.679242 | **−0.3800pp** |
| soupB+C 0.691330 → C 0.686964 | **−0.4366pp** |
| soupB+C 0.691330 → best_final 0.688002 | **−0.3328pp** |
| soupB R_C 0.8288pp → 本轮 R_C 0.7722pp | **−0.0566pp** |
| best_final 0.688002 vs 纪录 0.693096 | **−0.5094pp** |
| best_final vs 0.70 | −1.1998pp |

## 三、Gate 判定（冻结方案 §6.5）

| gate | 条件 | 实测 | 结果 |
|---|---|---|---|
| G1 | best ≥ 0.693096 | 0.688002 | ❌ |
| G2 | best ≥ 0.692819 | 0.688002 | ❌ |
| G3 | Z ≥ 0.6825 **且** R_C ≥ 1.00pp | Z=0.679242 ❌ / R_C=0.7722pp ❌ | ❌ |
| **G4** | 其余 | — | **→ STOP** |

## 四、机制判定：两个半边都失败，且方向相反

原假设：**soupB 抬 Z（0.683042） + KD 恢复 calibration response（往 1.2400 拉）**，串联。

实际：

```
Z 明显下降（−0.3800pp）
R_C 也没有恢复，反而略降（−0.0566pp）
```

**「average-init 的高 ZERO」能否与「KD 恢复的 fixed-bias response」共存于一个 checkpoint？
→ 3200 optimizer updates 内、此配方下**未实现**。**

### 4.1 初始化交互证据（**现象**，非因果结论）

同配方、**只换 init**：

| init | 无 KD 的 ZERO | KD3200 的 ZERO | ΔZ | 无 KD 的 R_C | KD3200 的 R_C | **ΔR_C** |
|---|---:|---:|---:|---:|---:|---:|
| s62 | 0.680973 | 0.681538 | +0.057pp | 0.5637pp | 1.0387pp | **+0.475pp** |
| **soupB** | 0.683042 | 0.679242 | **−0.380pp** | 0.8288pp | 0.7722pp | **−0.057pp** |

**可以写（现象描述）：**

> soupB-init + 当前完整 KD continuation 配方，在 3200 平台上**没有复现** s62-init 历史中的 R_C 恢复；
> 本次 R_C 从 soupB 的 0.8288pp 降到 0.7722pp，同时 ZERO 下降 0.3800pp。
> **说明该完整训练配方的效果对初始化/训练状态高度敏感。**

**不得写成（已证明的因果）：**

> ~~KD 本身的 calibration 恢复效应依赖初始化。~~

**为什么不能**：本轮**仍然没有**真正的 matched causal control ——

```
soupB init → 3200 步同训练，但 λ_kd = 0        ← 这条没有跑
```

因此混杂因素未被排除：**CE continuation 本身**、CWFA、scheduler、**从 soup 权重离开平均盆地**，
都与 KD 同时发生。当前只能锁死：

```
soupB-init + 当前完整 KD 训练配方，在 3200 平台上失败
```

不能进一步断言「一定是 KD 本身导致 soupB 的 Z/R 下降」。

**同样不能说「补上了 λ_kd=0 matched 对照所缺的信息」** ——
本节补的是 **initialization-interaction evidence（初始化交互证据）**，
**不是 matched causal control**。

**并且不需要为了归因再跑 λ=0**：比赛目标不是写论文，该对照不改变 STOP，也不改变下一步。

### 4.2 未验证的机制猜想（只作参考，不作结论）

soupB 成员 `{s62, s55b, s58, s64}` 与教师集 `{stage23B, s62, s58}` **重叠 2/4**
（s62、s58 既是成员又是教师）。KD 可能把 soupB 的权重平均几何往教师集成几何拉，
而 soupB 的 ZERO 优势恰恰来自那个"已偏离教师"的几何 → **拉回去 = 抹掉平均收益，又没换来校准收益。**

此猜想**未被检验**，不得当作结论引用。

## 五、冻结方案 §7 的事前警告（均已兑现）

冻结方案第七节自己写明「本路线未被证明的东西」，逐条对照：

| # | 事前警告 | 结果 |
|---|---|---|
| 1 | **无可加性证据**，「平均抬 Z + KD 抬 R_C → 串联」是待测假设 | ❌ **假设被否** |
| 2 | **KD 续训不抬 Z**（历史 Z 轨迹 0.6815/0.6778/0.6806/0.6782，均值低于 init 0.15pp）；soupB 的 0.683042 能保多少未知 | ❌ 未保住，−0.38pp |
| 3 | 教师集与 soupB 成员重叠 2/4 | ⚠️ 疑为机制原因，未检验 |
| 4 | **本地完全看不见** KD 的平台效应（本地 −0.043pp ↔ 平台 +0.53pp）；全程不得用本地 mIoU 判断 | ✅ 全程遵守 |

**事前把风险写对，不等于事后解释成功。第 1 条是承重假设，它被直接否掉了。**

## 六、Provenance（完整可复现）

| 项 | 值 |
|---|---|
| trainer | `_kd_train_soup_gate.py` md5 `c907861d870bc8634fe6bf3fa77ead1b` |
| 历史原始 trainer（未改） | `_kd_train.py` md5 `40160816e2645606bf57ebb8728d3996` |
| init | `soups/soupB.pt` sha256 `966139d11a62447c60518bfd27caed3244ccf08860859fd1ae9c7a79225f2f75` |
| teacher cache | `kd_soft_fp16_rebuild/probs/` 500 文件，`TEACHER_META.json` sha256 `7a8f0e2581a458e071cdd86d…` |
| checkpoint | `soupB_kd_gate_run1_20260916/checkpoint_step3200_gate.pt` sha256 `d434ed7fb064f4fdb087cfcc6a887da08eb938f8ba328832f6406921c0d91d6c`（567,600,877 B） |
| 冻结配方 | max_steps 8000 / offload_at 3200 / λ_kd 0.35 / T 2.0 / w_min 0.3 / kd_every 2 / bs 2 / accum 2 / lr 3e-5 / AdamW wd 0.01 / clip 1.0 / AMP on / CWFA 全程 ACTIVE |
| 实测步时 | 1.741 s/step（(5489−9)/3150） |
| checkpoint 时 lr / scaler | `1.800000e-05` / `32768.0` |
| 包 sha256 | ZERO `be5a58e7…` / C `0a846e3c…` / #13 `2401c47b…` |

## 七、保留产物（未删除）

```
soupB_kd_gate_run1_20260916/
  checkpoint_step3200_gate.pt      541.3 MiB
  KD_META.json / COMPLETED.json (status=gate_stop) / gate_STOP
  test_names.txt / splits_test1.txt

soupB_kd_gate_run1_20260916_tta8_3200/    17 GB   500 个 FP32 mean-logits + ARGMAX_ZERO.npy
soupB_kd_gate_run1_20260916_packs_3200/   三个 zip + PACK_META.json
提交包/soupB_kd_gate_run1_20260916_3200/  同上（交付副本）
soupB_kd_gate_run1_20260916.log / ..._tta8_3200.log
```

本地：`提交包/待提交/` 下 127 / 128 / 129 三个 zip 保留（已提交，可移入已提交区）。

GPU2 已完全释放（15 MiB，0 计算进程）。

## 八、下一步（按 GPT 与用户判定）

**不围绕 soupKD 配方做小修小补。** 不改 λ、不续跑、不换 soup、不补 4800 probe、不跑 λ=0 归因对照。
**回到路线层面重新选 P1。**

### 8.1 本轮最重要的新事实（不是"KD 坏了"）

```
soupB 的高 ZERO 在本轮完整 continuation 配方下被明显破坏；
并且在当前已核实的相同训练配方下，s62-init 历史中的 R_C 恢复没有迁移到 soupB-init。
```

**表述边界（必须守住）**：

- 不得推广成「parameter soup 普遍很脆」—— 本轮只证明 **soupB** 这一例、
  在这套 **3200-step continuation 配方**下被破坏。
- 「s62-KD 的 R 恢复不能迁移」的准确写法是
  **「在当前已核实的相同训练配方下，s62-init 历史中的 R_C 恢复没有迁移到 soupB-init」**，
  不能写成对 KD 机制的一般性断言。

这会直接影响下一轮对 **Stage23-family continuation / old proven optimizer routes /
Bridge-type learned coupling** 的排序。

### 8.2 下一轮的候选重审五项（不再因"某机制未验证"就抬成 P1）

1. 承重假设
2. **历史最强反例**
3. 与旧实验**真正的区别**
4. 为什么收益量级可能够到 ≥ 0.69pp
5. 明确失败条件

### 8.3 不该做的事

失败后最容易犯的错：立刻说「那换个 λ / 换个 soup / 跑个 CE-only 看看」——
这些都属于**失败路线内部的小修**。

**下一步：路线级重排，不动 GPU（GPU2 保持空闲）。**
