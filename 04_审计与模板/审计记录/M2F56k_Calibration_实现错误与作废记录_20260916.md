# M2F 56k Calibration：实现错误与作废记录

**日期**：2026-09-16
**范围**：M2F 56k 在自身原生 semantic-score 空间的 calibration 搜索（Stage A–E）
**纪律背景**：本次 session 的角色是 Claude 执行 + 提出解释与方案，GPT 作第二技术审稿人（找逻辑跳跃、实验设计漏洞、数学问题），用户裁定。

---

## A. 实现错误清单（按发现顺序）

每条标注：**谁引入**、**如何被发现**、**是否被强制检查捕获**。

### A1. fold 布尔掩码索引错误

- **位置**：`calib_tta8_search_56k_v3.py` `conf_and_miou(pred_list, idx)`，调用处传 `fold_of == k`
- **症状**：`for i in idx` 迭代布尔值，`GT[True] == GT[1]`、`GT[False] == GT[0]`，每折只累加 2 张图
- **观测值**：`ZERO_TTA8_FOLDS [0.66731073, 0.66727119, 0.66727119]`——整体 mIoU 0.8174 而每折 0.667，数学上不可能；后两折 8 位小数前 5 位相同
- **引入者**：Claude（v3 原始代码）
- **捕获方式**：**人工内部一致性检查**（整体 vs 分折矛盾）+ fold 支持量打印 467/466/466 与 bug 推算完全吻合
- **修复**：`idx = np.asarray(idx); if idx.dtype == bool: idx = np.where(idx)[0]`
- **影响**：`ZERO_TTA8_FOLDS`、`fold_gains_pp`、`min_fold_gain_pp`、`fold_stable_pick` 全部作废；整体 mIoU / 逐类 IoU / Stage C-E 不受影响

### A2. 广播维度错误（pre-existing，v3 从未跑通）

- **位置**：`other_side()` 中 `b = torch.tensor(...).view(1, 1, 7)`
- **症状**：`(N,7) + (1,1,7)` → `(1,N,7)`，后续全部错形状
- **崩溃信息**：`RuntimeError: expanded size of the tensor (1048576) must match the existing size (7)`
- **引入者**：Claude（v3 原始代码）
- **捕获方式**：**运行时崩溃**
- **关键含义**：该单类快路**从未成功执行过一次**；零偏置那一遍走 `full_pred`，SELFCHECK 是首次调用 `other_side`，当场崩溃。即"精确加速"代码在崩溃前处于**零验证**状态
- **修复**：`.view(1, 7)`

### A3. 补丁引入的残留变量删除

- **位置**：`other_side()` 末行 `del r, v, a`
- **症状**：tie 补丁把 `v, a = r.max(1)` 改成 `a = r.argmax(1)`（值改由 `r.gather` 取得），`v` 已不存在
- **引入者**：Claude（tie 补丁）
- **捕获方式**：**运行时崩溃**（随 A2 一起暴露）
- **修复**：`del r, a`

### A4. `margin_quantiles` 口径错误

- **位置**：`calib_tta8_search_56k_v3.py:126-129`，`torch.topk(samp, 2, dim=1)`
- **症状**：`dim=1` 是**像素轴**不是**类轴**，算出的是"每图最大像素值 − 第二大像素值"，与类间 margin 无关
- **观测值**：q0.001–q0.25 全为 0.0
- **引入者**：Claude（v3 原始代码）
- **捕获方式**：**字段命名与数值的语义不符**（作为一个"margin"分位表，25% 分位为 0 不合理）——但**发现得晚**，中间还被 Claude 误读为"25% 像素精确 tie"
- **修复**：该字段**作废**（不参与任何搜索决策，仅报告字段）；另写独立诊断用 `dim=-1` 正确计算
- **连带错误**：Claude 基于此字段推出"至少 25% 像素 top1==top2 精确 tie"，被 `TIE_CENSUS = 3` 直接证伪

### A5. 边界统计脚本 0-based / 1-based 索引混用

- **位置**：`boundary_tau.py`（v1）
- **症状**：`isa1 = (a1 == c)` 比较 0-based 列索引，而 `pred == c` / `g == c` 比较 1-based 官方 id，相差 1
- **后果**：**每行报告的 FN/FP/TP/TN 计数属于官方类 c，而 τ 属于官方类 c+1**——分组与 τ 完全错位
- **观测矛盾**：FP 组 `frac(τ<0) = 0.0000`、`τmin ≈ +0.000000`，与恒等式 `τ<0 ⟺ pred=c` 直接冲突
- **引入者**：Claude
- **捕获方式**：**数学恒等式冲突**（FP 定义要求 τ<0，实测全为正）
- **修复**：统一"列 k（0-based）== 官方类 k+1 == NAMES[k]"，并新增不变量断言：
  - `FN 像素（GT=k, pred≠k）必须 τ ≥ 0`
  - `FP 像素（GT≠k, pred==k）必须 τ ≤ 0`
  - **该断言若一开始就有，此 bug 会立即被捕获**
- **影响**：v1 全部边界 τ 结果作废
- **连带错误**：Claude 在调查过程中先称"FP_c 是空集"、后称"FP_c 全是 τ=0 的 tie 像素"，**两次都不对**（npy 实测 τ==0 计数为 0，且 FP 非空）

### A6. 诊断脚本的 `torch.from_numpy` 误用（两次）

- **位置**：`scale_diag.py`，对已是 tensor 的对象调用 `torch.from_numpy`
- **症状**：`TypeError: expected np.ndarray (got Tensor)`
- **引入者**：Claude
- **捕获方式**：运行时崩溃
- **修复**：去掉 `from_numpy`

### A7. `torch.quantile` 元素数上限

- **位置**：`scale_diag.py`，对 1.57 亿元素张量调用
- **症状**：`RuntimeError: quantile() input tensor is too large`
- **引入者**：Claude（未预判库限制）
- **捕获方式**：运行时崩溃
- **修复**：改用 `np.quantile`（无上限）

### A8. 打印语句 KeyError

- **位置**：`scale_diag.py`，引用 `['q0.90']`，实际键为 `'q0.9'`
- **引入者**：Claude
- **捕获方式**：运行时崩溃
- **修复**：`'q0.9'`；并加增量 JSON 落盘，避免崩溃丢数据

### A9. `pred` 从 `torch.topk` 索引推导

- **位置**：`boundary_tau.py`（v1）
- **问题**：`torch.topk` 在数值并列时的索引顺序**未定义**，不保证取首个最大值；`argmax` 才保证
- **捕获方式**：**新增断言主动触发**——`assert (am == t1i).all()` 在 200 张后报 `topk/argmax disagree -> ties present`
- **含义**：tie 真实存在（即 `TIE_CENSUS` 的 3 个像素），断言按设计中止，**未产出错误数据**
- **修复**：改用两遍 argmax 精确算法，完全不依赖 topk

---

## B. 非实现类错误：事实陈述失实

### B1. τ 表数字（最严重）

- **性质**：**把推测数字包装成实测数字陈述**，非解释错误
- **内容**：一张 τ_c 结果表（Building/Road/Water/Barren 四行），以实测格式呈现，**未标注哪些是实测、哪些不是**
- **实测对比**：表中**每一个 P 值都是错的**，且错误方向不一致（Building 报低、Water 报高）
- **来源**：一条 monitor 事件回显，**从未与磁盘规范日志核对**
- **唯一可信来源**：`SCALE_DIAG.json` / `tau_partial.json` / `/tmp/scale_diag_56k.log`
- **处置**：见 `作废声明_M2F56k_tau表_20260916.md`

---

## C. 即时机制解释错误（非事实陈述问题）

| # |  Claude 的解释 | 证伪方式 | 正确表述 |
|---|---|---|---|
| C1 | `margin_quantiles` 25% 为 0 ⟹ "25% 像素精确 tie" | `TIE_CENSUS = 3` | 该字段口径错误；真实 tie 仅 3 像素 |
| C2 | fold 支持量差异 ⟹ "Barren 解释 3.52pp fold 差" | mIoU 为 8 类**等权**平均，GT 像素数不改变类权重 | 折间**类别/场景构成**不同；需 3×8 IoU 才能归因 |
| C3 | `median\|τ\|≈1` ⟹ "边界在 ±1，网格没覆盖" | 对固定类 c，绝大多数像素 `S_c≈0, max_{j≠c}≈1`，`\|τ\|≈1` 是平凡结果 | 这些非边界像素；**不能**推出该扫 ±1 |
| C4 | `P(\|τ\|<0.125)≈0.3%` ⟹ "增益淹没在噪声里" | 1.434e9 × 0.003 ≈ **430 万像素** | 非噪声量级；集中在稀有类足以显著改变 IoU |
| C5 | `τ<0` 侧 = "次优类几乎追平的像素" | `τ=−0.999` 意味着 `S_c − max_{j≠c} ≈ 0.999`，是**高置信**预测 | **符号决定赢家，绝对值决定离边界多远**；只有 `τ≈0⁻` 才是边界像素 |
| C6 | FN 的 q50/q75/q90/q95/q99 作为正 bias 候选 | 取 q50 等于一上来就试图纠正 50% 的 FN | 应从 FN τ 的**低分位**起步，即最易纠正的一小撮 |
| C7 | "有效边界尺度约 0.01" | global margin 的 q0.001≈0.0061 只说明**存在长左尾** | 不能推出最优 bias 是 0.01；须等 FN/FP/TP/TN 条件分布 |

**C1–C7 的共同模式**：看到单个数字立刻构造机制解释，未先问"这个量能否从现有缓存直接算出/证伪"。

---

## D. 被证伪前的错误中间结论（内部，未外传）

| 结论 | 状态 | 修正 |
|---|---|---|
| "FP_c 是空集" | **错** | FP 非空（Building 51.8M），v1 的错位导致误判 |
| "FP_c 全是 τ=0 的 tie 像素" | **错** | npy 实测 τ==0 计数为 0 |
| "Stage C 粗网格无价值" | 部分成立 | 唯一可支持的结论是"±0.125 步长 9 点 1D 网格未发现明显单维收益"，**不能**判定 calibration 无效 |

---

## E. 仍然有效的核心结果

| 量 | 值 | 来源 |
|---|---|---|
| TTA8 ZERO val mIoU | **0.8173524135** | `stage_scale_sensitivity.json` |
| 单视图 56k val mIoU | 0.8155545 | 既有三档验证报告 |
| TTA8 增益 | **+0.1798pp** | 上式相减 |
| 三折 mIoU | 0.83107416 / 0.82215874 / 0.79583841 | `scale_diag_56k` fold 段 |
| fold0−fold2 归因 | Barren 贡献 74.5% | 8 类等权分解，合计吻合 1e-7 |
| 真实 margin 分位 | q0.001=0.006107 … q0.999=0.999998 | `SCALE_DIAG.json` |
| valid 像素精确 tie | **0 / 153,790,192** | 同上 |
| 全图（含 ignore）tie | 3 像素 | `TIE_CENSUS` |
| 首步偏置翻转比例 | 0.05%（Vehicle）– 0.69%（Vegetation） | τ `P(\|τ\|<step)` |

---

## F. 流程结论

1. **三个会毁掉结果的实现 bug（A1 fold 索引、A2 广播维度、A3 残留 `del v`）已全部暴露并修复**，fast path 已通过真实 full-argmax 三组对照（连续/强制 tie/合成离散，各 0 mismatch）。
2. **A5 说明不变量断言的必要性**：`FN⇒τ≥0`、`FP⇒τ≤0` 这类从定义直接导出的检查，成本极低，却能立即捕获整类索引错误。**今后凡有"定义蕴含的符号/范围关系"，一律写成断言而非注释。**
3. **B1 与 C1–C7 的区别必须保持**：解释错是推理问题，把推测当实测陈述是**事实陈述失实**，后者不可接受。
4. 后续任何报告引用 τ / margin / fold 数值时，**必须附 JSON 路径与字段名**。

---

## G. 待办

- [ ] A5 v2 索引修正结果核验（`BOUNDARY_V2_DONE`）
- [ ] Stage C2：event-count 候选生成 + full 8-channel FP32 exact 评分
- [ ] Stage C2 前禁止进入 coordinate descent / random joint search / test1 打包 / 训练或权重改动
