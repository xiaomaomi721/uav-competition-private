# TAST-E Phase C 纠偏与 CPU-only 重跑指令

**日期**：2026-09-26  
**范围**：只纠正 Phase C eligibility diagnostic；不启动 GPU、不触碰 B4、不改训练方案。  
**当前决定**：`INVALID_DIAGNOSTIC_IMPLEMENTATION / FAIL_CLOSED_OPERATIONALLY`。

---

# 0. 立即执行决定

本次已报告的：

```text
spectrum source→test2 = 1.323
source-source p95 = 0.677
TAST spectrum reduction = 39.3%
clipping = 0.177%
```

**全部暂时撤销为不可解释数值**，不能用于：

- 判定方法前提存在或不存在；
- 宣称 TAST-E FAIL；
- 放行 GPU；
- 修改 beta/rho；
- 选择训练 recipe。

保留原始 `ELIGIBILITY_RESULT.json`、脚本、日志和原 `FAIL_CLOSED` 记录，不覆盖、不删除、不反写为 PASS。新增纠偏状态：

```text
SCIENTIFIC_STATUS = INVALID_DIAGNOSTIC_IMPLEMENTATION
OPERATIONAL_STATUS = FAIL_CLOSED_PENDING_CORRECTED_CPU_RERUN
GPU_AUTHORIZED = false
```

这不是放宽预注册 gate，也不是结果出来后救方法；原因是执行代码没有实现预注册的测量对象。

---

# 1. 已确认的关键实现错误

## 1.1 FFT 空间轴错误（致命）

实际输入是 HWC：`[H,W,3]`，原脚本却使用：

```python
np.fft.fft2(img, axes=(-2, -1))
```

这会沿 `W×C` 变换，而不是 `H×W`。因此所谓中心低频窗口同时混入图像行/颜色通道含义，不是二维空间频谱。

纠正后统一使用：

```python
SPATIAL_AXES = (0, 1)  # for HWC only
fft = np.fft.fft2(image, axes=SPATIAL_AXES)
amp_shifted = np.fft.fftshift(np.abs(fft), axes=SPATIAL_AXES)
# inverse uses the same SPATIAL_AXES
```

若实现者选择先转 CHW，才可用 `(-2,-1)`；必须在代码中 assert shape/layout，不得隐式假设。

## 1.2 原 `mmd2` 不是 MMD

原实现只是：

```python
mean((a - b) ** 2)
```

它把两组任意顺序的第 i 张图逐行配对，打乱顺序数值就变，不是排列不变的分布距离。

纠正版本 primary 使用 **energy distance**（在 source-only 标准化后的 feature vectors 上），或真正 kernel MMD；本指令推荐 energy distance，避免用 test2 调 bandwidth。

```text
E(X,Y) = 2 mean ||Xi-Yj|| - mean ||Xi-Xj|| - mean ||Yi-Yj||
```

必须以块计算/抽样计算避免内存膨胀，并用小数组单元测试验证排列不变性、X==Y 接近0。

## 1.3 source-source bootstrap 不互斥且规模不匹配

原脚本独立抽两组250，平均约125张重叠；source-test2 却比较500-vs-500。

纠正版本 primary：

- 每次从 5597 real-GT source 中无放回抽1000张；
- 前500为source A，后500为source B，严格互斥；
- source-test2 同样500-vs-500；
- 固定 seed42；建议1000次 bootstrap；
- 保存每次 sample IDs 和距离，不能只存 percentile。

## 1.4 transform reduction 不是配对比较

原脚本 transformed 和 raw 使用了不同 content/donor。纠正版本先冻结 tuples：

```text
(content_source_id, donor_test2_id, rho, eval_reference_ids)
```

同一 tuple 比较 raw content 与 transformed content；donor/evaluation reference 不变。推荐 donor test2 与 evaluation-reference test2 互斥，避免仅证明“更像刚刚拷贝的 donor”。

## 1.5 source pool 错误

原脚本读取 `train_ids_5597pseudo.txt`，该 split 实际是6097（5597 real GT + 500 pseudo），不是预注册的5597 real-GT source。

纠正版本必须：

- 使用 frozen 5597 real-GT split；
- assert count=unique=5597；
- 禁止 7000–7499 pseudo/test1 IDs；
- missing image 直接 fail，不得用 `if exists` 静默过滤；
- 写 split path/SHA 和文件 manifest SHA。

## 1.6 统计族未按手册实现

纠正后至少：

1. spectrum：固定 radial bands，log amplitude mean/std，DC 单列或排除，source-only standardization；
2. color：RGB + Lab mean/std 和预注册 fixed-bin histograms；
3. illumination/texture：luminance mean/std/p10/p90、contrast、gradient energy、low/high-frequency ratio。

所有 bands/bins/features 在重跑前写入 config 并锁 SHA；不得看结果后选择。

## 1.7 clipping 与结构审计不完整

总 clipping：

```python
low = np.mean(out < 0)
high = np.mean(out > 1)
total = np.mean((out < 0) | (out > 1))  # low + high
```

必须另记 per-image mean/p95/max、pre-clamp min/max、NaN/Inf、inverse FFT imaginary residual。

需增加固定结构检查和32组 visual grid。visual review 只检查错位/伪影，不允许改变数值门槛。

## 1.8 `save_eligibility.py` 不是证据

禁止再通过硬编码数字生成或覆盖结果。最终 JSON 只能由 diagnostic 本身根据 raw manifests/bootstrap artifacts 生成，并绑定所有 SHA。

---

# 2. 不变项：不得借纠偏调参

以下仍冻结：

```text
beta = 0.01
rho distribution = Uniform(0.25, 0.50)
seed = 42
source pool = 5597 real-GT
primary sample size = 500 vs 500
source-source sets = disjoint
GPU use = prohibited in this rerun
B4 access = prohibited
```

窗口实现应修复为明确的二维、共轭对称 mask，并记录实际 side length；这是数学正确性修复，不是改 beta。必须检查 inverse FFT imaginary residual，而不是无条件 `.real` 后忽略。

---

# 3. 新的冻结 CPU diagnostic 设计

## 3.1 输入审计

### Source

- exact split path/SHA；
- count/unique=5597；
- 逐文件存在性、size/mode、SHA；
- pseudo/test1 ID count=0。

### Test2

- count/unique=1300；
- 连续 `test2_1.png`～`test2_1300.png`；
- 原始均为1024×1024 RGB；
- 逐文件 byte size/SHA manifest。

任何不符在输出目录创建正式结果前 fail closed。

## 3.2 冻结配置

重跑前创建并 SHA 锁定 `ELIGIBILITY_CONFIG_V2.json`：

- image layout/FFT axes；
- beta、window mask rule、actual side length；
- rho distribution；
- feature families/bands/bins；
- source-only standardization rule；
- energy-distance implementation；
- bootstrap count=1000；
- primary n=500；
- tuple count（建议至少500）；
- donor/evaluation reference disjoint rule；
- structure metrics；
- PASS conditions；
- software versions/seed。

## 3.3 单元测试

至少：

1. HWC FFT 等价于逐channel H×W FFT。
2. identity transform（rho=0）逐像素误差在数值容差内。
3. amplitude替换保留 source phase；conjugate symmetry 与 imaginary residual 通过。
4. energy distance 对行重排不变。
5. energy distance `X==X` 接近0。
6. bootstrap A/B 永不重叠，且每组500。
7. raw/transformed comparison 使用相同 tuple。
8. clipping total=low+high。
9. source manifest 恰好5597且无 pseudo IDs。
10. result writer 不接受手填 summary，必须从 raw artifacts 计算。

全部 PASS 才运行 full CPU diagnostic。

## 3.4 Primary source-shift evidence

对每个统计族：

- 用 source-only 参数标准化；
- 计算1000个 disjoint source500-vs-source500 energy distances；
- 计算对应 resampled source500-vs-test2-500 distances；
- 保存分布、p95、差值 bootstrap CI；
- primary 判据：source-test2 超过 source-source p95，且 `D_ST-D_SS` 的95% bootstrap CI下界 >0。

不能只报告单个数字“刚好高/低于 p95”。

## 3.5 Paired transform effectiveness

预先冻结 tuples 和 rho；每个 tuple：

- raw source；
- 使用该 tuple donor 的 transformed source；
- 对同一个 held-out test2 reference distribution 计算距离。

用 paired bootstrap 计算：

```text
reduction = (D_raw - D_transformed) / D_raw
```

报告 mean/CI，不能只报告点估计。

## 3.6 Safety/structure

- finite input/FFT/output/features；
- total clipping mean/p95/max ≤预注册阈值（原总阈值0.5%保留）；
- imaginary residual threshold 在 config 预注册；
- edge/gradient preservation 与 phase structure；
- 固定32组 visual grid；
- corrupt-image/normalization/dimension checks。

---

# 4. PASS/FAIL 规则

为避免再次把“广义外观假设”与“频谱算子假设”混在一起，V2 报告必须分别输出两个结论，不能事后偷换：

## 4.1 原 TAST-E broad gate

原规则继续保留：至少2个统计族存在显著 shift。它若失败，仍写：

```text
TASTE_BROAD_GATE = FAIL
```

绝不把历史 FAIL 改写为 PASS。

## 4.2 TAST-S spectral-specific gate

这是一个**新命名、透明登记的窄假设**，不是篡改原 gate。它只在 corrected implementation 下判断：

1. corrected spectrum source→test2 超过 disjoint source-source p95；
2. `D_ST-D_SS` bootstrap 95% CI 下界 >0；
3. paired transform spectrum reduction 的95% CI下界 ≥20%；
4. 其他 primary families 无 >10% 系统性恶化（其 paired worsening CI上界不得超过10%）；
5. total clipping ≤0.5%，finite/imaginary/structure checks 全 PASS。

全部满足时只可写：

```text
TASTS_SPECTRAL_GATE = PASS
GPU_ELIGIBILITY = PASS_PENDING_EXPLICIT_USER_APPROVAL
GPU_AUTHORIZED = false
```

这表示窄频谱假设有资格进入 matched smoke/formal，不等于自动授权 GPU，也不保证线上涨分。

任一失败：

```text
TASTS_SPECTRAL_GATE = METHOD_PRECONDITION_NOT_MET
GPU_ELIGIBILITY = FAIL
GPU_AUTHORIZED = false
```

不能再开第三版 gate 救结果。

---

# 5. 产物与证据要求

新建输出目录，禁止覆盖原 `02_eligibility`。建议：

```text
<TASTE_ROOT>/02_eligibility_v2_corrected/
├── CORRECTION_NOTICE.md
├── ELIGIBILITY_CONFIG_V2.json
├── SOURCE_MANIFEST.json
├── TEST2_MANIFEST.json
├── SAMPLE_PLAN.json
├── PAIRING_PLAN.json
├── UNIT_TEST_RESULTS.json
├── bootstrap_metrics.npz
├── paired_transform_metrics.npz
├── visual_grid/
├── ELIGIBILITY_RESULT_V2.json
├── ELIGIBILITY_REPORT_V2.md
├── ARTIFACT_HASHES.json
└── EXECUTION_REPORT.md
```

`ELIGIBILITY_RESULT_V2.json` 必须包含：

- 原 broad gate 与新 spectral gate，分别报告；
- 脚本/config/source/test2/sample/pair/bootstrap SHA；
- 全部 CI 和 PASS 子条件；
- `gpu_authorized=false`；
- `platform_submission_performed=false`；
- 明确没有访问 B4。

原目录追加一个非覆盖的 `SUPERSEDED_FOR_SCIENTIFIC_DECISION.md` 即可，说明旧结果因 implementation defect 不再用于科学判定；不要删除旧结果。

---

# 6. 给执行 AI 的直接请求

```text
只执行 TAST-E Phase C corrected CPU rerun。禁止启动 GPU，禁止触碰 B4，禁止进入 smoke/formal，禁止自动提交。

旧 Phase C 的 FAIL_CLOSED 文件必须保留；新增状态是 INVALID_DIAGNOSTIC_IMPLEMENTATION，不得反写旧结果。

必须修复：
1. HWC FFT 使用空间轴(0,1)；
2. 使用排列不变的真实 distribution distance（按指令用 source-standardized energy distance）；
3. source-source bootstrap 为严格互斥500-vs-500，1000次；
4. 使用冻结的5597 real-GT source，禁止6097 pseudo split；
5. raw/transformed 使用相同 content/donor/rho/reference tuples；
6. beta=.01和rho~Uniform(.25,.50)保持不变；
7. 实现完整三族特征、稳定性CI、其他族恶化≤10%、total clipping、finite、imaginary residual和structure checks；
8. 禁止硬编码结果写JSON。

先创建并锁定 ELIGIBILITY_CONFIG_V2/SAMPLE_PLAN/PAIRING_PLAN，再运行测试和诊断。无论结果如何生成 EXECUTION_REPORT.md。最终停止，只回报 broad gate、spectral gate及证据路径；gpu_authorized 必须保持 false，等待用户决定。
```

---

# 7. 当前项目结论

这次事件再次说明，问题不只是“我们门槛太严格”，还包括执行 AI 可能交回一个表面符合手册、实际测量对象错误的报告。今后任何 gate 在采信前必须同时审核：

1. **门槛是否与方法机制一致**；
2. **代码是否真的测了门槛声称的量**；
3. **原始产物是否可复算，报告是否由计算生成而非手填**。

当前 GPU2 保持空闲是正确的；但正确理由是“Phase C 诊断无效、等待 CPU 重跑”，不是“TAST-E 已被数据否定”。
