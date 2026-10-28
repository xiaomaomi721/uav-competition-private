# Stage65 pseudo round2 历史法医原始取证（2026-09-16）

审计对象：`stage65_w1_p1_pseudolabel_round2_60ep_seed42`
审计目的：回答「Stage65 效果不够好，是否可能来自实现/数据/绑定错误，而不是 pseudo round2 方案本身？」
采集方式：**严格只读**。未启动 GPU、未重训、未重生成 pseudo、未修改或删除任何历史文件。
判决词表：**VERIFIED / NOT_FOUND / UNBOUND / UNKNOWN**（全文只用这四种）

> 术语约定：凡称「VERIFIED」，指本次有独立证据（重算、重扫、逐位比对）验证。
> 凡证据链缺一环，一律标 **UNBOUND**，不补脑。

---

## 0. 结论速览

| 环节 | 判决 |
|---|---|
| teacher 身份（3 个 checkpoint SHA） | VERIFIED |
| pseudo 生成本身可复现（独立重算融合） | **VERIFIED（500/500 完全一致）** |
| official ↔ train ID 映射 | VERIFIED |
| image ↔ pseudo mask 对齐 | VERIFIED（0 错配） |
| confidence filtering 真实语义 | VERIFIED（但与脚本文档不一致，见 E） |
| 训练数据组成 6097 = 5597+500 | VERIFIED |
| 与 val 无重叠 | VERIFIED |
| augmentation 几何一致性 | VERIFIED（同一代码路径） |
| init / optimizer / scheduler | VERIFIED（LR 逐步吻合） |
| checkpoint 选择 | VERIFIED |
| package → checkpoint 绑定 | VERIFIED（文件级） |
| package → 平台分数绑定 | **B 级**（非平台文件名绑定） |
| **round-1（stage56）mask 重链脚本** | **UNBOUND** |
| **round-1 teacher 身份** | **UNBOUND** |

---

## A. Stage65 身份链

### A.1 关键文件（路径 / 大小 / mtime / SHA256）

| 文件 | bytes | mtime | SHA256 |
|---|---:|---|---|
| `configs/stage65/wave1_exp65_p1_round2_60ep.yaml` | 840 | 2026-09-07 22:01 | `a9fd16f8c5f4cb2def3ba20f858ddd094d779474f0891d11a2cdfb4ac2afc6c0` |
| `stage65_runs/…/config.yaml`（**解析后**运行配置） | 2610 | 2026-09-08 08:10:28 | `851a23e851ceec89bd6f6bda1f63106f8942cdec24cc504152c3d96712eee6f8` |
| `stage65_runs/…/run_manifest.json` | 15109 | 2026-09-09 14:11 | `53e40066de2b41d93b23df9f72533d2dccd6bcaff8bcc911c8b05d1f9103c212` |
| `stage65_runs/…/train_log.csv` | 9619116 | 2026-09-09 14:09 | `0f13547f3a0ca277ee695199e11b6e496f7db2aa5a8565c32b7a82a174cbbc42` |
| `stage65_runs/…/metrics.json` | 5206 | 2026-09-09 14:11 | `0128a971dd975ca1e1af7f29a25be9a62bb54470fa05208f7ef25a199e7d7123` |
| `src/trainer.py` | 52211 | — | `3f3168fff21f498b273dd296d3b15184ed874c04397924a001cfdd45d76052f3` |
| `src/dataset.py` | 11590 | — | `5467112f0f44df85a4e43790aa25bf7818350df5d650db8f5c06ab3ba75cfc16` |
| `src/transforms.py` | 10982 | — | `0f28c0f308ec7203267d442e5a9f3df89bf009cc89dedfbf88341b0c5aaab28d` |
| `src/label_mapping.py` | 1889 | — | `bb4ec52832e3afae314fd2bf5010fb364fa340b8b29d697d902d1614c38c76c6` |
| `src/rare_crop.py` | 7723 | — | `78964a0f42c9f14de96deddc988624653449f7f03a3000e53477295f226d3f64` |
| `/home/jhy/gen_pseudo_labels.py` | 10507 | — | `739e19e97d5f2781a5cc53e40af44568971c0d0fb820dd932b083b1fa37a685c` |
| `/home/jhy/p1_pseudolabel_full.sh` | 3209 | — | `95b5975ec4355c0190443d51a974c261166853b547e0fa1dbe051391fbeb139c` |
| `/home/jhy/relink_pseudo_v2.sh` | 1349 | — | `8b28328ee7c594733089f21f085edf9ff23c74f9a3b1ea4bef86d46035d1ea0a` |
| `/home/jhy/launch_stage65.sh` | 1519 | 2026-09-07 22:01 | `9ed502278fc82c0d08c0d79cb8cc05bb43020438d28685ea0f0f52c013757e7d` |
| `/home/jhy/stage65_standard.zip` | 3828685 | 2026-09-09 17:17 | `480c5e0288f00ba245be1ee79d44137365aec0f96ef253692851a50098b028dc` |
| `/home/jhy/stage65_calib.zip` | 3774200 | 2026-09-09 17:21 | `d87a233dfb9f3134b5c91b4a55cf8de6e845637be1aa4ba2b5a99835ca2f48b7` |

### A.2 teacher checkpoint SHA（3 教师，VERIFIED 存在且可读）

| 教师 | SHA256 |
|---|---|
| stage23 `best_miou.pt` | `d3740b4ef73fbb3ac57ddeedc3547ce4ce1ba31442dc7c70e1fe96b1fab3e228` |
| stage56 `best_miou.pt` | `a2adfb906d53d86b1f8ce2eb6bfa3bdb7c040e025bc187ee4165c09858ac78f2` |
| stage62 `best_miou.pt` | `e3c80e48fc71a8ac21ca54391b780da2b343fb67ce063d854841b4adfba4e9a8` |

### A.3 时间线（精确到纳秒，用于判定 relink 与开训的先后）

```
2026-09-07 22:01:35.74  launch_stage65.sh 创建
2026-09-08 04:34:03.20  pseudo_labels_v2_multi/masks_official/test1_1.png 生成
2026-09-08 04:34:12.73   ... test1_500.png
2026-09-08 04:34:13     p1_pseudolabel.log 记录 "P1 pseudo-label round-2 done"
2026-09-08 08:10:19.035  第 1 个 relink（7000.png）
2026-09-08 08:10:19.645  第 500 个 relink（耗时 0.61 s）
2026-09-08 08:10:28.609  run_manifest.start_time（= 训练启动）
2026-09-09 06:11:14     run_manifest.end_time（COMPLETED）
```

**relink 早于训练启动 8.96 秒 → Stage65 训练时读到的确实是 v2（round-2）伪标签。 VERIFIED**

### A.4 身份链

```
teacher: stage23 / stage56 / stage62 best_miou.pt   ← VERIFIED（SHA 见 A.2）
        ↓  gen_pseudo_labels.py (SHA 739e19e9…)   ← 路径由 p1_pseudolabel_full.sh 硬编码
        ↓  region 模式, threshold 0.90, TTA8, 每教师独立输出
        ↓  三教师全票一致融合（p1_pseudolabel_full.sh 内联 python）
pseudo dataset: /home/jhy/pseudo_labels_v2_multi/masks_official/  ← VERIFIED（500 张）
        ↓  relink_pseudo_v2.sh（SHA 8b28328e…）
        ↓  symlink 7000-7499 → masks_official/test1_1..500.png
Stage65 训练: configs/stage65/…yaml (SHA a9fd16f8…)  ← VERIFIED
        ↓  initialization.mode = published_pretrained  ← ⚠️ 非续训
init: nvidia/segformer-b3-finetuned-ade-512-512 (/home/jhy/models/segformer-b3-ade512)
        ↓
Stage65 checkpoint: best_miou.pt (step 81200) SHA f0a6c7ec…  ← VERIFIED
        ↓  run_config.yaml 记录 checkpoint_sha256
submission: 45.zip=standard SHA 480c5e02… / 46.zip=calib SHA d87a233d…  ← VERIFIED（文件级）
        ↓
leaderboard: 0.680077 / 0.683137   ← **UNBOUND 至平台文件名级**（见 J）
```

---

## B. Teacher / pseudo 生成链

`p1_pseudolabel_full.sh` 硬编码三个教师路径，逐个跑 `gen_pseudo_labels.py --mode region --threshold 0.90`，各输出到 `/home/jhy/p1_t{1,2,3}/masks_region_t0p90/`。

生成配置（来自 `p1_pseudolabel.log`）：

```
=== keep ratio by threshold ===
conf>=0.7:  kept 94.2%
conf>=0.8:  kept 88.6%
conf>=0.9:  kept 83.2%
conf>=0.95: kept 77.9%
DONE mode=region saved_t=0.9 failures=0
```

**注意**：这里 83.2% 是**单教师**的 region-keep 比例；最终三教师全票融合后降到 71.39%（见 C）。

融合（内联 python，逐张）：

```
valid = (A!=255)&(B!=255)&(C!=255)
agree = (A==B)&(B==C)&valid
out   = where(agree, A+1, 0)        # official: 一致→1..8, 否则→0(ignore)
```

日志输出：

```
unanimous pixels: 374269029 / valid 375497692 = 99.7%
images with ZERO consensus pixels (all-ignore): 19/500
```

### B.1 独立复算融合结果 —— VERIFIED

我按上述规则**重新实现一遍**，与磁盘上的 500 张 shipped mask 逐张比对：

```
basename lists identical across t1/t2/t3 : True
  -> zip-by-position 实际是按名对齐（三教师输出顺序一致）
re-derived merge == shipped mask, all 500 : True  (500/500)
```

**这是本次审计最强的一条证据**：融合过程可被完全复现，无错位、无残留。

### B.2 三教师中间产物仍在（可用于日后复验）

`/home/jhy/p1_t1`、`/home/jhy/p1_t2`、`/home/jhy/p1_t3` 均存在，各 500 张 `masks_region_t0p90/*.png`。

### B.3 pseudo 的底图是竞赛 test1 的 500 张测试图

```
images/7000.png -> /home/jhy/datasets/2026-低空图像语义分割赛道-训练集/images/test1_1.png
masks/7000.png  -> /home/jhy/pseudo_labels_v2_multi/masks_official/test1_1.png
```

`gen_pseudo_labels.py` 头部自述："Official-compliant: allowed to label official test images with own model and add them to training; final submission is a single model."
→ 这是**已获官方答复允许**的路线（用户 2026-09-15 转达官方回复）。**记录事实，不作合规判断。**

---

## C. Label mapping / Ignore 法医（全部 500 张静态扫描，非抽样）

### C.1 扫描结果

```
masks scanned : 500
shapes        : {(1024, 1024): 500}
dtypes        : {'uint8': 500}

value : pixel_count
     0 :  150,018,971
     1 :   99,153,930
     2 :   74,744,154
     3 :   32,694,575
     4 :   32,058,289
     5 :    1,098,996
     6 :  105,933,919
     7 :   26,921,055
     8 :    1,664,111

distinct values present : [0, 1, 2, 3, 4, 5, 6, 7, 8]
files containing value 0   (official Ignore) : 494
files containing value 255 (train Ignore)    : 0
values > 8 present (illegal in official space) : []
```

**判定：pseudo mask 处于 OFFICIAL 空间（0=Ignore, 1..8=类）。VERIFIED**

### C.2 与真实 GT 同空间

```
real 0: values=[1, 2, 3, 5, 6]        (official mask 0000.png)
real 1: values=[1, 2, 3, 4, 5, 6]
real 2: values=[1, 2]
OFFICIAL GT 0000.png: values=[1, 2, 3, 5, 6]   ← 与 real 0 逐值一致（同一文件）
```

真实掩码只含 official 1..8（**不含 0**）；pseudo 含 official 0..8。
→ 两者同空间，走**同一个** `official_to_train`。**VERIFIED**

### C.3 映射函数实际行为

`src/label_mapping.py`（SHA `bb4ec528…`）：

```python
def official_to_train(mask):
    """Map official 0..8 to train 255,0..7 without modifying the input."""
    values = set(int(x) for x in np.unique(array))
    illegal = values - OFFICIAL_IDS          # OFFICIAL_IDS = set(range(9))
    if illegal:
        raise ValueError(...)
    mapped = np.full(array.shape, IGNORE_INDEX, dtype=np.uint8)   # 255
    valid = array > 0
    mapped[valid] = array[valid] - 1
    return mapped
```

- official 0 → **255**（internal ignore）✓
- official 1..8 → 0..7 ✓
- 非法 ID（>8）→ **直接抛错**，不会静默通过 ✓
- 无 +1/-1 双重偏移；单一映射点

**VERIFIED**

### C.4 一致性交叉验证

我扫描得到的 official 1..8 像素总和 = 524,288,000 − 150,018,971 = **374,269,029**，
与融合日志的 `unanimous pixels: 374269029` **逐位相同**。**VERIFIED**

---

## D. Image ↔ pseudo mask 对齐

```
images entries : 7496
masks  entries : 7496
name sets equal: True

pseudo stems (>=7000): 500
  min=7000 max=7499 contiguous=True
  duplicates: 0

pairs checked            : 500
index mismatches         : 0        ← 按 resolve 后的 test1_N 索引比对
unresolvable to test1_N  : 0
```

**关键点**：比对的是**软链解析后的目标文件名中的 test1_N 索引**，而非文件数量——数量相等本身不能证明对齐。7000→test1_1, 7001→test1_2, …, 7499→test1_500 逐对一致。

**VERIFIED**

> 注：`masks/` 下 7496 个**全部是 symlink**（regular files = 0）。mtime 分两批：**6997 个 9/5**（真实 GT，指向官方 train/masks）+ **501 个 9/8**（500 伪标签 + 1 个目录项）。与 relink 时间线自洽。

---

## E. Confidence filtering 是否真的生效

### E.1 真实代码语义（`gen_pseudo_labels.py`）

```python
def region_keep_mask(argmax, conf, threshold):
    """Whole connected component kept iff >= 80% of its pixels have conf >= threshold."""
    above = (conf >= threshold).astype(np.int64)
    for cls in range(8):
        cls_mask = argmax == cls
        labels, n = ndimage.label(cls_mask)      # 逐类连通域
        comp_sum  = ndimage.sum(above, labels=labels, index=np.arange(1, n+1))
        comp_size = ndimage.sum(ones,  labels=labels, index=np.arange(1, n+1))
        frac = comp_sum / np.maximum(comp_size, 1)
        keep_comp = np.where(frac >= 0.80)[0] + 1
        keep |= np.isin(labels, keep_comp)
    return keep
```

- 阈值：**0.90**（命令行 `--threshold 0.90`，日志 `saved_t=0.9`）
- 判据：**逐类连通域整块保留，当且仅当该块 ≥80% 像素 conf ≥ 0.90**
- 置信度定义：softmax 的**逐像素 max-prob**（脚本头部 "5-softmax → per-pixel confidence"）
- filtering 时机：**生成时**做，不是在训练时
- 被过滤像素：写入 `IGNORE` 值（脚本内 `mask = np.full(argmax.shape, IGNORE)`）

### E.2 ⚠️ 文档与代码不一致（记录，非判决为 bug）

`region_keep_mask` 的 **docstring 说 "mean confidence"**，但**代码实现的是 "≥80% of pixels ≥ threshold"**。
两者在本数据上不等价。这是**文档/代码漂移**，非执行错误——实际生效的是代码行为。

判决：**VERIFIED**（代码行为已确认），但**注释具误导性**。

### E.3 filtering 实际发生量的独立复核

由 C.1 全量扫描重算：

```
round2 non-ignore pixels: 374,269,029
round2 keep ratio: 71.39%   (374269029 / 524288000)
```

融合日志自报 `unanimous pixels: 374269029 / valid 375497692 = 99.7%`。

→ **两个比例含义不同**：99.7% 是"三教师有共识的像素占三教师都非 ignore 的像素"；71.39% 是"最终保留像素占全部像素"。两者不矛盾。**VERIFIED**

### E.4 19 张零共识图像（全部列出）

```
test1_3, 24, 32, 119, 122, 185, 207, 247, 258, 312, 353, 372,
381, 406, 418, 420, 455, 483, 492
```

这 19 张最终 mask **全为 official 0 = 全 ignore**。
→ 训练时经 `official_to_train` 变成**全 255**，`CrossEntropyLoss(ignore_index=255)` 下**该图贡献 0 loss**。
→ 它们被计入 `train_sample_count=6097` 并照常参与采样，但实际是无梯度样本。

**VERIFIED**（事实确认，是否为缺陷留给 L）

---

## F. Stage65 实际训练数据组成

### F.1 精确数字

```
train_ids_5597pseudo.txt: 6097 行
  <7000 : 5597   (真实 GT)
  >=7000:  500   (伪标签 test1_1..500)
  unique: 6097   (无重复；read_stems 遇重复会直接 raise)
```

与 val 的关系：

```
train list n = 6097   val n = 1399
overlap train & val : 0
val entries >=7000  : 0        ← val 中无任何伪标签
val range           : 0006 .. 6986
```

**VERIFIED：无 val 泄漏，伪标签与验证集完全不相交**（伪标签底图为 test1，val 全为真实训练图）。

### F.2 采样机制与比例

```
train_loader = DataLoader(..., shuffle=True, ...)      src/trainer.py:291-294
val_loader   = DataLoader(..., shuffle=False, ...)     src/trainer.py:301-304
```

**均匀随机采样**（非 weighted sampler、非 dataset 加权、非 concat 重复）。
→ 伪标签在训练中的期望占比 = **500 / 6097 = 8.20%**

### F.3 实际采样量交叉验证（**修订：差异已解释**）

```
crop_statistics.json:
  total_crops     = 335,945
  random_crops    = 251,808
  rare_crops      =  84,137
```

> **修订说明**：初版把本项标为「遍历完整性 VERIFIED」，那是在 55 的差额未解释前就下了结论。GPT 指出这不成立，现予撤回并补上差额来源。**差额来源现已确定，见下。**

`total_crops` 的定义（`src/trainer.py:598-599`）：在 accumulation 内层循环中，
`crop_statistics.update_batch(metadata)` 逐 micro-batch 累加**该 micro-batch 内的实际样本数**。

**差额 55 的完整算术闭合：**

```
数据集 N = 6097，batch_size = 2，drop_last = False   (src/trainer.py:297)

每个 epoch 的 micro-batch 构成：
  满批   = floor(6097/2) = 3048 个  → 6096 个样本
  残批   = 1 个（含 1 个样本，6097 mod 2 = 1）
  合计   = 3049 micro-batches / epoch

总 micro-batches = 84000 steps × 2 = 168000
  168000 ÷ 3049 = 55 个完整 epoch，余 305 个 micro-batch
  → 共经历 55 个残批

样本暴露 = (168000 − 55) × 2 + 55 × 1
         = 335890 + 55
         = 335945            ← 与 total_crops 精确相等
```

**逐项对应 GPT 要求区分的口径：**

| 口径 | 值 | 说明 |
|---|---:|---|
| data iterations（数据步） | 84,000 | `completed_steps` |
| optimizer updates | 83,971 | `optimizer_update_count` |
| AMP skips | 29 | `skipped_optimizer_steps`；**每个 skip 仍消耗了一个完整 micro-batch 的数据** |
| micro-batches consumed | 168,000 | = 84000 × 2 |
| **sample exposures（crop 计数）** | **335,945** | 每 micro-batch 计其实际样本数 |
| 朴素期望（假设每 micro-batch 都满） | 336,000 | 若 6097 能被 2 整除 |
| **差额** | **55** | **= 跑完的 epoch 数，即残批个数** |

**结论**：
- 差额 **不是** AMP skip（skip 不减少数据消费，29 次 skip 已计入 168000 内）
- **不是**日志截断
- 是 `6097 mod 2 = 1` 导致**每个 epoch 必然产生一个单样本残批**的确定性结果
- 55 恰为奇数，与"每个残批只贡献 1 而非 2"完全自洽

**判决：差异来源 VERIFIED（算术可复现）。** 数据遍历完整。

### F.4 「配置写 500 不等于实际用到 500」的检验

在本例中**成立**：500 个伪标签 stem 全部在清单内、全部存在、无重复，且伪标签图片以 symlink 形式全部可解析。
唯一实质削弱是 E.4 的 19 张零共识图 → 实际**有梯度贡献的伪标签图 = 481 张**。

**VERIFIED**（实际参与训练的伪标签图 = 481 张有监督 + 19 张零梯度）

---

## G. Augmentation 几何一致性

### G.1 代码考察（`src/dataset.py` + `src/transforms.py`）

- `UAVSegmentationDataset.__getitem__` 对**所有** stem 走同一个 `self.transform`：
  ```python
  image, mask = self.load_raw(stem)          # 统一 official_to_train
  pixel_values, labels, crop_metadata = self.transform(image, mask, ...)
  ```
  **不存在 real / pseudo 两套 Dataset 或 Transform 的分支。 VERIFIED**

- crop 原点由 `RareClassCropper.choose_with_metadata(mask, ...)` 基于 **mask** 决定，
  再 `apply_geometric_ops(image, mask, crop_box=...)` **同时对 image 与 mask** 施加裁剪。
  → pseudo mask 与 image **共用同一 crop 原点**。**VERIFIED**

- 翻转/旋转：`apply_geometric_ops(image, mask, horizontal_flip=..., vertical_flip=..., rotate_k=...)`
  内部 `np.flip` / `np.rot90` **同时作用于 image 与 mask**。**VERIFIED**

- mask 缩放插值：`resize_pair()` 中
  ```python
  resized_image = Image.resize(..., BILINEAR)
  resized_mask  = Image.resize(..., NEAREST)     ← mask 用最近邻
  ```
  → **不会因插值产生非整数类别 ID**。**VERIFIED**

- 颜色增强：`_color_jitter` / `_domain_image_augment` **只接收 image**，不接触 mask。**VERIFIED**

- `rare_crop` 对伪标签同样生效（对 mask 统一判断 priority_ids `[7,4,6]`）。
  → 伪标签中的 Vehicle/Barren/Agricultural 区域会同样触发 rare crop。**VERIFIED**

- ID 合法性断言：`TrainTransform.__call__` 末尾
  ```python
  illegal = values - set(range(8)) - {self.ignore_index}
  if illegal: raise ValueError(...)
  ```
  → 若插值或映射产生非法值会**立即抛错**，不会静默污染。**VERIFIED**

### G.2 结论

image 与 pseudo mask **几何同步**，不存在"对 pseudo mask 单独 resize 造成错位"的路径。
**VERIFIED**

---

## H. Init / optimizer / scheduler

| 项 | 值 | 来源 |
|---|---|---|
| initialization | `published_pretrained`，`nvidia/segformer-b3-finetuned-ade-512-512` | run-dir `config.yaml` |
| 是否续训 | **否**（`inherit_stage03_optimizer=false`, `inherit_stage03_scheduler=false`） | 同上 |
| optimizer | AdamW, lr 6e-05, weight_decay 0.01，冷启 | 同上 |
| scheduler | polynomial, power 1.0 | 同上 |
| batch / accumulation | 2 / 2（有效 4） | 同上 |
| max_steps | 84000 | 同上 |
| CWFA | p=0.3 eps=4.0 norm_eps=1e-12 start_step=28001 | 同上 |
| AMP | true | 同上 |
| loss | cross_entropy | 同上 |
| seed | 42 | 同上 |
| EMA | 无（`ema_enabled` 未启用） | metrics.json |
| 实际完成 | `completed_steps=84000`，`optimizer_update_count=83971`，`skipped=29` | metrics.json |

### H.1 LR 反推交叉验证 —— VERIFIED

从 `train_log.csv` 读实际 LR，与 `lr = 6e-5 × (1 − step/84000)` 比对：

| step | 日志 LR | 配置期望 | ratio |
|---:|---|---:|---:|
| 1 | 5.9999285714e-05 | 5.9999285714e-05 | **1.000000** |
| 20 | 5.9985714286e-05 | 5.9985714286e-05 | **1.000000** |
| 100 | 5.9928571429e-05 | 5.9928571429e-05 | **1.000000** |
| 1000 | 5.9285714286e-05 | 5.9285714286e-05 | **1.000000** |
| 28001 | 4.0007142857e-05 | 3.9999285714e-05 | 1.000196 |
| 84000 | 2.0714285714e-08 | 0（分母归零） | — |

**前 1000 步 ratio 精确为 1.000000**，scheduler 从 step 0 起步，冷启无残留。
step 28001 的 1.000196 与 step 84000 的非零尾值，是主线"`scheduler.step()` 只在成功更新时推进"的已知行为（29 次 skip 造成），**Stage23 / Full6996 同样如此**，非 Stage65 特有。

### H.2 CWFA 门控 —— VERIFIED

```
first step with cwfa_active=1 : 28001     ← 与 config start_step 精确一致
count cwfa on / off           : 56000 / 28000
```

---

## I. Checkpoint 选择

| checkpoint | step | bytes | mtime | SHA256 |
|---|---:|---:|---|---|
| `best_miou.pt` | **81200** | 189186507 | 2026-09-09 13:12:23 | `f0a6c7ec5a6a92e4bd9ef4c4bbcba7cdc0f655df976f4e103c7183ba4dc4e168` |
| `last.pt` | 84000 | 567426665 | 2026-09-09 14:09:13 | `383a62699efce21aff77bb6cc52a04e3c88e69367681bf4638216cd8839dc440` |

`metrics.json`: `best_miou=0.8129741638309373`, `best_step=81200`。

**两个提交包使用的都是 `best_miou.pt`（step 81200）**，由 `run_config.yaml` 的 `checkpoint_sha256` 绑定。
**未使用 final / last.pt 出包。 VERIFIED**

**metric 是否可用于该训练设定**：该 best 基于 1399 张真实验证图，与训练集无重叠（F.1），
该 metric 在**其自身定义域内**可用。

---

## J. Package → leaderboard score 绑定

| 包 | SHA256（已与盘上文件逐位核对） | checkpoint | 偏置 | 平台分 |
|---|---|---|---|---|
| `45.zip` = `stage65_standard.zip` | `480c5e0288f00ba245be1ee79d44137365aec0f96ef253692851a50098b028dc` | `best_miou.pt` step 81200 (`f0a6c7ec…`) | 全 0（ZERO） | 0.680077 |
| `46.zip` = `stage65_calib.zip` | `d87a233dfb9f3134b5c91b4a55cf8de6e845637be1aa4ba2b5a99835ca2f48b7` | 同上 | CHAMPION 向量 | 0.683137 |

### J.1 证据分级

| 链接 | 级别 | 依据 |
|---|---|---|
| 包文件 ↔ 包 SHA | **A** | 我用 `sha256sum` 直接重算，与 `45_46_prep.json` 记录逐位一致 |
| 包 ↔ checkpoint | **A** | `run_config.yaml` 内含 `checkpoint_path` + `checkpoint_sha256`；两份 `run_config.yaml` 指向**同一个** checkpoint SHA |
| checkpoint ↔ 训练运行 | **A** | run-dir 路径唯一，`run_manifest` 记录 `best_checkpoint_sha256` 与之一致 |
| **包 ↔ 平台分数** | **B** | `提交清单.json` 与 `45_46_prep.json` 记录 45.zip→0.680077、46.zip→0.683137；**无平台文件名/截图级绑定** |
| `70`/`71`（若存在） | — | 未在本次取证范围内发现独立记录 |

**判决：B 级。不写成 A。**
（这与 Full6996 那批的 C 级不同——那批 69997 的三个分数在原始回填中无标签，只能靠顺序推断；本批 45/46 在本地记录里是**逐文件名**写明的。）

### J.2 两包差异

`official_class_pixel_ratios` 不同（standard vs calib 的类别像素占比分布不同），确认两者确为**不同偏置下的两份输出**，非同一文件的副本。

---

## K. 与直接基线做 diff

Stage65 的 `_base_` 链：`stage65 → stage56 → stage23 → stage18 → stage14 → stage06 → stage05 → _base_b0_4090 → full/stage03`。

**直接基线 = stage56**（同为「5597 真 + 500 伪标签」的 round-1）。
两者 `train_split` / `dataset_root` / `train_split_sha256` **完全相同**：

```
stage56 run_manifest: train_split_sha256 = 0fbcc0fca84b68a88bb978996a3d6312c39bb7968276d8e8762695c1831efccc
stage65 run_manifest: train_split_sha256 = 0fbcc0fca84b68a88bb978996a3d6312c39bb7968276d8e8762695c1831efccc
```

### K.1 变更项表

| 变更项 | 预期行为 | 实际代码行为 | 证据一致 | 风险 |
|---|---|---|---|---|
| **pseudo mask 内容** | round-1 单教师 stage23 → round-2 三教师全票融合 | symlink 7000-7499 于 9/8 08:10 重指 `pseudo_labels_v2_multi`；SHA 与时间戳可证 | 是 | 低 |
| **pseudo mask 编码** | 均应为 official（0=ignore） | round-2 = official ✓；round-1 源为 train-space，但存在 `masks_official` 且 500/500 转换正确 | 部分 | 见 K.2 |
| max_steps | 56000 → 84000（60 轮） | `max_steps: 84000`，`steps_per_epoch: 1400` | 是 | 低 |
| steps_per_epoch | 保持 1400 | 保持 1400 | 是 | 低 |
| cwfa.start_step | 保持 28001 | 保持 28001（日志证实 28001 激活） | 是 | 低 |
| loss / optimizer / scheduler | 保持 CE / AdamW 6e-5 / poly | 全部保持，LR 逐步吻合 | 是 | 低 |
| transforms / rare_crop | 保持 | 保持（hflip/vflip/rot90 0.5，color_jitter 0.5，domain_aug off，rare_crop 0.5） | 是 | 低 |
| initialization | 均 published_pretrained 冷启 | stage65 = published_pretrained，scheduler 从 0 | 是 | 低 |
| threshold / 融合 | 新增三教师全票一致 | 代码 + 重算 500/500 证实 | **是** | 低 |
| seed | 42 | 42 | 是 | 低 |

### K.2 ⚠️ Round-1 编码链存在 UNBOUND 环节

`prepare_pseudo_train.sh`（mtime 2026-09-05 08:03:12）第 2 段明确链接的是：

```bash
PL=/home/jhy/pseudo_labels_conf09/masks_region_t0p90     # ← train-space（含 255）
ln -sf "$PL/test1_${i}.png" "$DST/train/masks/${newid}.png"
```

而 **train-space mask 含 255**，`official_to_train` 遇到 255 会 `raise ValueError` —— stage56 却成功跑完。

同目录下的 `masks_official/`（**转换正确，500/500 逐位验证通过**）mtime = **2026-09-05 08:07:09**，
**晚于** `prepare_pseudo_train.sh` 的 08:03:12。

→ **在 08:03 与 08:07 之间必然发生过一次重链（把 symlink 改指 `masks_official`），但该步骤没有任何脚本或日志留存。**

**判决：UNBOUND**。round-1 实际使用的 mask 编码**无法从现存材料证明**，只能从其成功运行**反推**为 official。
（此项影响 stage56 的 provenance，**不影响 Stage65**——Stage65 的 mask 已被我直接全量扫描确认为 official。）

---

## M. 补充取证（2026-09-16 第二轮，只读）

### M.1 Matched control 搜索结果

**搜索方法**：扫描 bundle 下**全部** `run_manifest.json`，按 GPT 给定的逐项条件筛选
（同一 ADE20K 冷启 / 84000 步 / batch+accum / LR+scheduler / augmentation+CWFA / validation），
**不以实验名称推断**。

**全部 max_steps ≥ 56000 的运行分布：**

| max_steps | 运行数 | 代表 |
|---:|---:|---|
| 56000 | 28 | stage18/23/31/33/34/39–50/55b/57/58/59/61/63/64/66 |
| 67960 | 2 | stage51, stage52 |
| 70000 | 1 | stage62 |
| **84000** | **1** | **stage65（唯一）** |

**结论：**

> ### NO_MATCHED_CONTROL_FOUND

不存在任何与 Stage65 使用相同 84000 步预算、且其余条件一致、仅去掉 round2 pseudo 的实验。
**84000 步这个预算在全部历史实验中只出现过一次，就是 Stage65 自己。**

**最接近的候选与逐项差异**（均**不能**充当 matched control）：

| 实验 | max_steps | init | seed | cwfa_start | train_split | 是否可作对照 |
|---|---:|---|---:|---:|---|---|
| **stage65** | **84000** | published_pretrained | 42 | 28001 | train_ids_5597pseudo | —— |
| stage56 | 56000 | published_pretrained | 42 | 28001 | train_ids_5597pseudo | ✗ 步数 56000≠84000；pseudo 为 round-1 |
| stage61 | 56000 | published_pretrained | 42 | 28001 | train_ids_5597pseudo | ✗ 步数不符；且加了 MaxEnt |
| stage23 | 56000 | published_pretrained | 42 | 28001 | train.txt（无 pseudo） | ✗ 步数不符、数据不符 |

→ **stage23 是「stage56 的对照」，不是「stage65 的对照」。**
按 GPT 的指示，不得以 Stage23/Stage62 充当 Stage65 的 matched control。

### M.2 结论 4 处实验设计混杂（GPT 第二轮指出 + 本次新发现）

**(1) 这是 transductive self-training，不是普通 pseudo-labeling —— 且合规性未在本轮复核**

500 张 pseudo 的底图**就是 test1 本身**（`images/7000.png -> …/images/test1_1.png`）。
因此 Stage65 实际做的是：

```
用教师给 test1 的 500 张生成伪标签 → 把同一批 test1 加回训练 → 再在同一批 test1 上线上评分
```

**算法后果（确定性事实）：**
- 本地 1399 val 与这批 test1 **完全不相交**（F.1 已验证 overlap=0）
- → **pseudo 对本地 val 的影响与对 test1 的影响不是同一个量**
- → "本地 0.81297 历史最高、线上反而更低"**不能**直接说明 pseudo 思路失败

**合规性**：`gen_pseudo_labels.py` 头部自述 "Official-compliant: allowed to label official test images with own model and add them to training"。
**但本轮未重新核对比赛规则原文**是否允许用 test1 无标签图做自训练。
→ 判决：**UNKNOWN**（不在本审计范围内作合规判断，但必须标记为"未经复核的假设"，不得默认为等价于普通 pseudo-labeling）。

**(2) cold-start 混淆 —— 本次审计中最大的设计混杂项**

Stage65 是 `published_pretrained`（ADE20K）冷启，**不是**从 s62/Stage23 续训。
→ 它检验的命题是"**从 ADE20K 重新训练一个 B3，数据中混入 ~8% round2 pseudo**"，
而**不是**"把 pseudo 信息注入当前最强单模型"。
→ 若关心的是后者，**Stage65 没有回答它**。

**(3) 有效伪监督占比远低于样本占比 8.20%**

样本级：500 / 6097 = 8.20%。
逐层折损后（GPT 的粗算，非严格训练权重）：

```
481 / 6097 × 0.7139 ≈ 5.63%
```

（19 张全 ignore → 481 张有监督；保留像素率 71.39%）
crop 采样与类别构成还会进一步改变有效比例。
→ **本次实际有效的伪监督很可能只有约几个百分点量级**，检验力度偏弱。

**(4) 新发现：CWFA 相对激活位置被步数变化改变**

```
stage65 config 注释: "CWFA 保持28001起(继承), 60轮内仍只后段激活"
实际:               cwfa.start_step = 28001（绝对步，未随预算重标定）

相对位置:
  stage23 基线 : 28001 / 56000 = 50.00%
  stage65      : 28001 / 84000 = 33.33%
```

→ 配置注释所称的"后段"与实际相对位置不符；CWFA 覆盖比例由 **50% 变为 66.7%**。
→ 与 Full6996 形成鲜明对比：**Full6996 明确做了进度保持重标定（28001→35000, 50.000%→50.002%），Stage65 没有。**
→ 判决：**VERIFIED**（事实确认，非错误断言；是否为设计意图未知）

### M.3 对结论强度的修订

GPT 将「pseudo round2 方案无效」的证据强度由 medium 下调为 **medium-low**。
本审计接受该修订，理由为上述 4 项混杂：Stage65 检验的是一个
**"ADE20K 冷启 + 5597 GT + 500 test1 pseudo + 8.2% 样本占比 + 更低的有效像素监督 + conf0.90 三教师一致 + 84k 步"** 的非常具体的 recipe。

它**不能**否定：从 s62/Stage23 续训、更高 pseudo 权重、更合理筛选、curriculum 式后半程加入、更强 teacher、confidence calibration、其他 pseudo recipe。

### M.4 对初版两处表述的修订

| 初版表述 | 修订后 |
|---|---|
| "本地 0.81297 是历史最高——**训练确实更强了，线上反而更低。这个背离不是执行错误造成的，是真实的。**" | "**本地验证表现和按现有 B 级绑定重建的线上表现呈背离；当前未发现可解释该背离的执行错误。**"（平台绑定仅 B 级，不足以说"平台级证实"） |
| "19 张全 ignore **可能显著降低效果**" | **降级**。19/500 占伪标签 3.8%，占全数据集 0.31%。是低效率与采样浪费，**单独很难成为 Stage65 不够强的主要原因**。真正重要的是整体伪监督稀疏 + recipe 设计。 |

---

## L. 最终判决（只允许四种）

### L.1 逐项

| # | 项 | 判决 |
|---|---|---|
| 1 | 三教师 checkpoint 存在性与 SHA | VERIFIED |
| 2 | pseudo 生成过程可复现（融合重算 500/500） | VERIFIED |
| 3 | pseudo mask 编码空间（official 0..8） | VERIFIED |
| 4 | `official_to_train` 0→255 映射 | VERIFIED |
| 5 | image↔mask 对齐（0 错配） | VERIFIED |
| 6 | confidence filtering 实际执行 | VERIFIED |
| 7 | 训练数据 6097 = 5597 + 500，无重复 | VERIFIED |
| 8 | 与 val 无重叠 | VERIFIED |
| 9 | 采样比例 8.20%（均匀 shuffle） | VERIFIED |
| 10 | 遍历完整性（差异 55 已算术闭合 = epoch 数 / 残批数） | VERIFIED |
| 10b | matched control 是否存在 | **NO_MATCHED_CONTROL_FOUND** |
| 10c | CWFA 相对激活位置由 50% 变为 33.3% | VERIFIED |
| 10d | test1 自训练合规性复核 | **UNKNOWN**（本轮未核规则原文） |
| 11 | augmentation 几何一致性 | VERIFIED |
| 12 | LR / scheduler（逐步 ratio 1.000000） | VERIFIED |
| 13 | CWFA 门控（28001 精确激活） | VERIFIED |
| 14 | checkpoint 选择（best step 81200） | VERIFIED |
| 15 | package ↔ checkpoint 绑定 | VERIFIED |
| 16 | package ↔ 平台分数绑定 | **B 级** |
| 17 | round-1（stage56）mask 编码链 | **UNBOUND** |
| 18 | round-1 teacher 身份（`pseudo_stats.json` 记 stage23，mtime 9/4；`build_pseudo_dataset.sh` 用 conf09） | **UNBOUND** |
| 19 | `gen_pseudo_labels.py` docstring 写 "mean" 而代码是 "≥80% pixels" | VERIFIED（漂移确认） |
| 20 | 19 张零共识图为全 ignore、训练中零梯度 | VERIFIED（事实） |

### L.2 四个必答问题

**Q1：是否发现足以让 Stage65 负结果失效的实现错误？**

**未发现。**
pseudo 生成可 500/500 复现；编码、对齐、映射、采样、增强、LR、CWFA、checkpoint 选择全部 VERIFIED。
在**可恢复证据范围内**未发现能使该负结果失效的实现错误。（不为"实现一定正确"背书。）

**Q2：是否发现可能显著降低效果、但影响量未知的问题？**

**发现 2 项，影响量均未知：**

1. **19/500 张伪标签图为全 ignore**（E.4）。这 19 张计入 6097 但贡献零梯度，
   即实际有监督的伪标签图为 **481 张**，噪声占比相应上升。是否为缺陷取决于设计意图，**影响量无法从现有材料估计**。
2. **`region_keep_mask` docstring 与实现不一致**（E.2）。实际生效的是代码（≥80% 像素），
   若下游有人按 docstring 的 "mean confidence" 解读 keep 行为，会误判过滤强度。**对本次训练无直接影响，对后续复现有影响**。

此外，**伪标签占比仅 8.20%** 是一个可被注意的量级事实（VERIFIED），但它属于**方案设计参数**，不属于实现缺陷。

**Q3：是否存在 teacher / pseudo / training / checkpoint / submission 任一处身份绑定不完整？**

**存在 3 处，均不影响 Stage65 主链：**

1. **round-1（stage56）mask symlink 重链无存档脚本**（K.2）—— 影响 stage56 provenance。
2. **round-1 teacher 身份未绑定**—— `pseudo_stats.json` 记 stage23，但该文件属 conf09 目录（mtime 9/4），
   与 stage56 实际使用的 mask 之间无直接绑定记录。
3. **package ↔ 平台分数为 B 级**（J.1）—— 无平台文件名级绑定。

Stage65 自身链路（teacher→pseudo→mask→training→checkpoint→package）**逐环节已绑定**。

**Q4：现有 Stage65 负结果对「pseudo round2 方案本身无效」的证据强度？**

**medium-low**（GPT 第二轮裁定，本审计接受；初版给 medium，已下调）

理由与限制：

- **支持"实现干净"的部分**：Q1 未发现执行错误；pseudo 融合可 500/500 复现；
  编码/对齐/映射/采样/增强/LR/CWFA/checkpoint 选择全部 VERIFIED。
- **下调的主要理由不是发现实现错误，而是实验设计混杂与检验力度不足**（详见 M.2）：
  1. **transductive**：pseudo 底图就是 test1，本地 val 与 test1 不相交，两者表现不是同一个量
  2. **cold-start 混淆**：ADE20K 冷启重训，未回答"把 pseudo 注入最强单模型"这一命题——**最大混杂项**
  3. **有效伪监督稀疏**：样本级 8.20%，折损后粗算 ≈5.63%，检验力度弱
  4. **CWFA 相对位置漂移**：50% → 33.3%，未做进度重标定
  5. **NO_MATCHED_CONTROL_FOUND**：84000 步预算在全部历史中仅此一次，无同预算对照
- **限制**：package↔分数仅 B 级绑定。

支持的只是"**这一条具体 recipe 未显示收益**"，**不能外推为"pseudo-label 方法一般无效"**。
（与 Full6996 的边界处理一致。）

---

## 附：未能解决的开放项

1. round-1 mask 重链脚本缺失（UNBOUND）
2. round-1 teacher 身份未绑定（UNBOUND）
3. 19 张零共识图的影响量无法估计
4. `0.680077 / 0.683137` 无平台文件名级绑定
5. 若日后需要，`/home/jhy/p1_t{1,2,3}/` 三教师中间产物**仍在盘上**，可用于进一步复验

---

*本取证严格只读。未启动 GPU、未重训、未重生成 pseudo、未修改或删除任何历史文件。*
