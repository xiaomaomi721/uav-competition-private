# 迁移规格：B3 + SEMEDA → TTA8 → Stage23 bias → mask

> 2026-09-15 服务器实测核实。管线四段：**模型(exp47) → TTA8 → bias(来自Stage23) → 输出mask**。

---

## 零、先澄清三个容易搞混的点

| 容易搞混 | 事实 |
|---|---|
| "SEMEDA 模型"有额外参数吗 | **没有**。SEMEDA 是**训练期 loss**，不加参数。exp47 权重 = 纯 SegFormer-B3，644 个参数，与 Stage23 的 CWFA 权重**键集完全相同**。源码注释原话："Inference is unchanged: the encoder belongs to the criterion, not the model." |
| "Stage23 的 bias"是哪个 | **冠军向量** = 08 号拿 0.692812 用的那个，硬编码在代码里，**服务器上不存在任何 .npy 文件** |
| SEMEDA 的 `semeda_edge_s01.pt` 推理要吗 | **不要**。它是训练期 criterion 的一部分。只有**重训**才需要 |

---

## 一、管线四段的确切定义（逐段核实）

### 段1：B3 + SEMEDA — 模型

```
$B/stage47_runs/stage47_w1_exp47_b3_semeda_formal_seed42/best_miou.pt
```
`$B = /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1`

- 本地验证 mIoU **0.8095688**，best_step = 56000
- checkpoint 顶层键：`model_state_dict / step / best_miou / config / optimizer_update_count / skipped_optimizer_steps`
- 训练配置：SegFormer-B3，1024×1024，CE + λ0.5×L1(embedding)，
  `edge_encoder_path: /home/jhy/models/semeda_edge_s01.pt`，`feature_size: 256`，`gt_noise_sigma: 0.1`
- **推理时按普通 SegFormer-B3 加载即可**，不需要 semeda.py

### 段2：TTA8 — 八视图

定义在 `/home/jhy/gpu2_day_queue/tta_anatomy_v2.py`，原文核实：

```python
VIEWS = [(k, hf) for k in range(4) for hf in (False, True)]
# → rot0, rot0_mirror, rot90, rot90_mirror, rot180, rot180_mirror, rot270, rot270_mirror
```

每个视图：**先做几何变换 → 前向 → 逆变换回 canonical → 累积**。
八视图全部回到原方向后取 **float32 均值**，得到 `mean_logits`，形状 `(8, H, W)`。

```python
views = forward_views(model, px, 1)   # → (8, 8, H, W) float32
mean  = views.mean(dim=0, dtype=torch.float32)   # → (8, H, W)
```

**注意**：是 **logits 平均**（不是 softmax 后平均，不是投票）。均值必须用 float32。

### 段3：Stage23 bias — 冠军向量

```python
CHAMPION = [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5]
```

**索引语义（关键）**：`索引 i = 训练类别 i = 官方类别 i+1`

| 索引 | 训练类别名 | 官方类别 | bias |
|---:|---|---|---:|
| 0 | Background | 1 | −0.2 |
| 1 | Building | 2 | 0.1 |
| 2 | Road | 3 | 0.0709 |
| 3 | Water | 4 | 0.7 |
| 4 | Barren | 5 | 3.6 |
| 5 | Vegetation | 6 | 0.4 |
| 6 | Agricultural | 7 | 3.3215 |
| 7 | Vehicle | 8 | −0.5 |

加法位置：**在 TTA 均值之后、argmax 之前**。

```python
biased = mean + bias.view(8, 1, 1)
```

**这个向量是全局平移算子**——实测在每个模型上都涨 +1.0~1.5pp，与模型本身好坏无关。
它**不是** exp47 专属的，用在 exp47 上属于跨模型迁移，效果未验证（见 §五）。

### 段4：输出 mask

```python
pred = (mean + bias).argmax(dim=0).add(1).to(torch.uint8)   # 内部0-7 → PNG值1-8
Image.fromarray(pred.numpy(), mode="L").save(path, format="PNG")
```

- PNG 取值 **1–8**，**没有 0**（官方 0 = Ignore，预测不该产出）
- PNG 尺寸 **必须与输入图一致**（1024×1024）
- 500 张测试图，文件名与测试集一致

---

## 二、要迁移的文件

### A. 推理用（必需）

```
$B/src/models.py                  ← 模型定义
$B/src/transforms.py              ← normalize_to_tensor 归一化
$B/src/utils.py                   ← checkpoint 加载
$B/configs/                       ← 整个下（配置有继承链）
$B/stage47_runs/stage47_w1_exp47_b3_semeda_formal_seed42/best_miou.pt    181 MB
$B/stage47_runs/stage47_w1_exp47_b3_semeda_formal_seed42/config.yaml     ← 全展开配置
/home/jhy/models/segformer-b3-ade512/                                     181 MB
/home/jhy/gpu2_day_queue/tta_anatomy_v2.py     ← TTA8 权威实现，照抄别自研
```

### B. 重训才需要（推理不需要）

```
$B/src/semeda.py                 5.4 KB   ← SEMEDA 主模块
$B/src/losses.py                11.6 KB   ← SEMEDA loss 在这里
$B/src/trainer_formal_semeda_cwfa.py  53 KB  ← SEMEDA 训练器
$B/src/dataset.py               11.6 KB
$B/scripts/train_once_no_oom_retry.py    ← 训练入口
$B/splits/                       ★容易漏，没有它数据集切不出来
$B/requirements-segmentation.txt
$B/launch_exp47_formal.sh
/home/jhy/models/semeda_edge_s01.pt   32 KB  ★训练必需，在 bundle 之外，最易漏
/home/jhy/pretrain_semeda_edge.py
```

### C. 不要下

```
✗ last.pt (567 MB)    只续训用
✗ test1 的 500 张图    那是官方测试集
✗ 训练数据集 (13 GB)   /home/jhy/datasets/2026-低空图像语义分割赛道-训练集
✗ *.bak / *.bak2 / *.before_* / __pycache__
```

**总下载量约 365 MB。**

---

## 三、环境版本（Stage23 与 exp47 同快照，实测）

```json
{ "python": "3.10.12",
  "torch": "2.13.0+cu130",  "torch_cuda_build": "13.0",
  "transformers": "5.13.0", "cudnn": 92000,
  "gpu": "RTX 4090 24GB (capability 8.9)" }
```

⚠️ **transformers 必须是 5.x**。本地如果装成 4.x，SegFormer 接口对不上会报错。
⚠️ torch 2.13 只有 cu130 构建，**3060Ti 需要另装匹配的 torch**（30 系是 sm_86，
cu130 官方 wheel 是否含 sm_86 需队友实测；不行就降到 cu121/cu124 的 torch）。

---

## 四、⚠️ 3060Ti 上的三个硬问题

### 1. 显存：1024² + batch2 = 必 OOM

服务器训练配置是 4090 上 `crop_size=1024, batch_size=2, gradient_checkpointing=false`。
**3060Ti 只有 8 GB**，推理虽然比训练省显存，但 **TTA8 需要同时容纳 8 个视图的 logits**：

```
8 视图 × 8 类 × 1024 × 1024 × 4 bytes = 268 MB   ← 仅 logits
```

再加模型激活。建议：**`view_batch_size=1`**（`forward_views` 支持 1/2/4/8），
逐视图算完立刻逆变换并累加，不要 `torch.cat` 八个视图。

### 2. torch 版本：cu130 在 3060Ti 上可能装不上

30 系是 sm_86 架构。**先确认本地 torch 能识别到卡**，再谈跑不跑得动。

### 3. **最重要：exp47 权重 + Stage23 的 bias 是跨模型迁移，没人验证过**

冠军向量是在 **Stage23** 上拟合出来的。我们实测过它在每个模型上都是 +1.0~1.5pp 的平移，
但**那是针对用冠军向量评过分的模型**（Stage23、s62、KD 系列）。

**exp47 从来没被这样评过分。** 它的最优 bias 可能完全不同——
exp47 的本地 per-class IoU 是
`[0.7477, 0.8702, 0.8460, 0.9002, 0.6079, 0.8946, 0.7921, 0.8178]`，
与 Stage23 的 `[0.7500, 0.8719, 0.8411, 0.8994, 0.6113, 0.8975, 0.8013, 0.8154]` 很接近，
**所以冠军向量大概率是个合理起点，但不能假定它是最优的。**

**队友第一版应该：① 用冠军向量出一版；② 同时出一版零 bias 做对照。**
两版对比才知道 bias 迁移过来是帮忙还是帮倒忙。

---

## 五、直接可用的推理核心代码（已按核实过的契约写）

```python
import numpy as np, torch
from PIL import Image
from pathlib import Path

NUM_CLASSES = 8
CHAMPION = [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5]

def predict_one(model, img_path, device, bias_vector=CHAMPION, view_batch=1):
    img = np.asarray(Image.open(img_path).convert("RGB")).copy()
    px = normalize_to_tensor(img).unsqueeze(0).to(device)        # 1x3xHxW

    with torch.inference_mode():
        views = forward_views(model, px, view_batch)             # 8x8xHxW float32
        mean = views.mean(dim=0, dtype=torch.float32)            # 8xHxW

        bias = torch.tensor(bias_vector, dtype=torch.float32, device=device)
        pred = (mean + bias.view(NUM_CLASSES, 1, 1)) \
                 .argmax(dim=0).add(1).to(torch.uint8)           # 内部0-7 → 1-8

    return pred.cpu().numpy()
```

`forward_views` 从 `tta_anatomy_v2.py` 导入，`normalize_to_tensor` 从 `src.transforms` 导入。
**不要自己重写这两处**——TTA 的逆变换顺序和归一化参数错了，结果会静默偏差。

---

## 六、打包命令（流式，不占服务器磁盘）

服务器 `/dev/sda4` 只剩 **4.6 GB**，且是共享的。**不要在服务器上生成 tar**：

```bash
B=/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1
R=/home/jhy
S=$B/stage47_runs/stage47_w1_exp47_b3_semeda_formal_seed42

ssh teacher-4090 "tar -czf - \
  --exclude='*.bak' --exclude='*.bak2' --exclude='*.before_*' --exclude='*.bak_*' \
  --exclude='__pycache__' --exclude='*.pyc' \
  $B/src $B/configs $B/scripts $B/splits \
  $B/requirements-segmentation.txt $B/launch_exp47_formal.sh $B/launch_exp47_smoke.sh \
  $S/best_miou.pt $S/config.yaml $S/run_manifest.json $S/metrics.json \
  $R/models/semeda_edge_s01.pt $R/models/semeda_edge_s01.json \
  $R/models/segformer-b3-ade512 \
  $R/pretrain_semeda_edge.py \
  $R/gpu2_day_queue/tta_anatomy_v2.py" > Exp47_transfer.tar.gz
```

---

## 七、三个问题的答复

**1. 代码是否包含 SEMEDA 模块？**
✅ 包含，在 `src/semeda.py` + `src/losses.py` + `src/trainer_formal_semeda_cwfa.py` 三个文件，
另需 bundle 外的 `/home/jhy/models/semeda_edge_s01.pt`（32 KB，**最易漏**）。
⚠️ 但注意：**`src/` 下同时有 `trainer.py`（冻结版）和 `trainer_formal_semeda_cwfa.py`（SEMEDA 版），
别拿错**。另 `/home/jhy/losses_exp47.py` 与 bundle 内 `src/losses.py` md5 不同，是两份文件，以 bundle 内为准。
**推理不需要 SEMEDA 任何代码。**

**2. 最佳 checkpoint？**
✅ `$S/best_miou.pt`（181 MB），本地验证 mIoU 0.8095688，best_step 56000。
这是 **B3+SEMEDA** 的权重，是这条管线的段1。

**3. bias 校准文件？**
❌ 服务器全盘无任何 `bias*.npy` / `class_bias*` / `calibration*.json`（零命中）。
校准是推理时实时算的，冠军向量硬编码为
`[-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5]`。
要复现校准逻辑需要 `/home/jhy/gpu2_day_queue/constrained_calibration_priority_v2.py`。

---

## 八、必须转告队友的一句话

**本地验证 mIoU 不能预测线上分。** 已实测：本地 0.8071–0.8100 的四个 checkpoint，
线上排名完全不同。队友在 3060Ti 上复现出 0.8096，只说明**训练管线对了**，
**不能据此判断这个方案好坏**——线上表现必须另测。

同样，**exp47 的线上表现我们目前完全没有数据**（它从没被提交过）。
这条管线跑出来的 mask 交上去是什么分，是未知数。
