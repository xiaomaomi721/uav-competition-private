# GPU3 Formal Experiment
## Barren/Background Targeted Region Contrast v2

本实验与GPU2正在运行的：

`SEMEDA → Delayed CWFA 56000`

完全独立。

不要停止、修改、读取写入GPU2正在运行的实验目录。

本实验只允许：

`CUDA_VISIBLE_DEVICES=3`

---

# 1. 实验问题

历史Exp58已经验证过Region Contrast机制。

历史pair：

`[[6,0],[1,0]]`

即：

- Agricultural ↔ Background
- Building ↔ Background

当前Stage1–4诊断显示：

最高价值混淆pair已经明确为：

`Barren ↔ Background`

第二优先：

`Agricultural ↔ Background`

因此本实验只改变contrast pair：

`[[4,0],[6,0]]`

内部train id：

- Background = 0
- Building = 1
- Road = 2
- Water = 3
- Barren = 4
- Vegetation = 5
- Agricultural = 6
- Vehicle = 7

---

# 2. 实验基准

必须找到并审计历史：

`Exp58 Region Contrast mode C`

真实config、代码和结果。

必须确认历史：

`pairs = [[6,0],[1,0]]`

以及真实：

- model
- crop size
- CE
- RareCrop
- CWFA状态与参数
- contrast loss weight
- temperature
- feature layer
- sampling strategy
- start step（如有）
- optimizer
- scheduler
- seed
- max_steps

禁止根据记忆重写Exp58。

以服务器真实Exp58实现为准。

---

# 3. 新实验唯一算法变量

复制Exp58正式配置。

保持所有其他变量完全一致。

只修改：

`contrast_pairs`

从：

`[[6,0],[1,0]]`

改为：

`[[4,0],[6,0]]`

禁止同时修改：

- contrast weight
- temperature
- sample count
- positive/negative sampling
- feature level
- normalization
- optimizer
- LR
- scheduler
- RareCrop
- CWFA
- augmentation
- resolution
- model
- seed

本轮必须是严格pair-replacement experiment。

---

# 4. 特别禁止

不要加入：

- SEMEDA
- DINO distillation
- pseudo label
- MaxEnt
- MixUp
- MixStyle
- Dice/Lovasz
- UPerNet
- 新attention
- 新calibration

GPU2正在测试SEMEDA组合。

GPU3只研究：

`Targeted Region Contrast`

---

# 5. 启动前历史身份审计

输出：

`EXP58_IDENTITY_AUDIT`

至少包括：

历史Exp58：

- experiment ID
- config path
- checkpoint
- online score
- local best
- contrast pairs
- contrast loss全部参数
- base experiment identity

如果无法确认真实Exp58配置：

输出：

`EXP58_IDENTITY_BLOCKED`

停止。

禁止猜配置。

---

# 6. 工程静态门禁

新建独立config和run目录。

不得覆盖Exp58。

比较：

`old Exp58 config`
vs
`new Targeted-RC config`

diff必须证明实验变量差异只有：

`[[6,0],[1,0]] → [[4,0],[6,0]]`

允许：
- 实验名
- 输出路径
- 归档字段

变化。

---

# 7. 20-step smoke

正式训练前仅做工程smoke：

确认：

- GPU3
- train=5597
- val=1399
- B3/真实Exp58 backbone身份
- 1024/真实Exp58 resolution
- contrast loss正常finite
- pair中确实出现4↔0、6↔0
- 不出现历史1↔0 pair
- no NaN / Inf
- backward正常
- optimizer step正常

Smoke结束后删除/隔离smoke状态。

正式训练必须从Exp58相同正式初始化重新开始。

不得resume smoke。

---

# 8. 正式训练

如果以上全部PASS：

自动启动完整正式实验。

训练长度：

沿用Exp58正式长度。

如果Exp58正式为：

`56000 steps`

则本实验同样：

`56000 steps`

不得因为GPU3而缩短。

---

# 9. Validation

完全沿Exp58正式口径。

至少记录：

- mIoU
- 8类IoU
- Precision
- Recall
- confusion matrix

额外重点记录：

### Barren
- IoU
- Recall
- Precision
- GT pixels
- predicted pixels
- Barren→Background
- Background→Barren

### Agricultural
- Agricultural→Background
- Background→Agricultural

### Background
- IoU / Recall / Precision

---

# 10. 必须增加Targeted Pair诊断

每次正式validation至少额外记录：

`Barren→Background pixels`

`Background→Barren pixels`

定义：

`Barren_BG_mutual_confusion =
Barren→Background + Background→Barren`

以及：

`Agricultural_BG_mutual_confusion`

最终必须和历史Exp58、Exp23进行同口径比较。

---

# 11. checkpoint保存

除best/last外，至少保留：

- step44000
- step48000
- step52000
- step56000

原因：

local-best不能代表online-best。

允许纯工程性trainer instrumentation。

不得改变训练计算。

---

# 12. GPU隔离

只使用：

`CUDA_VISIBLE_DEVICES=3`

正式启动前确认GPU3没有他人进程。

如果GPU3出现非本实验未知进程：

不得kill别人。

停止启动并输出：

`GPU3_RESOURCE_BLOCKED`

如果实验已经运行中，禁止自行迁移到GPU2。

GPU2正在执行另一正式实验。

---

# 13. 训练完成后

先做完整1399 source-val验收。

然后对以下至少三个checkpoint：

- local best
- step52000
- step56000

执行：

## 500 Test raw audit

single-view
no TTA
no calibration

输出：

- 8类predicted shares
- Barren share
- Agricultural share
- Background share
- 与Exp23 champion core disagreement

---

# 14. TTA/8D

允许生成：

`TTA8`

和：

`TTA8 + frozen champion 8D`

冻结向量：

`[-0.2,0.1,0.0709,0.7,3.6,0.4,3.3215,-0.5]`

但：

禁止自动线上提交。

禁止重新scan bias。

等待ChatGPT选择checkpoint。

---

# 15. 最终需要比较的历史锚点

至少列出：

Exp23 CWFA raw online:
`67.371405`

Exp58 historical Region Contrast:
`67.6376`

Champion final:
`69.281290`

注意：

69.281290才是当前最终冠军。

---

# 16. 成功标准

本轮不能只看local mIoU。

我们最关心：

### S1
Barren→Background显著减少。

### S2
Background→Barren没有等量或更大爆增。

### S3
Barren IoU提高来自更好的semantic separation，
而不是简单扩大Barren预测面积。

### S4
Agricultural↔Background不恶化。

### S5
最终Target线上能否超过历史Exp58，并逼近/超过69.281290。

---

# 17. 最终输出

训练完成后返回：

`TARGETED_REGION_CONTRAST_V2_56000_PASS`

以及：

- experiment ID
- GPU3
- Exp58历史身份
- 配置diff
- pair确认
- local best step/mIoU
- step44000/48000/52000/56000 mIoU
- Barren IoU/Recall/Precision
- Barren→Background
- Background→Barren
- mutual confusion
- Agricultural对应指标
- raw Test shares
- candidate checkpoint paths
- SHA256
- TTA8 package生成状态
- frozen8D package生成状态

不要自动提交。