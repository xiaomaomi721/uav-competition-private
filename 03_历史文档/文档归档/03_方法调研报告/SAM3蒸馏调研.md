# SAM3 蒸馏调研（关键依据）

> 用途：SAM3 当教师做蒸馏的可行性调研结论，作为蒸馏方案的决策依据。
> 更新：2026-08-29

---

## 一、SAM3 是什么（官方 README 实证）

**全称**：SAM 3: Segment Anything with Concepts（Meta Superintelligence Labs）
**论文**：arxiv 2511.16719，官网 https://github.com/facebookresearch/sam3
**许可证**：SAM License（非商业友好，需留意比赛合规性——赛题允许学术公开权重，SAM License 属于学术公开，可用）

### 架构（关键，决定蒸馏怎么做）
- **848M 参数**，由 **detector + tracker 组成，共享一个 vision encoder**。
- detector 是 **DETR-based**，条件于 **文本(text)、几何(geometry)、图像 exemplar** 提示。
- tracker 继承 SAM 2 的 transformer encoder-decoder。
- **它不是传统语义分割模型**：不直接输出"8 类每像素 logits"。它是 **promptable concept segmentation（可提示的概念分割）**：
  - 输入一个文本短语（如 "农田"），输出该概念的所有实例 mask + box + score。
  - 是 **open-vocabulary 开放词表**，能分割任意文本概念（270K 概念，SA-CO 基准）。

### 对蒸馏的关键含义
- SAM3 的 **vision encoder 能输出稠密图像特征（dense image embeddings）**，这是可蒸馏的"教师特征"。
- 但 SAM3 **没有我们 8 类的语义 logits**，不能直接当 soft-label 教师。
- → 蒸馏只可能是 **特征蒸馏（feature KD）**：用 SAM3 图像编码器的特征，蒸馏到 SegFormer 学生的 encoder 特征上。

---

## 二、下载障碍（实测本地网络）

| 目标 | 状态 |
|---|---|
| github.com 主站 | ❌ 不通（被墙/限制） |
| huggingface.co | ❌ 不通 |
| **raw.githubusercontent.com** | ✅ 通（已抓到 SAM3 README） |
| **hf-mirror.com**（HF 国内镜像） | ✅ 通（200 OK） |
| modelscope.cn | ✅ 通 |

**结论**：SAM3 权重在 HuggingFace（`facebook/sam3` 和 `facebook/sam3.1`），本地直连 HF 不通，但 **hf-mirror.com 镜像通**，可走镜像下载。

### 权重情况
- SAM 3.1 是 2026-03-27 最新发布（Object Multiplex）。
- 权重在 HF：`facebook/sam3.1`（约 3.5GB 核心权重，另有配套 tokenizer/configs）。
- **⚠️ gated repo（需申请）**：README 明确写"Before using SAM 3, please request access to the checkpoints on the SAM 3 Hugging Face repo. Once accepted, you need to be authenticated"。
- 即：需要 HF 账号申请 access + `hf auth login` 认证，才能下载。这是**第一道坎**。

---

## 三、服务器环境（决定性障碍）

| 项 | 服务器现状 | SAM3 要求 | 是否满足 |
|---|---|---|---|
| Python | **3.10.12** | **3.12+** | ❌ **不满足** |
| PyTorch | 2.13.0+cu130 | 2.7+ | ✅ 够新 |
| CUDA | 13.0 | 12.6+ | ✅ |
| 磁盘 | 356G 剩余（90%满） | 权重 3.5G | ✅ 够 |
| 内存 | 251G | — | ✅ |

**关键结论**：服务器 Python 3.10 < SAM3 要求的 3.12，装不了 SAM3 的官方包。要跑 SAM3 需要：
- ① 升级服务器 Python 到 3.12（有风险，可能影响现有 uavseg 环境）；或
- ② 用独立 conda 环境装 Python 3.12（推荐，不影响现有环境）。

---

## 四、蒸馏可行性初步判断

### 方案（特征蒸馏）
1. 下载 SAM3 权重（走 hf-mirror，需先过 gated 认证）。
2. 服务器建独立 conda 环境（Python 3.12 + PyTorch 2.7+ cu128）。
3. 用 SAM3 vision encoder 提取 6996 张训练图的稠密特征（一次性离线，存盘）。
4. 训练 SegFormer 学生时，加一个特征蒸馏损失（学生 encoder 特征对齐 SAM3 特征）。

### 待解决的技术问题（做之前必须想清）
1. **维度对齐**：SAM3 848M 的 encoder 特征维度 vs SegFormer B3/B4/B5 encoder 特征维度不同，需要投影层。
2. **特征语义对齐**：SAM3 是 open-vocab concept 特征，SegFormer 是 8 类语义特征，对齐可能需要 adapter。
3. **显存**：SAM3 848M 参数，推理 1024×1024 图，单张 4090 24G 能否跑？需测试（可能要用 fp16 或降分辨率）。
4. **合规确认**：SAM License 是否满足"学术公开权重"要求，建议赛题群再确认一次（虽然官方参考资源明确列了 SAM3，应无问题）。

### 优先级
- 蒸馏是**合规的大分方向**（用户最推荐，官方允许），但**前置障碍多**（Python 3.12、gated 下载、特征蒸馏实现）。
- 建议：**先并行解决下载 + 环境**，同时读 SAM3 论文搞清 vision encoder 结构，再定蒸馏代码。
