# PMT-L / DINOv3 独立模型实验（2026-10-26）

> 2026-10-27 状态更新：已完成2000步后因同卡其他进程触发保护退出，未恢复训练。补评source1399：最终无mask协议66.044216；保留检查点mask状态68.531338。具体诊断、逐类结果和保留意见见工作区“队友方案审核交接_20261027/PMT实施记录/当前状态与评估.md”。下文启动记录为历史状态。

目标是验证新的表征与分割架构能否补足 H75；目前不能据此承诺线上 72 分。正式训练状态以服务器 status.json 和 formal.log 为准。

已完成：两个权重哈希验证、全部 6996 张源域数据完整性检查、转换特征数值对照、真实数据试跑。256/512 分辨率四层特征的最大相对 RMSE 为 9.02e-7；真实训练峰值分配显存 12984.74 MiB（约 12.68 GiB），nvidia-smi 进程占用约 14 GiB。正式训练已在 2026-10-26 晚间启动，独立于 smoke 重新初始化。尚无完整验证及线上成绩。实测文件见 results/。

## 资源与目录

- 旧服务器 teacher-4090，独占使用 GPU0，UUID GPU-bc304d79-c807-968a-eaad-d2c2ecc5a984。检测到同卡其他计算进程时，只退出本实验。
- 实验目录：/home/jhy/audits/codex_pmt_run_20261026。
- Python：/home/jhy/.venvs/uavseg/bin/python；timm 1.0.24 来自隔离 vendor 目录。
- 不修改队友模型及训练目录，不占用新服务器 GPU。

## 权重来源与核验

1. timm/vit_large_patch16_dinov3.lvd1689m，版本 30c1109559f65dea34316b0d4842d35c5771fe11，model.safetensors，1,212,347,640 字节。SHA256：45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941。
2. tue-mps/ade20k_semantic_pmt_large_512_dinov3，版本 506d077f4bab814567f2573790537071cf458e7c，pytorch_model.bin，215,489,811 字节。SHA256：5039b0090a15a836c6217fc6d3ebced96fc217b1fd6908f1fbb4a274fce92566。

均为公开发布的模型，通过 hf-mirror.com 下载固定版本；下载后和服务器加载时分别核对 SHA256。timm 参数逐项转换到 Transformers DINOv3，所有源键必须被消费；缺失的零 Q/V bias 和不用于推理的 mask token 显式填零。在 256 和 512 分辨率对比四层特征，误差超限就不训练。此测试验证两份实现的数值对应，不能替代模型准确率评估。

PMT 代码来自 https://github.com/tue-mps/pmt ，保留 MIT 许可证与代码署名；DINOv3 许可证另存。只使用预训练参数，不加入外部训练图片。

## 正式配置

- 冻结 DINOv3-L，使用第 6/12/18/24 层归一化特征；保持主干 eval，避免随机位置编码增强。
- 加载 PMT-L 解码器与四层 lateral BN；150 类分类头重置为 8 类加 no-object。
- 官方源域 train 5597 / val 1399；不使用验证标签训练。本轮不使用目标域伪标签。
- 全图 1024，D4 几何增强，BF16，batch 1，累积 4，8000 次优化器更新。
- AdamW：预训练解码器与 BN 学习率 2e-5，新分类头 2e-4，weight decay 0.05；400 次 warmup，之后 cosine；梯度范数上限 1。
- 分类 CE / mask BCE / Dice 系数 2/5/5；no-object 权重 0.1；每层匹配及损失采样 12544 点。7 层输出均监督。
- 标注 0 为忽略，1—8 映射到模型 0—7。匹配与 mask loss 的采样均排除忽略区；不把忽略区当作负样本。
- mask annealing 起止比例来自作者配置，按最长结束步 16826 归一化到 8000 次更新，幂次 0.9；验证统一关闭 attention mask，保证确定的推理协议。这是本项目适配，并非作者原始训练配置的完整复现。
- 每 1000 步保存断点，另存 100 步检查点；2000/4000/8000 步验证全部 1399 张，并报告 605 张未发现训练重叠样本的结果。后者也不能当成目标域成绩。
- 推理：原图分辨率、单视角；先上采样 mask logits，再 sigmoid 和类别 softmax 融合；不沿用 H75 Bstar。

## 启动和复现

launch.py 顺序执行：主干特征对应测试 → 16 张真实图 / 4 次更新的 smoke → 重新初始化后的正式训练。任一阶段失败即停止；smoke 权重不会作为正式初始化。

状态文件 status.json；阶段日志 conversion.log / smoke.log / formal.log。formal/config.json 保存配置，formal/train.jsonl 保存损失、耗时、显存与梯度范数。源域完整性结果 source_integrity.json。

断点保存可训练参数、优化器、随机状态及更新数；冻结主干仍需上述单独的 SHA 验证文件。恢复使用 train.py --mode train --resume <本实验检查点>。部署时需将主干和解码器一起提供，它们构成一个模型。

训练损失下降及源域验证改善均不等于线上提分。本轮尚无提交结果，需与半决赛基线按同一推理协议评估后再决定提交。
