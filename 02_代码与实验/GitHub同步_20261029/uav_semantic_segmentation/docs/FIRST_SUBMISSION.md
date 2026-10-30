# 第一次初赛探路提交记录

## 提交性质

第一次提交是探路提交，用于验证：

- 模型推理链路
- 训练 ID 到官方 ID 的还原
- 单通道 PNG 格式
- 测试文件命名
- 扁平 ZIP 结构
- 平台评测链路

它不是最终竞争模型。当前只确认本地提交包生成与严格格式检查通过；实际上传时间和平台分数应由队长在获得结果后补充。

## 使用模型

| 项目 | 内容 |
|---|---|
| 实验 | step06_domain_aug |
| 模型 | SegFormer-B0 |
| 权重记录 | 对应实验的 best_miou.pt，仅保存在被忽略的本地 outputs 中 |
| 训练性质 | 100-step quick pilot |
| 本地验证 mIoU | 0.256315 |
| TTA | 未使用 |
| 模型融合 | 未使用 |

文档不记录本机绝对 checkpoint 路径，Git 也不提交权重文件。

## 测试推理

- 测试图片：500 张
- 成功预测：500 张
- 失败：0 张
- 输出 PNG：500 张
- 尺寸：全部 1024×1024
- 模式：全部 L
- dtype：全部 uint8
- Palette：0 张
- 实际像素 ID：1–8
- 测试标签：未使用
- 随机增强：未使用

## ZIP 检查

- ZIP 文件数量：500
- 外层目录：无
- 嵌套路径：无
- 文件名集合：与测试图完全一致
- ZipFile.testzip()：None
- ZIP 大小：7,785,878 bytes，约 7.43 MiB
- SHA256：7f92851b2e398482a410ba0e54d466d6c36e316b5f196a0ea5654b6c4aebbffd
- ZIP 内随机抽检：10 张，均为 1024×1024、L、uint8、无 Palette

ZIP、500 张预测和可视化均位于被 Git 忽略的 outputs/ 中，不得提交仓库。

## 预测分布

| 官方类别 | 像素比例 |
|---|---:|
| Background | 52.449844% |
| Building | 15.959855% |
| Road | 3.247460% |
| Water | 1.044244% |
| Barren | 0.015313% |
| Vegetation | 26.563492% |
| Agricultural | 0.712472% |
| Vehicle | 0.007319% |

## 结论与限制

- 没有出现全局单类塌缩，主导类别占比未超过 90%。
- Background 明显偏高。
- Barren 和 Vehicle 预测比例极低。
- 该分布符合 quick 模型未充分收敛的特征。
- 本次模型不应作为最终竞争模型，也不能替代 Stage 03 正式基线。

本记录来源于本地 run_config.yaml、prediction_statistics.csv、submission_check.json 和 ZIP 二次复检结果；这些原始运行产物继续保留在本地忽略目录中。
