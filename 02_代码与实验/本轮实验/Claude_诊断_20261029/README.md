# Claude 诊断与候选 2026-10-29

服务器工作目录：teacher-4090 `~/aic2026_claude_diag_20261029/`（提交包 `pack/`、训练权重 `train/`，约 1.3 GB）。
方案说明与交接：`01_当前比赛/Claude方案与诊断/Claude_无人机交接_20261029_方案与本地成绩.md`。
今天做的包都没有提交平台。

## 脚本

| 文件 | 用途 |
|---|---|
| `zip_compare.py` | 候选 ZIP 与冠军逐像素对比：各类面积、类别迁移、出现图数 |
| `per_image.py`、`sheet.py`、`sheet2.py` | 逐图类别比例、失败样例图、Source 随机样例图 |
| `domain_feat.py`、`dom_an.py` | 外观特征聚类，Source1399 分簇 mIoU |
| `green_check.py` | 绿色场景里各预测类的颜色和教师权重 |
| `loss_decomp.py` | Source1399 丢分按混淆对精确拆解 |
| `cache_raw.py` | 给某个 checkpoint 缓存 Target 的 D4 原始 logits（fp16） |
| `mm_fit.py` | 面积对齐：拟合 8 维 bias，使 Target 各类面积等于冠军 |
| `pack.sh` | 调用冻结打包脚本 `package_semifinal_bias.py`（SHA 206cc80d…）出 ZIP 并做 QA |
| `c1_chain.sh` | C1 缓存→拟合→打包 |
| `mix_train.py` | C1 配方 + 跨域 ClassMix 训练 |
| `soup.py` | 同配方 checkpoint 参数平均 |
| `arm_chain.sh` | MIX 一臂：平均→缓存→拟合→打包→Source 检查 |
| `src_eval.py` | Source1399 D4 mIoU（只做崩坏检测） |
| `region_gate_eval.py` | 按连通块大小筛选 Bstar 改判的评估 |
| `gate_pack.py` | 生成 H75_BARREN_GATE4096 |
| `mixviz.py` | ClassMix 训练样本可视化 |

## 结果（`results/`）

- `zip_compare_out.json`：冠军 / CTRL / C1 / M2F / H75-Agri0 对比
- `zip_compare_mm.json`、`zip_compare_gate.json`、`zip_compare_b26.json`：C1_MM / CTRL_MM、GATE4096、队友 BARREN_2P6 与冠军对比
- `region_gate_eval.json`：块大小与改判正确率、各阈值的 Source mIoU 和 Target 出现图数
- `mm/`、`bias/`：拟合过程和 bias 配置
- `pack_reports/`：5 个包的 QA 和打包清单
- `train_summaries/`、`logs/`：MIX 4 次训练、参数平均、Source 评估
- `domain_feat.json`、`cluster_K3_*.npy`：逐图外观特征和聚类标签

## 图

- `fail_sheet.jpg`：冠军漏判植被/道路的 Target 样例
- `src_sheet.jpg`：Source 随机样例与 GT
- `mixviz.jpg`：FDA β=0.01 出现彩色斑块（已否决）
- `mixviz2.jpg`：改用均值/方差迁移后的结果

## 已删除

`cache_C1/`、`cache_CTRL/`（共 7.2 GB 原始 logits 缓存，10-29 22:00 为腾磁盘空间删除）。需要时用 `cache_raw.py` 重建，单卡约 22 分钟。
