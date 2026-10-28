# ATDA-71 最终单模型溯源（Q 节）

## 最终单模型（唯一）

- **模型**：`composed_atda_8k_a1p25_folded.pt`
- **SHA256**：`5be48fa5048b97c67aa7fc486fdb6619661d2ff2f08af9f92998ef4ac811d2ba`
- **平台分**：**mIoU = 0.706154**（提交包 278，`atda_8k_a1p25_Bstar.zip`）
- **定义**：`theta261 + 1.25*(theta_atda_8k − theta261)`，再将 b_star 折叠进 `decode_head.classifier.bias`

## 平台成绩（272–278，提交顺序 = 编号顺序）

| 编号 | 变体 | mIoU |
|---|---|---|
| 272 | 2k a1.0 | 0.700504 |
| 273 | 2k a0.5 | 0.703288 |
| 274 | 4k a1.0 | 0.705031 |
| 275 | 4k a0.5 | 0.704711 |
| 276 | 8k a1.0 | 0.706048 |
| 277 | 8k a0.75 | 0.705655 |
| 278 | 8k a1.25 | **0.706154** |

相对 theta261 基线 0.702836，winner = **+0.003318（+0.33pp）**，强成功（≥+0.20pp）。

## base 身份

- **theta261** = `phase0_candidates_20260918/A_cleanSEM_M_k100_th209.pt`
- **base SHA256** = `9ffdd0beb8480a1efd361503d637d3d53cf716c808537c551fefae09f7fe2786`
- 平台 0.702836（包 261）

## 训练 checkpoint

- 8k 训练 checkpoint `run_s2s3_base261/checkpoint_step8000.pt`
- SHA256 `c3e87f6c51bd1fb32e6249783618d9bec7f1fc431a4a36d4ef3066d54e480eac`
- 8000 步，`frozen_identical_to_theta261=True`、`theta_ref_unchanged=True`、`amp_skips=2`

## 折叠验证

- 张量级：`only_classifier_bias_changed=True`，bias delta vs b_star = `2.98e-08`（fp32）
- 逐像素：500 张 test1，5.24 亿像素，**0 不匹配**（折叠前后 argmax 完全一致）

## config

- 训练 config：`configs/atda/atda_s2s3.yaml`（`atda71_s2s3_base261_seed42`）
- 模型架构 config：`stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml`（cwfa 关闭）
- lr=1e-5, wd=0.01, AdamW, poly power=1.0, total_steps=10000, batch 1 source+1 target
- λ_hard=0.5, λ_cons=0.25, ramp=500, EMA 0.999
- trainable scope: `segformer.stages.2.*`, `segformer.stages.3.*`

## 文件清单

- 代码：`atda_20260918/_atda_train.py`、`_atda_ref261_cache.py`、`_atda_export.py`、`_atda_finalize.py`、`_atda_launch.sh`、`_atda_eval.sh`
- source 源集：`source_ids_5597.txt`（5597 官方 GT，SHA `ea765c43…`）
- target：`2026-低空图像语义分割赛道-训练集/images`（test1_1..500，500 张）
- reference cache：`ref261_cache/scores/`（native 256×256 D4 TTA8 canonical）

## 随机种子 / 版本

- seed = 42
- torch 2.13.0+cu130, CUDA 13.0, cudnn 92000, transformers 5.13.0, numpy 2.2.6, python 3.10.12
- GPU: RTX 4090, driver 580.126.09

## 推理协议（部署）

D4 TTA8（4 旋转 × 2 翻转），native stride-4 logits，FP32 view mean，cwfa 关闭，strict load。
折叠后模型直接 `argmax` 输出 official class 1..8，无需再 +b_star。

## 日志

- 训练：`run_s2s3_base261/atda.log`、`run_s2s3_base261.console.log`
- smoke：`smoke_200/SMOKE_REPORT.json`
- 评测：`eval.log`、`a1p0_backfill.log`
- finalize：`finalize_8k.log`、`finalize/*/FINALIZE_REPORT.json`
