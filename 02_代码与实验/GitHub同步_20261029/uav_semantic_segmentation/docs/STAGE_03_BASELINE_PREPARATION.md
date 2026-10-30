# Stage 03: baseline preparation

## 1. 阶段目标

建立完整数据、固定划分、纯 CE 的 SegFormer-B0 正式基线。Stage 03 当前状态是正式配置已准备，等待队员完成 20-step 冒烟测试、人工审核和完整训练。

## 2. 队员分工

- 谢诗博：仓库、配置、Codex 任务拆解、结果检查
- 曾昭虹：RTX 4060 环境验证、20-step 冒烟测试、正式基线训练
- 蒋鸿宇：阶段审核和进度确认
- RTX 4090：必要时用于完整训练或加速复现

## 3. 数据使用

- 官方训练图像 6996 张
- train 5597
- val 1399
- 测试图 500 张不参与 Stage 03
- 固定 quick validation 文件位于 `splits/fixed_quick_val.txt`，只服务于 quick 复现和完整性检查

## 4. 正式配置

- 配置文件：`configs/full/stage03_b0_ce_512_seed42.yaml`
- experiment_id：`formal_b0_ce_512_seed42`
- model：SegFormer-B0
- model_name：`nvidia/segformer-b0-finetuned-ade-512-512`
- num_classes：8
- ignore_index：255
- crop_size：512
- loss：plain Cross Entropy
- optimizer：AdamW，learning_rate=0.00006，weight_decay=0.01
- batch_size：2
- gradient_accumulation_steps：2
- effective_batch_size：4
- amp：true
- seed：42
- num_workers：0
- max_steps：56000
- validation_interval：1400
- train_split：`splits/train.txt`
- validation_split：`splits/val.txt`
- max_train_samples：0，表示不限制
- max_val_samples：0，表示不限制
- rare_crop.enabled：false
- transforms.domain_aug.enabled：false
- quick_pilot：false

## 5. 执行顺序

1. clone 或 pull 仓库
2. 配置官方数据路径
3. check_env
4. check_split
5. 三个回归测试
6. check_stage03_config
7. 20-step 冒烟
8. 人工审核
9. 正式训练
10. 汇总结果

## 6. 冒烟测试命令

Windows CMD 多行命令：

```cmd
python scripts\train.py ^
--config configs\full\stage03_b0_ce_512_seed42.yaml ^
--dataset-root "<DATASET_ROOT>" ^
--output-dir outputs\stage03\formal_b0_ce_512_seed42_smoke ^
--device cuda ^
--max-steps 20 ^
--max-train-samples 128 ^
--max-val-samples 64
```

## 7. 正式训练命令

Windows CMD 多行命令：

```cmd
python scripts\train.py ^
--config configs\full\stage03_b0_ce_512_seed42.yaml ^
--dataset-root "<DATASET_ROOT>" ^
--output-dir outputs\stage03\formal_b0_ce_512_seed42 ^
--device cuda
```

## 8. 禁止项

- 不重新划分数据
- 不使用测试集训练
- 不直接加载 step06_domain_aug 权重
- 不加入 Weighted CE
- 不加入 Dice/Focal
- 不加入 Rare Crop
- 不加入 Domain Augmentation
- 不使用 TTA
- 不使用 ensemble
- 不上传权重和 outputs

## 9. 输出验收

必须得到：

- `config.yaml`
- `environment.json`
- `train_log.csv`
- `metrics.json`
- `per_class_iou.csv`
- `confusion_matrix.npy`
- `best_miou.pt`
- `last.pt`
- `completion.json`
- 至少 8 张预测可视化

## 10. 汇报指标

- best mIoU
- best step
- Pixel Accuracy
- 8 类 IoU
- Vehicle IoU
- Barren IoU
- Road IoU
- Agricultural IoU
- peak VRAM
- average step time
- OOM 情况
- 是否完整完成 56000 step

## 11. max_steps 与 epoch 换算

当前 Trainer 仍以 `max_steps` 作为训练终止条件。5597 张训练图、`batch_size=2`、`gradient_accumulation_steps=2` 时，一个近似完整 epoch 约为 1400 个 optimizer step；`56000` step 约等于 40 个 epoch；`validation_interval=1400` 约等于每个 epoch 验证一次。
