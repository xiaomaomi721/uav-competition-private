# SHBR：Selective Heterogeneous Blind Recall

当前冠军：Arm B `71.045315`。本目录实现一条 fail-closed 条件链：只有 Swin-L/Mask2Former donor 在 H300 开发、R1099 冻结检查以及 test2 无 GT 支持域检查中全部通过，才向 Arm B 的 A05 selected-mask soft KL 增加弱 donor blind-class KL。

固定处理项：

```text
G ∩ M = ∅
beta = 0.25
L = 0.1*CE_source
    + ramp*[sum_M KL(A05||student) + beta*sum_G KL(donor||student)] / |M|
```

`G` 只可召回 Water/Barren/Agricultural，且 A05 与 Arm B 都未给出 donor 的目标类。没有 generic complement 正监督，没有 M2F Bstar，没有推理期 teacher/ensemble。

## 运行

完整要求见 [`执行手册_交队友.md`](执行手册_交队友.md)。固定部署位置为 `/data/uav/shbr_bundle/`，唯一命令：

```bash
bash run_all.sh
```

任一阶段失败都会生成最终 STOP 文件。只有 `READY_FOR_MANUAL_PAIRED_SUBMISSION` 才允许人工提交冻结 CONTROL 和新 TREAT 两包；代码不含上传功能。

## 证据边界

- TCA V2 source本地 `+0.493082pp`，线上却 `−0.153165pp`；CE020 本地 `+0.133756pp`，线上却 `−0.068270pp`。因此 source mIoU 不用于预测 test2 排名。
- source gate 只验证 donor 是否携带可识别的新信息，不能证明 test2 准确率符号。
- 最终若 READY，只代表值得做一次配对在线因果试验，不代表预计必涨。

## 方法启发（不是论文复现）

- U²PL, CVPR 2022：不把所有 unreliable pixels 强制成正伪标签。  
  https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Semi-Supervised_Semantic_Segmentation_Using_Unreliable_Pseudo-Labels_CVPR_2022_paper.html
- GoodSAM, CVPR 2024：利用异构区域表示与语义 teacher 的互补。  
  https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_GoodSAM_Bridging_Domain_and_Capacity_Gaps_via_Segment_Anything_Model_CVPR_2024_paper.html
- Category-Adaptive Thresholding：按类别冻结不同阈值。  
  https://arxiv.org/abs/2103.15467

## 关键文件

- `shbr_common.py`：冻结公式、D4、GT映射、门控与审计合约
- `donor_adapter.py`：MMSeg Mask2Former score-space adapter
- `build_prediction_cache.py`：source/test2 D4 cache
- `discover_and_validate_gate.py`：H300开发→冻结→R1099单次检查
- `build_target_cache.py`：test2 支持域与训练 cache
- `calibrate_fixed_beta.py`：固定 beta 的无更新量级门
- `train_shbr.py`：CONTROL/TREAT 共用训练路径
- `evaluate_source_health.py`：只作 collapse guard
- `package_and_audit.py`：单 checkpoint 打包和无 GT 行为审计
- `run_all.sh`：唯一入口
