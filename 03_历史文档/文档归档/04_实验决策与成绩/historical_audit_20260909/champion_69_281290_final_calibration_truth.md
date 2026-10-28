# Champion 69.281290 — Final Calibration Truth

> 2026-09-09。代码级证据锁死，非推断。

## 事实
- online_score = **69.281290**
- submission = **calib_build_p100** (zip 6be4e8b0...ee04)
- checkpoint = stage23 B3+CWFA (sha d3740b4ef73...e228)
- TTA = rot_tta.py D4 (4旋转×2镜像=8视图)
- **final_8d_vector (train_id order) = [-0.2000, +0.1000, 0.0709, 0.7000, 3.6000, 0.4000, 3.3215, -0.5000]**
- class_order = Background/Building/Road/Water/Barren/Vegetation/Agricultural/Vehicle
- calibration_space = LOGITS; formula = logits += bias (无scale当用--prior-bias)
- TTA顺序: 8视图平均logits → +bias → argmax
- build_p100 = Building bias -0.0384 → +0.10 (第8个挤出维度)

## 冲突解决
- 6维版(-0.0384) vs 8维版(+0.10): 差在第2维建筑; 最终=+0.10版
