# 全部推理、后处理、校准方案与线上分

范围：本地线上成绩记录表、historical_audit及指定复盘文档；线上分为历史账本记录，未独立核验平台回执。

## 校准来源必须分开

Stage23完成原始8D扫描；其他模型复用Stage23向量不等于本模型重新扫描。部分模型确有局部重扫，下表分calibration_origin记录。

| 阶段 | 完整向量：背景/建筑/道路/水体/荒地/植被/农田/车辆 | 历史分 |
|---|---|---|
| 旧锁定 | [-0.2, -0.0384, 0.0709, 0.7, 3.6, 0.1932, 3.3215, -0.5] |69.276229|
| 中间植被+.4 | [-0.2, -0.0384, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5] |69.278739|
| 最终build_p100 | [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5] |69.281290|

## 全部分数变体（标准版行仅作对照）

### 一、B3 基线系列（Exp18，无 CWFA）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| Exp18_B3_标准版 | 无后处理 | 线上分=66.739544 | NONE | 线上成绩记录表.md:13 |
| Exp18_B3_先验校正版 | prior-shift | 线上分=66.895 | EXP18_PRIOR_OR_CALIBRATION_SEARCH | 线上成绩记录表.md:14 |
| Exp18_B3_先验系数0.7版 | prior×0.7 | 线上分=67.030056 | EXP18_PRIOR_OR_CALIBRATION_SEARCH | 线上成绩记录表.md:15 |
| Exp18_B3_先验系数1.3版 | prior×1.3 | 线上分=67.205006 | EXP18_PRIOR_OR_CALIBRATION_SEARCH | 线上成绩记录表.md:16 |
| Exp18_B3_先验系数1.5版 | prior×1.5 | 线上分=67.2502 | EXP18_PRIOR_OR_CALIBRATION_SEARCH | 线上成绩记录表.md:17 |
| Exp18_B3_先验系数1.7版 | prior×1.7 | 线上分=67.289126 | EXP18_PRIOR_OR_CALIBRATION_SEARCH | 线上成绩记录表.md:18 |
| Exp18_B3_先验系数2.0版 | prior×2.0 | 线上分=67.336993 | EXP18_PRIOR_OR_CALIBRATION_SEARCH | 线上成绩记录表.md:19 |
| Exp18_B3_先验系数2.5版 | prior×2.5 | 线上分=67.383838 | EXP18_PRIOR_OR_CALIBRATION_SEARCH | 线上成绩记录表.md:20 |
| Exp18_B3_先验系数3.0版 | prior×3.0 | 线上分=67.393058 | EXP18_PRIOR_OR_CALIBRATION_SEARCH | 线上成绩记录表.md:21 |
| Exp18_B3_先验系数3.5版 | prior×3.5 | 线上分=67.356169 | EXP18_PRIOR_OR_CALIBRATION_SEARCH | 线上成绩记录表.md:22 |
| Exp18_B3_旋转TTA8视图_标准版 | 8视图TTA | 线上分=67.209674 | EXP18_PRIOR_OR_CALIBRATION_SEARCH | 线上成绩记录表.md:23 |
| Exp18_B3_旋转TTA8视图_先验2.5版 | +prior2.5 | 线上分=67.996011 | EXP18_PRIOR_OR_CALIBRATION_SEARCH | 线上成绩记录表.md:24 |
| Exp18_B3_旋转TTA8视图_先验3.0版 | +prior3.0 | 线上分=68.009701 | EXP18_PRIOR_OR_CALIBRATION_SEARCH | 线上成绩记录表.md:25 |
| Exp18_B3_旋转TTA8视图_先验3.5版 | +prior3.5 | 线上分=67.990004 | EXP18_PRIOR_OR_CALIBRATION_SEARCH | 线上成绩记录表.md:26 |
| Exp18_B3_旋转TTA8视图_按类校准V1 | +校准V1 | 线上分=68.025335 | EXP18_PRIOR_OR_CALIBRATION_SEARCH | 线上成绩记录表.md:27 |
| Exp18_B3_旋转TTA8视图_按类校准V1.5 | V1.5 | 线上分=67.963641 | EXP18_PRIOR_OR_CALIBRATION_SEARCH | 线上成绩记录表.md:28 |
| Exp18_B3_旋转TTA8视图_按类校准V1b | V1b | 线上分=67.861075 | EXP18_PRIOR_OR_CALIBRATION_SEARCH | 线上成绩记录表.md:29 |
| Exp18_B3_旋转TTA8视图_按类校准V1c | V1c | 线上分=68.028673 | EXP18_PRIOR_OR_CALIBRATION_SEARCH | 线上成绩记录表.md:30 |
| Exp18_B3_旋转TTA8视图_按类校准V2 | V2 | 线上分=67.831069 | EXP18_PRIOR_OR_CALIBRATION_SEARCH | 线上成绩记录表.md:31 |
| Exp18_B3_旋转TTA8视图_背景压到0.005 | V1d | 线上分=68.030798 | EXP18_PRIOR_OR_CALIBRATION_SEARCH | 线上成绩记录表.md:32 |
| Exp18_B3_旋转TTA8视图_背景压到底 | V1e | 线上分=68.032596 | EXP18_PRIOR_OR_CALIBRATION_SEARCH | 线上成绩记录表.md:33 |
| Exp18_B3_V1f_背景负0.005 | V1f | 线上分=68.034643 | EXP18_PRIOR_OR_CALIBRATION_SEARCH | 线上成绩记录表.md:34 |
| Exp18_B3_V1f_背景负0.010 | V1f | 线上分=68.036412 | EXP18_PRIOR_OR_CALIBRATION_SEARCH | 线上成绩记录表.md:35 |
| Exp18_B3_V1f_背景负0.020 | V1f | 线上分=68.039691 | EXP18_PRIOR_OR_CALIBRATION_SEARCH | 线上成绩记录表.md:36 |

### 2.1 基础

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| Exp23_B3CWFA_标准版 | 无后处理 | 线上分=67.371405 | NONE | 线上成绩记录表.md:43 |
| Exp23_B3CWFA_旋转TTA8视图_校准版 | +8视图+V1e | 线上分=68.686044 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:44 |

### 2.2 背景负值梯度（单调上升，背景压是有效方向）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| V1e复验 | 0.000 | 线上分=68.686044 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:49 |
| 背景负0.005 | −0.005 | 线上分=68.688338 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:50 |
| 背景负0.010 | −0.010 | 线上分=68.690102 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:51 |
| 背景负0.020 | −0.020 | 线上分=68.693969 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:52 |
| 背景负0.030 | −0.030 | 线上分=68.698192 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:53 |
| 背景负0.040 | −0.040 | 线上分=68.702114 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:54 |
| 背景负0.050 | −0.050 | 线上分=68.705735 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:55 |
| 背景负0.060 | −0.060 | 线上分=68.709247 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:56 |
| 背景负0.070 | −0.070 | 线上分=68.712814 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:57 |
| 背景负0.080 | −0.080 | 线上分=68.716121 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:58 |
| 背景负0.100 | −0.100 | 线上分=68.722245 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:59 |
| 背景负0.120 | −0.120 | 线上分=68.728744 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:60 |
| 背景负0.150 | −0.150 | 线上分=68.736794 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:61 |
| 背景负0.200 | −0.200 | 线上分=68.749433 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:62 |
| 背景负0.250荒地3.5 | −0.250 | 线上分=69.253142 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:63 |
| 背景负0.300荒地3.5 | −0.300 | 线上分=69.248491 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:64 |
| 背景负0.400荒地3.5 | −0.400 | 线上分=69.231327 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:65 |

### 2.3 荒地梯度（荒地=第二个"背景"，抬升单调）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| 荒地1.40 | 1.40 | 线上分=68.807208 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:70 |
| 荒地1.70 | 1.70 | 线上分=68.904357 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:71 |
| 荒地1.80 | 1.80 | 线上分=68.946521 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:72 |
| 荒地2.00 | 2.00 | 线上分=68.998581 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:73 |
| 荒地2.30 | 2.30 | 线上分=69.062468 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:74 |
| 荒地2.60 | 2.60 | 线上分=69.126146 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:75 |
| 荒地2.90 | 2.90 | 线上分=69.177881 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:76 |
| 荒地3.20 | 3.20 | 线上分=69.221335 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:77 |
| 荒地3.50 | 3.50 | 线上分=69.25571 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:78 |
| 荒地3.60 | 3.60 | 线上分=69.257555 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:79 |
| 荒地3.80 | 3.80 | 线上分=69.251513 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:80 |
| 荒地4.20 | 4.20 | 线上分=69.21782 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:81 |

### 2.4 背景−0.2 × 荒地组合（背景锁定 −0.200）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| 背景负0.2荒地2.3 | 2.30 | 线上分=69.081147 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:86 |
| 背景负0.2荒地2.6 | 2.60 | 线上分=69.140589 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:87 |
| 背景负0.2荒地2.9 | 2.90 | 线上分=69.188055 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:88 |
| 背景负0.2荒地3.2 | 3.20 | 线上分=69.229683 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:89 |
| 背景负0.2荒地3.6 | 3.60 | 线上分=69.257555 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:90 |

### 2.5 农田梯度（农田压有效但到顶，抬全负）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| 农田压2.80 | 2.80 | 线上分=68.746461 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:95 |
| 农田压2.40 | 2.40 | 线上分=68.750914 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:96 |
| 农田压2.00 | 2.00 | 线上分=68.751878 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:97 |
| 农田压1.60 | 1.60 | 线上分=68.713302 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:98 |
| 农田3.5 | 3.5(抬) | 线上分=68.668192 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:99 |
| 农田3.7 | 3.7(抬) | 线上分=68.641946 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:100 |

### 2.6 剩余类校准（第3个可调类=水体）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| 道路0.15 | road 0.15 | 线上分=69.253252 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:105 |
| 道路0.25 | road 0.25 | 线上分=69.246306 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:106 |
| 水体0.85 | water 0.85 | 线上分=69.26623 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:107 |
| 水体0.70 | water 0.70 | 线上分=69.268985 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:108 |
| 建筑0.05 | building 0.05 | 线上分=69.25904 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:109 |
| 建筑0.12 | building 0.12 | 线上分=69.258702 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:110 |
| 车辆负0.50 | veh −0.50 | 线上分=69.265051 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:111 |
| 车辆负0.85 | veh −0.85 | 线上分=69.239758 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:112 |

### 2.8 第4维扫描（水体/车辆组合，判决：校准彻底到顶）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| 水体0.55 | water 0.55 | 线上分=69.268927 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:117 |
| 水体0.40 | water 0.40 | 线上分=69.267179 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:118 |
| 车辆负0.40 | veh −0.40 | 线上分=69.274406 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:119 |
| 车辆负0.30 | veh −0.30 | 线上分=69.270475 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:120 |
| 水体0.70车辆0.50 | water0.70+veh−0.50 | 线上分=69.276229 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:121 |
| 水体0.55车辆0.50 | 组合 | 线上分=69.276136 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:122 |
| 水体0.70车辆0.40 | 组合 | 线上分=69.274406 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:123 |
| 水体0.55车辆0.40 | 组合 | 线上分=69.274306 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:124 |

### 2.7 prior 系数重扫（判决：被校准碾压，放下）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| prior系数2.0 | 2.0 | 线上分=68.418181 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:131 |
| prior系数2.5 | 2.5 | 线上分=68.501772 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:132 |
| prior系数3.0 | 3.0 | 线上分=68.553899 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:133 |
| prior系数3.5 | 3.5 | 线上分=68.572398 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:134 |
| prior系数4.0 | 4.0 | 线上分=68.565254 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:135 |

### 3.1 基础

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| Exp26_B4_标准版 | 无后处理 | 线上分=66.951058 | NONE | 线上成绩记录表.md:144 |
| Exp26_B4_旋转TTA8视图_V1e校准版 | +8视图+V1e | 线上分=68.030036 | UNKNOWN_EXACT_ORIGIN | 线上成绩记录表.md:145 |

### 3.2 校准重扫（B4 荒地上限更高）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| V1e迁移 | V1e | 线上分=68.030036 | EXP26_MODEL_SPECIFIC_RESCAN_AFTER_INITIAL_TRANSFER | 线上成绩记录表.md:150 |
| 背景负0.2荒地3.4 | 3.40 | 线上分=68.625691 | EXP26_MODEL_SPECIFIC_RESCAN_AFTER_INITIAL_TRANSFER | 线上成绩记录表.md:151 |
| 背景负0.2荒地3.6 | 3.60 | 线上分=68.65144 | EXP26_MODEL_SPECIFIC_RESCAN_AFTER_INITIAL_TRANSFER | 线上成绩记录表.md:152 |
| 背景负0.2荒地3.8 | 3.80 | 线上分=68.677898 | EXP26_MODEL_SPECIFIC_RESCAN_AFTER_INITIAL_TRANSFER | 线上成绩记录表.md:153 |
| 背景负0.2荒地4.0 | 4.00 | 线上分=68.699969 | EXP26_MODEL_SPECIFIC_RESCAN_AFTER_INITIAL_TRANSFER | 线上成绩记录表.md:154 |
| 背景负0.2荒地4.2 | 4.20 | 线上分=68.711309 | EXP26_MODEL_SPECIFIC_RESCAN_AFTER_INITIAL_TRANSFER | 线上成绩记录表.md:155 |
| 背景负0.2荒地4.5 | 4.50 | 线上分=68.722425 | EXP26_MODEL_SPECIFIC_RESCAN_AFTER_INITIAL_TRANSFER | 线上成绩记录表.md:156 |
| 背景负0.2荒地4.8 | 4.80 | 线上分=68.724874 | EXP26_MODEL_SPECIFIC_RESCAN_AFTER_INITIAL_TRANSFER | 线上成绩记录表.md:157 |
| 背景负0.3荒地4.0 | bg−0.3 | 线上分=68.689993 | EXP26_MODEL_SPECIFIC_RESCAN_AFTER_INITIAL_TRANSFER | 线上成绩记录表.md:158 |
| 背景负0.1荒地3.6 | bg−0.1 | 线上分=68.645683 | EXP26_MODEL_SPECIFIC_RESCAN_AFTER_INITIAL_TRANSFER | 线上成绩记录表.md:159 |

### 3.3 B4 剩余类（纯 CE 校准空间有限）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| 道路0.15 | road 0.15 | 线上分=68.721815 | EXP26_MODEL_SPECIFIC_RESCAN_AFTER_INITIAL_TRANSFER | 线上成绩记录表.md:164 |
| 道路0.25 | road 0.25 | 线上分=68.715 | EXP26_MODEL_SPECIFIC_RESCAN_AFTER_INITIAL_TRANSFER | 线上成绩记录表.md:165 |
| 水体0.85 | water 0.85 | 线上分=68.731587 | EXP26_MODEL_SPECIFIC_RESCAN_AFTER_INITIAL_TRANSFER | 线上成绩记录表.md:166 |
| 水体0.70 | water 0.70 | 线上分=68.732808 | EXP26_MODEL_SPECIFIC_RESCAN_AFTER_INITIAL_TRANSFER | 线上成绩记录表.md:167 |
| 建筑0.05 | building 0.05 | 线上分=68.723412 | EXP26_MODEL_SPECIFIC_RESCAN_AFTER_INITIAL_TRANSFER | 线上成绩记录表.md:168 |
| 建筑0.12 | building 0.12 | 线上分=68.721429 | EXP26_MODEL_SPECIFIC_RESCAN_AFTER_INITIAL_TRANSFER | 线上成绩记录表.md:169 |
| 车辆负0.50 | veh −0.50 | 线上分=68.759823 | EXP26_MODEL_SPECIFIC_RESCAN_AFTER_INITIAL_TRANSFER | 线上成绩记录表.md:170 |
| 车辆负0.85 | veh −0.85 | 线上分=68.685353 | EXP26_MODEL_SPECIFIC_RESCAN_AFTER_INITIAL_TRANSFER | 线上成绩记录表.md:171 |

### 3.4 B4+CWFA（Exp29，2026-08-29 判决：CWFA 在 B4 上有害 −0.95）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| Exp29_B4CWFA_标准版 | 无后处理 | 线上分=66.001109 | NONE | 线上成绩记录表.md:178 |
| Exp29_B4CWFA_旋转TTA8校准版 | +TTA+校准 | 线上分=67.459865 | TRANSFERRED_B3_STAGE23_VECTOR_NOT_PROVEN_MODEL_SPECIFIC8D_SEARCH | 线上成绩记录表.md:179 |

### 3.5 B4+no-decay（Exp30，2026-08-29 判决：no-decay 在 B4 上也有害 −0.37）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| Exp30_B4nodecay_标准版 | 无后处理 | 线上分=66.584083 | NONE | 线上成绩记录表.md:186 |
| Exp30_B4nodecay_旋转TTA8校准版 | +TTA+校准 | 线上分=7.926311 | TRANSFERRED_B3_STAGE23_VECTOR_NOT_PROVEN_MODEL_SPECIFIC8D_SEARCH | 线上成绩记录表.md:187 |

### 3.6 B4+低lr+drop_path0.2（Exp35，2026-08-31 判决：参数调对没能救回 B4，−0.118）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| Exp35_标准版 | lr 2e-5 + dp 0.2，无后处理 | 线上分=66.832587 | NONE | 线上成绩记录表.md:201 |
| Exp35_旋转TTA8 | +TTA 无校准 | 线上分=67.972872 | NONE | 线上成绩记录表.md:202 |

### 2.9 植被 bias 补扫（第6维，此前从未单独调，2026-08-30）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| baseline | 0.1932 | 线上分=69.276229 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:217 |
| veg_000 | 0.0 | 线上分=69.261544 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:218 |
| veg_m100 | −0.1 | 线上分=69.24977 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:219 |
| veg_m200 | −0.2 | 线上分=69.235613 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:220 |
| veg_m300 | −0.3 | 线上分=69.219421 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:221 |
| veg_p400 | +0.4 | 线上分=69.278739 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:222 |
| veg_p600 | +0.6 | 线上分=69.268621 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:223 |

### 2.10 蒸馏λ/MixStyleα 减步数扫描（14000步，2026-08-30，仅横向排序）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| 蒸馏 λ=0.1 | λ=0.1 | 线上标准=66.225255; 线上校准=66.736261 | TRANSFERRED_B3_CALIBRATION_EXACT_VECTOR_UNKNOWN_NOT_STAGE23_ORIGINAL_SCAN | 线上成绩记录表.md:230 |
| 蒸馏 λ=0.3 | λ=0.3 | 线上标准=66.043135; 线上校准=66.922685 | TRANSFERRED_B3_CALIBRATION_EXACT_VECTOR_UNKNOWN_NOT_STAGE23_ORIGINAL_SCAN | 线上成绩记录表.md:231 |
| MixStyle α=0.05 | α=0.05 | 线上标准=66.243589; 线上校准=66.634921 | TRANSFERRED_B3_CALIBRATION_EXACT_VECTOR_UNKNOWN_NOT_STAGE23_ORIGINAL_SCAN | 线上成绩记录表.md:232 |
| MixStyle α=0.2 | α=0.2 | 线上标准=65.841038; 线上校准=66.522947 | TRANSFERRED_B3_CALIBRATION_EXACT_VECTOR_UNKNOWN_NOT_STAGE23_ORIGINAL_SCAN | 线上成绩记录表.md:233 |

### 2.11 道路/建筑残渣扫描（第7/8维，最后两个未扫维度，2026-09-01）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| road_base | road 0.0709 | 线上分=69.278739 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:248 |
| road_m100 | road −0.10 | 线上分=69.273667 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:249 |
| road_m200 | road −0.20 | 线上分=69.266583 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:250 |
| road_m300 | road −0.30 | 线上分=69.254496 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:251 |
| road_m500 | road −0.50 | 线上分=69.218904 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:252 |
| build_m200 | build −0.20 | 线上分=69.270673 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:253 |
| build_m100 | build −0.10 | 线上分=69.276359 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:254 |
| build_p100 | build +0.10 | 线上分=69.28129 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:255 |
| build_p300 | build +0.30 | 线上分=69.276256 | STAGE23_ORIGINAL_SEARCH_OR_RECHECK | 线上成绩记录表.md:256 |

### 四、B5 系列（Exp27，纯 CE）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| Exp27_B5_标准版 | 无后处理 | 线上分=66.405239 | NONE | 线上成绩记录表.md:273 |
| Exp27_B5_旋转TTA8视图_V1e校准版 | +V1e | 线上分=67.200599 | EXP27_RESCAN_AFTER_V1E_TRANSFER | 线上成绩记录表.md:274 |
| 校准_V1e迁移 | V1e | 线上分=67.200599 | EXP27_RESCAN_AFTER_V1E_TRANSFER | 线上成绩记录表.md:275 |
| 校准_背景负0.2荒地3.6 | 3.60 | 线上分=67.67578 | EXP27_RESCAN_AFTER_V1E_TRANSFER | 线上成绩记录表.md:276 |
| 校准_背景负0.2荒地4.0 | 4.00 | 线上分=67.665084 | EXP27_RESCAN_AFTER_V1E_TRANSFER | 线上成绩记录表.md:277 |
| 校准_背景负0.2荒地4.5 | 4.50 | 线上分=67.679726 | EXP27_RESCAN_AFTER_V1E_TRANSFER | 线上成绩记录表.md:278 |
| 校准_背景负0.1荒地4.0 | bg−0.1 | 线上分=67.677944 | EXP27_RESCAN_AFTER_V1E_TRANSFER | 线上成绩记录表.md:279 |

### 4.1 B5+LoRA（Exp32，2026-08-30 判决：LoRA 在 B5 上灾难性有害 −2.72）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| Exp32_B5LoRA_标准版 | 无后处理 | 线上分=63.68103 | NONE | 线上成绩记录表.md:286 |
| Exp32_B5LoRA_旋转TTA8校准版 | +TTA+校准 | 线上分=63.159382 | TRANSFERRED_B3_STAGE23_VECTOR_NOT_PROVEN_MODEL_SPECIFIC8D_SEARCH | 线上成绩记录表.md:287 |

### 四-B、B3+CopyPaste 系列（Exp33，2026-08-31 判决：线下涨线上跌，泛化失败）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| Exp33_CopyPaste_标准版 | 无后处理 | 线上分=66.427993 | NONE | 线上成绩记录表.md:297 |
| Exp33_CopyPaste_旋转TTA8校准版 | +TTA+B3六维校准 | 线上分=68.775101 | TRANSFERRED_B3_STAGE23_VECTOR_NOT_PROVEN_MODEL_SPECIFIC8D_SEARCH | 线上成绩记录表.md:298 |

### 4.2 B3+蒸馏λ0.1+MixStyleα0.05 集成（Exp39，2026-09-01 判决：+0.177 微正，未过 CWFA）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| Exp39_标准版 | 蒸馏λ0.1+MixStyleα0.05 无后处理 | 线上分=66.915682 | NONE | 线上成绩记录表.md:315 |
| Exp39_旋转TTA8校准版 | +TTA+B3六维校准 | 线上分=68.725136 | TRANSFERRED_B3_STAGE23_VECTOR_NOT_PROVEN_MODEL_SPECIFIC8D_SEARCH | 线上成绩记录表.md:316 |

### 4.3 B3+注意力融合解码头（Exp40，2026-09-01 判决：−0.760 有害，MCCA 适配 SegFormer 失败）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| Exp40_标准版 | MCCA 式 CCA+AFF 解码头替代 concat+1×1 | 线上分=65.978999 | NONE | 线上成绩记录表.md:332 |
| Exp40_旋转TTA8校准版 | +TTA+B3六维校准 | 线上分=67.40267 | TRANSFERRED_B3_STAGE23_VECTOR_NOT_PROVEN_MODEL_SPECIFIC8D_SEARCH | 线上成绩记录表.md:333 |

### 4.4 B3+边界分支BCE（Exp41，2026-09-01：真无TTA判决 +0.314 ✅，训练期第三正收益但<CWFA）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| Exp41_真无TTA标准版 | 单视图无校准（模型本体） | 线上分=67.053 | NONE | 线上成绩记录表.md:349 |
| Exp41_TTA8无校准 | 旋转TTA8，无校准 | 线上分=67.480463 | NONE | 线上成绩记录表.md:350 |
| Exp41_旋转TTA8校准版 | +TTA+八维锁定校准 | 线上分=68.79692 | TRANSFERRED_B3_STAGE23_VECTOR_NOT_PROVEN_MODEL_SPECIFIC8D_SEARCH | 线上成绩记录表.md:351 |

### 4.5 空间先验后处理：车辆道路约束（2026-09-01，灾难性失败）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| exp41_spatial_veh_clear | Exp41校准预测中，距道路/建筑5px外的车辆全部改背景 | 线上分=60.0346 | HARD_MASK_POSTPROCESS_AFTER_CALIBRATION | 线上成绩记录表.md:373 |

### 4.6 B3+加权CE（Exp42，2026-09-01：官方点名第一，线上微负−0.049）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| Exp42_真无TTA标准版 | B3 + 训练集频率逆平方根权重，clip[0.5,2.5] | 线上分=66.6902 | NONE | 线上成绩记录表.md:392 |

### 4.7 B3+CE+Dice（Exp44，2026-09-02：官方损失三件套第三个，负收益−0.176）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| Exp44_CE_Dice_标准版 | B3 + CE+Dice 联合 | 线上分=66.5635 | NONE | 线上成绩记录表.md:407 |

### 4.8 B3+no-decay 校准重扫（Exp25v2，2026-09-02：盲区验证，顶点 69.2638 仍<CWFA锁定）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| 老版 barren380 | 3.8 | 线上分=69.2257 | EXP25V2_MODEL_SPECIFIC_RESCAN_OF_TRANSFERRED_STAGE23_VECTOR | 线上成绩记录表.md:421 |
| 老版 barren360 | 3.6 | 线上分=69.2124 | EXP25V2_MODEL_SPECIFIC_RESCAN_OF_TRANSFERRED_STAGE23_VECTOR | 线上成绩记录表.md:422 |
| lock_barren40 | 4.0 | 线上分=69.2638 | EXP25V2_MODEL_SPECIFIC_RESCAN_OF_TRANSFERRED_STAGE23_VECTOR | 线上成绩记录表.md:423 |
| lock_base | 3.6 | 线上分=69.2548 | EXP25V2_MODEL_SPECIFIC_RESCAN_OF_TRANSFERRED_STAGE23_VECTOR | 线上成绩记录表.md:424 |
| lock_barren42 | 4.2 | 线上分=69.2524 | EXP25V2_MODEL_SPECIFIC_RESCAN_OF_TRANSFERRED_STAGE23_VECTOR | 线上成绩记录表.md:425 |
| lock_barren45 | 4.5 | 线上分=69.2265 | EXP25V2_MODEL_SPECIFIC_RESCAN_OF_TRANSFERRED_STAGE23_VECTOR | 线上成绩记录表.md:426 |

### 4.9 B3+边界分支+CWFA 叠加（Exp46，2026-09-02：两个正收益互相摧毁 −0.450）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| Exp46_边界CWFA_标准版 | B3 + 边界分支BCE + CWFA | 线上分=66.2893 | NONE | 线上成绩记录表.md:437 |

### 4.10 Exp43 Focal + Exp45 MixUp（2026-09-03：官方三件套全负结案 + MixUp 第6个正收益）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| Exp43_Focal_标准版 | B3 + Focal Loss | 线上分=66.397 | NONE | 线上成绩记录表.md:451 |
| Exp45_MixUp_标准版 | B3 + 温和 MixUp | 线上分=67.0098 | NONE | 线上成绩记录表.md:452 |

### 4.11 补做 TTA8+锁定校准版（Exp42-47，2026-09-03：双层判决完整，无超锁定）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| Exp42 加权CE | 66.690 | 标准版=66.69; TTA8+锁定校准=68.303 | STAGE23_VECTOR_REUSED_NOT_MODEL_SPECIFIC8D_SCAN | 线上成绩记录表.md:461 |
| Exp43 Focal | 66.397 | 标准版=66.397; TTA8+锁定校准=67.9802 | STAGE23_VECTOR_REUSED_NOT_MODEL_SPECIFIC8D_SCAN | 线上成绩记录表.md:462 |
| Exp44 CE+Dice | 66.564 | 标准版=66.564; TTA8+锁定校准=68.4128 | STAGE23_VECTOR_REUSED_NOT_MODEL_SPECIFIC8D_SCAN | 线上成绩记录表.md:463 |
| Exp45 MixUp | 67.010 | 标准版=67.01; TTA8+锁定校准=69.1399 | STAGE23_VECTOR_REUSED_NOT_MODEL_SPECIFIC8D_SCAN | 线上成绩记录表.md:464 |
| Exp46 边界+CWFA | 66.289 | 标准版=66.289; TTA8+锁定校准=68.1674 | STAGE23_VECTOR_REUSED_NOT_MODEL_SPECIFIC8D_SCAN | 线上成绩记录表.md:465 |
| Exp47 SEMEDA | 67.796 | 标准版=67.796; TTA8+锁定校准=69.0776 | STAGE23_VECTOR_REUSED_NOT_MODEL_SPECIFIC8D_SCAN | 线上成绩记录表.md:466 |

### 三合一校准版（补充）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| B3+CWFA+no-decay 三合一 | 67.691 | 校准版=67.691 | TRANSFERRED_B3_CALIBRATION_NOT_MODEL_SPECIFIC8D_SCAN | 线上成绩记录表.md:541 |

### 八、Late-CWFA staged screening 线上分（2026-09-10，Target Transfer Audit）

| 方案 | 参数 | 线上分(百分数) | 校准来源 | 证据行 |
|---|---|---|---|---|
| C7_TTA8 | step7000 · TTA8-only | 线上分=67.9246 | NONE | 线上成绩记录表.md:559 |
| C7_TTA8_FIXED8D | step7000 · +冻结8D | 线上分=69.0699 | STAGE23_FINAL_VECTOR_FROZEN_TRANSFER | 线上成绩记录表.md:560 |
| C8_TTA8 | step8000 · TTA8-only | 线上分=67.8534 | NONE | 线上成绩记录表.md:561 |
| C8_TTA8_FIXED8D | step8000 · +冻结8D | 线上分=69.121 | STAGE23_FINAL_VECTOR_FROZEN_TRANSFER | 线上成绩记录表.md:562 |

## 其他推理变体、无分原因及建议

### Exp18 horizontal flip TTA
- 状态：TRIED_DELTA_ONLY；模型：Exp18 B3
- 协议：Historical summary says hflip; distinct from4-viewHV and D4eight
- 分数：{"delta_pp": 0.275, "absolute_percent": null}
- 校准来源：NONE
- 无分/限制：Only +0.275 summary; exact independent score missing in scope
- 证据：method_effect_ledger.csv:4 旋转TTA8,测试时推理增广(旋转8视图),B3 66.739,B3+rotTTA8 67.210,,0.47,N/A,8视图(0/90/180/270x flip),UNKNOWN,B3,1024,+校准=TTA层已顶,CONFIRMED_USEFUL,HIGH,翻转TTA的升级版
- 证据：method_effect_ledger.csv:5 翻转TTA,测试时推理增广(水平翻转),B3 66.739,flipTTA,,0.275,N/A,hflip,UNKNOWN,B3,1024,被旋转TTA8取代,CONFIRMED_USEFUL,MEDIUM,"曾误判全负,实为净正"
- 证据：method_effect_ledger.csv:19 多尺度TTA,测试时多尺度输入,B3,multi-scale,,全负,N/A,UNKNOWN,UNKNOWN,B3,1024,被简单翻转TTA取代,IMPLEMENTATION_FAILURE,HIGH,复杂版失败但简单D4/TTA8正收益

### Exp18 HV flip4 variants
- 状态：ARTIFACTS_REPORTED_SCORE_UNKNOWN；模型：Exp18 B3
- 协议：orig/hflip/vflip/hflip+vflip; standard/prior3.0 plus prior2.5/3.5/4.0; hflip-only standard/prior3.0
- 分数：null
- 校准来源：EXP18_PRIOR_SHIFT
- 无分/限制：Artifact inventory but no exact per-variant online score
- 证据：raw_master_exp15_25_partial.md:4 > 统一超参(所有行config验证)：lr=6e-05, wd=0.01, AdamW, poly power1.0, bs2×accum2, amp, planned_epochs=40, steps/epoch1400, max_steps=56000, seed42, crop1024, rare_crop p0.5 ids[7,4,6], domain_aug disabled, flip/jitter p0.5。预训练=/home/jhy/models/segformer-bX-ade512。
- 证据：raw_master_exp15_25_partial.md:48 - flip_tta.py: 4视图(orig/hflip/vflip/hflip+vflip), 含 PRIOR_LOG_BIAS=[0.0119,-0.0128,0.0197,0.2970,0.2455,-0.0356,0.7381,-0.2320]
- 证据：raw_master_exp15_25_partial.md:53 - **flip-TTA输出均来自Exp18(非Exp23)**: fliptta_b3_hv_standard/prior3.0, fliptta_b3_prior2.5/3.5/4.0, hflip_b3_standard/prior3.0 (各run_config记 checkpoint_sha256=66667268...88fd)
- 证据：raw_master_exp15_25_partial.md:54 - **冠军推理=Exp23 ckpt + rotta8 + prior-shift/校准**; flip-TTA仅在Exp18验证

### Exp23 independent flip TTA
- 状态：NOT_FOUND_IN_SCOPED_RECORDS；模型：Exp23
- 协议：Standalone flip control distinct from GoldenD4
- 分数：null
- 校准来源：NONE
- 无分/限制：Cannot borrow Exp18 flip gain as Exp23 measurement
- 证据：RAW_EVIDENCE_REPORT.md:28 - flip TTA独立提交在Exp18验证(非Exp23)；冠军推理=Exp23 ckpt(d3740b4e)+rotta8+8维校准。**无flip TTA在Exp23上的独立提交记录**。
- 证据：RAW_EVIDENCE_REPORT.md:36 - stage18_submission_standard/prior(.zip), exp23_submission_standard/rotta8_calib(.zip,含V1e=0,-0.0384,0.0709,1.0692,1.1048,0.1932,3.3215,-0.6960), flip/rotTTA目录族, exp23calib_* 73个校准zip。来源=/home/jhy ls -la。
- 证据：RAW_EVIDENCE_REPORT.md:52  /  冠军提交zip  /  /home/jhy/stage18_submission*, exp23_submission*, exp23calib_*(73), fliptta_b3_*, rotta_b3_*, hflip_b3_*  / 

### High-resolution/resize inference
- 状态：TRIED_SUMMARY_ONLY；模型：B3
- 协议：B3 nominal1024 upscale x1.5/x2.0 including2048; interpolation details UNKNOWN
- 分数：{"delta_pp_range": [-8.0, -4.2]}
- 校准来源：NONE
- 无分/限制：No exact absolute or per-variant score table
- 证据：线上成绩记录表.md:498  /  高分辨率推理  /  −4.2~−8.0  /  灾难  / 
- 证据：method_effect_ledger.csv:10 高分辨率推理,推理分辨率2048,B3,upscale,,-4.2,N/A,x1.5/x2.0,UNKNOWN,B3,1024,—,TRANSFER_FAILURE,HIGH,细路像素仅0.013%非真瓶颈;2048分布外
- 证据：失败方案深度归因与翻案清单.md:29  /  4  /  **高分辨率推理** (−4.2~−8.0)  /  细窄道路被 32× 下采样抹掉（79.9%）  /  细路像素占比仅 0.013%，对 mIoU 影响 <0.01；且 2048 是分布外输入  /  攻击了**真实但次要**的难点，且用错手段  /  死路确认（连通域分桶已定位真凶=中小目标误判背景）  / 
- 证据：失败方案深度归因与翻案清单.md:91 3. **判死不是终点，翻案要有数据依据**：翻案1（伪标签贴测试分布）、翻案2（约束CopyPaste）、翻案3（噪声鲁棒）是三个有实证依据的翻案，其余确认死路（高分辨率/色彩增广/注意力头/B4B5容量线）。

### Multiscale/complex TTA
- 状态：TRIED_SUMMARY_ONLY；模型：B3
- 协议：Multiple inference scales; exact scales/views/fusion UNKNOWN
- 分数：null
- 校准来源：NONE
- 无分/限制：Recorded all negative, no absolute scores; simpleD4 positive is different implementation
- 证据：线上成绩记录表.md:499  /  多尺度 TTA  /  全负  /  有害  / 
- 证据：失败方法根因复盘.md:17  /  5  /  多尺度TTA  /  尺度差异  /  全负  /  判死（后来发现翻转TTA其实+0.275，第一次判死判错了）  / 

### Sliding-window inference
- 状态：NOT_ESTABLISHED_IN_SCOPED_SOURCES；模型：UNKNOWN
- 协议：Window/stride/fusion UNKNOWN
- 分数：null
- 校准来源：NONE
- 无分/限制：No explicit variant+score; cannot infer tried or failed

### Test-time contrast normalization
- 状态：ANALYSIS_ONLY_ABANDONED；模型：UNKNOWN
- 协议：Dataset contrast assessment; transform params UNKNOWN
- 分数：null
- 校准来源：NONE
- 无分/限制：Ledger says no room(data evidence), abandoned; not a scored experiment
- 证据：线上成绩记录表.md:502  /  测试时对比度归一化  /  无空间(数据证)  /  放弃  / 

### CLAHE
- 状态：NOT_ESTABLISHED_IN_SCOPED_SOURCES；模型：UNKNOWN
- 协议：Specific protocol UNKNOWN
- 分数：null
- 校准来源：NONE
- 无分/限制：No explicit tested variant+online score in named sources; training jitter/MixStyle/sharpen/edge loss is not this inference experiment

### gamma correction
- 状态：NOT_ESTABLISHED_IN_SCOPED_SOURCES；模型：UNKNOWN
- 协议：Specific protocol UNKNOWN
- 分数：null
- 校准来源：NONE
- 无分/限制：No explicit tested variant+online score in named sources; training jitter/MixStyle/sharpen/edge loss is not this inference experiment

### test-time sharpen
- 状态：NOT_ESTABLISHED_IN_SCOPED_SOURCES；模型：UNKNOWN
- 协议：Specific protocol UNKNOWN
- 分数：null
- 校准来源：NONE
- 无分/限制：No explicit tested variant+online score in named sources; training jitter/MixStyle/sharpen/edge loss is not this inference experiment

### test-time style normalization/transfer
- 状态：NOT_ESTABLISHED_IN_SCOPED_SOURCES；模型：UNKNOWN
- 协议：Specific protocol UNKNOWN
- 分数：null
- 校准来源：NONE
- 无分/限制：No explicit tested variant+online score in named sources; training jitter/MixStyle/sharpen/edge loss is not this inference experiment

### soft boundary refinement / CRF / morphology
- 状态：NOT_ESTABLISHED_IN_SCOPED_SOURCES；模型：UNKNOWN
- 协议：Specific protocol UNKNOWN
- 分数：null
- 校准来源：NONE
- 无分/限制：No explicit tested variant+online score in named sources; training jitter/MixStyle/sharpen/edge loss is not this inference experiment

### Spatial-prior soft vehicle reassignment
- 状态：SUGGESTED_NOT_EXECUTED；模型：UNKNOWN
- 协议：GT A/B first; raw-logit second-best reassignment; change gate5-10% target pixels
- 分数：null
- 校准来源：NONE
- 无分/限制：Suggestion after failed60.034600 hard deletion, not tested replacement
- 证据：线上成绩记录表.md:385 - 空间先验并非整体判死，**硬规则版判死**。若要重审，必须：①在有真值训练/验证集做规则A/B；②用原始 logits/次高类软重分配；③像素改动门禁≤目标类5~10%；④图级先验只做置信度软抑制，不做像素硬删除。

### Small connected-component removal
- 状态：SUGGESTED_EXECUTION_NOT_ESTABLISHED；模型：UNKNOWN
- 协议：Components<N pixels relabel to surrounding majority; N unspecified
- 分数：null
- 校准来源：NONE
- 无分/限制：Proposal only, unlike actually executed vehicle-hard-deletion variant
- 证据：空间先验后处理方案_突破69.28关键.md:46 2. **细碎孤点清理**：连通域 <N 像素的预测块，重标为周边多数类。实证：中小目标系统性误判背景，细碎块=噪声。

### Water isolated-pixel cleanup
- 状态：SUGGESTED_EXECUTION_NOT_ESTABLISHED；模型：UNKNOWN
- 协议：Isolated single water pixels relabel to surrounding class
- 分数：null
- 校准来源：NONE
- 无分/限制：Proposal only; no submitted result found
- 证据：空间先验后处理方案_突破69.28关键.md:47 3. **水体连片约束**：孤立单像素水体重标为周边类。实证：水体是连片区域。

### Exp41 model-specific partial calibration rescan
- 状态：TRIED_SUMMARY_SCORE_ONLY；模型：Exp41
- 协议：background-.10/barren3.0; remaining coordinates UNKNOWN
- 分数：{"percent": 68.812, "precision": "rounded prose"}
- 校准来源：EXP41_PARTIAL_RESCAN_NOT_COMPLETE8D
- 证据：线上成绩记录表.md:362 3. **校准后 68.797** 仍不及锁定 69.281：边界分支模型专属校准最高也就 68.812（背景−0.10/荒地3.0），校准空间比 B3+CWFA 小。**标准版高≠校准空间大**——校准空间由 logits 的系统性偏置决定，CWFA 的偏置更大所以校准拉得动。

### EMA decay0.999 inference checkpoint comparison
- 状态：TRIED_DELTA_MODEL_ID_CONFLICT；模型：Exp10 B2/768
- 协议：EMA vsraw; source config inventory Exp10B2/768, derivative summary mislabelsB3/1024
- 分数：{"delta_pp": -0.445, "absolute_percent": null}
- 校准来源：NONE
- 无分/限制：Exact online absolute unknown; no synthesis from B3 baseline
- 证据：线上成绩记录表.md:484  /  EMA (Exp10)  /  −0.445  /  有害  / 
- 证据：raw_master_exp01_47_partial.md:27  /  10  /  B2  /  768  /  CE  /  rare+ema(.999)  /  0.787047@56000(raw>ema0.786823)  /  e7cbf65b  / 

### EMA decay0.99 /0.995
- 状态：SUGGESTED_NOT_EXECUTED；模型：UNKNOWN
- 协议：Alternative smoothing dose
- 分数：null
- 校准来源：NONE
- 无分/限制：Explicitly untried revisit proposal
- 证据：historical_method_audit_report.md:62 - **EMA**: decay0.999太钝≈没平滑（剂量错，0.99/0.995未试）
- 证据：historical_method_audit_report.md:66 - EMA锚定训练分布（时机/平滑钝）

### Exp35 dedicated calibration search
- 状态：SUGGESTED_NOT_EXECUTED；模型：Exp35
- 协议：B4-specific calibration after67.972872TTA
- 分数：null
- 校准来源：PROPOSED_MODEL_SPECIFIC
- 无分/限制：Hypothetical68.5+ prediction is not measured score
- 证据：线上成绩记录表.md:210 4. **校准空间未挖**：66.83 标准版的校准版还没扫（B4纯CE 校准到过 68.76）。按 B4 的校准规律（荒地抬更高），Exp35 校准后可能到 68.5+，但仍不会超过 B3 系的 69.28。**不值得花提交额度，除非复赛需要。**

### Exp54 sharpen training
- 状态：SCORED_TRAINING_NOT_INFERENCE；模型：Exp54
- 协议：ResumeExp49 training10000steps with sharpen
- 分数：{"percent": 66.8485}
- 校准来源：NONE
- 证据：raw_master_experiment_ledger.csv:55 exp54,stage54,exp54_sharpen_ce,configs/stage54,81d0e357,B3,1024,CE(锐化resume exp49),"10000步,cwfa start5000",6e-05,0.01,10000,42,0.803651,10000,66.8485,UNKNOWN,exp49,NO,NO,NO,server

### Stage55b calibration residual/rescan
- 状态：HISTORICAL_MENTION_EXACT_SCAN_UNKNOWN；模型：Stage55b
- 协议：Report mentions pseudo-training followed by residual8D; exact scan variants/bias/results missing here
- 分数：null
- 校准来源：MODEL_CALIBRATION_MENTIONED_DETAILS_UNKNOWN
- 无分/限制：Do not globally say never scanned; per-submission linkage not closed
- 证据：historical_method_audit_report.md:18  /  6  /  伪标签(贴测试分布,stage55b)  /  +0.68  /  复验(stage56/62续)  /  CONFIRMED(翻案版)  / 
- 证据：historical_method_audit_report.md:91 - **伪标签(贴测试) + 8维校准(残差级)**: 训练期注入×测试时先验，正交（stage55b已证可合但校准微降→校准降为残差）

### Exp49 calibration/rescan
- 状态：UNKNOWN_IN_THIS_SCOPED_AUDIT；模型：Exp49
- 协议：Core67.7711 not calibrated/rescan score
- 分数：null
- 校准来源：UNKNOWN_REQUIRE_RESCAN_SOURCES
- 无分/限制：Exact rescan table absent here; other evidence may establish actual scans, do not globally call unscanned
- 证据：raw_master_experiment_ledger.csv:51 stage49,stage49,exp49_maxent_l100,configs/stage49,db02f5dc,B3,1024,CE+MaxEnt,λ1.0,6e-05,0.01,56000,42,0.809339,UNKNOWN,67.7711,UNKNOWN,Exp23,NO,NO,NO,server

### Multi-model inference ensemble
- 状态：NOT_ESTABLISHED_IN_SCOPED_SOURCES；模型：UNKNOWN
- 协议：Multiple checkpoint-logit fusion unspecified
- 分数：null
- 校准来源：NONE
- 无分/限制：Exp39 training-method combination called 集成 is not multiple-model inference ensemble
- 证据：线上成绩记录表.md:311 ### 4.2 B3+蒸馏λ0.1+MixStyleα0.05 集成（Exp39，2026-09-01 判决：+0.177 微正，未过 CWFA）

## 数据质量与不可外推事项

- 线上分是历史本地账本记录，不是本次独立核验的平台回执；不自动实验。
- Stage23原始8D扫描与跨模型复用严格区分。B4/B5/Exp25v2/Exp41存在局部专属重扫例外；Exp49/Stage55b不能一概说没扫。
- 旧锁定建筑-.0384/植被.1932、中间-.0384/.4、最终+.1/.4三个向量不等价。FIXED8D_EXP41_STYLE是Stage23最终向量的历史别名，不证明源于Exp41搜索。
- 早期表只给部分坐标时不补造完整向量；不同阶段的扫描不是严格同底座factorial。
- 原表页眉69.278739/Aug30已过时，Sep1build_p10069.281290是后续值；附录待跑PID是旧记录非当前状态。
- 派生raw_online_submission_ledger.csv/calibration_raw_history.csv遗漏加粗最佳行并错抓14k标准/校准列；此输出直接解析原Markdown全部表，保留加粗分数。
- B4no-decay7.926311保留；其机制解释是历史归因推测，不是独立实验证明。
- Exp25v2老barren380同时写69.2257与生成未提交，保留状态冲突。
- LateCWFA四行fraction转换为百分数；C7fixed相对Exp47fixed=-.0077pp而非原文-.08pp，C8fixed相对Golden=-.16029pp；C7/C8不是Stage66。
- EMA原始config证据Exp10B2/768优先于二次汇总B3/1024；不从-.445倒推出不存在的绝对分。
- 提分构成锚点中的~68.0和+.6~.8是近似拆解，不是已证Exp23TTA-only独立提交分，禁止填成精确对照。
- CLAHE/gamma/滑窗等在指定范围未找到记录是UNKNOWN，不是全局未做证明。
- 空间规则方案文档是先期建议，后续60.034600硬删除失败记录优先；细碎连通域/水体清理未因此自动变为已试。
