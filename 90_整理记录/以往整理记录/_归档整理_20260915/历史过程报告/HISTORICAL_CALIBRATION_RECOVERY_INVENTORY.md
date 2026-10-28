# HISTORICAL_CALIBRATION_RECOVERY_INVENTORY

**任务：HISTORICAL_POSITIVE_CALIBRATION_RECOVERY_V2**

## 分类汇总

 | 分类 | 数量 |
 |---|---:|
| RUN_NOW | 0 |
| QUEUE_TTA | 10 |
| RULE_BLOCKED | 2 |
| SKIP | 3 |

## 一览

| candidate | experiment_id | step | eligibility | reason |
|---|---|---:|---|---|
| Stage23_Golden_build_p100 | stage23_w1_exp23_b3_1024_cwfa_ce_formal_seed42 | 56000 | QUEUE_TTA | Exact complete FP32 full1399 cache unavailable in inspected candidate; Historical actual execution used AMP, so artifact identity does not certify new FP32 protocol |
| Exp47_standalone_SEMEDA | stage47_w1_exp47_b3_semeda_formal_seed42 | 56000 | QUEUE_TTA | Standalone identity verified, but complete val/test TTA8 logit cache not verified; Historical predictions/scores support inspection only, NOT RUN_NOW; new bias sweeps require proper logits |
| Stage66_best51800 | stage66_formal_b3_1024_ce_semeda_l05_rarecrop_cwfa_p03_e4_s28001_seed42 | 51800 | QUEUE_TTA | Distinct best51800 checkpoint; no verified re-biasable full cache for it |
| Stage66_checkpoint52000_C | stage66_formal_b3_1024_ce_semeda_l05_rarecrop_cwfa_p03_e4_s28001_seed42 | 52000 | QUEUE_TTA | SKIP existing FP16 cache for downstream calibration: explicit all1399 precision FAIL; Historical reference metrics remain inspectable without forward; Need authorized exact FP32 cache for new calibration |
| Stage66_checkpoint56000 | stage66_formal_b3_1024_ce_semeda_l05_rarecrop_cwfa_p03_e4_s28001_seed42 | 56000 | QUEUE_TTA | No verified re-biasable cache for exact56000 checkpoint; do not substitute best or52000 |
| Exp45 | stage45_w1_exp45_b3_mixup_formal_seed42 | 51800 | QUEUE_TTA | Historical core positive +0.2703; calibration gain is not core gain.; no complete provenance-verified TTA8 cache established in scoped audit. Original eligibility: POSITIVE_ONLINE_ELIGIBLE_CACHE_UNKNOWN |
| Exp34 | stage34_w1_exp34_b3_1024_mixstyle_ce_formal_seed42 | 56000 | QUEUE_TTA | Full56000 alpha0.1 positive +0.492; do not blacklist method using weaker14000-step alpha scans.; no complete provenance-verified TTA8 cache established in scoped audit. Original eligibility: POSITIVE_ONLINE_ELIGIBLE_CACHE_UNKNOWN |
| Exp31 | stage31_w1_exp31_b3_1024_dino2distill_ce_formal_seed42 | 47600 | QUEUE_TTA | Use submitted CLEAN checkpoint identity, not raw manifest best. Both hashes recomputed; clean reports strict_load true644 tensors. Tensor-level equivalence not independently compared.; no complete provenance-verified TTA8 cache established in scoped audit. Original eligibility: POSITIVE_ONLINE_ELIGIBLE_CACHE_UNKNOWN |
| Exp41_core | stage41_w1_exp41_b3_edge_branch_bce_formal_seed42 | 54600 | QUEUE_TTA | True no-TTA core +0.313456. exp41_submission_standard.zip misleadingly uses rot_tta.py/TTA8; true core metadata is exp41_true_standard_v2. FIXED8D_EXP41_STYLE cannot identify this core.; no complete provenance-verified TTA8 cache established in scoped audit. Original eligibility: POSITIVE_ONLINE_ELIGIBLE_CACHE_UNKNOWN |
| Exp49 | stage49_w1_exp49_b3_cwfa_ce_maxent_l100_seed42 | 56000 | QUEUE_TTA | Positive approximately+0.400 versus historical CWFA baseline. Effective configured lambda1, not method-name assumption; current build_loss wires CE-lambda*H to trainer criterion.; no complete provenance-verified TTA8 cache established in scoped audit. Original eligibility: POSITIVE_ONLINE_ELIGIBLE_CACHE_UNKNOWN |
| Exp48 | stage48_w1_exp48_b3_cwfa_ce_maxent_l010_seed42 | 53200 | SKIP | Below same-family CWFA baseline; does not exclude effective lambda1.; no complete provenance-verified TTA8 cache established in scoped audit. Original eligibility: EXCLUDE_FAILED_VERSION_ONLY |
| Exp50 | stage50_w1_exp50_b3_cwfa_ce_maxent_l500_seed42 | 56000 | SKIP | Below same-family CWFA baseline; does not exclude effective lambda1.; no complete provenance-verified TTA8 cache established in scoped audit. Original eligibility: EXCLUDE_FAILED_VERSION_ONLY |
| Exp52 | stage52_w1_exp52_b3_cwfa_maxent_l100_full6796_seed42 | 62863 | SKIP | 6796train/200val,67960steps differs from original5597/1399; do not auto-add as effective Exp49 lambda1 or conflate full6996.; no complete provenance-verified TTA8 cache established in scoped audit. Original eligibility: UNKNOWN_POSITIVE_NOT_ESTABLISHED_DIFFERENT_PROTOCOL |
| Stage55 | stage55_w1_exp55_short14k | 13592 | RULE_BLOCKED | No documented contest test/transductive-training permission closure. Published-pretrained init, not assumed resume; pseudo_train6796 train_ids,200val; CWFA starts7000.; no complete provenance-verified TTA8 cache established in scoped audit. Original eligibility: RULE_BLOCKED |
| Stage55b | stage55b_w1_exp55b_b3_cwfa_5597_pseudo500_seed42 | 54600 | RULE_BLOCKED | Historical positive cannot override absent documented contest rule closure. Actual manifest stage55b differs from directory stage56; published-pretrained init.; no complete provenance-verified TTA8 cache established in scoped audit. Original eligibility: RULE_BLOCKED |

## 各候选逐字段证据

UNKNOWN是未能闭环，不等于不存在或线上负效益。记录型线上分与平台原始回执分开。历史single-view/local/TTA协议之间不作伪matched delta。缓存manifest SHA不冒充全部payload SHA。

### Stage23_Golden_build_p100

| 字段 | 值 |
|---|---|
| candidate | Stage23_Golden_build_p100 |
| experiment_id | stage23_w1_exp23_b3_1024_cwfa_ce_formal_seed42 |
| method | B3 CWFA; historical test D4 TTA8 plus explicit final8D |
| checkpoint_path | /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage23_runs/stage23_w1_exp23_b3_1024_cwfa_ce_formal_seed42/best_miou.pt |
| checkpoint_step | 56000 |
| checkpoint_sha256 | d3740b4ef73fbb3ac57ddeedc3547ce4ce1ba31442dc7c70e1fe96b1fab3e228 |
| historical_local | {"miou": 0.810986, "n": 1399, "protocol": "historical single-view AMP zero-bias; not D4 FP32"} |
| historical_online | {"score_percent": 69.28129, "association": "HISTORICALLY_RECORDED_ASSOCIATION_CLOSED", "official_platform_receipt": "UNVERIFIED_NOT_FOUND_IN_INSPECTED_SOURCES", "ledger_evidence": "online_ledger"} |
| historical_inference_protocol | rot_tta.py: configured AMP forward; D4 4rot x2mirror, inverse-transform then bilinear align_corners_false to1024, mean logits, explicit bias added once, argmax+1. Explicit --prior-bias overrides default prior_scale. |
| old_bias | [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5] |
| val_tta8_logits_cache | /home/jhy/gpu2_day_queue/tta_golden/mean_logits_fp16 |
| test_tta8_logits_cache | NOT_IDENTIFIED_IN_INSPECTED_PATHS; PNG/ZIP are not logits |
| cache_dtype | float16 |
| cache_shape | [1024, 1024, 8] |
| cache_sha | {"full_payload_sha": "UNKNOWN_not_rehashed", "metadata_not_payload": {"inventory_sha256_recorded": "a8a0498fad79dabb4a03651ce23441d5259b01cfe4e10872a82b95fc8c979c3d"}} |
| reusable | False |
| gpu_forward_needed | YES, only future authorized QUEUE_TTA; not launched |
| eligibility | QUEUE_TTA |
| reason | Exact complete FP32 full1399 cache unavailable in inspected candidate; Historical actual execution used AMP, so artifact identity does not certify new FP32 protocol |

**证据：** {"sources": {"golden_run_config": {"path": "/home/jhy/calib_build_p100/run_config.yaml", "sha256_fresh": "51058e97ca123bcec1fc9544d0eeb1235aa2aeb98461c330463c2398f43d6de3", "excerpt": "experiment_id Stage23; checkpoint_sha256 d374...; n_views8; prior_shifttrue; prior_scale3.0. Does not record actual vector."}, "golden_execution_log": {"path": "/home/jhy/calib_road_build.log", "excerpts": ["71255 ===== [build_p100] bias=-0.2000 0.1000 0.0709 0.7000 3.6000 0.4000 3.3215 -0.5000 =====", "71777 zip_path /home/jhy/calib_build_p100.zip", "71779 sha256 6be4e8b0b15aaa4a002bb99d1b1fe0677ec9fb67fe1ee32ac4e0fda5d763ee04", "81433 DONE [build_p100]", "81434 same ZIP SHA256"]}, "golden_scan_script": {"path": "/home/jhy/run_calib_road_build_scan.sh", "sha256_fresh": "545e85824480ba29c7aeb039d2b5eef13fd2be342ed1e33aff97b5c8778fea72"}, "rot_tta_source": {"path": "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/scripts/rot_tta.py", "sha256_fresh": "7d0bdcc838695c8405389bff03e15bafc4cdce4d87cc01f005e4d56eca44804e", "excerpts": ["138-142 explicit prior_bias overrides prior_scale default", "143 AMP from experiment config", "175 autocast enabled=amp", "185 logits=accum/n_views", "187 logits+=bias_tensor", "188 argmax+1"]}, "golden_truth_local": {"path": "C:\\Users\\Lenovo\\OneDrive\\Documents\\桌面\\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\\文档归档\\04_实验决策与成绩\\historical_audit_20260909\\champion_69_281290_final_calibration_truth.json"}, "online_ledger": {"path": "C:\\Users\\Lenovo\\OneDrive\\Documents\\桌面\\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\\文档归档\\04_实验决策与成绩\\线上成绩记录表.md", "excerpts": ["line255 build_p100 | build+0.10 |69.281290", "line258 final vector bg-.2/build+.10/road.0709/water.70/barren3.6/vegetation.40/agri3.3215/vehicle-.50", "line466 Exp47 SEMEDA standard67.796 calibrated69.0776"], "authority_limit": "Local historical ledger, not official score receipt"}, "golden_partial_reference": {"path": "/home/jhy/gpu2_day_queue/partial_v2/reference/PHASE_B_REFERENCE_SUFFICIENT.json", "sha256": "45eafdbbad0cb256b09597dda6451516518b4b76cfd4b8ae5f5628de04f14840"}, "golden_precision_gate": {"path": "/home/jhy/gpu2_day_queue/partial_v2/gpu_gates/FP16_LOGIT_CACHE_PRECISION_AUDIT.json", "sha256": "63d697c4fce1cd6aa627a192f728213b8d2320f39ace9651583c59685cf2758e"}}, "cache_admission_details": {"path": "/home/jhy/gpu2_day_queue/tta_golden/mean_logits_fp16", "split": "val", "actual_file_count": 1182, "expected": 1399, "missing": 217, "sample_header_shape": [1024, 1024, 8], "dtype": "float16", "precision_gate": "PASS n16 only against fresh float32; calibration_allowed=false; full_phase_b_complete=false", "inventory_sha256_recorded": "a8a0498fad79dabb4a03651ce23441d5259b01cfe4e10872a82b95fc8c979c3d", "full_cache_content_rehashed": false}, "test_artifacts": {"zip": "/home/jhy/calib_build_p100.zip", "zip_sha256_fresh": "6be4e8b0b15aaa4a002bb99d1b1fe0677ec9fb67fe1ee32ac4e0fda5d763ee04", "predictions": "/home/jhy/calib_build_p100/predictions", "count": 500, "zip_root_names_exact_prediction_names": true, "all_zip_png_bytes_match_prediction_files": true, "reuse": "historical fixed predictions/identity analysis only; NOT RUN_NOW for new8Dscan and not no-autocast FP32 test logits"}}

### Exp47_standalone_SEMEDA

| 字段 | 值 |
|---|---|
| candidate | Exp47_standalone_SEMEDA |
| experiment_id | stage47_w1_exp47_b3_semeda_formal_seed42 |
| method | B3 CE+SEMEDA embedding_weight0.5, RareCrop enabled; CWFA false; published pretrained initialization |
| checkpoint_path | /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage47_runs/stage47_w1_exp47_b3_semeda_formal_seed42/best_miou.pt |
| checkpoint_step | 56000 |
| checkpoint_sha256 | 27e1441c008ab1b90eaa18bb45f44d6e9163f635ea544f1b6469a39cbd63d13e |
| historical_local | {"miou": 0.809568813669471, "n": 1399, "valid_pixels": 1445194394, "protocol": "training/evaluation single-view metric, not D4"} |
| historical_online | {"standard_percent": 67.796, "tta8_historical_bias_percent": 69.0776, "source": "online_ledger lines466/472; local historical Exp47 calibration truth", "official_platform_receipt": "UNVERIFIED", "tta8_unbiased_score": "UNKNOWN"} |
| historical_inference_protocol | Historical launch_exp47_package.sh standard predict_submission_edge.py plus rot_tta.py D4 AMP explicit historical bias; calibrated output is NOT TTA-only |
| old_bias | [-0.2, -0.0384, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5] |
| val_tta8_logits_cache | NOT_IDENTIFIED_IN_INSPECTED_PATHS |
| test_tta8_logits_cache | NOT_IDENTIFIED_IN_INSPECTED_PATHS |
| cache_dtype | UNKNOWN |
| cache_shape | UNKNOWN |
| cache_sha | {"full_payload_sha": "UNKNOWN_not_rehashed", "metadata_not_payload": {}} |
| reusable | False |
| gpu_forward_needed | YES, only future authorized QUEUE_TTA; not launched |
| eligibility | QUEUE_TTA |
| reason | Standalone identity verified, but complete val/test TTA8 logit cache not verified; Historical predictions/scores support inspection only, NOT RUN_NOW; new bias sweeps require proper logits |

**证据：** {"sources": {"exp47_run": {"path": "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage47_runs/stage47_w1_exp47_b3_semeda_formal_seed42", "files_read": ["config.yaml", "metrics.json", "completed.json"], "excerpts": ["metrics best_step56000; cwfa_enabledfalse; rare_crop_enabledtrue; loss_namesemeda"]}, "exp47_launch": {"path": "/home/jhy/launch_exp47_package.sh", "sha256_fresh": "670636c980146d6f3d2bd868ea3fcf25e0a9abd39884865fac14b0126cbdf068", "excerpt": "--prior-shift --prior-bias -0.2000 -0.0384 0.0709 0.7000 3.6000 0.4000 3.3215 -0.5000"}, "exp47_run_config": {"path": "/home/jhy/calibv2_exp47/run_config.yaml", "sha256_fresh": "4d4312a73965da51e5c6e588bd040dde31036e5cc6df9e2d90039d67ab5d70d2"}, "exp47_package_log": {"path": "/home/jhy/exp47_package.log", "excerpt": "standard checkpoint_step56000; calibrated views8 priorTrue; DONE artifacts with recorded ZIP SHA256"}, "exp47_truth_local": {"path": "C:\\Users\\Lenovo\\OneDrive\\Documents\\桌面\\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\\文档归档\\04_实验决策与成绩\\historical_audit_20260909\\exp47_calibration_truth.json"}, "online_ledger": {"path": "C:\\Users\\Lenovo\\OneDrive\\Documents\\桌面\\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\\文档归档\\04_实验决策与成绩\\线上成绩记录表.md", "excerpts": ["line255 build_p100 | build+0.10 |69.281290", "line258 final vector bg-.2/build+.10/road.0709/water.70/barren3.6/vegetation.40/agri3.3215/vehicle-.50", "line466 Exp47 SEMEDA standard67.796 calibrated69.0776"], "authority_limit": "Local historical ledger, not official score receipt"}, "rot_tta_source": {"path": "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/scripts/rot_tta.py", "sha256_fresh": "7d0bdcc838695c8405389bff03e15bafc4cdce4d87cc01f005e4d56eca44804e", "excerpts": ["138-142 explicit prior_bias overrides prior_scale default", "143 AMP from experiment config", "175 autocast enabled=amp", "185 logits=accum/n_views", "187 logits+=bias_tensor", "188 argmax+1"]}}, "cache_admission_details": {"val_tta8_logits": "NOT_IDENTIFIED_IN_INSPECTED_PATHS", "test_tta8_logits": "NOT_IDENTIFIED_IN_INSPECTED_PATHS", "historical_val_predictions": "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/diagnostics/semeda_vs_cwfa/exp47_val_pred", "historical_confusion": "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/diagnostics/semeda_vs_cwfa/conf47.npy", "note": "These hard predictions/confusion do not provide re-biasable D4 FP32 logits"}, "test_artifacts": {"standard_dir": "/home/jhy/submit_exp47", "calibrated_dir": "/home/jhy/calibv2_exp47", "count_recorded_each": 500, "standard_zip": "/home/jhy/submit_exp47.zip", "standard_zip_sha256_recorded": "1844574211e81a3b9bb202aafe606c013f336654fa8678daaab1270f411f853e", "calibrated_zip": "/home/jhy/calibv2_exp47.zip", "calibrated_zip_sha256_recorded": "39ac886011b9567184429183161fb20a373cec628812ef418245e90e01f4d74f"}}

### Stage66_best51800

| 字段 | 值 |
|---|---|
| candidate | Stage66_best51800 |
| experiment_id | stage66_formal_b3_1024_ce_semeda_l05_rarecrop_cwfa_p03_e4_s28001_seed42 |
| method | SEMEDA0.5+RareCrop, delayed CWFA p0.3 eps4 start28001 |
| checkpoint_path | /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage66_runs/stage66_formal_b3_1024_ce_semeda_l05_rarecrop_cwfa_p03_e4_s28001_seed42/best_miou.pt |
| checkpoint_step | 51800 |
| checkpoint_sha256 | c50d4133d35d690b9735d8710c6eeeb50844db5924394789a00f9ee7eaa03331 |
| historical_local | {"training_metric_miou": 0.8125200681968625, "per_checkpoint_csv_miou": 0.8125296965690411, "note": "Separate recorded evaluation outputs; do not erase difference or label either D4 without source verification"} |
| historical_online | {"score": "UNKNOWN_NOT_FOUND_IN_INSPECTED_RECORDS"} |
| historical_inference_protocol | Historical test record.json states D4,500; zero-bias and Golden fixed8D variants. Record binds checkpoint path but omits checkpoint hash and precision. |
| old_bias | [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5] |
| val_tta8_logits_cache | NOT_VERIFIED_FOR_THIS_CHECKPOINT |
| test_tta8_logits_cache | NOT_IDENTIFIED_IN_INSPECTED_PATHS; PNG/ZIP are not logits |
| cache_dtype | UNKNOWN |
| cache_shape | UNKNOWN |
| cache_sha | {"full_payload_sha": "UNKNOWN_not_rehashed", "metadata_not_payload": {}} |
| reusable | False |
| gpu_forward_needed | YES, only future authorized QUEUE_TTA; not launched |
| eligibility | QUEUE_TTA |
| reason | Distinct best51800 checkpoint; no verified re-biasable full cache for it |

**证据：** {"sources": {"stage66_run": {"path": "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage66_runs/stage66_formal_b3_1024_ce_semeda_l05_rarecrop_cwfa_p03_e4_s28001_seed42", "files_read": ["config.yaml", "metrics.json", "completed.json"], "run_config_sha256_fresh": "8ec70aeab18126225109d7a8c80e8244de470c7bc811436b120fe975a5cf6e19"}, "stage66_per_checkpoint_metrics": {"path": "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage66_runs/stage66_formal_b3_1024_ce_semeda_l05_rarecrop_cwfa_p03_e4_s28001_seed42/per_checkpoint_val_metrics.csv"}, "stage66_test_records": {"directories": ["/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/autopack_stage66", "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/latecwfa_submission_stage66"], "files_read": "Each named prediction package record.json; checkpoint paths, D4, explicit bias or null, n500"}}, "cache_admission_details": {"val_test_tta8_logits": "NOT_VERIFIED_FOR_THIS_CHECKPOINT", "must_not_substitute": "Stage66 C val cache belongs to52000, not51800"}, "test_artifacts": {"dirs": ["/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/autopack_stage66/stage66_best_TTA8", "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/autopack_stage66/stage66_best_TTA8_FIXED8D", "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/latecwfa_submission_stage66/localbest_C_TTA8", "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/latecwfa_submission_stage66/localbest_C_TTA8_FIXED8D"], "identity_limit": "Different historical package families not assumed byte-equivalent"}}

### Stage66_checkpoint52000_C

| 字段 | 值 |
|---|---|
| candidate | Stage66_checkpoint52000_C |
| experiment_id | stage66_formal_b3_1024_ce_semeda_l05_rarecrop_cwfa_p03_e4_s28001_seed42 |
| method | SEMEDA0.5+RareCrop, delayed CWFA p0.3 eps4 start28001 |
| checkpoint_path | /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage66_runs/stage66_formal_b3_1024_ce_semeda_l05_rarecrop_cwfa_p03_e4_s28001_seed42/checkpoint_step52000.pt |
| checkpoint_step | 52000 |
| checkpoint_sha256 | 311919e410d43cfaef5138570c754010a62a3c5b84c8a3a736af016157ce8afb |
| historical_local | {"per_checkpoint_csv_miou": 0.8125501791930202, "historical_true_fp32_d4_n": 1399, "historical_true_fp32_d4_zero_miou": 0.8131277295176083, "historical_true_fp32_d4_golden_bias_miou": 0.7876702342447008, "historical_true_fp32_d4_locked_gate_bias_miou": 0.7875682049879659, "reuse": "Historical metadata metrics only; NOT RUN_NOW; do not infer persisted exact logits or arbitrary new-bias confusion"} |
| historical_online | {"score": "UNKNOWN_NOT_FOUND_IN_INSPECTED_RECORDS"} |
| historical_inference_protocol | tta_anatomy_v2 D4 float32 no-autocast reference, eight canonicalized views; disk standard mean logits only, FP16 |
| old_bias | [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5] |
| val_tta8_logits_cache | /home/jhy/gpu2_day_queue/tta_stage66_partial_v2/mean_logits_fp16 |
| test_tta8_logits_cache | NOT_VERIFIED |
| cache_dtype | float16 |
| cache_shape | [1024, 1024, 8] |
| cache_sha | {"full_payload_sha": "UNKNOWN_not_rehashed", "metadata_not_payload": {"config_sha256_fresh": "611dcc5a53bbfcef5ad794355f91b94b1f496afcbc26554888c7769c61e3495c", "anatomy_sha256_fresh": "9f9292da5863274c974674c9085d9f5e81d7dbaefc04a8fbc9f4a89211c5dee3"}} |
| reusable | False |
| gpu_forward_needed | YES, only future authorized QUEUE_TTA; not launched |
| eligibility | QUEUE_TTA |
| reason | SKIP existing FP16 cache for downstream calibration: explicit all1399 precision FAIL; Historical reference metrics remain inspectable without forward; Need authorized exact FP32 cache for new calibration |

**证据：** {"sources": {"stage66_run": {"path": "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage66_runs/stage66_formal_b3_1024_ce_semeda_l05_rarecrop_cwfa_p03_e4_s28001_seed42", "files_read": ["config.yaml", "metrics.json", "completed.json"], "run_config_sha256_fresh": "8ec70aeab18126225109d7a8c80e8244de470c7bc811436b120fe975a5cf6e19"}, "stage66_per_checkpoint_metrics": {"path": "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage66_runs/stage66_formal_b3_1024_ce_semeda_l05_rarecrop_cwfa_p03_e4_s28001_seed42/per_checkpoint_val_metrics.csv"}, "stage66_anatomy": {"path": "/home/jhy/gpu2_day_queue/tta_stage66_partial_v2/TTA_ANATOMY.json", "sha256_fresh": "9f9292da5863274c974674c9085d9f5e81d7dbaefc04a8fbc9f4a89211c5dee3", "excerpts": ["checkpoint_step52000 hash311919...; n_images1399; gpu_diagnostics.autocast_enabledfalse", "mean_logits_cache float16 shape1024x1024x8 certifiedfalse", "fp16_precision_gate FAIL; downstream_cache_use_allowedfalse; exact_fallback_availablefalse"]}, "stage66_test_records": {"directories": ["/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/autopack_stage66", "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/latecwfa_submission_stage66"], "files_read": "Each named prediction package record.json; checkpoint paths, D4, explicit bias or null, n500"}}, "cache_admission_details": {"path": "/home/jhy/gpu2_day_queue/tta_stage66_partial_v2/mean_logits_fp16", "split": "val", "actual_count": 1399, "filenames_exact_split": true, "dtype": "float16", "sample_header_shape": [1024, 1024, 8], "per_file_hash_entries_recorded": 1399, "recorded_hash_names_exact_files": true, "all_payload_hashes_rechecked": false, "precision_status": "FAIL", "downstream_cache_use_allowed": false, "exact_fallback_available": false, "mean_abs_logit_error": 0.001203498908565354, "max_abs_logit_error": 0.03125, "zero_bias_max_image_disagreement": 0.0004100799560546875, "zero_bias_max_image_miou_delta": 0.0005037214067898921, "gate_threshold": 0.0001, "config_path_bound": "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/configs/stage66/stage66_formal.yaml", "config_sha256_fresh": "611dcc5a53bbfcef5ad794355f91b94b1f496afcbc26554888c7769c61e3495c", "anatomy_sha256_fresh": "9f9292da5863274c974674c9085d9f5e81d7dbaefc04a8fbc9f4a89211c5dee3"}, "test_artifacts": {"zero_bias_dir": "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/latecwfa_submission_stage66/s52000_C_TTA8", "fixed_bias_dir": "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/latecwfa_submission_stage66/s52000_C_TTA8_FIXED8D", "count_recorded": 500, "test_tta8_logits": "NOT_VERIFIED"}}

### Stage66_checkpoint56000

| 字段 | 值 |
|---|---|
| candidate | Stage66_checkpoint56000 |
| experiment_id | stage66_formal_b3_1024_ce_semeda_l05_rarecrop_cwfa_p03_e4_s28001_seed42 |
| method | SEMEDA0.5+RareCrop, delayed CWFA p0.3 eps4 start28001 |
| checkpoint_path | /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage66_runs/stage66_formal_b3_1024_ce_semeda_l05_rarecrop_cwfa_p03_e4_s28001_seed42/checkpoint_step56000.pt |
| checkpoint_step | 56000 |
| checkpoint_sha256 | f70c03eceb1ca560bed8c59dbb2b9d790391a01c0a6bc0823e17b2538a708aa2 |
| historical_local | {"per_checkpoint_csv_miou": 0.8124672938336803} |
| historical_online | {"score": "UNKNOWN_NOT_FOUND_IN_INSPECTED_RECORDS"} |
| historical_inference_protocol | Historical test record.json D4 zero-bias/Golden fixed8D500; precision and serialized hash not recorded in record.json |
| old_bias | [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5] |
| val_tta8_logits_cache | NOT_VERIFIED_FOR_THIS_CHECKPOINT |
| test_tta8_logits_cache | NOT_IDENTIFIED_IN_INSPECTED_PATHS; PNG/ZIP are not logits |
| cache_dtype | UNKNOWN |
| cache_shape | UNKNOWN |
| cache_sha | {"full_payload_sha": "UNKNOWN_not_rehashed", "metadata_not_payload": {}} |
| reusable | False |
| gpu_forward_needed | YES, only future authorized QUEUE_TTA; not launched |
| eligibility | QUEUE_TTA |
| reason | No verified re-biasable cache for exact56000 checkpoint; do not substitute best or52000 |

**证据：** {"sources": {"stage66_run": {"path": "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage66_runs/stage66_formal_b3_1024_ce_semeda_l05_rarecrop_cwfa_p03_e4_s28001_seed42", "files_read": ["config.yaml", "metrics.json", "completed.json"], "run_config_sha256_fresh": "8ec70aeab18126225109d7a8c80e8244de470c7bc811436b120fe975a5cf6e19"}, "stage66_per_checkpoint_metrics": {"path": "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage66_runs/stage66_formal_b3_1024_ce_semeda_l05_rarecrop_cwfa_p03_e4_s28001_seed42/per_checkpoint_val_metrics.csv"}, "stage66_test_records": {"directories": ["/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/autopack_stage66", "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/latecwfa_submission_stage66"], "files_read": "Each named prediction package record.json; checkpoint paths, D4, explicit bias or null, n500"}}, "cache_admission_details": {"val_test_tta8_logits": "NOT_VERIFIED_FOR_THIS_CHECKPOINT"}, "test_artifacts": {"zero_bias_dir": "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/latecwfa_submission_stage66/s56000_C_TTA8", "fixed_bias_dir": "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/latecwfa_submission_stage66/s56000_C_TTA8_FIXED8D", "note": "autopack last artifacts point to last.pt, not checkpoint_step56000.pt; weight equality not tested here"}}

### Exp45

| 字段 | 值 |
|---|---|
| candidate | Exp45 |
| experiment_id | stage45_w1_exp45_b3_mixup_formal_seed42 |
| method | MixUp alpha0.1 probability0.5 start1 |
| checkpoint_path | /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage45_runs/stage45_w1_exp45_b3_mixup_formal_seed42/best_miou.pt |
| checkpoint_step | 51800 |
| checkpoint_sha256 | 172d8512ae1406621a91e3d21d74300a9c75eeb9c47422e6741592db2045e34d |
| historical_local | {"miou": 0.8112808706598829, "protocol": "historical recorded evaluation; NOT new matched TTA8"} |
| historical_online | {"core_score_percent": 67.0098, "comparator": 66.739544, "comparator_name": "B3 no-TTA", "calibrated_score_percent": 69.1399, "origin": "Historical project ledger evidence, not independently verified platform receipts. Positive eligibility is versus recorded same-protocol historical baseline, NOT versus Golden69.281290; no new DeltaCore computed.", "platform_receipt": "UNKNOWN"} |
| historical_inference_protocol | Single RGB1024 view, ImageNet normalization, 8 classes, TTA=false and no bias for cited core score; 500 test images according to run_config.yaml |
| old_bias | {"cited_core_score_bias": [0, 0, 0, 0, 0, 0, 0, 0], "historical_calibrated_variant": [-0.2, -0.0384, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5]} |
| val_tta8_logits_cache | UNKNOWN |
| test_tta8_logits_cache | UNKNOWN |
| cache_dtype | UNKNOWN |
| cache_shape | UNKNOWN |
| cache_sha | UNKNOWN |
| reusable | False |
| gpu_forward_needed | YES if TTA8 logits required; none executed |
| eligibility | QUEUE_TTA |
| reason | Historical core positive +0.2703; calibration gain is not core gain.; no complete provenance-verified TTA8 cache established in scoped audit. Original eligibility: POSITIVE_ONLINE_ELIGIBLE_CACHE_UNKNOWN |

**证据：** {"sources": ["/home/jhy/submit_exp45/run_config.yaml", "/home/jhy/calibv2_exp45/run_config.yaml", "/home/jhy/run_calibv2.sh:16-27,45-48", "online_ledger:448-469", "checkpoint sibling run_manifest.json and metrics.json"], "ledger": "C:\\Users\\Lenovo\\OneDrive\\Documents\\桌面\\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\\文档归档\\04_实验决策与成绩\\线上成绩记录表.md", "raw_ledger": "C:\\Users\\Lenovo\\OneDrive\\Documents\\桌面\\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\\文档归档\\04_实验决策与成绩\\historical_audit_20260909\\raw_master_experiment_ledger.csv", "cache_coverage": "Bounded remote /home/jhy historical manifest/report filename sweep, scoped formal run directory artifact listings, scoped submission directory metadata and diagnostics cache candidate sweep. Found prediction PNGs and confusion_matrix.npy, not validated reusable raw logits. Not a global proof that no caches exist. No remote cache created.", "details": {"candidate": "Exp45", "method": "MixUp alpha0.1 probability0.5 start1", "experiment_id": "stage45_w1_exp45_b3_mixup_formal_seed42", "checkpoint_path": "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage45_runs/stage45_w1_exp45_b3_mixup_formal_seed42/best_miou.pt", "checkpoint_step": 51800, "checkpoint_sha256": "172d8512ae1406621a91e3d21d74300a9c75eeb9c47422e6741592db2045e34d", "historical_local": 0.8112808706598829, "historical_online": 67.0098, "online_baseline": 66.739544, "online_baseline_name": "B3 no-TTA", "eligibility": "POSITIVE_ONLINE_ELIGIBLE_CACHE_UNKNOWN", "reason": "Historical core positive +0.2703; calibration gain is not core gain.", "calibration_online": 69.1399, "calibration_bias": [-0.2, -0.0384, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5]}}

### Exp34

| 字段 | 值 |
|---|---|
| candidate | Exp34 |
| experiment_id | stage34_w1_exp34_b3_1024_mixstyle_ce_formal_seed42 |
| method | MixStyle alpha0.1 probability0.5 |
| checkpoint_path | /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage34_runs/stage34_w1_exp34_b3_1024_mixstyle_ce_formal_seed42/best_miou.pt |
| checkpoint_step | 56000 |
| checkpoint_sha256 | d19752b8f0cfdb7db39c84e7012a02b8246c43da4c8302e89bf3cd1fc74102d0 |
| historical_local | {"miou": 0.8091444104209777, "protocol": "historical recorded evaluation; NOT new matched TTA8"} |
| historical_online | {"core_score_percent": 67.231, "comparator": 66.739, "comparator_name": "B3 historical no-TTA baseline (rounded ledger)", "calibrated_score_percent": "UNKNOWN", "origin": "Historical project ledger evidence, not independently verified platform receipts. Positive eligibility is versus recorded same-protocol historical baseline, NOT versus Golden69.281290; no new DeltaCore computed.", "platform_receipt": "UNKNOWN"} |
| historical_inference_protocol | Single RGB1024 view, ImageNet normalization, 8 classes, TTA=false and no bias for cited core score; 500 test images according to run_config.yaml |
| old_bias | {"cited_core_score_bias": [0, 0, 0, 0, 0, 0, 0, 0], "historical_calibrated_variant": "UNKNOWN exact vector; prior_shift=true prior_scale3 recorded, not equivalent to explicit 8D vector"} |
| val_tta8_logits_cache | UNKNOWN |
| test_tta8_logits_cache | UNKNOWN |
| cache_dtype | UNKNOWN |
| cache_shape | UNKNOWN |
| cache_sha | UNKNOWN |
| reusable | False |
| gpu_forward_needed | YES if TTA8 logits required; none executed |
| eligibility | QUEUE_TTA |
| reason | Full56000 alpha0.1 positive +0.492; do not blacklist method using weaker14000-step alpha scans.; no complete provenance-verified TTA8 cache established in scoped audit. Original eligibility: POSITIVE_ONLINE_ELIGIBLE_CACHE_UNKNOWN |

**证据：** {"sources": ["/home/jhy/exp34_mixstyle_submission_standard/run_config.yaml", "/home/jhy/exp34_mixstyle_submission_rotta8calib/run_config.yaml", "online_ledger:235,530", "raw_online_ledger:35", "checkpoint sibling run_manifest.json and metrics.json"], "ledger": "C:\\Users\\Lenovo\\OneDrive\\Documents\\桌面\\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\\文档归档\\04_实验决策与成绩\\线上成绩记录表.md", "raw_ledger": "C:\\Users\\Lenovo\\OneDrive\\Documents\\桌面\\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\\文档归档\\04_实验决策与成绩\\historical_audit_20260909\\raw_master_experiment_ledger.csv", "cache_coverage": "Bounded remote /home/jhy historical manifest/report filename sweep, scoped formal run directory artifact listings, scoped submission directory metadata and diagnostics cache candidate sweep. Found prediction PNGs and confusion_matrix.npy, not validated reusable raw logits. Not a global proof that no caches exist. No remote cache created.", "details": {"candidate": "Exp34", "method": "MixStyle alpha0.1 probability0.5", "experiment_id": "stage34_w1_exp34_b3_1024_mixstyle_ce_formal_seed42", "checkpoint_path": "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage34_runs/stage34_w1_exp34_b3_1024_mixstyle_ce_formal_seed42/best_miou.pt", "checkpoint_step": 56000, "checkpoint_sha256": "d19752b8f0cfdb7db39c84e7012a02b8246c43da4c8302e89bf3cd1fc74102d0", "historical_local": 0.8091444104209777, "historical_online": 67.231, "online_baseline": 66.739, "online_baseline_name": "B3 historical no-TTA baseline (rounded ledger)", "eligibility": "POSITIVE_ONLINE_ELIGIBLE_CACHE_UNKNOWN", "reason": "Full56000 alpha0.1 positive +0.492; do not blacklist method using weaker14000-step alpha scans.", "calibration_bias": "UNKNOWN exact vector; prior_shift=true prior_scale3 recorded, not equivalent to explicit 8D vector"}}

### Exp31

| 字段 | 值 |
|---|---|
| candidate | Exp31 |
| experiment_id | stage31_w1_exp31_b3_1024_dino2distill_ce_formal_seed42 |
| method | DINOv2-large cosine feature distillation lambda1 start1 |
| checkpoint_path | /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage31_runs/stage31_w1_exp31_b3_1024_dino2distill_ce_formal_seed42/best_miou_clean.pt |
| checkpoint_step | 47600 |
| checkpoint_sha256 | bad30e00e7dd0b0e57964735156747eea06e689e37efa99e4ad0a9660d2d6356 |
| historical_local | {"miou": 0.8124134723397496, "protocol": "historical recorded evaluation; NOT new matched TTA8"} |
| historical_online | {"core_score_percent": 66.991, "comparator": 66.739, "comparator_name": "B3 historical no-TTA baseline (rounded ledger)", "calibrated_score_percent": "UNKNOWN", "origin": "Historical project ledger evidence, not independently verified platform receipts. Positive eligibility is versus recorded same-protocol historical baseline, NOT versus Golden69.281290; no new DeltaCore computed.", "platform_receipt": "UNKNOWN"} |
| historical_inference_protocol | Single RGB1024 view, ImageNet normalization, 8 classes, TTA=false and no bias for cited core score; 500 test images according to run_config.yaml |
| old_bias | {"cited_core_score_bias": [0, 0, 0, 0, 0, 0, 0, 0], "historical_calibrated_variant": "UNKNOWN exact vector; prior_shift=true prior_scale3 recorded"} |
| val_tta8_logits_cache | UNKNOWN |
| test_tta8_logits_cache | UNKNOWN |
| cache_dtype | UNKNOWN |
| cache_shape | UNKNOWN |
| cache_sha | UNKNOWN |
| reusable | False |
| gpu_forward_needed | YES if TTA8 logits required; none executed |
| eligibility | QUEUE_TTA |
| reason | Use submitted CLEAN checkpoint identity, not raw manifest best. Both hashes recomputed; clean reports strict_load true644 tensors. Tensor-level equivalence not independently compared.; no complete provenance-verified TTA8 cache established in scoped audit. Original eligibility: POSITIVE_ONLINE_ELIGIBLE_CACHE_UNKNOWN |

**证据：** {"sources": ["/home/jhy/exp31_distill_submission_standard/run_config.yaml", "/home/jhy/exp31_distill_submission_rotta8calib/run_config.yaml", "online_ledger:235", "raw_online_ledger:32", "checkpoint sibling run_manifest.json, metrics.json and config.yaml"], "ledger": "C:\\Users\\Lenovo\\OneDrive\\Documents\\桌面\\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\\文档归档\\04_实验决策与成绩\\线上成绩记录表.md", "raw_ledger": "C:\\Users\\Lenovo\\OneDrive\\Documents\\桌面\\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\\文档归档\\04_实验决策与成绩\\historical_audit_20260909\\raw_master_experiment_ledger.csv", "cache_coverage": "Bounded remote /home/jhy historical manifest/report filename sweep, scoped formal run directory artifact listings, scoped submission directory metadata and diagnostics cache candidate sweep. Found prediction PNGs and confusion_matrix.npy, not validated reusable raw logits. Not a global proof that no caches exist. No remote cache created.", "details": {"candidate": "Exp31", "method": "DINOv2-large cosine feature distillation lambda1 start1", "experiment_id": "stage31_w1_exp31_b3_1024_dino2distill_ce_formal_seed42", "checkpoint_path": "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage31_runs/stage31_w1_exp31_b3_1024_dino2distill_ce_formal_seed42/best_miou_clean.pt", "checkpoint_step": 47600, "checkpoint_sha256": "bad30e00e7dd0b0e57964735156747eea06e689e37efa99e4ad0a9660d2d6356", "raw_training_best_sha256": "54812df1699afef3fc2768749ae9d19d64a24021547d27f2ac2d0b01dde71c2a", "historical_local": 0.8124134723397496, "historical_online": 66.991, "online_baseline": 66.739, "online_baseline_name": "B3 historical no-TTA baseline (rounded ledger)", "eligibility": "POSITIVE_ONLINE_ELIGIBLE_CACHE_UNKNOWN", "reason": "Use submitted CLEAN checkpoint identity, not raw manifest best. Both hashes recomputed; clean reports strict_load true644 tensors. Tensor-level equivalence not independently compared.", "teacher_path": "/home/jhy/models/dinov2-large", "teacher_sha256": "UNKNOWN", "calibration_bias": "UNKNOWN exact vector; prior_shift=true prior_scale3 recorded"}}

### Exp41_core

| 字段 | 值 |
|---|---|
| candidate | Exp41_core |
| experiment_id | stage41_w1_exp41_b3_edge_branch_bce_formal_seed42 |
| method | B3 auxiliary edge branch BCE; NOT MixStyle and NOT a calibration-vector identity |
| checkpoint_path | /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage41_runs/stage41_w1_exp41_b3_edge_branch_bce_formal_seed42/best_miou.pt |
| checkpoint_step | 54600 |
| checkpoint_sha256 | 98cb41c0f40e75dfde3d265c87aac29d5429ce8f4202b6252603b67b71babb3b |
| historical_local | {"miou": 0.8118399199741564, "protocol": "historical recorded evaluation; NOT new matched TTA8"} |
| historical_online | {"core_score_percent": 67.053, "comparator": 66.739544, "comparator_name": "B3 true no-TTA", "calibrated_score_percent": 68.79692, "origin": "Historical project ledger evidence, not independently verified platform receipts. Positive eligibility is versus recorded same-protocol historical baseline, NOT versus Golden69.281290; no new DeltaCore computed.", "platform_receipt": "UNKNOWN"} |
| historical_inference_protocol | Single RGB1024 view, ImageNet normalization, 8 classes, TTA=false and no bias for cited core score; 500 test images according to run_config.yaml |
| old_bias | {"cited_core_score_bias": [0, 0, 0, 0, 0, 0, 0, 0], "historical_calibrated_variant": [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5]} |
| val_tta8_logits_cache | UNKNOWN |
| test_tta8_logits_cache | UNKNOWN |
| cache_dtype | UNKNOWN |
| cache_shape | UNKNOWN |
| cache_sha | UNKNOWN |
| reusable | False |
| gpu_forward_needed | YES if TTA8 logits required; none executed |
| eligibility | QUEUE_TTA |
| reason | True no-TTA core +0.313456. exp41_submission_standard.zip misleadingly uses rot_tta.py/TTA8; true core metadata is exp41_true_standard_v2. FIXED8D_EXP41_STYLE cannot identify this core.; no complete provenance-verified TTA8 cache established in scoped audit. Original eligibility: POSITIVE_ONLINE_ELIGIBLE_CACHE_UNKNOWN |

**证据：** {"sources": ["/home/jhy/exp41_true_standard_v2/run_config.yaml", "/home/jhy/exp41_pred_std/run_config.yaml", "/home/jhy/exp41_pred_calib/run_config.yaml", "/home/jhy/launch_exp41_package.sh", "online_ledger:345-351", "checkpoint sibling run_manifest.json and metrics.json"], "ledger": "C:\\Users\\Lenovo\\OneDrive\\Documents\\桌面\\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\\文档归档\\04_实验决策与成绩\\线上成绩记录表.md", "raw_ledger": "C:\\Users\\Lenovo\\OneDrive\\Documents\\桌面\\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\\文档归档\\04_实验决策与成绩\\historical_audit_20260909\\raw_master_experiment_ledger.csv", "cache_coverage": "Bounded remote /home/jhy historical manifest/report filename sweep, scoped formal run directory artifact listings, scoped submission directory metadata and diagnostics cache candidate sweep. Found prediction PNGs and confusion_matrix.npy, not validated reusable raw logits. Not a global proof that no caches exist. No remote cache created.", "details": {"candidate": "Exp41_core", "method": "B3 auxiliary edge branch BCE; NOT MixStyle and NOT a calibration-vector identity", "experiment_id": "stage41_w1_exp41_b3_edge_branch_bce_formal_seed42", "checkpoint_path": "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage41_runs/stage41_w1_exp41_b3_edge_branch_bce_formal_seed42/best_miou.pt", "checkpoint_step": 54600, "checkpoint_sha256": "98cb41c0f40e75dfde3d265c87aac29d5429ce8f4202b6252603b67b71babb3b", "historical_local": 0.8118399199741564, "historical_online": 67.053, "online_baseline": 66.739544, "online_baseline_name": "B3 true no-TTA", "eligibility": "POSITIVE_ONLINE_ELIGIBLE_CACHE_UNKNOWN", "reason": "True no-TTA core +0.313456. exp41_submission_standard.zip misleadingly uses rot_tta.py/TTA8; true core metadata is exp41_true_standard_v2. FIXED8D_EXP41_STYLE cannot identify this core.", "tta8_no_bias_online": 67.480463, "tta8_comparator": 67.209674, "calibration_online": 68.79692, "calibration_bias": [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5], "calibration_bias_provenance": "Borrowed locked8D vector, not evidence of Exp41-trained calibration origin"}}

### Exp49

| 字段 | 值 |
|---|---|
| candidate | Exp49 |
| experiment_id | stage49_w1_exp49_b3_cwfa_ce_maxent_l100_seed42 |
| method | Effective MaxEnt lambda1 + delayed CWFA start28001 |
| checkpoint_path | /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage49_runs/stage49_w1_exp49_b3_cwfa_ce_maxent_l100_seed42/best_miou.pt |
| checkpoint_step | 56000 |
| checkpoint_sha256 | 5e5d4e719efe8943bc8af6a03cba99b527d905118731b3beb8c478517f826ee1 |
| historical_local | {"miou": 0.8093387203224842, "protocol": "historical recorded evaluation; NOT new matched TTA8"} |
| historical_online | {"core_score_percent": 67.7711, "comparator": 67.371405, "comparator_name": "B3 CWFA5597 standard", "calibrated_score_percent": "UNKNOWN", "origin": "Historical project ledger evidence, not independently verified platform receipts. Positive eligibility is versus recorded same-protocol historical baseline, NOT versus Golden69.281290; no new DeltaCore computed.", "platform_receipt": "UNKNOWN"} |
| historical_inference_protocol | Single RGB1024 view, ImageNet normalization, 8 classes, TTA=false and no bias for cited core score; 500 test images according to run_config.yaml |
| old_bias | {"cited_core_score_bias": [0, 0, 0, 0, 0, 0, 0, 0], "historical_calibrated_variant": "UNKNOWN exact variant; many exp49 calibration packages exist, none attributed to core score"} |
| val_tta8_logits_cache | UNKNOWN |
| test_tta8_logits_cache | UNKNOWN |
| cache_dtype | UNKNOWN |
| cache_shape | UNKNOWN |
| cache_sha | UNKNOWN |
| reusable | False |
| gpu_forward_needed | YES if TTA8 logits required; none executed |
| eligibility | QUEUE_TTA |
| reason | Positive approximately+0.400 versus historical CWFA baseline. Effective configured lambda1, not method-name assumption; current build_loss wires CE-lambda*H to trainer criterion.; no complete provenance-verified TTA8 cache established in scoped audit. Original eligibility: POSITIVE_ONLINE_ELIGIBLE_CACHE_UNKNOWN |

**证据：** {"sources": ["/home/jhy/exp49_maxent_l100_submission_standard/run_config.yaml", "raw_online_ledger:51", "checkpoint sibling run_manifest.json, metrics.json and config.yaml", "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/src/losses.py:75-107,201-209", "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/src/trainer.py:377,631"], "ledger": "C:\\Users\\Lenovo\\OneDrive\\Documents\\桌面\\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\\文档归档\\04_实验决策与成绩\\线上成绩记录表.md", "raw_ledger": "C:\\Users\\Lenovo\\OneDrive\\Documents\\桌面\\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\\文档归档\\04_实验决策与成绩\\historical_audit_20260909\\raw_master_experiment_ledger.csv", "cache_coverage": "Bounded remote /home/jhy historical manifest/report filename sweep, scoped formal run directory artifact listings, scoped submission directory metadata and diagnostics cache candidate sweep. Found prediction PNGs and confusion_matrix.npy, not validated reusable raw logits. Not a global proof that no caches exist. No remote cache created.", "details": {"candidate": "Exp49", "method": "Effective MaxEnt lambda1 + delayed CWFA start28001", "experiment_id": "stage49_w1_exp49_b3_cwfa_ce_maxent_l100_seed42", "checkpoint_path": "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage49_runs/stage49_w1_exp49_b3_cwfa_ce_maxent_l100_seed42/best_miou.pt", "checkpoint_step": 56000, "checkpoint_sha256": "5e5d4e719efe8943bc8af6a03cba99b527d905118731b3beb8c478517f826ee1", "historical_local": 0.8093387203224842, "historical_online": 67.7711, "online_baseline": 67.371405, "online_baseline_name": "B3 CWFA5597 standard", "eligibility": "POSITIVE_ONLINE_ELIGIBLE_CACHE_UNKNOWN", "reason": "Positive approximately+0.400 versus historical CWFA baseline. Effective configured lambda1, not method-name assumption; current build_loss wires CE-lambda*H to trainer criterion.", "calibration_bias": "UNKNOWN exact variant; many exp49 calibration packages exist, none attributed to core score"}}

### Exp48

| 字段 | 值 |
|---|---|
| candidate | Exp48 |
| experiment_id | stage48_w1_exp48_b3_cwfa_ce_maxent_l010_seed42 |
| method | MaxEnt lambda0.1 + CWFA |
| checkpoint_path | /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage48_runs/stage48_w1_exp48_b3_cwfa_ce_maxent_l010_seed42/best_miou.pt |
| checkpoint_step | 53200 |
| checkpoint_sha256 | 39eb76c5a3c8f015f43fc6864fe8d86bf6341fbb94e65179aac4665113c3cddc |
| historical_local | {"miou": 0.8119476670612262, "protocol": "historical recorded evaluation; NOT new matched TTA8"} |
| historical_online | {"core_score_percent": 67.1308, "comparator": 67.371405, "comparator_name": "UNKNOWN", "calibrated_score_percent": "UNKNOWN", "origin": "Historical project ledger evidence, not independently verified platform receipts. Positive eligibility is versus recorded same-protocol historical baseline, NOT versus Golden69.281290; no new DeltaCore computed.", "platform_receipt": "UNKNOWN"} |
| historical_inference_protocol | Single RGB1024 view, ImageNet normalization, 8 classes, TTA=false and no bias for cited core score; 500 test images according to run_config.yaml |
| old_bias | {"cited_core_score_bias": [0, 0, 0, 0, 0, 0, 0, 0], "historical_calibrated_variant": "UNKNOWN"} |
| val_tta8_logits_cache | UNKNOWN |
| test_tta8_logits_cache | UNKNOWN |
| cache_dtype | UNKNOWN |
| cache_shape | UNKNOWN |
| cache_sha | UNKNOWN |
| reusable | False |
| gpu_forward_needed | YES if TTA8 logits required; none executed |
| eligibility | SKIP |
| reason | Below same-family CWFA baseline; does not exclude effective lambda1.; no complete provenance-verified TTA8 cache established in scoped audit. Original eligibility: EXCLUDE_FAILED_VERSION_ONLY |

**证据：** {"sources": ["/home/jhy/exp48_maxent_l010_submission_standard/run_config.yaml", "raw_online_ledger:50", "checkpoint sibling run_manifest.json and metrics.json"], "ledger": "C:\\Users\\Lenovo\\OneDrive\\Documents\\桌面\\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\\文档归档\\04_实验决策与成绩\\线上成绩记录表.md", "raw_ledger": "C:\\Users\\Lenovo\\OneDrive\\Documents\\桌面\\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\\文档归档\\04_实验决策与成绩\\historical_audit_20260909\\raw_master_experiment_ledger.csv", "cache_coverage": "Bounded remote /home/jhy historical manifest/report filename sweep, scoped formal run directory artifact listings, scoped submission directory metadata and diagnostics cache candidate sweep. Found prediction PNGs and confusion_matrix.npy, not validated reusable raw logits. Not a global proof that no caches exist. No remote cache created.", "details": {"candidate": "Exp48", "method": "MaxEnt lambda0.1 + CWFA", "experiment_id": "stage48_w1_exp48_b3_cwfa_ce_maxent_l010_seed42", "checkpoint_path": "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage48_runs/stage48_w1_exp48_b3_cwfa_ce_maxent_l010_seed42/best_miou.pt", "checkpoint_step": 53200, "checkpoint_sha256": "39eb76c5a3c8f015f43fc6864fe8d86bf6341fbb94e65179aac4665113c3cddc", "historical_local": 0.8119476670612262, "historical_online": 67.1308, "online_baseline": 67.371405, "eligibility": "EXCLUDE_FAILED_VERSION_ONLY", "reason": "Below same-family CWFA baseline; does not exclude effective lambda1."}}

### Exp50

| 字段 | 值 |
|---|---|
| candidate | Exp50 |
| experiment_id | stage50_w1_exp50_b3_cwfa_ce_maxent_l500_seed42 |
| method | MaxEnt lambda5 + CWFA |
| checkpoint_path | /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage50_runs/stage50_w1_exp50_b3_cwfa_ce_maxent_l500_seed42/best_miou.pt |
| checkpoint_step | 56000 |
| checkpoint_sha256 | 79d2eeb4c0f388e8c74062dcf2d66308fd4f418a492cbc3fcd07fb9cfcffa09b |
| historical_local | {"miou": 0.8098133271368766, "protocol": "historical recorded evaluation; NOT new matched TTA8"} |
| historical_online | {"core_score_percent": 65.3961, "comparator": 67.371405, "comparator_name": "UNKNOWN", "calibrated_score_percent": "UNKNOWN", "origin": "Historical project ledger evidence, not independently verified platform receipts. Positive eligibility is versus recorded same-protocol historical baseline, NOT versus Golden69.281290; no new DeltaCore computed.", "platform_receipt": "UNKNOWN"} |
| historical_inference_protocol | Single RGB1024 view, ImageNet normalization, 8 classes, TTA=false and no bias for cited core score; 500 test images according to run_config.yaml |
| old_bias | {"cited_core_score_bias": [0, 0, 0, 0, 0, 0, 0, 0], "historical_calibrated_variant": "UNKNOWN"} |
| val_tta8_logits_cache | UNKNOWN |
| test_tta8_logits_cache | UNKNOWN |
| cache_dtype | UNKNOWN |
| cache_shape | UNKNOWN |
| cache_sha | UNKNOWN |
| reusable | False |
| gpu_forward_needed | YES if TTA8 logits required; none executed |
| eligibility | SKIP |
| reason | Below same-family CWFA baseline; does not exclude effective lambda1.; no complete provenance-verified TTA8 cache established in scoped audit. Original eligibility: EXCLUDE_FAILED_VERSION_ONLY |

**证据：** {"sources": ["/home/jhy/exp50_maxent_l500_submission_standard/run_config.yaml", "raw_online_ledger:52", "checkpoint sibling run_manifest.json and metrics.json"], "ledger": "C:\\Users\\Lenovo\\OneDrive\\Documents\\桌面\\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\\文档归档\\04_实验决策与成绩\\线上成绩记录表.md", "raw_ledger": "C:\\Users\\Lenovo\\OneDrive\\Documents\\桌面\\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\\文档归档\\04_实验决策与成绩\\historical_audit_20260909\\raw_master_experiment_ledger.csv", "cache_coverage": "Bounded remote /home/jhy historical manifest/report filename sweep, scoped formal run directory artifact listings, scoped submission directory metadata and diagnostics cache candidate sweep. Found prediction PNGs and confusion_matrix.npy, not validated reusable raw logits. Not a global proof that no caches exist. No remote cache created.", "details": {"candidate": "Exp50", "method": "MaxEnt lambda5 + CWFA", "experiment_id": "stage50_w1_exp50_b3_cwfa_ce_maxent_l500_seed42", "checkpoint_path": "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage50_runs/stage50_w1_exp50_b3_cwfa_ce_maxent_l500_seed42/best_miou.pt", "checkpoint_step": 56000, "checkpoint_sha256": "79d2eeb4c0f388e8c74062dcf2d66308fd4f418a492cbc3fcd07fb9cfcffa09b", "historical_local": 0.8098133271368766, "historical_online": 65.3961, "online_baseline": 67.371405, "eligibility": "EXCLUDE_FAILED_VERSION_ONLY", "reason": "Below same-family CWFA baseline; does not exclude effective lambda1."}}

### Exp52

| 字段 | 值 |
|---|---|
| candidate | Exp52 |
| experiment_id | stage52_w1_exp52_b3_cwfa_maxent_l100_full6796_seed42 |
| method | MaxEnt lambda1 full6796 different training protocol |
| checkpoint_path | /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage52_runs/stage52_w1_exp52_b3_cwfa_maxent_l100_full6796_seed42/best_miou.pt |
| checkpoint_step | 62863 |
| checkpoint_sha256 | bc4621edc366e7edefa20d7061948be725054941c97911f8113ef8b88047bceb |
| historical_local | {"miou": 0.8159307398683922, "protocol": "historical recorded evaluation; NOT new matched TTA8"} |
| historical_online | {"core_score_percent": 66.2421, "comparator": "UNKNOWN exact Exp51 comparator not audited", "comparator_name": "UNKNOWN", "calibrated_score_percent": "UNKNOWN", "origin": "Historical project ledger evidence, not independently verified platform receipts. Positive eligibility is versus recorded same-protocol historical baseline, NOT versus Golden69.281290; no new DeltaCore computed.", "platform_receipt": "UNKNOWN"} |
| historical_inference_protocol | Single RGB1024 view, ImageNet normalization, 8 classes, TTA=false and no bias for cited core score; 500 test images according to run_config.yaml |
| old_bias | {"cited_core_score_bias": [0, 0, 0, 0, 0, 0, 0, 0], "historical_calibrated_variant": "UNKNOWN"} |
| val_tta8_logits_cache | UNKNOWN |
| test_tta8_logits_cache | UNKNOWN |
| cache_dtype | UNKNOWN |
| cache_shape | UNKNOWN |
| cache_sha | UNKNOWN |
| reusable | False |
| gpu_forward_needed | YES if TTA8 logits required; none executed |
| eligibility | SKIP |
| reason | 6796train/200val,67960steps differs from original5597/1399; do not auto-add as effective Exp49 lambda1 or conflate full6996.; no complete provenance-verified TTA8 cache established in scoped audit. Original eligibility: UNKNOWN_POSITIVE_NOT_ESTABLISHED_DIFFERENT_PROTOCOL |

**证据：** {"sources": ["/home/jhy/exp52_maxentfull_submission_standard/run_config.yaml", "raw_online_ledger:54", "checkpoint sibling run_manifest.json and metrics.json"], "ledger": "C:\\Users\\Lenovo\\OneDrive\\Documents\\桌面\\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\\文档归档\\04_实验决策与成绩\\线上成绩记录表.md", "raw_ledger": "C:\\Users\\Lenovo\\OneDrive\\Documents\\桌面\\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\\文档归档\\04_实验决策与成绩\\historical_audit_20260909\\raw_master_experiment_ledger.csv", "cache_coverage": "Bounded remote /home/jhy historical manifest/report filename sweep, scoped formal run directory artifact listings, scoped submission directory metadata and diagnostics cache candidate sweep. Found prediction PNGs and confusion_matrix.npy, not validated reusable raw logits. Not a global proof that no caches exist. No remote cache created.", "details": {"candidate": "Exp52", "method": "MaxEnt lambda1 full6796 different training protocol", "experiment_id": "stage52_w1_exp52_b3_cwfa_maxent_l100_full6796_seed42", "checkpoint_path": "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage52_runs/stage52_w1_exp52_b3_cwfa_maxent_l100_full6796_seed42/best_miou.pt", "checkpoint_step": 62863, "checkpoint_sha256": "bc4621edc366e7edefa20d7061948be725054941c97911f8113ef8b88047bceb", "historical_local": 0.8159307398683922, "historical_online": 66.2421, "online_baseline": "UNKNOWN exact Exp51 comparator not audited", "eligibility": "UNKNOWN_POSITIVE_NOT_ESTABLISHED_DIFFERENT_PROTOCOL", "reason": "6796train/200val,67960steps differs from original5597/1399; do not auto-add as effective Exp49 lambda1 or conflate full6996."}}

### Stage55

| 字段 | 值 |
|---|---|
| candidate | Stage55 |
| experiment_id | stage55_w1_exp55_short14k |
| method | Pseudo short14k |
| checkpoint_path | /home/jhy/stage55_short_out/best_miou.pt |
| checkpoint_step | 13592 |
| checkpoint_sha256 | d65fbd96ac1ae2873b60cc6719726e41d8a629937aebd6add11a2277020b9bee |
| historical_local | {"miou": 0.7700222181356076, "protocol": "historical recorded evaluation; NOT new matched TTA8"} |
| historical_online | {"core_score_percent": "UNKNOWN", "comparator": "UNKNOWN", "comparator_name": "UNKNOWN", "calibrated_score_percent": "UNKNOWN", "origin": "Historical project ledger evidence, not independently verified platform receipts. Positive eligibility is versus recorded same-protocol historical baseline, NOT versus Golden69.281290; no new DeltaCore computed.", "platform_receipt": "UNKNOWN"} |
| historical_inference_protocol | Single RGB1024 view, ImageNet normalization, 8 classes, TTA=false and no bias for cited core score; 500 test images according to run_config.yaml |
| old_bias | {"cited_core_score_bias": [0, 0, 0, 0, 0, 0, 0, 0], "historical_calibrated_variant": "UNKNOWN"} |
| val_tta8_logits_cache | UNKNOWN |
| test_tta8_logits_cache | UNKNOWN |
| cache_dtype | UNKNOWN |
| cache_shape | UNKNOWN |
| cache_sha | UNKNOWN |
| reusable | False |
| gpu_forward_needed | YES if TTA8 logits required; none executed |
| eligibility | RULE_BLOCKED |
| reason | No documented contest test/transductive-training permission closure. Published-pretrained init, not assumed resume; pseudo_train6796 train_ids,200val; CWFA starts7000.; no complete provenance-verified TTA8 cache established in scoped audit. Original eligibility: RULE_BLOCKED |

**证据：** {"sources": ["/home/jhy/stage55_short_out/run_manifest.json", "/home/jhy/stage55_short_out/config.yaml", "/home/jhy/stage55_pseudo_submission_standard/run_config.yaml"], "ledger": "C:\\Users\\Lenovo\\OneDrive\\Documents\\桌面\\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\\文档归档\\04_实验决策与成绩\\线上成绩记录表.md", "raw_ledger": "C:\\Users\\Lenovo\\OneDrive\\Documents\\桌面\\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\\文档归档\\04_实验决策与成绩\\historical_audit_20260909\\raw_master_experiment_ledger.csv", "cache_coverage": "Bounded remote /home/jhy historical manifest/report filename sweep, scoped formal run directory artifact listings, scoped submission directory metadata and diagnostics cache candidate sweep. Found prediction PNGs and confusion_matrix.npy, not validated reusable raw logits. Not a global proof that no caches exist. No remote cache created.", "details": {"candidate": "Stage55", "method": "Pseudo short14k", "experiment_id": "stage55_w1_exp55_short14k", "checkpoint_path": "/home/jhy/stage55_short_out/best_miou.pt", "checkpoint_step": 13592, "checkpoint_sha256": "d65fbd96ac1ae2873b60cc6719726e41d8a629937aebd6add11a2277020b9bee", "historical_local": 0.7700222181356076, "historical_online": "UNKNOWN", "eligibility": "RULE_BLOCKED", "reason": "No documented contest test/transductive-training permission closure. Published-pretrained init, not assumed resume; pseudo_train6796 train_ids,200val; CWFA starts7000.", "train_split": "/home/jhy/datasets/pseudo_train_6796/train_ids.txt", "train_split_sha256": "56f4e28a752692feddee1487bd5cb1833d0dfdad096fddf2a3d6c3ab6aa1cfbb"}}

### Stage55b

| 字段 | 值 |
|---|---|
| candidate | Stage55b |
| experiment_id | stage55b_w1_exp55b_b3_cwfa_5597_pseudo500_seed42 |
| method | Pseudo5597real+500test + CWFA |
| checkpoint_path | /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage56_runs/stage56_w1_exp56_b3_cwfa_5597_pseudo500_seed42/best_miou.pt |
| checkpoint_step | 54600 |
| checkpoint_sha256 | a2adfb906d53d86b1f8ce2eb6bfa3bdb7c040e025bc187ee4165c09858ac78f2 |
| historical_local | {"miou": 0.8097902977110372, "protocol": "historical recorded evaluation; NOT new matched TTA8"} |
| historical_online | {"core_score_percent": 68.048, "comparator": 67.371405, "comparator_name": "UNKNOWN", "calibrated_score_percent": "UNKNOWN", "origin": "Historical project ledger evidence, not independently verified platform receipts. Positive eligibility is versus recorded same-protocol historical baseline, NOT versus Golden69.281290; no new DeltaCore computed.", "platform_receipt": "UNKNOWN"} |
| historical_inference_protocol | Single RGB1024 view, ImageNet normalization, 8 classes, TTA=false and no bias for cited core score; 500 test images according to run_config.yaml |
| old_bias | {"cited_core_score_bias": [0, 0, 0, 0, 0, 0, 0, 0], "historical_calibrated_variant": "Multiple variants; specific winning vector and score-link UNKNOWN; do not attach rescan baseline bias to core score"} |
| val_tta8_logits_cache | UNKNOWN |
| test_tta8_logits_cache | UNKNOWN |
| cache_dtype | UNKNOWN |
| cache_shape | UNKNOWN |
| cache_sha | UNKNOWN |
| reusable | False |
| gpu_forward_needed | YES if TTA8 logits required; none executed |
| eligibility | RULE_BLOCKED |
| reason | Historical positive cannot override absent documented contest rule closure. Actual manifest stage55b differs from directory stage56; published-pretrained init.; no complete provenance-verified TTA8 cache established in scoped audit. Original eligibility: RULE_BLOCKED |

**证据：** {"sources": ["/home/jhy/stage55b_pseudo_submit/run_config.yaml", "/home/jhy/stage55b_rotta8/run_config.yaml", "/home/jhy/stage55b_rescan_results.txt", "raw_online_ledger:56", "checkpoint sibling run_manifest.json and config.yaml"], "ledger": "C:\\Users\\Lenovo\\OneDrive\\Documents\\桌面\\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\\文档归档\\04_实验决策与成绩\\线上成绩记录表.md", "raw_ledger": "C:\\Users\\Lenovo\\OneDrive\\Documents\\桌面\\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\\文档归档\\04_实验决策与成绩\\historical_audit_20260909\\raw_master_experiment_ledger.csv", "cache_coverage": "Bounded remote /home/jhy historical manifest/report filename sweep, scoped formal run directory artifact listings, scoped submission directory metadata and diagnostics cache candidate sweep. Found prediction PNGs and confusion_matrix.npy, not validated reusable raw logits. Not a global proof that no caches exist. No remote cache created.", "details": {"candidate": "Stage55b", "method": "Pseudo5597real+500test + CWFA", "experiment_id": "stage55b_w1_exp55b_b3_cwfa_5597_pseudo500_seed42", "directory_alias": "stage56_w1_exp56_b3_cwfa_5597_pseudo500_seed42", "checkpoint_path": "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage56_runs/stage56_w1_exp56_b3_cwfa_5597_pseudo500_seed42/best_miou.pt", "checkpoint_step": 54600, "checkpoint_sha256": "a2adfb906d53d86b1f8ce2eb6bfa3bdb7c040e025bc187ee4165c09858ac78f2", "historical_local": 0.8097902977110372, "historical_online": 68.048, "online_baseline": 67.371405, "eligibility": "RULE_BLOCKED", "reason": "Historical positive cannot override absent documented contest rule closure. Actual manifest stage55b differs from directory stage56; published-pretrained init.", "train_split": "/home/jhy/datasets/pseudo_train_6796/train_ids_5597pseudo.txt", "train_split_sha256": "0fbcc0fca84b68a88bb978996a3d6312c39bb7968276d8e8762695c1831efccc", "calibration_bias": "Multiple variants; specific winning vector and score-link UNKNOWN; do not attach rescan baseline bias to core score"}}


## 磁盘与资源

- 当前约10.15GiB可用，正式训练仍在写入，未删除任何文件。
- 本轮审计报告预计1–10MiB，不复制logits、不产PNG/ZIP。
- 有可信现有缓存时，单core的133候选/1399image per-image confusion约91MiB原始；需独立预算，不把64MiB元数据预算误用成扫描预算。
- 未来补算原始FP32验证缓存43.71875GiB/core，500test缓存15.625GiB/core；合计59.34375GiB/core，仅logits，不含安全余量和出包。
- S1五个指定checkpoint如果全部同时保留新val+test缓存，原始logits合计296.71875GiB；这只是容量规划，不是获准批量生成，也不是本次CPU审计需求。
- 当前空间不足以直接新建单core全套FP32缓存；/ssd和/hdd无已确认的本项目可写目录，不自动使用。
- CPU共享资源非保留资源；如扫描获准，CUDA隐藏、nice19/ionice3、串行、1线程起步，避免给formal增加明显I/O竞争。

## 安全与范围

- GPU2 RareExposure/GPU3 Paramwise formal及其调度器均未修改、暂停或重启；此次历史审计未发送任何进程信号。
- 不补GPU TTA；QUEUE_TTA只登记。无训练、无删除、无自动线上提交。
- Stage59、Full6996和用户明确失败路线不进入本轮候选包生产。
- Pseudo/Target Adaptation未有规则闭环，保持RULE_BLOCKED。
- 未遍历其他用户目录；缓存缺失结论限本次明确审计范围。

## 后续

**RUN_NOW=0：没有启动8D CPU重扫。** 现有汇总指标/PNG可用于历史证据整理，但不能替代logits进入新bias扫描。已准备统一factory设计；不为制造进度而扫描部分或精度FAIL缓存。

## 配套证据文件

- `HISTORICAL_RECOVERY_S1_AUDIT.json`
- `s2_readonly_inventory_20260914.json`
- `HISTORICAL_GOLDEN_69281290_IDENTITY_RECOVERY.md`
- `HISTORICAL_STAGE66_REFERENCE_METRICS.md`
- `HISTORICAL_RECOVERY_RULE_AUDIT.md`
- `HISTORICAL_CALIBRATION_FACTORY_DESIGN_V2.md`
- `HISTORICAL_CALIBRATION_RECOVERY_EXECUTION_CONTRACT.md`
- `HISTORICAL_POSITIVE_CALIBRATION_RECOVERY_V2_RESOURCE_AUDIT.json`
- `HISTORICAL_RECOVERY_TRAINING_GUARD_02.json`
- `文档归档\04_实验决策与成绩\线上成绩记录表.md`
- `文档归档\04_实验决策与成绩\historical_audit_20260909\raw_master_experiment_ledger.csv`
