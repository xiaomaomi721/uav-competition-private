# NEXT_EXPERIMENT_FINAL_REPORT

**Status: FAILED — queue stopped before any new GPU smoke or formal training.**

## Why execution stopped
- Stage66 C completed 1399 images but exited 2: true FP32 versus FP16 precision gate FAIL.
- Plain maximum pixel disagreement: 0.0004100799560546875; threshold: 0.0001.
- Cache preserved, not certified for calibration. No thresholds relaxed and no rerun performed.
- Following the latest any-stage-failure rule, the independent experiment was NOT launched.

## Scheduling deviation
- The F65000 handoff window was missed. Existing pipeline started Stage66 C at09:48.
- At09:51:35 only identity-verified supervisor4134156 was SIGSTOPped. C was not signaled and ended naturally.
- Historical supervisor1178941 and supervisor4134156 remain stopped. No automatic resume.

## Full6996 delivered
- 69997/last tensor-deduplicated three variants plus65000 three variants: all six official checks and ZIP SHA PASS.
- See FULL6996_PRIORITY_PACKAGES_REPORT.md. No online submission.

## Next experiment prepared, NOT run
- Selected: Stage23 Golden + RareExposure v2, based on actual RareCrop1024 spatial no-op audit.
- Only sampling changes;4197uniform+700Barren+700Agri, within-stream no replacement, per-image cap3/epoch.
- All41 CPU index sequences match reference, fixed5597/1399 split, Stage23 settings,56000outersteps, fresh publishedpretrained.
- Optimizer backup audit passes but is not selected; no paramwise or warmup added.
- Source/config/dataset SHA, allowlisted single-block diff, real loader/collate, determinism and no-retry CPU tests PASS.
- GPU smoke NOT_STARTED; formal attempts0; no new metrics/checkpoints/ZIPs.
- No detached successor training runner is armed. Prepared bundle retained for a future decision.

## Machine-readable evidence

```json
{
  "status": "FAILED",
  "at": "2026-09-12T10:02:57.658952",
  "scope": "NEXT_EXPERIMENT_UNATTENDED_QUEUE_STOPPED_BEFORE_GPU_SMOKE",
  "reason": "STAGE66_C_FP16_PRECISION_GATE_FAILED_NO_PREREGISTERED_SAFE_RECOVERY",
  "stage66_exit_code": 2,
  "stage66_raw_wait_status": 512,
  "stage66_n_images": 1399,
  "stage66_precision_status": "FAIL",
  "stage66_max_plain_disagreement": 0.0004100799560546875,
  "threshold": 0.0001,
  "stage66_cache_preserved": true,
  "stage66_cache_use_allowed": false,
  "stage66_receipt": "/home/jhy/gpu2_day_queue/tta_stage66_partial_v2/TTA_ANATOMY.json",
  "stage66_receipt_sha256": "9f9292da5863274c974674c9085d9f5e81d7dbaefc04a8fbc9f4a89211c5dee3",
  "schedule_deviation": "MISSED_F65000_INTERCEPT_WINDOW_STAGE66_C_STARTED_AT_09_48; paused_exact_supervisor_only_09_51_35_child_finished_naturally",
  "selected_experiment": "Stage23 Golden + RareExposure v2",
  "rare_crop_audit": "PASS_SPATIAL_NO_OP_CONFIRMED_1000_ACTUAL_PAIRS",
  "bounded_sampling_gate": "PASS_41_EPOCHS_HARD_CAP_3",
  "optimizer_gate": "PASS_BACKUP_ONLY_NOT_SELECTED",
  "only_changed_variable": "training sampling/exposure;4197 uniform distinct +700 Barren distinct +700 Agricultural distinct per5597 epoch",
  "baseline": "Stage23 Golden; fixed5597/1399;publishedpretrained;56000outersteps;notFull6996",
  "cpu_preflight": "PASS",
  "entry_contract": "/home/jhy/gpu2_day_queue/next_experiment/bundle_v1/ENTRY_CONTRACT.json",
  "entry_contract_sha256": "cccda9c191402bb0d4a27396a9437cfc1ae09eb812a05a641775fbcae9da65b0",
  "cpu_preflight_path": "/home/jhy/gpu2_day_queue/next_experiment/bundle_v1/prepared/CPU_PREFLIGHT.json",
  "cpu_preflight_sha256": "fbf2f377bf7cd7fdbec84cc9d064d6eaaf9eee6228b9d2751073acc60058d3c7",
  "gpu_smoke": "NOT_STARTED",
  "formal_training": "NOT_STARTED",
  "formal_attempts": 0,
  "successor_production_runner": "NOT_DEPLOYED_OR_LAUNCHED; local draft only, no armed automatic training",
  "new_checkpoint": null,
  "new_validation": null,
  "new_test_packages": [],
  "priority_full6996_packages": "6_PASS",
  "priority_report": "/home/jhy/gpu2_day_queue/FULL6996_PRIORITY_PACKAGES_REPORT.md",
  "gpu2_process_query": "",
  "automatic_submission": false,
  "automatic_second_experiment": false,
  "old_supervisors": "1178941 and4134156 remain T; do not resume automatically",
  "deferred": [
    "Stage66 D calibration blocked by FP16 gate",
    "Full6996 60000/55000 packages",
    "next experiment GPU smoke/formal"
  ],
  "recovery": "NONE_REGISTERED; keep GPU2 idle; no threshold changes, no checkpoint/source/cache deletion"
}
```
