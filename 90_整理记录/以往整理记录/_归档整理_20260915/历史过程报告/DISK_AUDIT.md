# Disk audit - 2026-09-14

Read-only first pass. No deletion/moving/GPU/signals. Only owned /home/jhy file content traversed, same-device, no symlink following; owned process FD inode checks only. nice19 / ionice class3 / thread env2.

## Result
Eligible files: **0**. Reclaimed bytes: **0**. 31 identity-verified ZIPs (137,282,345 bytes) remain blocked by incomplete handle visibility. See DISK_CLEANUP_PLAN.md and CANDIDATES.json.

## Filesystem snapshot
{
  "total_bytes": 3745119428608,
  "used_bytes": 3543697408000,
  "free_bytes_including_reserved": 201422020608,
  "available_bytes_to_user": 11103801344,
  "source": "statvfs /home/jhy; mounted filesystem, not home-only usage"
}

## Classification (mutually exclusive, logical bytes)
| Category | Files | Logical bytes | Protected | Unverified | Safe yes |
|---|---:|---:|---:|---:|---:|
| ZIP | 373 | 1658288870 | 486271648 | 1172017222 | 0 |
| other | 296398 | 43835222785 | 28965828626 | 14869394159 | 0 |
| smoke_other | 1018 | 3724825 | 3724825 | 0 | 0 |
| visualizations | 8627 | 8185405551 | 6935370599 | 1250034952 | 0 |
| checkpoints_weights | 346 | 98194943952 | 98194943952 | 0 | 0 |
| logits_arrays_cache | 3827 | 52061302503 | 52061302503 | 0 | 0 |
| diagnostics_other | 23921 | 18506279395 | 18489760731 | 16518664 | 0 |

## Top-level du snapshots
Allocated bytes, du -x --max-depth=1, no symlink following. All depth1 entries for home/S/Q2/Q3 retained in JSON.

### /home/jhy
- 223334342656 bytes: /home/jhy
- 56936833024 bytes: /home/jhy/deploy_exp18
- 47290478592 bytes: /home/jhy/gpu2_day_queue
- 41750466560 bytes: /home/jhy/deploy_stage05
- 13560664064 bytes: /home/jhy/datasets
- 7122239488 bytes: /home/jhy/.vscode-server
- 6377291776 bytes: /home/jhy/deploy_exp26
- 5586026496 bytes: /home/jhy/.venvs
- 5301194752 bytes: /home/jhy/deploy_exp25
- 4350971904 bytes: /home/jhy/deploy_exp27
- 2381737984 bytes: /home/jhy/deploy_exp24
- 2183540736 bytes: /home/jhy/models
- 2026688512 bytes: /home/jhy/.cache
- 1904173056 bytes: /home/jhy/deploy_p1p2
- 1848049664 bytes: /home/jhy/deploy_exp17
- 1179561984 bytes: /home/jhy/deploy_exp16

### /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1
- 56936828928 bytes: /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1
- 6727233536 bytes: /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/diagnostics
- 3961442304 bytes: /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage65_runs
- 2783191040 bytes: /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage36_runs
- 2648788992 bytes: /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/gate_probe_runs
- 2375249920 bytes: /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage37_runs
- 1560027136 bytes: /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage59_runs
- 1560018944 bytes: /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage66_runs
- 1024417792 bytes: /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage42_runs
- 996450304 bytes: /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage31_runs
- 982163456 bytes: /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage39_runs
- 844247040 bytes: /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage40_runs
- 818978816 bytes: /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage46_runs
- 799662080 bytes: /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage41_runs
- 798826496 bytes: /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage31_smoke_runs
- 797011968 bytes: /home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage52_runs

### /home/jhy/gpu2_day_queue
- 47290478592 bytes: /home/jhy/gpu2_day_queue
- 23482941440 bytes: /home/jhy/gpu2_day_queue/tta_stage66_partial_v2
- 19837759488 bytes: /home/jhy/gpu2_day_queue/tta_golden
- 1770840064 bytes: /home/jhy/gpu2_day_queue/next_experiment
- 1351413760 bytes: /home/jhy/gpu2_day_queue/full6996_refit
- 775757824 bytes: /home/jhy/gpu2_day_queue/smoke_full6996
- 65777664 bytes: /home/jhy/gpu2_day_queue/packages_partial_v2
- 1732608 bytes: /home/jhy/gpu2_day_queue/v2_builds
- 1146880 bytes: /home/jhy/gpu2_day_queue/partial_v2
- 839680 bytes: /home/jhy/gpu2_day_queue/pipeline_partial_v2
- 204800 bytes: /home/jhy/gpu2_day_queue/v2_cpu_tests
- 77824 bytes: /home/jhy/gpu2_day_queue/tta_stage66_smoke_partial_v2
- 28672 bytes: /home/jhy/gpu2_day_queue/__pycache__
- 28672 bytes: /home/jhy/gpu2_day_queue/tta_golden_smoke
- 24576 bytes: /home/jhy/gpu2_day_queue/phase_b_early_stop_20260912T091520
- 24576 bytes: /home/jhy/gpu2_day_queue/priority_handoff_replacement_20260912T090940

### /home/jhy/gpu3_paramwise01
- 797609984 bytes: /home/jhy/gpu3_paramwise01
- 795308032 bytes: /home/jhy/gpu3_paramwise01/runs
- 1568768 bytes: /home/jhy/gpu3_paramwise01/bundle
- 483328 bytes: /home/jhy/gpu3_paramwise01/queue
- 155648 bytes: /home/jhy/gpu3_paramwise01/postprocess
- 69632 bytes: /home/jhy/gpu3_paramwise01/gpu_identity_fix
- 20480 bytes: /home/jhy/gpu3_paramwise01/protocol_v2

## Limitations
- Live filesystem snapshot; measurements can change during audit.
- Allocated bytes are not guaranteed reclaimable bytes: open descriptors, hardlinks, snapshots/reflinks, and concurrent writes can affect df.
- All training directories and entire Q2/Q3 trees preserved; only standalone ZIP candidates selected.
- No diagnostic array/cache deletion verified. No unique visualization deletion verified.
- Reports themselves consume a small amount of space.
- 20/25 GiB cleanup goal cannot be reached from eligible verified files; do not relax protection.
- FD visibility incomplete, not evidence of active candidate opens. sd-pam1172952 denied;2142096 exited;3803206/4162826 coordinator-known zombies. No escalation.
- Initial metadata inventory contained334510 owned regular files; totals are snapshot measurements, not a complete provenance audit of every file.
- Most diagnostics/calibration storage is unconfirmed arrays; no array/checkpoint giant hashing or cleanup.
- Full home du includes directory metadata and differs from regular-file category totals.
- Classification by filename/path is conservative and mutually exclusive, not semantic proof of disposability.
