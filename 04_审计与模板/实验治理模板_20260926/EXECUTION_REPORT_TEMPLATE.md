# EXECUTION_REPORT

## 1. Identity

- run_id:
- pair_id:
- role:
- status: `COMPLETED / FAILED / INTERRUPTED / RESULT_UNTRUSTED_MISSING_EVIDENCE`
- request SHA256:
- method card SHA256:
- entry contract SHA256:
- prepared/executed/reported by:

## 2. Requested versus actual

| Field | Requested | Actual | Match |
|---|---|---|---|
| command | | | |
| entrypoint SHA | | | |
| source snapshot SHA | | | |
| config SHA | | | |
| parent checkpoint SHA | | | |
| source split SHA/count | | | |
| donor domain/manifest | | | |
| seed | | | |
| target updates | 8000 | | |
| trainable scope | stage0/1 | | |
| frozen scope | stage2/3/head | | |

List every unplanned diff; write `NONE` if none.

## 3. GPU and process identity

- host:
- PID / PPID / SID:
- cmdline:
- cwd:
- Python executable/version:
- PyTorch/CUDA/cuDNN/driver:
- `CUDA_DEVICE_ORDER`:
- `CUDA_VISIBLE_DEVICES`:
- logical CUDA device:
- physical GPU index/UUID/PCI bus/name:
- matching nvidia-smi compute PID evidence:
- other processes touched: `NONE` required.

## 4. Training execution

- start/end UTC:
- exit code:
- loop steps attempted/completed:
- successful optimizer updates:
- AMP skip count and exact steps:
- OOM count/details:
- NaN/Inf events:
- losses and LR summary:
- runtime and peak VRAM:
- stop/restart/resume events:
- automatic retry performed: `NO` required unless separately authorized.

## 5. Parameter and mechanism audit

- trainable parameter names/count/hash before/after:
- frozen parameter and buffer hash before/after:
- frozen drift: `NONE` required.
- original/styled loss both active and finite:
- beta/rho/warmup actually executed:
- donor self-pair violations:
- control/treatment matched-sequence receipt:

## 6. Checkpoints

| Step | Successful updates | Path | SHA256 | Metric protocol | Metrics | Retention lock |
|---:|---:|---|---|---|---|---|

Selected checkpoint:
Selection rule frozen before run:
Reason selection obeys rule:

## 7. Evaluation

- evaluator path/SHA:
- split path/SHA:
- ignore handling:
- valid/ignore pixel counts:
- merged confusion-derived mIoU:
- per-class metrics artifact:
- purpose: implementation/collapse/mechanism guard only; not a test2 ranker.

## 8. Test2 behavior and package

- inference protocol and SHA:
- checkpoints loaded at inference: `1` required.
- test2 GT used: `NO` required.
- behavior metrics (descriptive only):
- prediction count/names/size/mode/labels:
- ZIP path/SHA:
- package audit result:
- platform submission performed: `NO` unless user separately confirmed.

## 9. Failures, exceptions, and recovery

- failure stage:
- last trusted step/update:
- exception text/path:
- root cause verified or hypothesis only:
- recovery performed:
- recovery authorization:
- evidence invalidated by recovery:

## 10. Artifact index

List all contract, config, source diff, runtime logs, manifests, checkpoints, evaluations, predictions, package, hashes, failure and retention records.

## 11. Conclusion boundaries

Allowed conclusions:

Forbidden conclusions:

- Never claim test2 no-GT behavior is accuracy.
- Never claim source delta predicts test2.
- Never generalize this recipe result to an entire method family.
- Never call a result platform-bound without a receipt tied to ZIP SHA.

## 12. Trust verdict

- required artifacts complete:
- deviations:
- final verdict: `TRUSTED_FOR_STATED_SCOPE / UNTRUSTED_DO_NOT_USE`
- submission recommendation (not submission action):
- explicit statement: `NO AUTOMATIC SUBMISSION WAS PERFORMED`.
