# A05 residual-D1 pair execution report

## 1. Administrative identity

- pair_id:
- run_id:
- role: `CONTROL / TREATMENT`
- final status:
- TAST GPU work performed: `NONE` required.
- B4 accessed or modified: `NO` required.
- automatic submission performed: `NO` required.

## 2. Frozen inputs

- A05 path/SHA:
- source split path/count/SHA:
- test2 manifest count/SHA:
- theta261 checkpoint/cache SHA and META audit:
- original theta278-export checkpoint/cache SHA and META audit:
- Bstar:
- replay manifest/records SHA:

## 3. Actual execution identity

- exact command:
- entrypoint/config/source SHA:
- host/PID/PPID/SID/cwd:
- Python/PyTorch/CUDA/cuDNN/driver:
- CUDA environment:
- physical GPU index/UUID/PCI/name:
- nvidia-smi compute PID binding:

## 4. Contract versus actual

List every field and every deviation. Only role/run/output/target coefficient may differ across pair members.

## 5. Training

- start/end/exit code:
- attempted loops: `2000` required for complete formal.
- successful optimizer updates:
- exact AMP skip loops:
- source/target batch sizes:
- optimizer groups/LRs/WD:
- trainable parameter names/count/hash:
- frozen parameters/buffers before/after:
- source CE and unweighted hard CE summaries:
- hard-mask coverage:
- weighted target contribution:
- NaN/Inf/OOM/gradient events:
- restart/resume history:

## 6. Pair replay

- loop receipts path/SHA:
- source/target/augmentation match result:
- control target-branch proof:
- `PAIR_VERIFICATION_REPORT.md` path/verdict:

## 7. Checkpoints

| Loop | Role | Successful updates | Path | SHA | Eligible for package |
|---:|---|---:|---|---|---|
| 1000 | audit | | | | NO |
| 2000 | primary | | | | YES |

No checkpoint selection, averaging, interpolation, or alpha scan.

## 8. Descriptive evaluation (never a gate)

- EVAL_PROTOCOL_V3 identity:
- H300/R1099/merged metrics:
- test2 no-GT behavior relative to A05:
- stagewise parameter drift:
- conclusions allowed/forbidden:

## 9. Package

- checkpoint count=1:
- D4 TTA8/FP32 mean/Bstar once:
- inference config SHA:
- PNG/ZIP audit:
- ZIP SHA:
- platform submission performed:

## 10. Failure and recovery

- failures/exceptions:
- root cause status:
- user authorization for recovery:
- artifacts invalidated:

## 11. Trust verdict

- required artifacts complete:
- unexpected deviations:
- final verdict: `TRUSTED_FOR_STATED_SCOPE / UNTRUSTED_DO_NOT_USE`.
- explicit statements:
  - no offline effect-based early stop;
  - no automatic retry;
  - no automatic submission;
  - no TAST GPU execution;
  - no B4 interaction.
