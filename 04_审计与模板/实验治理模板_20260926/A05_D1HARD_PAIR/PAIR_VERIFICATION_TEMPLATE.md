# PAIR VERIFICATION REPORT

## Identity

- pair_id:
- control run_id:
- treatment run_id:
- replay manifest SHA:
- records SHA:
- verifier path/SHA:

## Structural config diff

Allowed differences only:

```text
run_id
role
output_dir
target coefficient
```

Unexpected differences: `NONE` required.

## Loop-by-loop stream verification

Verify all 2,000 attempted loops:

- source stem;
- source transform seed/crop metadata;
- source tensor SHA;
- source label SHA;
- target filename;
- D4 index;
- all strong-photo scalars and blur flag;
- noise seed/tensor SHA;
- raw and augmented target SHA;
- teacher cache file SHA;
- pseudo-label mask SHA and pixel count.

| Check | Matching loops | Required | Result |
|---|---:|---:|---|
| Source identity | | 2000 | |
| Source transform/tensor | | 2000 | |
| Target identity | | 2000 | |
| D4/photo/noise | | 2000 | |
| Target tensor | | 2000 | |
| Teacher cache identity | | 2000 | |

The pseudo-label mask may diverge only if its construction depends on student output; in this recipe it is teacher-only and therefore must match exactly.

## Control branch proof

- target forward executed in control for all loops:
- unweighted hard CE calculated in control:
- nonzero hard coverage observed:
- weighted target contribution is exactly zero:
- no shortcut path detected:

## AMP and update accounting

- control attempted loops/successful updates/skip loops:
- treatment attempted loops/successful updates/skip loops:
- skip differences reported, not compensated:

## Verdict

`PAIR_MATCH_PASS / PAIR_MATCH_FAIL`

A failed pair cannot support target-loss causality or paired platform submission.
