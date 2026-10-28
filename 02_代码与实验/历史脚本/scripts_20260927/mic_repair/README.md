# MIC inference repair

This directory contains the additive replacement for the defective historical `_mic_infer.py`.

## Frozen intent

- Do not retrain.
- Load exactly one MIC TREAT step8000 checkpoint.
- Use the known-good A05 deployment protocol: numeric test2 names, eight `(rotation, horizontal_flip)` views, FP32 native-logit mean, Bstar once, bilinear 1024, argmax+1.
- Output basename must be inherited from the input basename.
- Require physical GPU2 and verify identities before output creation.
- Generate an input/output identity manifest and a separately audited FIXED ZIP.
- Never submit automatically.

## Files

- `mic_infer_fixed.py`: formal GPU inference entrypoint.
- `test_mic_infer_fixed.py`: CPU contracts and D4/name regression tests.
- `audit_submission_zip.py`: independent package auditor.

The historical source is retained unchanged at `../desktop_originals/_mic_infer.py` and is prohibited for future packages.
