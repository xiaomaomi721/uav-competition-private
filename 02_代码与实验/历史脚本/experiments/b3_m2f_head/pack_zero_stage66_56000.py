"""Stage66 step 56000 ZERO package, sourced from the run's own last.pt.

`checkpoint_step56000.pt` was deleted by the server reclamation pipeline, but
`last.pt` IS step 56000 and its SHA256 is attested by the training run itself:

    run_manifest.json -> "last_checkpoint_sha256":
        de6ffa3bf82ab507e731367ed7a784c4f61bb2ca21a5c89028ec42c68bc0f978
    last.pt (step=56000) -> same SHA256, recomputed on disk

This script does NOT reimplement the pipeline. It imports the already-verified
pack_zero_backfill module and registers one extra job, so the forward/package
code path is byte-for-byte the same one that produced the Stage63 / Stage66-52k
/ Stage66-51800 packages.
"""
import sys

sys.path.insert(0, '/home/jhy/rare_diag_20260915')
import pack_zero_backfill as pz

pz.JOBS['stage66_56000'] = (
    pz.S66 / 'config.yaml',
    pz.S66 / 'last.pt',
    56000,
    'de6ffa3bf82ab507e731367ed7a784c4f61bb2ca21a5c89028ec42c68bc0f978',
)

if __name__ == '__main__':
    mode, key = sys.argv[1], sys.argv[2]
    if key != 'stage66_56000':
        raise SystemExit(f'This driver only serves stage66_56000, got {key}')
    if mode == 'forward':
        pz.forward(key)
    elif mode == 'package':
        pz.package(key)
    else:
        raise SystemExit('Usage: pack_zero_stage66_56000.py forward|package stage66_56000')
