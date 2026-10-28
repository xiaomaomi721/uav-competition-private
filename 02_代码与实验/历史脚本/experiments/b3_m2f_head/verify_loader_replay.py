"""Prove that fast-forwarding a DataLoader lands on the SAME batch.

_kd_train.py's resume path rebuilds the loaders with the same seed and calls
next() `consumed` times. That is only correct if the shuffled stream is a fixed
sequence AND rebuild+replay reproduces it exactly. This script tests exactly
that, on both streams, using the batch `stem` lists -- no model, no GPU.

Run with no GPU visible; this is a pure data check.
"""
import sys
from pathlib import Path

R = Path('/home/jhy/rare_diag_20260915')
SNAP = Path('/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1')
DATASET = Path('/home/jhy/datasets/2026-低空图像语义分割赛道-训练集')
CONFIG = SNAP / 'stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml'
SOFT = R / 'kd_soft_fp16'

sys.path.insert(0, str(R))
sys.path.insert(0, str(SNAP))
sys.path.insert(0, '/home/jhy/gpu2_day_queue')

import _kd_train_patched as K

# Every value here mirrors _kd_launch.sh exactly.
BATCH, WORKERS, SEED = 2, 4, 42
N_FF = 137          # what we pretend was consumed before the park
N_AFTER = 6         # how many batches we compare afterwards


def build(config, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    # _kd_train.py reads splits/train.txt against project_root=SNAP (the deploy
    # snapshot), not against outdir; only splits_test1.txt is written to outdir.
    (outdir / 'splits_train.txt').write_text(
        (SNAP / 'splits/train.txt').read_text(encoding='utf-8'), encoding='utf-8')
    test_names = sorted(p.stem for p in (SOFT / 'probs').glob('*.npy'))
    (outdir / 'splits_test1.txt').write_text('\n'.join(test_names), encoding='utf-8')
    real = K.build_loader(config, DATASET, SNAP, 'splits/train.txt', None,
                          DATASET / 'train' / 'images', DATASET / 'train' / 'masks',
                          batch_size=BATCH, workers=WORKERS, seed=SEED)
    test = K.build_loader(config, DATASET, outdir, 'splits_test1.txt', None,
                          DATASET / 'images', DATASET / 'train' / 'masks',
                          prob_dir=SOFT / 'probs', batch_size=BATCH, workers=WORKERS,
                          seed=SEED + 1, use_probs=True)
    return real, test


def stems(it, n):
    return [next(it)['stem'] for _ in range(n)]


config = K.load_config(CONFIG)
tmp = R / '_loader_replay_tmp'
import shutil
if tmp.exists():
    shutil.rmtree(tmp)

print(f'crop_size={config["crop_size"]}  batch={BATCH}  workers={WORKERS}  seed={SEED}')
fails = []

# ---- STAGE 0: does iter() position actually select a different stream? -------
# The resume design assumes the k-th iter(loader) yields a different permutation
# and that training uses the one from its own iter() call. If two iter() calls
# gave the same stream, the ordering fix in _kd_train.py would be pointless.
new = build(config, tmp)
a = next(iter(new[0]))['stem']
b = next(iter(new[0]))['stem']
same = (a == b)
print('\n[stage0] iter() position selects the stream')
print(f'  iter #1 first batch : {a}')
print(f'  iter #2 first batch : {b}')
print(f'  identical           : {same}  (must be False for the design to matter)')
if same:
    fails.append('stage0-iter-position-does-not-matter')
del new

for name, idx in (('real', 0), ('test', 1)):
    # ---- pass 1: the ground truth (an uninterrupted stream) ----
    new = build(config, tmp)
    it = iter(new[idx])
    stems(it, N_FF)                       # the batches that "already happened"
    truth = stems(it, N_AFTER)            # the batches that must come next
    del it, new

    # ---- pass 2: a fresh rebuild + fast-forward ----
    new = build(config, tmp)
    it = iter(new[idx])
    fwd = stems(it, N_FF)                 # the fast-forward
    got = stems(it, N_AFTER)              # what actually follows

    ok_seq = got == truth
    print(f'\n[{name}] fast-forward {N_FF} batches, then compare {N_AFTER}')
    print(f'  first fast-forwarded batch : {fwd[0]}')
    print(f'  first batch after replay   : {got[0]}')
    print(f'  expected                   : {truth[0]}')
    print(f'  sequence identical         : {ok_seq}')
    if not ok_seq:
        for i, (a, b) in enumerate(zip(got, truth)):
            if a != b:
                print(f'    MISMATCH at +{i}: {a} != {b}')
        fails.append(name)

    # ---- pass 3: does epoch rollover replay too? (6996/2 = 3498 batches) ----
    EP = 3498
    it2 = iter(new[idx])
    stems(it2, EP + 5)                    # cross the epoch boundary once
    truth2 = stems(it2, 3)
    del it2, new
    new = build(config, tmp)
    it2 = iter(new[idx])
    stems(it2, EP + 5)
    got2 = stems(it2, 3)
    print(f'  epoch rollover replay      : {got2 == truth2}')
    if got2 != truth2:
        fails.append(name + '-rollover')
    del it2, new

# ---- STAGE 2: reproduce the ENGINE CHAIN after a park ------------------------
# _kd_train.py draws a permutation for the one-shot gradient diagnostic BEFORE
# the training loop draws its own. The resume has to replay that same chain:
# iter() for the diagnostic, iter() again for training, then fast-forward.
# This checks the pair of streams the training loop actually consumes.
print('\n[stage2] full engine chain (diagnostic iter, then training iter, then FF)')
FF2 = 40
for name, idx in (('real', 0), ('test', 1)):
    new = build(config, tmp)
    it_diag = iter(new[idx])              # consumed by the diagnostic
    it_train = iter(new[idx])             # consumed by the training loop
    stems(it_train, FF2)
    truth2 = stems(it_train, 3)
    del it_diag, it_train, new

    new = build(config, tmp)
    it_diag = iter(new[idx])
    it_train = iter(new[idx])
    stems(it_train, FF2)
    got2 = stems(it_train, 3)
    print(f'  [{name}] identical: {got2 == truth2}')
    if got2 != truth2:
        fails.append(name + '-engine-chain')
    del it_diag, it_train, new

shutil.rmtree(tmp, ignore_errors=True)

print('\n=== LOADER REPLAY ' + ('OK' if not fails else 'FAILED: ' + ','.join(fails)) + ' ===')
raise SystemExit(1 if fails else 0)
