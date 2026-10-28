"""Conservative disk cleanup: delete ONLY redundant per-run `last.pt`.

Safety envelope (all conditions must hold or the file is skipped and reported):
  1. path under /home/jhy/deploy_exp*/ ONLY  -- the rare_diag_20260915 tree is
     never traversed at all.
  2. a sibling `best_miou.pt` exists in the same run directory  -> the run's
     selected weights survive, so `last.pt` is redundant for any later use.
  3. the run directory is NOT in the protected list (assets we may still need:
     the M2F encoder donor chain, the historical ZERO-backfill bodies, and every
     run whose `last.pt` is the only surviving copy at a specific step).
  4. the file has not been modified in the last 24 h  -> nothing is writing.
  5. never a directory; never a symlink; regular file only.

Everything else -- reports, configs, completion.json, confusion matrices,
score checkpoints, M2F run1b, packages, the competition dataset -- is untouched.
Default is DRY RUN; pass --apply to actually unlink.
"""
import os
from pathlib import Path
import sys
import time

ROOTS = sorted(Path('/home/jhy').glob('deploy_exp*'))
PROTECTED_SUBSTRINGS = (
    # M2F encoder donor and its chain
    'stage62_',
    'stage23_',
    # historical ZERO-backfill bodies (56k especially: its last.pt was the
    # only surviving copy of step 56000 after the reclamation pipeline ran)
    'stage59_',
    'stage63_',
    'stage66_',
    # the run whose last.pt we resurrected on purpose
    'resume55b',
)
MIN_AGE_SEC = 24 * 3600
FLOOR_FREE = int(1.5 * 1024**3)


def main():
    apply = '--apply' in sys.argv[1:]
    now = time.time()
    print('mode:', 'APPLY (delete)' if apply else 'DRY RUN (no deletion)',
          flush=True)
    print('dev roots:', [str(r) for r in ROOTS], flush=True)
    print('protected run substrings:', PROTECTED_SUBSTRINGS, flush=True)
    print(flush=True)
    print(f'{"size":>9s}  {"mtime":10s}  path', flush=True)

    total = 0
    skipped = []
    victims = []
    for root in ROOTS:
        for lp in root.rglob('last.pt'):
            if not lp.is_file() or lp.is_symlink():
                continue
            run = lp.parent
            # (2) selected weights must survive
            if not (run / 'best_miou.pt').is_file():
                skipped.append((str(lp), 'no sibling best_miou.pt'))
                continue
            # (3) protected runs
            if any(s in run.name for s in PROTECTED_SUBSTRINGS):
                skipped.append((str(lp), 'PROTECTED run'))
                continue
            # (4) quiescence
            age = now - lp.stat().st_mtime
            if age < MIN_AGE_SEC:
                skipped.append((str(lp), f'modified {age/3600:.1f}h ago'))
                continue
            victims.append(lp)

    for lp in sorted(victims, key=lambda p: -p.stat().st_size):
        sz = lp.stat().st_size
        total += sz
        print(f'{sz/2**30:8.2f}G  '
              f'{time.strftime("%Y-%m-%d", time.localtime(lp.stat().st_mtime))}  '
              f'{lp}', flush=True)

    print(flush=True)
    print(f'candidates: {len(victims)}   reclaimable: {total/2**30:.2f} GiB',
          flush=True)
    print(f'skipped:    {len(skipped)}', flush=True)
    for p, why in skipped[:20]:
        print(f'   SKIP [{why}] {p}', flush=True)
    if len(skipped) > 20:
        print(f'   ... and {len(skipped)-20} more', flush=True)

    if not apply:
        print('\nDRY_RUN_DONE (nothing deleted)', flush=True)
        return

    freed = 0
    failed = []
    for lp in victims:
        sz = lp.stat().st_size
        try:
            # re-check the sibling still exists immediately before unlink
            if not (lp.parent / 'best_miou.pt').is_file():
                failed.append((str(lp), 'best_miou.pt vanished'))
                continue
            os.unlink(lp)
            freed += sz
        except OSError as e:
            failed.append((str(lp), repr(e)))
    print(f'\nDELETED {len(victims)-len(failed)} files, '
          f'freed {freed/2**30:.2f} GiB', flush=True)
    if failed:
        for p, why in failed:
            print(f'   FAILED [{why}] {p}', flush=True)
    import shutil
    free = shutil.disk_usage('/home/jhy').free
    print(f'free now: {free/2**30:.2f} GiB  (floor {FLOOR_FREE/2**30:.2f} GiB)',
          flush=True)
    if free < FLOOR_FREE:
        raise SystemExit('DISK FLOOR BREACHED')
    print('CLEANUP_DONE', flush=True)


if __name__ == '__main__':
    main()
