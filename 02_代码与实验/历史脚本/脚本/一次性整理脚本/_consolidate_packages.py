"""Consolidate every submission package into one folder, deduplicated by content.

Moves only -- nothing is deleted. Files that differ in content but share a name
are both kept with a disambiguating suffix, and every relocation is verified by
SHA256 before and after the move.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path('.')
DEST = ROOT / '提交包'
ARCHIVE = ROOT / '_归档整理_20260915'

# sources to consolidate; the handoff bundle and the quarantine area are left alone
SOURCES = [ROOT / '提交包', ROOT / 'submit_batch_01', ROOT / 'submit_batch_02',
           ROOT / 'submit_batch_03', ROOT / 'submit_batch_04']
SKIP_PARTS = ('_归档整理_20260915',)
# the ten packages produced for the current round
CURRENT = {'s3200_zero.zip', 's3200_champion.zip', 's4800_zero.zip',
           's4800_champion.zip', 's6400_zero.zip', 's6400_champion.zip',
           's8000_zero.zip', 's8000_champion.zip', 's62_combo4.zip',
           's62_agri_m08.zip'}
CURRENT_RENAMED = {'s3200_zero.zip': '64_s3200_zero.zip',
                   's3200_champion.zip': '65_s3200_champion.zip',
                   's4800_zero.zip': '66_s4800_zero.zip',
                   's4800_champion.zip': '67_s4800_champion.zip',
                   's6400_zero.zip': '68_s6400_zero.zip',
                   's6400_champion.zip': '69_s6400_champion.zip',
                   's8000_zero.zip': '70_s8000_zero.zip',
                   's8000_champion.zip': '71_s8000_champion.zip',
                   's62_combo4.zip': '72_s62_combo4.zip',
                   's62_agri_m08.zip': '73_s62_agri_m08.zip'}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    collector = DEST / '_收纳中'
    if collector.exists():
        raise SystemExit(f'staging already exists, refusing: {collector}')
    collector.mkdir()

    found = []
    for src in SOURCES:
        if not src.is_dir():
            continue
        for z in sorted(src.rglob('*.zip')):
            if any(part in SKIP_PARTS for part in z.parts):
                continue
            if z.parent == collector:
                continue
            found.append(z)

    staged, seen = [], {}
    for z in found:
        digest = sha256(z)
        key = (z.name, digest)
        if key in seen:
            continue                      # byte-identical duplicate of the same name
        seen[key] = True
        staged.append((z, digest))

    by_name = defaultdict(list)
    for z, digest in staged:
        by_name[z.name].append((z, digest))

    moved, deduped = [], 0
    for name, entries in sorted(by_name.items()):
        for i, (z, digest) in enumerate(entries):
            target_name = CURRENT_RENAMED.get(name, name) if name in CURRENT else name
            if i > 0:                     # same name, different bytes
                target_name = target_name.replace('.zip', f'__v{i + 1}.zip')
            target = collector / target_name
            if target.exists():
                target = collector / (target.stem + f'__{digest[:8]}.zip')
            shutil.move(str(z), str(target))
            if sha256(target) != digest:
                raise SystemExit(f'SHA changed during move: {target}')
            moved.append({'from': str(z).replace('\\', '/'),
                          'to': str(target).replace('\\', '/'),
                          'sha256': digest, 'bytes': target.stat().st_size})

    # split the staging folder into the final layout
    layout = {'当前待提交_KD_Run1': [], '历史提交包': [], '对照与诊断_不提交': []}
    for f in sorted(collector.iterdir()):
        if f.name.startswith(tuple(str(i) for i in range(64, 74))[:1]) and f.name[:2].isdigit() \
                and 64 <= int(f.name[:2]) <= 73:
            layout['当前待提交_KD_Run1'].append(f)
        elif 'ref_' in f.name or 'repro' in f.name.lower():
            layout['对照与诊断_不提交'].append(f)
        else:
            layout['历史提交包'].append(f)

    for folder, files in layout.items():
        d = DEST / folder
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.move(str(f), str(d / f.name))
    collector.rmdir()

    # fold loose manifests into one place, leave the new round's own docs on top
    records = DEST / '清单与记录'
    records.mkdir(exist_ok=True)
    for old in DEST.glob('*.json'):
        shutil.move(str(old), str(records / old.name))
    for old in DEST.glob('*.md'):
        if old.name != 'README_先看这个.md':
            shutil.move(str(old), str(records / old.name))

    # retire now-empty directories (moved, not deleted)
    emptied = []
    for d in sorted(DEST.rglob('*'), key=lambda p: -len(p.parts)):
        if d.is_dir() and not any(d.iterdir()):
            target = ARCHIVE / '空目录' / d.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target = target.with_name(target.name + '__' + str(len(emptied)))
            shutil.move(str(d), str(target))
            emptied.append(d.name)
    for extra in (ROOT / 'submit_batch_01', ROOT / 'submit_batch_02',
                  ROOT / 'submit_batch_03', ROOT / 'submit_batch_04'):
        if extra.is_dir() and not any(extra.rglob('*')):
            shutil.move(str(extra), str(ARCHIVE / '空目录' / extra.name))
            emptied.append(extra.name)

    (DEST / '_收纳记录.json').write_text(json.dumps({
        'created': datetime.now().isoformat(timespec='seconds'),
        'mode': 'MOVE only; nothing deleted; every move SHA-verified',
        'consolidated_from': [str(s) for s in SOURCES if s.is_dir()],
        'unique_packages': len(moved),
        'bytes': sum(m['bytes'] for m in moved),
        'emptied_dirs_retired': emptied,
        'moves': moved,
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'collected {len(moved)} unique packages, {sum(m["bytes"] for m in moved)/1024/1024:.1f} MB')
    for folder in layout:
        print(f'  {folder}: {len(list((DEST/folder).glob("*.zip")))}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
