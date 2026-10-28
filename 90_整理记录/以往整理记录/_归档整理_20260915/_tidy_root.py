"""Reversible tidying: move historical process reports into an archive subfolder.

Nothing is deleted. Every move is recorded in MOVED_MANIFEST.json with its
original name, destination and size, so the whole operation is undone by moving
the files back.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path('.')
OUT = ROOT / '_归档整理_20260915'

PREFIXES = ('ALL_SCHEMES_', 'HISTORICAL_', 'POST_RARE', 'STORAGE_AWARE_',
            'DISK_', 's2_readonly_', 'PHASE_B_', 'EIGHT_D_', 'HANDOFF_LIVE_',
            'FACT_HANDOFF_', 'EVIDENCE_SUPPLEMENT_', 'CANONICAL_CALIBRATION_IDENTITIES',
            'POSTPROCESS_MANIFEST', 'RARE_REDOWNLOAD_RECEIPT',
            'PARAMWISE01_PROTOCOL_UPDATE_ACK', 'GPU3：')

# files that stay in the root as the live working set
KEEP = {'正式路线与预注册规则_20260915.md', 'B3_M2F_实现前测试合同_20260915.md',
        '决策资料库导航.md', 'README_文件索引.md', '突破70_大方案_20260915.md',
        '决策记录_20260915_第二轮评审定案.md', '清理记录_20260915.md'}


def main() -> int:
    if OUT.exists():
        raise SystemExit(f'target already exists, nothing changed: {OUT}')
    reports = OUT / '历史过程报告'
    reports.mkdir(parents=True)

    moved, kept = [], []
    for path in sorted(ROOT.iterdir()):
        if not path.is_file() or path.name in KEEP:
            continue
        if any(path.name.startswith(pre) for pre in PREFIXES):
            target = reports / path.name
            if target.exists():
                raise SystemExit(f'would overwrite, aborted: {target}')
            shutil.move(str(path), str(target))
            moved.append({'from': path.name, 'to': str(target).replace('\\', '/'),
                          'bytes': target.stat().st_size})
        else:
            kept.append(path.name)

    for directory in sorted(ROOT.iterdir()):
        if not directory.is_dir() or directory.name in ('_归档整理_20260915', '.claude'):
            continue
        if any(f.is_file() for f in directory.rglob('*')):
            continue
        target = OUT / '空目录' / directory.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(directory), str(target))
        moved.append({'from': directory.name + '/', 'to': str(target).replace('\\', '/'),
                      'bytes': 0})

    (OUT / 'MOVED_MANIFEST.json').write_text(json.dumps({
        'created': datetime.now().isoformat(timespec='seconds'),
        'mode': 'MOVE (reversible); nothing deleted',
        'moved_count': len(moved),
        'moved_bytes': sum(item['bytes'] for item in moved),
        'moved': moved,
        'kept_in_root': kept,
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'moved {len(moved)} items, {sum(i["bytes"] for i in moved) / 1024:.0f} KB archived')
    print(f'root now holds {len(kept)} loose files:')
    for name in kept:
        print('   ', name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
