"""Third tidy pass: batch dirs, leftover scripts, root README.

MOVE only -- nothing deleted. The submit_batch_* zips are byte-identical
(SHA256 verified) to copies already in 提交包/历史提交包/, so the
filename+SHA platform-receipt binding is preserved after the move.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path('.')
LOG: list = []


def move(src: Path, dst: Path) -> None:
    if not src.exists():
        LOG.append({'from': src.as_posix(), 'status': 'missing'})
        return
    if dst.exists():
        dst = dst.with_name(dst.stem + '__dup' + dst.suffix)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    LOG.append({'from': src.as_posix(), 'to': dst.as_posix(), 'status': 'moved'})


def main() -> int:
    # batch dirs keep their receipt pairing -- move to one labelled place
    for d in ['submit_batch_01', 'submit_batch_02', 'submit_batch_03',
              'submit_batch_04']:
        move(ROOT / d, ROOT / '提交包' / '清单与记录' / '批次目录_原始' / d)

    # tidy scripts belong with the other one-off scripts
    for name in ['_group_scripts.py', '_tidy_root2.py']:
        move(ROOT / name, ROOT / '脚本' / '一次性整理脚本' / name)
    move(ROOT / '_tidy_root2.json', ROOT / '归档_整理记录' / '_tidy_root2.json')

    (ROOT / '归档_整理记录' / '_tidy_root3.json').write_text(json.dumps({
        'created': datetime.now().isoformat(timespec='seconds'),
        'mode': 'MOVE only; nothing deleted',
        'moves': LOG,
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps({'moved': sum(1 for m in LOG if m['status'] == 'moved'),
                      'missing': [m['from'] for m in LOG if m['status'] == 'missing']},
                     ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
