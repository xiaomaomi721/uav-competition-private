"""Group the loose scripts in the workspace root into a 脚本/ tree.

Moves only. Local copies are backups of scripts that ran on the server, so
relocating them cannot affect anything running remotely.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path('.')
GROUPS = {
    '训练与KD': ['_kd_train.py', '_kd_launch.sh', '_kd_restart.sh',
                '_kd_precompute_teachers.py', '_kd_diagnose.py', '_make_teachers.py',
                '_hard_stop.sh', '_combo_queue.sh', '_kd_pack_eight.py',
                '_s62_axis_probe.py', '_s62_combo_probe.py'],
    '校准与TTA': ['tta_anatomy.py', 'tta_anatomy_v2.py', 'tta_partial_gates.py',
                 'tta_priority_gates.py', 'tta_v2_gates.py',
                 'constrained_calibration.py', 'constrained_calibration_v2.py',
                 'constrained_calibration_priority_v2.py',
                 'canonical_paramwise_optimizer.py'],
    '审计与检查': ['_check_cache_matches_ensemble.py', '_check_distribution.py',
                 '_check_geom_alignment.py', '_check_kd_math.py', '_check_quant_gap.py',
                 '_error_morphology.py', '_error_morphology_v2.py',
                 '_morphology_report.py', '_val_predict.py'],
    '一次性整理脚本': ['_quarantine_move.py', '_rescue_from_archive.py',
                    '_tidy_root.py', '_consolidate_packages.py'],
}
KEEP_AT_ROOT = {'README_从这里开始.md'}


def main() -> int:
    dest = ROOT / '脚本'
    if dest.exists():
        raise SystemExit(f'already exists, refusing: {dest}')
    dest.mkdir()
    moved, missing = [], []
    for group, names in GROUPS.items():
        d = dest / group
        d.mkdir()
        for name in names:
            src = ROOT / name
            if not src.is_file():
                missing.append(name)
                continue
            shutil.move(str(src), str(d / name))
            moved.append({'from': name, 'to': f'脚本/{group}/{name}'})
    (dest / '_整理记录.json').write_text(json.dumps({
        'created': datetime.now().isoformat(timespec='seconds'),
        'mode': 'MOVE only; nothing deleted',
        'moved': moved, 'not_found': missing,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'moved {len(moved)} scripts; not found: {missing or "none"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
