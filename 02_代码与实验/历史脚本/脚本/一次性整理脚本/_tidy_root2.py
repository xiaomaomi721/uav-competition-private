"""Second tidy pass: group root scripts, fold loose material into 文档归档/.

MOVE only. Nothing is deleted by this script; deletion candidates are listed
in --report output for the user to approve separately.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path('.')
DOCS = ROOT / '文档归档'

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


def move(src: Path, dst: Path, log: list) -> None:
    if not src.exists():
        log.append({'from': src.as_posix(), 'to': None, 'status': 'missing'})
        return
    if dst.exists():
        dst = dst.with_name(dst.stem + '__dup' + dst.suffix)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    log.append({'from': src.as_posix(), 'to': dst.as_posix(), 'status': 'moved'})


def main() -> int:
    log: list = []

    # 1. root scripts -> 脚本/<group>/
    dest = ROOT / '脚本'
    dest.mkdir(exist_ok=True)
    for group, names in GROUPS.items():
        for name in names:
            if name == '_tidy_root2.py':
                continue
            move(ROOT / name, dest / group / name, log)

    # 2. loose review/report material -> 文档归档/07_汇报与复盘/
    review = DOCS / '07_汇报与复盘'
    for name in ['无人机低空航拍图像语义分割-2.pdf']:
        move(ROOT / name, review / name, log)
    for d in ['尺度分析可视化', '官方图片给你看', '_从归档救出_20260915']:
        move(ROOT / d, review / d, log)

    # 3. loose decision/report notes -> 文档归档/04_实验决策与成绩/
    decisions = DOCS / '04_实验决策与成绩'
    for name in ['决策记录_20260915_第二轮评审定案.md', '决策资料库导航.md',
                 '清理记录_20260915.md', '突破70_大方案_20260915.md',
                 '正式路线与预注册规则_20260915.md',
                 'B3_M2F_实现前测试合同_20260915.md']:
        move(ROOT / name, decisions / name, log)

    # 4. CANDIDATES.json is a disk-cleanup candidate sheet, not a result
    move(ROOT / 'CANDIDATES.json', ROOT / '归档_整理记录' / 'CANDIDATES.json', log)
    move(ROOT / 'README_文件索引.md', ROOT / '归档_整理记录' / 'README_文件索引_旧版.md', log)

    # 5. the two CPU audit dirs have no home yet
    move(ROOT / 'CPU形态审计_20260915', ROOT / '审计记录' / 'CPU形态审计_20260915', log)
    move(ROOT / '新head工程准备_20260915', ROOT / '审计记录' / '新head工程准备_20260915', log)

    (ROOT / '_tidy_root2.json').write_text(json.dumps({
        'created': datetime.now().isoformat(timespec='seconds'),
        'mode': 'MOVE only; nothing deleted',
        'moves': log,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'moved': sum(1 for m in log if m['status'] == 'moved'),
                      'missing': [m['from'] for m in log if m['status'] == 'missing']},
                     ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
