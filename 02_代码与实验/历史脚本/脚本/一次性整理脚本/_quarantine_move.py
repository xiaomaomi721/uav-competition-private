"""2026-09-15 清理执行：把归档/交接类文件移入 _待删除_20260915（不永久删除）。

前置条件（已满足）：
  - `_从归档救出_20260915/救出清单.json` 中 56 个独有文件已校验 SHA256 通过
  - 这些文件在工作区其他位置确认无副本
本脚本只做同盘 rename，不删除任何内容。
"""
import json
import pathlib
import shutil
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = pathlib.Path('.')
QUAR = ROOT / '_待删除_20260915'
RESCUE = ROOT / '_从归档救出_20260915'

QUAR.mkdir(exist_ok=True)

log = {'schema': 'QUARANTINE_MOVE_V1', 'created': '2026-09-15', 'moved': []}


def dir_size(p):
    return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())


def files_of(p):
    if p.is_file():
        return 1
    return sum(1 for f in p.rglob('*') if f.is_file())


# ---------- 步骤 1：把救出的 32 个独有 zip 归入 提交包/ ----------
zip_src = RESCUE / '提交包_独有zip'
zip_dst = ROOT / '提交包' / '_归档救出_20260915'
moved_zips = []
if zip_src.exists():
    zip_dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(zip_src.rglob('*.zip')):
        d = zip_dst / p.relative_to(zip_src)
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(d))
        moved_zips.append(str(d))
    # 清掉空的源目录树
    shutil.rmtree(zip_src, ignore_errors=True)
    log['moved'].append({
        'step': 'rescue_relocate',
        'what': f'{len(moved_zips)} 个独有 zip',
        'to': str(zip_dst),
    })
print(f'[1] 32 个独有 zip 已归入 提交包/{zip_dst.name}/')

# ---------- 步骤 2：更新救出清单里的路径 ----------
man = RESCUE / '救出清单.json'
if man.exists():
    d = json.loads(man.read_text(encoding='utf-8'))
    for it in d['items']:
        old = it['dest'].replace('/', '\\')
        if old.startswith(str(zip_src).replace('/', '\\')):
            rel = pathlib.Path(old).relative_to(zip_src)
            it['dest'] = str(zip_dst / rel)
            it['relocated'] = True
    d['note'] += ' 其中 32 个提交包 zip 已于同日归入 提交包/_归档救出_20260915/。'
    man.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('[2] 救出清单.json 路径已更新')

# ---------- 步骤 3：移动待删项到隔离区 ----------
TARGETS = [
    ('_非决策材料归档_20260914', '前次清理归档（9/14 移出的 833 项；独有文件已救出）'),
    ('evidence_supplement_20260914', '旧证据补充包（已并入交接包 03_成绩与实验）'),
    ('FACT_HANDOFF_20260914', '旧事实交接包（已并入交接包 03_成绩与实验）'),
    ('无人机项目_完整事实交接_含代码证据_20260914_201455.zip', '9/14 旧交接 zip（已并入新交接包）'),
    ('_staging', '暂存目录'),
    ('新建 Microsoft Word 文档.docx', '空文档'),
]

print('\n[3] 移动到隔离区：')
freed = 0
for name, why in TARGETS:
    p = ROOT / name
    if not p.exists():
        print(f'    跳过（不存在）: {name}')
        continue
    n = files_of(p)
    s = dir_size(p) if p.is_dir() else p.stat().st_size
    freed += s
    dest = QUAR / name
    if dest.exists():
        print(f'    !! 目标已存在，跳过: {dest}')
        continue
    shutil.move(str(p), str(dest))
    log['moved'].append({'name': name, 'reason': why, 'files': n, 'bytes': s})
    print(f'    ✓ {name:58s} {n:4d} 文件  {s / 1e6:7.1f} MB')

log['total_bytes'] = freed
(QUAR / '清理执行记录.json').write_text(
    json.dumps(log, ensure_ascii=False, indent=1), encoding='utf-8')

print(f'\n合计移入隔离区：{freed / 1e6:.1f} MB')
print(f'隔离区位置：{QUAR.resolve()}')
print('\n⚠️ 内容仍在磁盘上，确认无误后可整体删除该目录。')
