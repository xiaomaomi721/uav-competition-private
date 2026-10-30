"""Apply the prepared additive sync, verify bytes, and write a receipt."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
plan = json.loads((ROOT / 'sync_plan.json').read_text(encoding='utf-8'))
desktop = Path(plan['destination']).resolve()
record_relative = Path('90_整理记录/GitHub补齐_20261029')

def target_path(relative):
    path = desktop.joinpath(*relative.replace('\\', '/').split('/'))
    path.resolve().relative_to(desktop)
    return path

def sha256(content):
    return hashlib.sha256(content).hexdigest()

# Stop before any writes if a previously existing file changed after planning.
for item in plan['preserve']:
    assert sha256(target_path(item['path']).read_bytes()) == item['sha256'], item['path']

copied = []
for item in plan['additions']:
    content = (ROOT / 'payload' / item['destination_relative']).read_bytes()
    assert sha256(content) == item['sha256'], item['path']
    destination = target_path(item['destination_relative'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert destination.is_file() and sha256(destination.read_bytes()) == item['sha256'], destination
        action = 'identical_already_present'
    else:
        with destination.open('xb') as stream:
            stream.write(content)
        action = 'added'
    assert sha256(destination.read_bytes()) == item['sha256'], destination
    copied.append(dict(item, action=action))

for item in plan['preserve']:
    assert sha256(target_path(item['path']).read_bytes()) == item['sha256'], item['path']

receipt = {
    'date': datetime.now().astimezone().isoformat(), 'repository': plan['repository'],
    'commit': plan['commit'], 'tree': plan['tree'], 'branch': plan['branch'],
    'destination': str(desktop), 'copied': copied,
    'addition_sha256_checks_passed': len(copied),
    'existing_files_unchanged_before_index_append': len(plan['preserve']),
    'different_files_preserved': [x['path'] for x in plan['files'] if x['state'] == 'different_preserve'],
    'intentionally_skipped': ['.gitignore'],
    'repository_control_notes_relocated_to': (record_relative / '仓库附带记录').as_posix(),
}
(ROOT / '同步回执.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
lines = [
    '# GitHub 缺失资料补齐记录', '', '日期：2026-10-29（北京时间）', '',
    f"- 来源：{plan['repository']}", f"- 分支：`{plan['branch']}`",
    f"- 固定提交：`{plan['commit']}`", f"- 树 SHA：`{plan['tree']}`",
    '- GitHub连接器读取远端提交及完整目录树，确认与本机Git对象一致；从Git对象提取原始字节。',
    '- 用户选择：只补齐桌面缺少的文档、代码和实验记录。',
    '- 远端866文件：830份与桌面相同，5份有差异，31个原路径缺失。',
    '- 补齐27份历史资料（赛题Word及26张分析/汇报图片）；归档3份仓库说明/清单，共30份，91,289,722字节。',
    '- .gitignore属于仓库管理文件，不加入桌面项目；仓库README和99_GitHub同步清单放入本记录目录，避免混淆当前入口。',
    '- 30份新增文件的SHA256逐一核验通过；同步前后核查835份已有文件内容未被覆盖。之后仅追加本助手的同步索引条目。',
    '- 代码和实验记录已存在，无需重复复制；未复制模型权重、提交ZIP或缓存，未执行训练、未推送远端。',
    '- 截至本次读取，仓库名称虽含private，GitHub API显示visibility=public；本次没有修改仓库权限。',
    '- 来源是10月28日快照，不将其中历史成绩或执行状态写成当前新结果。', '',
    '## 已有差异文件（保留桌面版本）', '',
]
lines.extend('- `' + x + '`' for x in receipt['different_files_preserved'])
lines.extend(['', '## 补齐清单', '', '| 仓库路径 | 桌面相对路径 | 字节数 | SHA256 |', '|---|---|---:|---|'])
lines.extend(f"| {x['path']} | {x['destination_relative']} | {x['bytes']} | `{x['sha256']}` |" for x in copied)
(ROOT / '同步记录.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

record_dir = target_path(record_relative.as_posix())
record_dir.mkdir(parents=True, exist_ok=True)
for name in ('comparison.json', 'sync_plan.json', '同步回执.json', '同步记录.md',
             'compare_repository.py', 'stage_missing.py', 'apply_missing.py'):
    content = (ROOT / name).read_bytes()
    destination = record_dir / name
    if destination.exists():
        assert destination.read_bytes() == content, destination
    else:
        with destination.open('xb') as stream:
            stream.write(content)
    assert destination.read_bytes() == content

entry = ('\n- 2026-10-29：从 xiaomaomi721/uav-competition-private 补齐30份缺失资料及仓库附带记录；'
         '详见 [同步记录](90_整理记录/GitHub补齐_20261029/同步记录.md)。已有文件保留，新增文件SHA256核验通过。\n')
for index in (desktop / '00_Codex工作同步索引.md', ROOT.parent / '00_Codex工作同步索引.md'):
    original = index.read_bytes()
    if entry.encode('utf-8') not in original:
        with index.open('ab') as stream:
            stream.write(entry.encode('utf-8'))
        assert index.read_bytes() == original + entry.encode('utf-8')
for item in plan['preserve']:
    path = target_path(item['path'])
    if item['path'] == '00_Codex工作同步索引.md':
        content = path.read_bytes()
        assert content.endswith(entry.encode('utf-8'))
        assert sha256(content[:-len(entry.encode('utf-8'))]) == item['sha256']
    else:
        assert sha256(path.read_bytes()) == item['sha256'], item['path']
print(json.dumps({'added_or_verified': len(copied), 'sha256_passed': len(copied),
                  'existing_preserved': len(plan['preserve']), 'receipt': str(record_dir / '同步记录.md')}, ensure_ascii=False))
