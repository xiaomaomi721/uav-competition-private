"""Copy only prepared additions, preserving existing files and provenance."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
plan = json.loads((ROOT / 'sync_plan.json').read_text(encoding='utf-8'))
desktop = Path(plan['destination']).resolve()
def digest(content):
    return hashlib.sha256(content).hexdigest()
def target(relative):
    path = desktop.joinpath(*relative.replace('\\', '/').split('/'))
    path.resolve().relative_to(desktop)
    return path
def create_only(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.is_file() and path.read_bytes() == content, str(path)
    else:
        with path.open('xb') as stream:
            stream.write(content)
    assert path.read_bytes() == content

preserve = {}
for item in plan['files']:
    if item['action'] == 'already_present_elsewhere':
        preserve[item['existing_paths'][0]] = item['sha256']
    elif item['action'] in ('different_preserve', 'identical_at_target'):
        preserve[item['destination_relative']] = item['existing_sha256']
def verify_existing():
    for relative, expected in preserve.items():
        assert digest(target(relative).read_bytes()) == expected, relative
verify_existing()
copied = []
for item in plan['files']:
    if item['action'] != 'add_missing':
        continue
    content = (ROOT / 'payload' / item['destination_relative']).read_bytes()
    assert digest(content) == item['sha256'], item['source_path']
    create_only(target(item['destination_relative']), content)
    copied.append(item)
verify_existing()
counts = {action: sum(x['action'] == action for x in plan['files']) for action in sorted({x['action'] for x in plan['files']})}
receipt = dict(date=datetime.now().astimezone().isoformat(), repository=plan['repository'],
               commit=plan['commit'], tree=plan['tree'], destination=str(desktop),
               counts=counts, bytes_added=sum(x['bytes'] for x in copied),
               sha256_verified=len(copied), existing_files_preservation_checked=len(preserve), copied=copied)
(ROOT / '同步回执.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
lines = [
    '# XSB 仓库缺失资料同步记录', '', '日期：2026-10-29（北京时间）', '',
    f"- 仓库：{plan['repository']}", f"- 固定提交：`{plan['commit']}`", f"- 树 SHA：`{plan['tree']}`",
    '- GitHub连接器读取最新提交及779个文件的完整树；本机Git仅fetch对象，没有切换、修改或推送源仓库。',
    '- 从本机Git对象按固定提交提取文件；779条路径、Git blob SHA与远端目录逐条一致。',
    f"- 实际新增 {len(copied)} 份文件，{receipt['bytes_added']:,} 字节；逐一核验SHA256通过。",
    '- 其中95份为桌面原先缺少的基线代码工程、配置、固定划分、报告和最新半决赛文档；工程相对目录保持完整。',
    '- 其余76份为缺少的历史提交记录、归档索引与10月28–29日近期材料。',
    f"- 519份资料在桌面已有相同字节内容，未重复复制；89份对应文件有差异，保留桌面版本。共核验{len(preserve)}个已有文件未被改写。",
    f"- 代码与半决赛档案：`{plan['code_root']}`。这不是包含所有旧桌面归档的完整Git克隆。",
    f"- 无历史映射的新归档：`{plan['new_archive_root']}`；已有旧路径映射的缺失文件按整理后的路径补齐。",
    '- 原文中的线上分数、已执行/未执行状态按其快照日期理解；同步并不代表重新训练、评分或独立复核赛事成绩。',
    '- 未复制模型权重、提交ZIP、缓存或官方数据；未执行仓库中的训练/推理脚本。', '',
    '## 入口', '',
    '- 最新半决赛档案：`02_代码与实验/GitHub同步_20261029/uav_semantic_segmentation/docs/semifinal/README.md`',
    '- 近期工作归档：`01_当前比赛/XSB新增归档_20261029/近期同步归档/20261029/README_入口.md`',
    '- 另一仓库同步记录：`90_整理记录/GitHub补齐_20261029/同步记录.md`', '',
    '## 新增清单', '', '| 仓库路径 | 桌面相对路径 | 字节数 | SHA256 |', '|---|---|---:|---|',
]
lines.extend(f"| {x['source_path']} | {x['destination_relative']} | {x['bytes']} | `{x['sha256']}` |" for x in copied)
lines.extend(['', '## 保留桌面版本的差异清单', '', '| 仓库路径 | 桌面相对路径 |', '|---|---|'])
lines.extend(f"| {x['source_path']} | {x['destination_relative']} |" for x in plan['files'] if x['action'] == 'different_preserve')
(ROOT / '同步记录.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
record_dir = target('90_整理记录/XSB仓库补齐_20261029')
for name in ('remote_manifest.json', 'sync_plan.json', '同步回执.json', '同步记录.md', 'prepare_xsb.py', 'apply_xsb.py'):
    create_only(record_dir / name, (ROOT / name).read_bytes())

entry = ('\n- 2026-10-29：从 2856643972xsb/uav_semantic_segmentation 补齐171份代码、半决赛文档与归档记录；'
         '详见 [XSB同步记录](90_整理记录/XSB仓库补齐_20261029/同步记录.md)。519份相同内容跳过，89份差异保留桌面版本。\n')
for index in (desktop / '00_Codex工作同步索引.md', ROOT.parent / '00_Codex工作同步索引.md'):
    original = index.read_bytes()
    if entry.encode('utf-8') not in original:
        with index.open('ab') as stream:
            stream.write(entry.encode('utf-8'))
        assert index.read_bytes() == original + entry.encode('utf-8')
verify_existing()
for item in copied:
    assert digest(target(item['destination_relative']).read_bytes()) == item['sha256']
print(json.dumps({key: receipt[key] for key in ('counts', 'bytes_added', 'sha256_verified', 'existing_files_preservation_checked')}, ensure_ascii=False))
