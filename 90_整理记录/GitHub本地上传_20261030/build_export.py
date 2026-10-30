"""Build a reviewed, additive desktop export and exact Git staging manifest."""
import collections
import hashlib
import json
from pathlib import Path
import subprocess

BASE = Path(__file__).resolve().parent
WORK = BASE.parent
plan = json.loads((BASE / 'upload_plan.json').read_text(encoding='utf-8'))
SOURCE = Path(plan['source']).resolve()
REPO = Path(plan['checkout']).resolve()
RECORD = Path('90_整理记录/GitHub本地上传_20261030')
assert not plan['credential_candidates'], 'Credential candidates require review.'
def digest(content): return hashlib.sha256(content).hexdigest()
def git(*args, **kwargs): return subprocess.check_output(['git', '-C', str(REPO), *args], **kwargs)
assert git('rev-parse', 'HEAD').decode().strip() == plan['parent_commit']
assert not git('status', '--porcelain=v1').strip(), 'Checkout changed after audit.'

# Freeze reviewed file bytes before adding the operation's own index entry.
reviewed = {}
for item in plan['files']:
    if item['action'] == 'local_only': continue
    content = (SOURCE / item['path']).read_bytes()
    assert digest(content) == item['sha256'], f'Source changed after audit: {item["path"]}'
    reviewed[item['path']] = content

local = [x for x in plan['files'] if x['action'] == 'local_only']
local_lines = ['# 本次保留在本地的文件', '', '日期：2026-10-30（北京时间）。', '',
               f'源目录：`{SOURCE}`。下表保留路径、大小和SHA256；文件原位置不变。', '',
               '| 相对路径 | 字节数 | SHA256 | 排除原因 |', '|---|---:|---|---|']
local_lines.extend(f'| `{x["path"]}` | {x["bytes"]} | `{x["sha256"]}` | {x["reason"]} |' for x in local)
(BASE / '本地保留文件清单.md').write_text('\n'.join(local_lines) + '\n', encoding='utf-8')

summary = [
    '# 桌面无人机上传GitHub：执行记录', '', '日期：2026-10-30（北京时间）', '',
    f'- 本地源目录：`{SOURCE}`。', f'- 目标仓库：{plan["repository"]}。',
    f'- 分支：`{plan["branch"]}`。父提交：`{plan["parent_commit"]}`。',
    '- 用户授权：将本地无人机项目文件夹同步到自己的GitHub；未向队友仓库推送。',
    '- 原始差异：340份新增、5份更新、857份不变；新增/更新文件合计8,339,010字节。',
    '- 另写入本次同步清单、脚本、记录和索引；最终提交统计及提交SHA见本目录上传完成回执。',
    '- 297个本地模型、二进制/压缩包等文件（2,113,778,333字节）排除，.claude目录排除；原文件未删除。',
    '- 新增/修改的文本与Office XML凭据扫描无候选；图片/PDF未做OCR，扫描不等于逐文件人工内容审计。',
    '- 来源文件不改写；仅追加本助手自己的同步索引。远端独有文件保留，不执行镜像删除、不改写已有提交历史。',
    '- GitHub API本次返回visibility=public；仓库名称中的private不是当前权限。本次未修改可见性。',
    '- 文档中的实测、用户报分、作者判断和未执行方案保留其原始口径；上传不表示重新运行实验或核验赛事成绩。',
    '- 官方数据集、模型权重和缓存不新增入Git；原有报告配图及历史材料按已有归档范围保留。', '',
    '## 本轮更新文件', '',
]
summary.extend('- `' + x['path'] + '`' for x in plan['files'] if x['action'] == 'update')
summary.extend(['', '## 原始本地新增清单', '', '| 相对路径 | 字节数 | SHA256 |', '|---|---:|---|'])
summary.extend(f'| {x["path"]} | {x["bytes"]} | `{x["sha256"]}` |' for x in plan['files'] if x['action'] == 'add')
(BASE / '同步执行记录.md').write_text('\n'.join(summary) + '\n', encoding='utf-8')

generated = {}
for name in ('upload_plan.json', '本地保留文件清单.md', '同步执行记录.md', 'prepare_upload.py', 'build_export.py'):
    generated[(RECORD / name).as_posix()] = (BASE / name).read_bytes()
entry = ('\n- 2026-10-30：将桌面无人机的新增/更新资料同步到 xiaomaomi721/uav-competition-private；'
         '原始差异340份新增、5份更新；模型与压缩包留本地。详见 '
         '[执行记录](90_整理记录/GitHub本地上传_20261030/同步执行记录.md)及同目录上传完成回执。\n').encode('utf-8')
index = '00_Codex工作同步索引.md'
assert entry not in reviewed[index]
generated[index] = reviewed[index] + entry

# Desktop writes are restricted to new dated records and appending the assistant index.
for relative, content in generated.items():
    target = SOURCE / relative
    target.resolve().relative_to(SOURCE)
    target.parent.mkdir(parents=True, exist_ok=True)
    if relative == index:
        assert target.read_bytes() == reviewed[index]
        with target.open('ab') as stream: stream.write(entry)
    else:
        assert not target.exists(), f'Existing dated record will not be overwritten: {target}'
        with target.open('xb') as stream: stream.write(content)
    assert target.read_bytes() == content
    reviewed[relative] = content
workspace_index = WORK / index
assert entry not in workspace_index.read_bytes()
with workspace_index.open('ab') as stream: stream.write(entry)

# Add/update source material in the managed export checkout; leave remote-only files intact.
for relative, content in reviewed.items():
    target = REPO / relative
    target.resolve().relative_to(REPO)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.read_bytes() != content:
        target.write_bytes(content)
    assert target.read_bytes() == content

readme = '''# 无人机语义分割比赛资料

本仓库归档桌面“无人机”文件夹中的文档、代码、配置、实验记录和轻量报告。最新同步日期：**2026-10-30（北京时间）**。

## 项目入口

- [桌面项目导航](README_从这里开始.md)
- [当前比赛、方案与线上成绩记录](01_当前比赛/)
- [代码与实验](02_代码与实验/)
- [XSB仓库半决赛档案（2026-10-29快照）](02_代码与实验/GitHub同步_20261029/uav_semantic_segmentation/docs/semifinal/README.md)
- [Claude近期诊断与交接](01_当前比赛/Claude方案与诊断/)
- [历史文档与复盘](03_历史文档/)
- [审计与模板](04_审计与模板/)
- [提交包相关记录](提交包/)
- [整理记录与旧路径映射](90_整理记录/)

## 同步范围

本轮同步新增/修改资料，并保留远端独有文件；没有进行镜像删除。原始差异为340份新增、5份更新，另归档本次操作记录。具体内容与指纹见[本次执行记录](90_整理记录/GitHub本地上传_20261030/同步执行记录.md)和[文件清单](90_整理记录/GitHub本地上传_20261030/upload_plan.json)。

模型权重、比赛提交ZIP、二进制缓存等297份文件保留在本地；路径、大小和SHA256见[本地保留文件清单](90_整理记录/GitHub本地上传_20261030/本地保留文件清单.md)。机器设置与凭据不纳入同步。初次归档清单仍保留在[99_GitHub同步](99_GitHub同步/)。

这是手动资料归档，不会自动同步之后的本地修改，也不表示所有历史实验都可以直接克隆运行。文档中的“当前”“待提交”、成绩与执行状态只对应原记录时间；实测数据、用户报分、作者分析和未执行计划须分开理解。本次上传没有运行训练、推理或比赛提交。

2026-10-30核验时仓库公开可见；名称中的private来自历史命名。请以GitHub当前仓库权限为准。
'''.encode('utf-8')
(REPO / 'README.md').write_bytes(readme)
reviewed['README.md'] = readme
ignore_path = REPO / '.gitignore'
ignore = ignore_path.read_bytes()
ignore += b'\n# Additional local-only model/cache containers\n*.npy\n*.npz\n*.bin\n*.pkl\n*.pickle\n*.h5\n*.hdf5\n*.parquet\n*.mp4\n*.avi\n*.mov\n*.p12\n*.pfx\n'
ignore_path.write_bytes(ignore)
reviewed['.gitignore'] = ignore

expected = {path: dict(sha256=digest(content), git_blob_sha=hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest(), bytes=len(content))
            for path, content in reviewed.items()}
(BASE / 'expected_export.json').write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding='utf-8')
stage_paths = [x['path'] for x in plan['files'] if x['action'] in ('add', 'update')]
stage_paths += list(generated) + ['README.md', '.gitignore']
stage_paths = sorted(set(stage_paths))
(BASE / 'stage_paths.nul').write_bytes(b'\0'.join(x.encode('utf-8') for x in stage_paths) + b'\0')
print(json.dumps({'reviewed_export_files': len(expected), 'stage_candidates': len(stage_paths),
                  'desktop_records_added': len(generated) - 1, 'source_index_appended': True}, ensure_ascii=False))
