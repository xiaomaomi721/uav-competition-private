"""Record the authorized, verified public upload and stage its audit receipt."""
import hashlib
import json
from pathlib import Path
import subprocess

BASE = Path(__file__).resolve().parent
WORK = BASE.parent
DESKTOP = Path(r'C:\Users\Lenovo\OneDrive\Documents\桌面\无人机')
REPO = WORK / '_github_private/repo'
RELATIVE = Path('90_整理记录/GitHub本地上传_20261030')
COMMIT = '2f6253fc157276c4d32950918838478590c0df8f'
TREE = 'bd35ce594fca66562cd750919bd128dc9827a277'

def git(*args): return subprocess.check_output(['git', '-C', str(REPO), *args])
assert git('rev-parse', 'HEAD').decode().strip() == COMMIT
assert git('rev-parse', 'HEAD^{tree}').decode().strip() == TREE
assert not git('status', '--porcelain=v1').strip()
result = json.loads((BASE / '上传状态回执.json').read_text(encoding='utf-8'))
assert result['status'] == 'local_commit_ready_push_rejected_awaiting_user_approval'
result['previous_status'] = result['status']
result.update(status='uploaded_and_verified', push_attempt_executed=True,
              authorization='用户明确确认：保持公开并授权上传',
              remote_head_verified=COMMIT, remote_tree_verified=TREE, remote_files_verified=1211,
              verification_method='GitHub connector read latest commit and full recursive tree; commit and tree root match local Git exactly.',
              completion_record='The receipt is committed separately after the source-material commit.',
              source_material_commit=COMMIT)
result['previous_rejection'] = result.pop('rejection')
result.pop('required_user_choice', None)
(BASE / '上传状态回执.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
completion = f'''# GitHub公开上传完成回执

日期：2026-10-30（北京时间）。

**桌面无人机资料已按用户明确授权公开上传，内容提交已由GitHub连接器独立读取核验。**

- 仓库：https://github.com/xiaomaomi721/uav-competition-private
- 分支：`codex/project-archive-20261028`。
- 内容提交：[`{COMMIT}`](https://github.com/xiaomaomi721/uav-competition-private/commit/{COMMIT})。
- 内容树：`{TREE}`。
- 内容提交：345个新增、7个更新；包含同步清单与索引。
- 远端完整目录树共1,211个文件，未截断，提交及树哈希与本机完全一致。
- 原始本地差异为340个新增、5个更新；其余为本次归档清单和仓库说明。
- 297个模型、压缩包等本地产物保留原位置，未上传。
- 没有删除远端独有资料，没有强制推送，没有修改仓库可见性。

## 授权与状态沿革

首次推送被自动审批拒绝，未执行网络上传。用户随后明确回复“保持公开并授权上传”，故本次正常推送获准并成功。

本回执和状态修订作为后续归档提交上传；它们是在上述内容提交验证后产生，不将回执本身伪称为内容提交的一部分。

## 对应证据

- `upload_plan.json`：原始本地清单及排除项。
- `staged_verification.json`：导出文件字节核验。
- `上传状态回执.json`：当前已上传状态及上次审批拒绝记录。
- `本地保留文件清单.md`：留在本地的路径、大小和SHA256。
'''
(BASE / '上传完成回执.md').write_text(completion, encoding='utf-8')
pending_path = BASE / '上传待确认记录.md'
pending = pending_path.read_text(encoding='utf-8')
pending = ('> **状态已解除（2026-10-30）：** 用户已明确授权保持公开并上传，内容提交已推送并核验成功。'
           '下文保留首次审批拒绝时的历史快照；当前以《上传完成回执.md》为准。\n\n' + pending)
pending_path.write_text(pending, encoding='utf-8')

entry = ('\n- 2026-10-30 上传完成：用户明确授权保持公开，内容提交2f6253f已推送到 '
         'xiaomaomi721/uav-competition-private，远端1,211文件目录树与本地匹配；'
         '详见 [上传完成回执](90_整理记录/GitHub本地上传_20261030/上传完成回执.md)。\n').encode('utf-8')
for index in (DESKTOP / '00_Codex工作同步索引.md', WORK / '00_Codex工作同步索引.md'):
    before = index.read_bytes()
    assert entry not in before
    with index.open('ab') as stream: stream.write(entry)
    assert index.read_bytes() == before + entry

# Preserve source data and update only assistant-owned operation records.
files = ['verify_export.py', 'save_pending_receipt.py', 'record_success.py', 'expected_export.json',
         'staged_verification.json', '上传状态回执.json', '上传待确认记录.md', '上传完成回执.md']
for name in files:
    content = (BASE / name).read_bytes()
    desktop_path = DESKTOP / RELATIVE / name
    desktop_path.parent.mkdir(parents=True, exist_ok=True)
    desktop_path.write_bytes(content)
    checkout_path = REPO / RELATIVE / name
    checkout_path.parent.mkdir(parents=True, exist_ok=True)
    checkout_path.write_bytes(content)
    assert desktop_path.read_bytes() == content == checkout_path.read_bytes()
(REPO / '00_Codex工作同步索引.md').write_bytes((DESKTOP / '00_Codex工作同步索引.md').read_bytes())
readme = REPO / 'README.md'
readme_content = readme.read_text(encoding='utf-8')
readme_content += '\n本次内容已公开上传并核验完成，见[上传完成回执](90_整理记录/GitHub本地上传_20261030/上传完成回执.md)。\n'
readme.write_text(readme_content, encoding='utf-8')
paths = [(RELATIVE / name).as_posix() for name in files] + ['00_Codex工作同步索引.md', 'README.md']
git('--literal-pathspecs', 'add', '--', *paths)
index = {}
for record in git('ls-files', '--stage', '-z').split(b'\0'):
    if record:
        meta, path = record.split(b'\t', 1)
        index[path.decode('utf-8')] = meta.decode().split()[1]
for relative in paths:
    content = (REPO / relative).read_bytes()
    expected = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
    assert index[relative] == expected, relative
print(json.dumps({'status': result['status'], 'verified_content_commit': COMMIT,
                  'receipt_files_staged': len(paths), 'desktop_receipt': str(DESKTOP / RELATIVE / '上传完成回执.md')}, ensure_ascii=False))
