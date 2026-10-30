"""Persist the prepared commit and rejected push status; never retry a push."""
import hashlib
import json
from pathlib import Path
import subprocess

BASE = Path(__file__).resolve().parent
WORK = BASE.parent
DESKTOP = Path(r'C:\Users\Lenovo\OneDrive\Documents\桌面\无人机')
REPO = WORK / '_github_private/repo'
commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD']).decode().strip()
tree = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD^{tree}']).decode().strip()
assert commit == '2f6253fc157276c4d32950918838478590c0df8f'
verification = json.loads((BASE / 'staged_verification.json').read_text(encoding='utf-8'))
result = dict(date='2026-10-30', status='local_commit_ready_push_rejected_awaiting_user_approval',
              repository='https://github.com/xiaomaomi721/uav-competition-private', visibility='public',
              branch='codex/project-archive-20261028', local_commit=commit, tree=tree,
              remote_head_last_verified='eac684521375d03bb1dc3e90b603b7eedd23e7c8',
              push_attempt_executed=False,
              rejection='Automatic approval rejected public publication because synchronization authorization did not explicitly authorize publishing internal project materials publicly.',
              required_user_choice=['authorize_public_upload', 'change_repository_to_private_then_upload'],
              added=345, modified=7, export_files_hash_verified=1209, local_only_files=297,
              verification=verification)
(BASE / '上传状态回执.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
text = f'''# GitHub上传状态：待用户确认公开范围

日期：2026-10-30（北京时间）。

**当前只完成本地提交，尚未推送到GitHub。**

- 仓库：{result['repository']}
- 当前可见性：Public / 公开；名称中的private为历史命名。
- 分支：`{result['branch']}`。
- 已准备的本地提交：`{commit}`。
- 提交树：`{tree}`。
- 最后核验的远端提交：`{result['remote_head_last_verified']}`。
- 暂存核验：345个新增、7个更新；1,209份导出文件的Git blob与来源逐字节一致。
- 297个模型、压缩包等文件保留本地，新增/修改文本与Office XML凭据扫描无候选。
- 格式检查识别原归档文档中的34处尾部空白；为保留原始字节，没有重写这些材料。

## 推送阻断原因

本次 `git push origin HEAD:refs/heads/codex/project-archive-20261028` 在进程执行前被自动审批拒绝。理由：该操作会公开发布项目内部资料，而用户只明确授权同步到自己的Git，尚未明确授权公开发布。没有改用其他通道绕过拒绝，没有强制推送，也没有擅自修改仓库权限。

## 下一步所需确认

用户需选择“保持公开并授权本次上传”或“先将此仓库改为私有再上传”。收到明确确认后，重新核对远端头提交与本地变化，再执行授权动作。

完整文件清单与本地排除项见本目录 `upload_plan.json`、`本地保留文件清单.md`、`staged_verification.json`。本回执是提交完成后产生的操作记录，尚不在上述待推送提交中。
'''
(BASE / '上传待确认记录.md').write_text(text, encoding='utf-8')
record = DESKTOP / '90_整理记录/GitHub本地上传_20261030'
record.mkdir(parents=True, exist_ok=True)
for name in ('verify_export.py', 'save_pending_receipt.py', 'expected_export.json',
             'staged_verification.json', '上传状态回执.json', '上传待确认记录.md'):
    content = (BASE / name).read_bytes()
    path = record / name
    assert not path.exists(), f'Refuse overwriting an existing receipt: {path}'
    with path.open('xb') as stream: stream.write(content)
    assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(content).digest()
entry = ('\n- 2026-10-30 上传状态更正：本地提交2f6253f已准备，向公开仓库推送被自动审批拒绝，尚未上传；'
         '待用户确认公开上传或先改私有。详见 '
         '[待确认记录](90_整理记录/GitHub本地上传_20261030/上传待确认记录.md)。\n').encode('utf-8')
for path in (DESKTOP / '00_Codex工作同步索引.md', WORK / '00_Codex工作同步索引.md'):
    before = path.read_bytes()
    assert entry not in before
    with path.open('ab') as stream: stream.write(entry)
    assert path.read_bytes() == before + entry
print(json.dumps({'status': result['status'], 'local_commit': commit,
                  'desktop_record': str(record / '上传待确认记录.md')}, ensure_ascii=False))
