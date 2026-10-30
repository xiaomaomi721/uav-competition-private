"""Verify staged bytes against the reviewed desktop export; report bounded stats."""
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess

BASE = Path(__file__).resolve().parent
REPO = BASE.parent / '_github_private/repo'
expected = json.loads((BASE / 'expected_export.json').read_text(encoding='utf-8'))
plan = json.loads((BASE / 'upload_plan.json').read_text(encoding='utf-8'))
def git(*args): return subprocess.check_output(['git', '-C', str(REPO), *args])
index = {}
for item in git('ls-files', '--stage', '-z').split(b'\0'):
    if item:
        meta, path = item.split(b'\t', 1)
        mode, sha, stage = meta.decode().split()
        assert stage == '0'
        index[path.decode('utf-8')] = sha
missing, mismatched = [], []
for path, info in expected.items():
    if path not in index: missing.append(path)
    elif index[path] != info['git_blob_sha']: mismatched.append(path)
assert not missing and not mismatched, json.dumps({'missing': missing, 'mismatched': mismatched}, ensure_ascii=False)
assert all(path in index for path in plan['remote_only_preserved']), 'Remote-only file removed.'
changed = git('diff', '--cached', '--name-status', '-z', '--no-renames').split(b'\0')
changes = []
for i in range(0, len(changed) - 1, 2):
    status, path = changed[i].decode(), changed[i + 1].decode('utf-8')
    assert status in ('A', 'M') and path in expected, (status, path)
    changes.append({'status': status, 'path': path, **expected[path]})
blocked = {x['path'] for x in plan['files'] if x['action'] == 'local_only'}
assert not any(x['path'] in blocked for x in changes)
assert not any(re.search(r'(^|/)(?:\.claude|\.aws|\.ssh|\.env)(?:/|$)', x['path']) for x in changes)
check = subprocess.run(['git', '-C', str(REPO), '-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check'], capture_output=True)
warning_counts = collections.Counter()
for line in check.stdout.decode('utf-8', 'replace').splitlines():
    if re.search(r':\d+: ', line): warning_counts[line.rsplit(': ', 1)[-1]] += 1
result = dict(export_files_verified=len(expected), index_files=len(index),
              changed_counts=dict(collections.Counter(x['status'] for x in changes)),
              changed_bytes=sum(x['bytes'] for x in changes), remote_only_preserved=len(plan['remote_only_preserved']),
              whitespace_check_exit=check.returncode, whitespace_warning_counts=dict(warning_counts),
              whitespace_policy='Preserve original archive bytes; whitespace is not rewritten.',
              changed_files=changes)
(BASE / 'staged_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({key: value for key, value in result.items() if key != 'changed_files'}, ensure_ascii=False, indent=2))
