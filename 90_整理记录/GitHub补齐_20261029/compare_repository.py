"""Compare verified Git snapshot with desktop; create an additive sync plan."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent / '_github_private' / 'repo'
DESKTOP = Path(r'C:\Users\Lenovo\OneDrive\Documents\桌面\无人机')
COMMIT = 'eac684521375d03bb1dc3e90b603b7eedd23e7c8'
TREE = '4e19f2032cbd125682f14a6c40f6fe827e2beb59'

def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args])

assert git('rev-parse', COMMIT + '^{tree}').decode().strip() == TREE
records = []
for entry in git('ls-tree', '-rz', COMMIT).split(b'\0'):
    if not entry:
        continue
    meta, path = entry.split(b'\t', 1)
    mode, kind, sha = meta.decode().split()
    assert kind == 'blob' and mode in ('100644', '100755')
    relative = path.decode('utf-8')
    target = DESKTOP.joinpath(*relative.split('/'))
    target.resolve().relative_to(DESKTOP.resolve())
    if target.is_file():
        content = target.read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        state = 'identical' if actual == sha else 'different_preserve'
    elif target.exists():
        state = 'path_conflict_preserve'
    else:
        state = 'missing'
    records.append({'path': relative, 'git_blob_sha': sha, 'mode': mode, 'state': state})
result = {'repository': 'https://github.com/xiaomaomi721/uav-competition-private',
          'branch': 'codex/project-archive-20261028', 'commit': COMMIT, 'tree': TREE,
          'remote_tree_verified_via': 'GitHub connector recursive tree, truncated=false',
          'source': str(REPO), 'destination': str(DESKTOP), 'files': records}
(ROOT / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'counts': {s: sum(r['state'] == s for r in records) for s in sorted({r['state'] for r in records})},
                  'missing': [r['path'] for r in records if r['state'] == 'missing'],
                  'different': [r['path'] for r in records if r['state'] == 'different_preserve']}, ensure_ascii=False))
