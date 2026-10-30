"""Prepare missing archive material without changing desktop files."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / 'comparison.json').read_text(encoding='utf-8'))
source = Path(data['source'])
desktop = Path(data['destination'])
payload = ROOT / 'payload'

def blob_sha(content):
    return hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()

pending = []
for item in data['files']:
    if item['state'] != 'missing':
        continue
    if item['path'] == '.gitignore':
        item['sync_action'] = 'skip_repository_control_file'
        continue
    content = subprocess.check_output(['git', '-C', str(source), 'cat-file', 'blob', item['git_blob_sha']])
    assert blob_sha(content) == item['git_blob_sha']
    item.update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    pending.append((item, content))

# Check relocated files by exact Git blob hash, reading only matching sizes.
sizes = {len(content) for _, content in pending}
by_hash = {}
for directory, dirs, files in os.walk(desktop):
    dirs[:] = [d for d in dirs if d not in ('.git', '.claude', '__pycache__')]
    for name in files:
        path = Path(directory) / name
        if path.stat().st_size in sizes:
            digest = blob_sha(path.read_bytes())
            by_hash.setdefault(digest, []).append(path.relative_to(desktop).as_posix())

additions = []
for item, content in pending:
    if item['git_blob_sha'] in by_hash:
        item.update(sync_action='already_present_elsewhere', existing_paths=by_hash[item['git_blob_sha']])
        continue
    relative = item['path']
    if relative == 'README.md' or relative.startswith('99_GitHub同步/'):
        relative = '90_整理记录/GitHub补齐_20261029/仓库附带记录/' + relative
    target = payload.joinpath(*relative.split('/'))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    item.update(sync_action='add_missing', destination_relative=relative)
    additions.append({key: item[key] for key in ('path', 'destination_relative', 'bytes', 'git_blob_sha', 'sha256')})

# Record every existing repository-path file for a post-sync preservation check.
preserve = []
for item in data['files']:
    if item['state'] in ('identical', 'different_preserve'):
        path = desktop.joinpath(*item['path'].split('/'))
        preserve.append({'path': item['path'], 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
data.update(additions=additions, preserve=preserve)
(ROOT / 'sync_plan.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'additions': len(additions), 'bytes': sum(x['bytes'] for x in additions),
                  'relocated': [x['path'] for x in data['files'] if x.get('sync_action') == 'already_present_elsewhere'],
                  'prepared': [x['destination_relative'] for x in additions]}, ensure_ascii=False, indent=2))
