"""Prepare XSB repository additions; deduplicate the archived desktop material."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent / 'uav_semantic_segmentation'
DESKTOP = Path(r'C:\Users\Lenovo\OneDrive\Documents\桌面\无人机')
manifest = json.loads((ROOT / 'remote_manifest.json').read_text(encoding='utf-8'))
assert not manifest['truncated']
commit = manifest['commit']
def git(*args, **kwargs):
    return subprocess.check_output(['git', '-C', str(REPO), *args], **kwargs)
def blob_sha(content):
    return hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
assert git('rev-parse', commit + '^{tree}').decode().strip() == manifest['tree']
local = {}
for record in git('ls-tree', '-rz', commit).split(b'\0'):
    if record:
        meta, path = record.split(b'\t', 1)
        mode, kind, sha = meta.decode().split()
        assert kind == 'blob' and mode in ('100644', '100755')
        local[path.decode('utf-8')] = sha
assert local == {x['path']: x['sha'] for x in manifest['files']}

files = manifest['files']
batch = git('cat-file', '--batch', input=('\n'.join(x['sha'] for x in files) + '\n').encode())
contents = {}
pos = 0
for item in files:
    end = batch.index(b'\n', pos)
    sha, kind, size = batch[pos:end].decode().split()
    count = int(size)
    content = batch[end + 1:end + 1 + count]
    assert sha == item['sha'] and kind == 'blob' and count == item['size'] and blob_sha(content) == sha
    contents[item['path']] = content
    pos = end + count + 2
assert pos == len(batch)

sizes = {x['size'] for x in files if x['path'].startswith('无人机/')}
existing = {}
for directory, dirs, names in os.walk(DESKTOP):
    dirs[:] = [name for name in dirs if name not in ('.git', '.claude', '__pycache__')]
    for name in names:
        path = Path(directory) / name
        if path.stat().st_size in sizes:
            existing.setdefault(blob_sha(path.read_bytes()), []).append(path.relative_to(DESKTOP).as_posix())

with (DESKTOP / '90_整理记录/20261028/新旧路径映射.csv').open(encoding='utf-8-sig', newline='') as stream:
    mapping = {x['old_path'].replace('\\', '/'): x['new_path'].replace('\\', '/') for x in csv.DictReader(stream)}
core_root = '02_代码与实验/GitHub同步_20261029/uav_semantic_segmentation'
archive_root = '01_当前比赛/XSB新增归档_20261029'
records = []
for item in files:
    record = dict(source_path=item['path'], git_blob_sha=item['sha'], bytes=item['size'])
    content = contents[item['path']]
    record['sha256'] = hashlib.sha256(content).hexdigest()
    if item['path'].startswith('无人机/'):
        old_path = item['path'][len('无人机/'):]
        if item['sha'] in existing:
            record.update(action='already_present_elsewhere', existing_paths=existing[item['sha']])
            records.append(record)
            continue
        relative = mapping.get(old_path, archive_root + '/' + old_path)
    else:
        # Keep this absent code project together so relative imports/config paths work.
        relative = core_root + '/' + item['path']
    destination = DESKTOP.joinpath(*relative.split('/'))
    destination.resolve().relative_to(DESKTOP.resolve())
    record['destination_relative'] = relative
    if destination.exists():
        assert destination.is_file(), destination
        record['existing_sha256'] = hashlib.sha256(destination.read_bytes()).hexdigest()
        record['action'] = 'identical_at_target' if record['existing_sha256'] == record['sha256'] else 'different_preserve'
    else:
        record['action'] = 'add_missing'
        stage = ROOT / 'payload' / relative
        stage.parent.mkdir(parents=True, exist_ok=True)
        stage.write_bytes(content)
    records.append(record)

plan = dict(repository=manifest['repository'], commit=commit, tree=manifest['tree'],
            destination=str(DESKTOP), code_root=core_root, new_archive_root=archive_root, files=records)
(ROOT / 'sync_plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
summary = {'counts': {action: sum(x['action'] == action for x in records) for action in sorted({x['action'] for x in records})},
           'add_bytes': sum(x['bytes'] for x in records if x['action'] == 'add_missing'),
           'core_additions': sum(x['action'] == 'add_missing' and not x['source_path'].startswith('无人机/') for x in records),
           'archive_additions': [x['source_path'] for x in records if x['action'] == 'add_missing' and x['source_path'].startswith('无人机/')],
           'different': [x['source_path'] for x in records if x['action'] == 'different_preserve']}
print(json.dumps(summary, ensure_ascii=False, indent=2))
