"""Audit desktop-to-Git changes without modifying desktop or Git checkout."""
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import zipfile

BASE = Path(__file__).resolve().parent
WORK = BASE.parent
SOURCE = Path(r'C:\Users\Lenovo\OneDrive\Documents\桌面\无人机')
REPO = WORK / '_github_private/repo'
spec = importlib.util.spec_from_file_location('prior_uav_export', WORK / 'GitHub私有归档_20261028/prepare_snapshot.py')
rules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rules)

def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args])
def blob_hash(path):
    size = path.stat().st_size
    h = hashlib.sha1(b'blob ' + str(size).encode() + b'\0')
    h256 = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
            h256.update(chunk)
    return h.hexdigest(), h256.hexdigest()

assert not git('status', '--porcelain=v1').strip(), 'Local Git checkout contains changes; review first.'
head = git('rev-parse', 'HEAD').decode().strip()
remote = git('rev-parse', 'origin/codex/project-archive-20261028').decode().strip()
assert head == remote, 'Refresh checkout to remote before preparing export.'
tracked = {}
for record in git('ls-tree', '-rz', 'HEAD').split(b'\0'):
    if record:
        meta, path = record.split(b'\t', 1)
        tracked[path.decode('utf-8')] = meta.decode().split()[2]

entries, findings, excluded_dirs = [], [], []
extra_binary = {'.npy', '.npz', '.bin', '.pkl', '.pickle', '.h5', '.hdf5', '.parquet', '.mp4', '.avi', '.mov'}
for directory, dirs, names in os.walk(SOURCE):
    parent = Path(directory)
    for name in dirs[:]:
        if name.lower() in rules.BLOCK_DIRS or (parent / name).is_symlink():
            dirs.remove(name)
            excluded_dirs.append((parent / name).relative_to(SOURCE).as_posix())
    for name in sorted(names):
        path = parent / name
        relative = path.relative_to(SOURCE).as_posix()
        reason = None
        if path.is_symlink(): reason = 'symbolic_link'
        elif name == '.env' or name.startswith('.env.') or path.suffix.lower() in {'.pem', '.key', '.p12', '.pfx'}: reason = 'credential_file'
        elif path.suffix.lower() in rules.LOCAL_EXT | extra_binary: reason = 'local_binary_artifact'
        elif path.stat().st_size > 95 * 1024**2: reason = 'oversize_file'
        blob, sha256 = blob_hash(path)
        entry = dict(path=relative, bytes=path.stat().st_size, sha256=sha256, git_blob_sha=blob,
                     action='local_only' if reason else ('unchanged' if tracked.get(relative) == blob else ('update' if relative in tracked else 'add')),
                     reason=reason)
        entries.append(entry)
        if entry['action'] not in ('add', 'update'):
            continue
        content = None
        if path.suffix.lower() in rules.TEXT_EXT | {'.xml', '.html', '.js', '.ts', '.bat', '.cmd'} or name.startswith('.'):
            try: content = path.read_text(encoding='utf-8-sig')
            except UnicodeDecodeError: content = path.read_text(encoding='gb18030', errors='replace')
        elif path.suffix.lower() in {'.docx', '.pptx', '.xlsx'}:
            with zipfile.ZipFile(path) as archive:
                content = '\n'.join(archive.read(n).decode('utf-8', 'replace') for n in archive.namelist() if n.endswith('.xml'))
        if content is not None:
            for kind, pattern in rules.STRONG + [('possible_literal_credential', rules.GENERIC)]:
                for match in pattern.finditer(content):
                    findings.append(dict(path=relative, line=content.count('\n', 0, match.start()) + 1,
                                         kind=kind, match_length=len(match.group(0))))

counts = dict(collections.Counter(x['action'] for x in entries))
plan = dict(repository='https://github.com/xiaomaomi721/uav-competition-private',
            branch='codex/project-archive-20261028', parent_commit=head,
            source=str(SOURCE), checkout=str(REPO), files=entries,
            remote_only_preserved=sorted(set(tracked) - {x['path'] for x in entries}),
            excluded_directories=excluded_dirs, credential_candidates=findings,
            summary=dict(counts=counts, upload_bytes=sum(x['bytes'] for x in entries if x['action'] in ('add', 'update')),
                         local_only_bytes=sum(x['bytes'] for x in entries if x['action'] == 'local_only')))
(BASE / 'upload_plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'summary': plan['summary'], 'credential_candidates': findings,
                  'excluded_directories': excluded_dirs,
                  'updates': [x['path'] for x in entries if x['action'] == 'update'],
                  'new_top_folders': dict(collections.Counter('/'.join(x['path'].split('/')[:2]) for x in entries if x['action'] == 'add'))},
                 ensure_ascii=False, indent=2))
