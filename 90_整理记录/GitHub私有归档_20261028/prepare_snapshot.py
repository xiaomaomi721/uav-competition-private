"""Export a private-repository snapshot; source files are never modified."""
import os, re, json, hashlib, shutil, zipfile, collections
from pathlib import Path

BASE = Path(__file__).resolve().parent
SOURCE = Path(r'C:\Users\Lenovo\OneDrive\Documents\桌面\无人机')
DEST = BASE.parent / '_github_private' / 'repo'
BLOCK_DIRS = {'.git', '.claude', '.codex', '.agents', '.aws', '.ssh', '__pycache__', '.ruff_cache', '.pytest_cache', '.venv', 'venv'}
LOCAL_EXT = {'.zip', '.rar', '.7z', '.tar', '.gz', '.pt', '.pth', '.ckpt', '.safetensors', '.onnx', '.pyc'}
TEXT_EXT = {'.py', '.md', '.json', '.yaml', '.yml', '.txt', '.csv', '.jsonl', '.sh', '.ps1', '.log', '.toml', '.ini', '.cfg'}
STRONG = [
    ('private_key', re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----')),
    ('github_token', re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b')),
    ('huggingface_token', re.compile(r'\bhf_[A-Za-z0-9]{25,}\b')),
    ('openai_key', re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}\b')),
    ('credential_url', re.compile(r'https?://[^\s/:]+:[^\s/@]+@[^\s/]+')),
]
GENERIC = re.compile(r'''(?im)(?:password|passwd|api_key|access_token|auth_token|secret_key|密码)\s*[=:：]\s*["']?([^\s"'`,;\)\]}]{4,})''')

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    files, findings, excluded_dirs = [], [], []
    for directory, dirs, names in os.walk(SOURCE):
        here = Path(directory)
        for name in dirs[:]:
            if name in BLOCK_DIRS or (here / name).is_symlink():
                dirs.remove(name)
                excluded_dirs.append((here / name).relative_to(SOURCE).as_posix())
        for name in sorted(names):
            p = here / name
            rel = p.relative_to(SOURCE).as_posix()
            reason = None
            if p.is_symlink(): reason = 'symbolic_link'
            elif p.name == '.env' or p.name.startswith('.env.') or p.suffix.lower() in {'.pem', '.key', '.p12', '.pfx'}: reason = 'credential_file'
            elif p.suffix.lower() in LOCAL_EXT: reason = 'local_binary_artifact'
            elif p.stat().st_size > 95 * 1024**2: reason = 'oversize_file'
            entry = {'path': rel, 'bytes': p.stat().st_size, 'sha256': digest(p), 'action': 'local_only' if reason else 'upload', 'reason': reason}
            files.append(entry)
            if reason: continue
            content = None
            if p.suffix.lower() in TEXT_EXT or p.name.startswith('.'):
                try: content = p.read_text(encoding='utf-8-sig')
                except UnicodeDecodeError: content = p.read_text(encoding='gb18030', errors='replace')
            elif p.suffix.lower() in {'.docx', '.pptx'}:
                with zipfile.ZipFile(p) as z:
                    content = '\n'.join(z.read(n).decode('utf-8', 'replace') for n in z.namelist() if n.endswith('.xml'))
            if content is not None:
                for kind, pattern in STRONG + [('possible_literal_credential', GENERIC)]:
                    for m in pattern.finditer(content):
                        findings.append({'path': rel, 'line': content.count('\n', 0, m.start()) + 1, 'kind': kind, 'match_length': len(m.group(0))})
    report = {'source': str(SOURCE), 'snapshot': str(DEST), 'files': files, 'excluded_directories': excluded_dirs,
              'credential_scan': findings, 'summary': {'included_files': sum(f['action']=='upload' for f in files), 'included_bytes': sum(f['bytes'] for f in files if f['action']=='upload'), 'local_only_files': sum(f['action']=='local_only' for f in files), 'local_only_bytes': sum(f['bytes'] for f in files if f['action']=='local_only'), 'credential_candidates': len(findings)}}
    (BASE / 'snapshot_plan.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'summary': report['summary'], 'excluded_directories': excluded_dirs, 'credential_scan': findings}, ensure_ascii=False))

if __name__ == '__main__': main()
