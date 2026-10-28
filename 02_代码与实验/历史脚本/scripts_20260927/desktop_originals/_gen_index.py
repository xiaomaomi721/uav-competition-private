#!/usr/bin/env python3
"""Generate AUDIT_INDEX.json."""
import os, json, hashlib
from pathlib import Path

AUDIT = Path('/home/jhy/rare_diag_20260915/b4_migration_6296/audit_20260927')

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

# Standard name -> old prefix name mapping
files = [
    'RETROSPECTIVE_EXECUTION_REPORT.md',
    'B4_B3_COMPARABILITY.json',
    'CHECKPOINT_AUDIT.json',
    'PSEUDO_PAYLOAD_MANIFEST.json',
    'ARTIFACT_HASHES.json',
    'RETENTION_MANIFEST.json',
]

# Old prefix files (retained, superseded)
old_files = [
    'b4_audit_RETROSPECTIVE_EXECUTION_REPORT.md',
    'b4_audit_B4_B3_COMPARABILITY.json',
    'b4_audit_CHECKPOINT_AUDIT.json',
    'b4_audit_ARTIFACT_HASHES.json',
    'b4_audit_RETENTION_MANIFEST.json',
]

index = {
    'audit_dir': str(AUDIT),
    'standard_files': {},
    'old_prefix_files': {},
    'pseudo_payload': {
        'count': 500,
        'root': '/home/jhy/pseudo_labels_v2_multi/masks_official',
        'all_symlink': False,
        'note': 'Files are regular files, not symlinks. This freezes B4-era payload; does not prove B3 read same payload.'
    },
    'pending_fields': [],
    'self_gpu_modified': False,
    'base_probe_modified': False,
    'b3_touched': False,
}

for f in files:
    p = AUDIT / f
    if p.exists():
        index['standard_files'][f] = {
            'path': str(p),
            'sha256': sha256_file(str(p)),
            'size': os.path.getsize(str(p)),
        }

for f in old_files:
    p = AUDIT / f
    if p.exists():
        index['old_prefix_files'][f] = {
            'path': str(p),
            'sha256': sha256_file(str(p)),
            'superseded_by': f.replace('b4_audit_', ''),
        }

# AUDIT_INDEX self-excluded
index['self_sha256'] = None
index['note'] = 'AUDIT_INDEX.json itself is not hashed.'

(AUDIT / 'AUDIT_INDEX.json').write_text(json.dumps(index, indent=2))
print(f"Wrote AUDIT_INDEX.json")
print(f"Standard files: {len(index['standard_files'])}")
print(f"Old prefix files: {len(index['old_prefix_files'])}")
print(f"Pseudo count: {index['pseudo_payload']['count']}")

# List all files in audit dir
print("\n=== Final audit dir ===")
for f in sorted(AUDIT.iterdir()):
    print(f"  {f.name}  {f.stat().st_size} bytes")
