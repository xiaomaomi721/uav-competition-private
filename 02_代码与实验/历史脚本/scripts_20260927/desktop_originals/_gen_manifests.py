#!/usr/bin/env python3
"""Generate complete ARTIFACT_HASHES.json and RETENTION_MANIFEST.json."""
import os, json, hashlib
from pathlib import Path

AUDIT = Path('/home/jhy/rare_diag_20260915/b4_migration_6296/audit_20260927')
BASE = Path('/home/jhy/rare_diag_20260915/b4_migration_6296/base_probe')

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

# Files to hash
files = {
    'launch_config': '/home/jhy/rare_diag_20260915/b4_migration_6296/b4_probe_config.yaml',
    'resolved_config': str(BASE / 'config.yaml'),
    'run_manifest': str(BASE / 'run_manifest.json'),
    'completed': str(BASE / 'completed.json'),
    'metrics': str(BASE / 'metrics.json'),
    'validation_history': str(BASE / 'validation_history.json'),
    'train_log': str(BASE / 'train_log.csv'),
    'best_miou': str(BASE / 'best_miou.pt'),
    'last': str(BASE / 'last.pt'),
    'train_split': '/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/splits/train.txt',
}

hashes = {}
for key, path in files.items():
    if os.path.exists(path):
        hashes[key] = {'path': path, 'sha256': sha256_file(path), 'size': os.path.getsize(path)}
    else:
        hashes[key] = {'path': path, 'sha256': 'MISSING', 'size': 0}

# Pseudo manifest
pseudo_path = str(AUDIT / 'PSEUDO_PAYLOAD_MANIFEST.json')
hashes['pseudo_payload_manifest'] = {'path': pseudo_path, 'sha256': sha256_file(pseudo_path), 'size': os.path.getsize(pseudo_path)}

# Audit files (standard names)
audit_files = [
    'RETROSPECTIVE_EXECUTION_REPORT.md',
    'B4_B3_COMPARABILITY.json',
    'CHECKPOINT_AUDIT.json',
    'RETENTION_MANIFEST.json',
]
for af in audit_files:
    p = AUDIT / af
    if p.exists():
        hashes[f'audit_{af}'] = {'path': str(p), 'sha256': sha256_file(str(p)), 'size': os.path.getsize(str(p))}

# Build ARTIFACT_HASHES
artifact = {
    'self_excluded': True,
    'note': 'ARTIFACT_HASHES.json itself is not hashed (self-reference excluded).',
    'files': hashes,
    'launch_config_sha256': '38a7de390319a43c5a957a77cc4031fb1d513dc8723e49addb23b49f0ec52112',
    'resolved_config_sha256': '12eb53e666a1954aeabd665b9aad960f6a0db4247a3acc35887e68afaf2aa917',
    'train_split_sha256': '0fbcc0fca84b68a88bb978996a3d6312c39bb7968276d8e8762695c1831efccc',
    'val_split_sha256': '2fc7f230a4ce389767dfc9cba7be6b48b219a0606ac136c31cbd861f3ccd1e74',
}

(AUDIT / 'ARTIFACT_HASHES.json').write_text(json.dumps(artifact, indent=2))
print("Wrote ARTIFACT_HASHES.json")

# Build RETENTION_MANIFEST with absolute paths and hashes
retention = {
    'locked_items': [],
    'note': 'All paths absolute. locked=true means do not delete/modify.',
}

lock_items = [
    ('B4 best checkpoint', str(BASE / 'best_miou.pt'), 'selected_for_zero_bias_package'),
    ('B4 last checkpoint', str(BASE / 'last.pt'), 'historical_record'),
    ('launch config', '/home/jhy/rare_diag_20260915/b4_migration_6296/b4_probe_config.yaml', 'provenance'),
    ('resolved config', str(BASE / 'config.yaml'), 'provenance'),
    ('run manifest', str(BASE / 'run_manifest.json'), 'provenance'),
    ('completed', str(BASE / 'completed.json'), 'provenance'),
    ('metrics', str(BASE / 'metrics.json'), 'results'),
    ('train log', str(BASE / 'train_log.csv'), 'provenance'),
    ('validation history', str(BASE / 'validation_history.json'), 'results'),
    ('pseudo payload manifest', pseudo_path, 'data_provenance'),
]

for name, path, reason in lock_items:
    if os.path.exists(path):
        retention['locked_items'].append({
            'name': name,
            'path': path,
            'sha256': sha256_file(path),
            'locked': True,
            'may_auto_delete': False,
            'reason': reason,
        })
    else:
        retention['locked_items'].append({
            'name': name,
            'path': path,
            'sha256': 'MISSING',
            'locked': True,
            'may_auto_delete': False,
            'reason': reason + ' (FILE MISSING)',
        })

# Audit files
for af in ['RETROSPECTIVE_EXECUTION_REPORT.md', 'B4_B3_COMPARABILITY.json', 'CHECKPOINT_AUDIT.json', 'ARTIFACT_HASHES.json', 'RETENTION_MANIFEST.json', 'PSEUDO_PAYLOAD_MANIFEST.json', 'AUDIT_INDEX.json']:
    p = AUDIT / af
    if p.exists():
        retention['locked_items'].append({
            'name': f'audit/{af}',
            'path': str(p),
            'sha256': sha256_file(str(p)),
            'locked': True,
            'may_auto_delete': False,
            'reason': 'audit_record',
        })

(AUDIT / 'RETENTION_MANIFEST.json').write_text(json.dumps(retention, indent=2))
print("Wrote RETENTION_MANIFEST.json")
print(f"Locked {len(retention['locked_items'])} items")
