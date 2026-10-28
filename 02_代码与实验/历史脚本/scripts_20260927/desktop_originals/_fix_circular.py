#!/usr/bin/env python3
"""Fix circular hashing.
Order:
1. ARTIFACT_HASHES: exclude self, RETENTION, AUDIT_INDEX
2. RETENTION: lock ARTIFACT_HASHES SHA, exclude self and AUDIT_INDEX
3. AUDIT_INDEX: lock both, exclude self
"""
import json, hashlib, os
from pathlib import Path

AUDIT = Path('/home/jhy/rare_diag_20260915/b4_migration_6296/audit_20260927')
BASE = Path('/home/jhy/rare_diag_20260915/b4_migration_6296/base_probe')

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

# === STEP 1: ARTIFACT_HASHES (exclude self, RETENTION, AUDIT_INDEX) ===
TRAIN_SPLIT = '/home/jhy/datasets/pseudo_train_6796/train_ids_5597pseudo.txt'
VAL_SPLIT = '/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/splits/val.txt'

ah = {
    'self_excluded': True,
    'note': 'ARTIFACT_HASHES.json, RETENTION_MANIFEST.json, and AUDIT_INDEX.json are excluded from this manifest to avoid circular hashing.',
    'excluded_files': ['ARTIFACT_HASHES.json', 'RETENTION_MANIFEST.json', 'AUDIT_INDEX.json'],
    'files': {
        'launch_config': {
            'path': '/home/jhy/rare_diag_20260915/b4_migration_6296/b4_probe_config.yaml',
            'sha256': '38a7de390319a43c5a957a77cc4031fb1d513dc8723e49addb23b49f0ec52112',
        },
        'resolved_config': {
            'path': str(BASE / 'config.yaml'),
            'sha256': sha256_file(str(BASE / 'config.yaml')),
        },
        'run_manifest': {'path': str(BASE / 'run_manifest.json'), 'sha256': sha256_file(str(BASE / 'run_manifest.json'))},
        'completed': {'path': str(BASE / 'completed.json'), 'sha256': sha256_file(str(BASE / 'completed.json'))},
        'metrics': {'path': str(BASE / 'metrics.json'), 'sha256': sha256_file(str(BASE / 'metrics.json'))},
        'validation_history': {'path': str(BASE / 'validation_history.json'), 'sha256': sha256_file(str(BASE / 'validation_history.json'))},
        'train_log': {'path': str(BASE / 'train_log.csv'), 'sha256': sha256_file(str(BASE / 'train_log.csv'))},
        'best_miou': {'path': str(BASE / 'best_miou.pt'), 'sha256': 'd7a63b0c1124f5ab213b027125322ace8ce7cf834f48fcfb45e34b20fdef6000'},
        'last': {'path': str(BASE / 'last.pt'), 'sha256': '7e4d3376ec3fc12a0aa3c7eb418b6dd7f5b805eacedb180069522b5614a7f259'},
        'train_split_6097': {
            'path': TRAIN_SPLIT, 'count': 6097, 'unique': 6097,
            'sha256': '0fbcc0fca84b68a88bb978996a3d6312c39bb7968276d8e8762695c1831efccc',
        },
        'val_split_1399': {
            'path': VAL_SPLIT, 'count': 1399, 'unique': 1399,
            'sha256': '2fc7f230a4ce389767dfc9cba7be6b48b219a0606ac136c31cbd861f3ccd1e74',
        },
        'pseudo_payload_manifest': {
            'path': str(AUDIT / 'PSEUDO_PAYLOAD_MANIFEST.json'),
            'sha256': sha256_file(str(AUDIT / 'PSEUDO_PAYLOAD_MANIFEST.json')),
        },
        'retrospective_report': {
            'path': str(AUDIT / 'RETROSPECTIVE_EXECUTION_REPORT.md'),
            'sha256': sha256_file(str(AUDIT / 'RETROSPECTIVE_EXECUTION_REPORT.md')),
        },
        'comparability': {
            'path': str(AUDIT / 'B4_B3_COMPARABILITY.json'),
            'sha256': sha256_file(str(AUDIT / 'B4_B3_COMPARABILITY.json')),
        },
        'checkpoint_audit': {
            'path': str(AUDIT / 'CHECKPOINT_AUDIT.json'),
            'sha256': sha256_file(str(AUDIT / 'CHECKPOINT_AUDIT.json')),
        },
    },
}

(AUDIT / 'ARTIFACT_HASHES.json').write_text(json.dumps(ah, indent=2))
ah_sha = sha256_file(str(AUDIT / 'ARTIFACT_HASHES.json'))
print(f"Step 1: ARTIFACT_HASHES.json SHA={ah_sha[:16]}...")
print(f"  excludes self, RETENTION, AUDIT_INDEX")

# === STEP 2: RETENTION_MANIFEST (lock ARTIFACT_HASHES SHA, exclude self and AUDIT_INDEX) ===
lock_items = [
    ('B4 best checkpoint', str(BASE / 'best_miou.pt')),
    ('B4 last checkpoint', str(BASE / 'last.pt')),
    ('launch config', '/home/jhy/rare_diag_20260915/b4_migration_6296/b4_probe_config.yaml'),
    ('resolved config', str(BASE / 'config.yaml')),
    ('run manifest', str(BASE / 'run_manifest.json')),
    ('completed', str(BASE / 'completed.json')),
    ('metrics', str(BASE / 'metrics.json')),
    ('train log', str(BASE / 'train_log.csv')),
    ('validation history', str(BASE / 'validation_history.json')),
    ('actual 6097 train split', TRAIN_SPLIT),
    ('1399 validation split', VAL_SPLIT),
    ('pseudo payload manifest', str(AUDIT / 'PSEUDO_PAYLOAD_MANIFEST.json')),
    ('retrospective report', str(AUDIT / 'RETROSPECTIVE_EXECUTION_REPORT.md')),
    ('comparability', str(AUDIT / 'B4_B3_COMPARABILITY.json')),
    ('checkpoint audit', str(AUDIT / 'CHECKPOINT_AUDIT.json')),
    ('artifact hashes', str(AUDIT / 'ARTIFACT_HASHES.json')),
    # NOTE: excludes RETENTION_MANIFEST itself and AUDIT_INDEX
]

rm = {'locked_items': [], 'note': 'All paths absolute, unique. Excludes self (RETENTION_MANIFEST.json) and AUDIT_INDEX.json to avoid circular hashing.'}
seen = set()
for name, path in lock_items:
    if path in seen: continue
    seen.add(path)
    if os.path.exists(path):
        rm['locked_items'].append({
            'name': name, 'path': path,
            'sha256': sha256_file(path),
            'locked': True, 'may_auto_delete': False,
        })

(AUDIT / 'RETENTION_MANIFEST.json').write_text(json.dumps(rm, indent=2))
rm_sha = sha256_file(str(AUDIT / 'RETENTION_MANIFEST.json'))
print(f"Step 2: RETENTION_MANIFEST.json SHA={rm_sha[:16]}...")
print(f"  {len(rm['locked_items'])} unique items, excludes self and AUDIT_INDEX")

# === STEP 3: AUDIT_INDEX (lock both, exclude self) ===
standard_files = [
    'RETROSPECTIVE_EXECUTION_REPORT.md',
    'B4_B3_COMPARABILITY.json',
    'CHECKPOINT_AUDIT.json',
    'PSEUDO_PAYLOAD_MANIFEST.json',
    'ARTIFACT_HASHES.json',
    'RETENTION_MANIFEST.json',
]

ai = {
    'audit_dir': str(AUDIT),
    'standard_files': {},
    'old_prefix_files': {},
    'train_split': {'path': TRAIN_SPLIT, 'count': 6097, 'sha256': '0fbcc0fca84b68a88bb978996a3d6312c39bb7968276d8e8762695c1831efccc'},
    'val_split': {'path': VAL_SPLIT, 'count': 1399, 'sha256': '2fc7f230a4ce389767dfc9cba7be6b48b219a0606ac136c31cbd861f3ccd1e74'},
    'training_mask_entries': {'count': 500, 'all_symlinks': True},
    'resolved_payload': {'count': 500, 'all_regular_files': True},
    'retention_unique_paths': len(rm['locked_items']),
    'pending_fields': [],
    'self_gpu_modified': False,
    'base_probe_modified': False,
    'b3_touched': False,
    'self_sha256': None,
    'circular_hash_policy': 'ARTIFACT_HASHES excludes self/RETENTION/INDEX; RETENTION locks ARTIFACT_HASHES but excludes self/INDEX; INDEX locks both but excludes self.',
}

for f in standard_files:
    p = AUDIT / f
    if p.exists():
        ai['standard_files'][f] = {'path': str(p), 'sha256': sha256_file(str(p))}

old_files = [
    'b4_audit_RETROSPECTIVE_EXECUTION_REPORT.md',
    'b4_audit_B4_B3_COMPARABILITY.json',
    'b4_audit_CHECKPOINT_AUDIT.json',
    'b4_audit_ARTIFACT_HASHES.json',
    'b4_audit_RETENTION_MANIFEST.json',
]
for f in old_files:
    p = AUDIT / f
    if p.exists():
        ai['old_prefix_files'][f] = {'path': str(p), 'superseded_by': f.replace('b4_audit_', '')}

(AUDIT / 'AUDIT_INDEX.json').write_text(json.dumps(ai, indent=2))
ai_sha = sha256_file(str(AUDIT / 'AUDIT_INDEX.json'))
print(f"Step 3: AUDIT_INDEX.json SHA={ai_sha[:16]}...")
print(f"  locks both ARTIFACT_HASHES and RETENTION, excludes self")

print(f"\n=== Final SHAs ===")
print(f"ARTIFACT_HASHES: {ah_sha}")
print(f"RETENTION_MANIFEST: {rm_sha}")
print(f"AUDIT_INDEX: {ai_sha}")
print(f"\nRetention paths: {len(rm['locked_items'])}")
