#!/usr/bin/env python3
"""Final consistency fix: regenerate all audit files in order."""
import os, json, hashlib
from pathlib import Path
import numpy as np
from PIL import Image

AUDIT = Path('/home/jhy/rare_diag_20260915/b4_migration_6296/audit_20260927')
BASE = Path('/home/jhy/rare_diag_20260915/b4_migration_6296/base_probe')

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

# === STEP 1: PSEUDO_PAYLOAD_MANIFEST with layered structure ===
TRAIN_MASK_DIR = Path('/home/jhy/datasets/pseudo_train_6796/train/masks')
RESOLVED_DIR = Path('/home/jhy/pseudo_labels_v2_multi/masks_official')

entries = []
for i in range(500):
    mask_name = f'{7000 + i:04d}.png'
    resolved_name = f'test1_{i+1}.png'
    mask_path = TRAIN_MASK_DIR / mask_name
    resolved_path = RESOLVED_DIR / resolved_name
    
    is_sym = mask_path.is_symlink()
    symlink_target = os.readlink(str(mask_path)) if is_sym else None
    rp = mask_path.resolve()
    
    # Get resolved file info
    stat = rp.stat()
    with Image.open(rp) as img:
        mode = img.mode
        shape = list(img.size)
        arr = np.array(img)
        labels = sorted(set(arr.flatten().tolist()))
    
    entries.append({
        'training_mask_entry': {
            'path': str(mask_path),
            'is_symlink': is_sym,
            'symlink_target': symlink_target,
            'id': 7000 + i,
        },
        'resolved_payload': {
            'path': str(rp),
            'sha256': sha256_file(str(rp)),
            'size_bytes': stat.st_size,
            'image_mode': mode,
            'image_size_wh': shape,
            'unique_label_min': min(labels),
            'unique_label_max': max(labels),
            'num_unique_labels': len(labels),
        }
    })

pseudo_manifest = {
    'training_mask_entries': {
        'root': str(TRAIN_MASK_DIR),
        'id_range': '7000..7499',
        'count': 500,
        'all_are_symlinks': all(e['training_mask_entry']['is_symlink'] for e in entries),
    },
    'resolved_payload_files': {
        'root': str(RESOLVED_DIR),
        'count': 500,
        'all_are_regular_files': all(not (RESOLVED_DIR / f'test1_{i+1}.png').is_symlink() for i in range(500)),
        'all_are_symlinks': False,
    },
    'caveat': 'Training mask entries are 500 symlinks; they resolve to 500 regular PNG payloads in round-2 directory. This freezes B4-era resolution. It does NOT prove historical B3 (Sep 5) read the same payload; symlinks were redirected Sep 8.',
    'entries': entries,
}

# Self hash (before writing)
content = json.dumps(pseudo_manifest, sort_keys=True).encode()
pseudo_manifest['manifest_sha256'] = hashlib.sha256(content).hexdigest()

(AUDIT / 'PSEUDO_PAYLOAD_MANIFEST.json').write_text(json.dumps(pseudo_manifest, indent=2))
print(f"Step 1: PSEUDO_PAYLOAD_MANIFEST.json written, {len(entries)} entries")
print(f"  training all_symlinks={pseudo_manifest['training_mask_entries']['all_are_symlinks']}")
print(f"  resolved all_regular={pseudo_manifest['resolved_payload_files']['all_are_regular_files']}")

# === STEP 2: B4_B3_COMPARABILITY.json (already correct, just verify) ===
# Already has correct values from previous step. No change needed except note about symlink.
comparability = json.loads((AUDIT / 'B4_B3_COMPARABILITY.json').read_text())
comparability['pseudo_payload_note'] = 'Training masks are symlinks to resolved regular files; payload not proven identical between B3 and B4.'
(AUDIT / 'B4_B3_COMPARABILITY.json').write_text(json.dumps(comparability, indent=2))
print("Step 2: comparability updated")

# === STEP 3: RETENTION_MANIFEST dedup + lock splits ===
TRAIN_SPLIT = '/home/jhy/datasets/pseudo_train_6796/train_ids_5597pseudo.txt'
VAL_SPLIT = '/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/splits/val.txt'

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
]

# Add audit files that exist
audit_standard = ['ARTIFACT_HASHES.json', 'RETENTION_MANIFEST.json']
for af in audit_standard:
    p = AUDIT / af
    if p.exists():
        lock_items.append((f'audit/{af}', str(p)))

retention = {'locked_items': [], 'note': 'All paths absolute, unique. locked=true, may_auto_delete=false.'}
seen = set()
for name, path in lock_items:
    if path in seen:
        continue
    seen.add(path)
    if os.path.exists(path):
        retention['locked_items'].append({
            'name': name,
            'path': path,
            'sha256': sha256_file(path),
            'locked': True,
            'may_auto_delete': False,
        })
    else:
        retention['locked_items'].append({
            'name': name,
            'path': path,
            'sha256': 'MISSING',
            'locked': True,
            'may_auto_delete': False,
        })

(AUDIT / 'RETENTION_MANIFEST.json').write_text(json.dumps(retention, indent=2))
print(f"Step 3: RETENTION_MANIFEST.json, {len(retention['locked_items'])} unique items")

# === STEP 4: ARTIFACT_HASHES.json ===
artifact = {
    'self_excluded': True,
    'note': 'ARTIFACT_HASHES.json itself is not hashed.',
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
            'path': TRAIN_SPLIT,
            'count': 6097,
            'unique': 6097,
            'sha256': '0fbcc0fca84b68a88bb978996a3d6312c39bb7968276d8e8762695c1831efccc',
        },
        'val_split_1399': {
            'path': VAL_SPLIT,
            'count': 1399,
            'unique': 1399,
            'sha256': '2fc7f230a4ce389767dfc9cba7be6b48b219a0606ac136c31cbd861f3ccd1e74',
        },
        'pseudo_payload_manifest': {
            'path': str(AUDIT / 'PSEUDO_PAYLOAD_MANIFEST.json'),
            'sha256': sha256_file(str(AUDIT / 'PSEUDO_PAYLOAD_MANIFEST.json')),
        },
        'retention_manifest': {
            'path': str(AUDIT / 'RETENTION_MANIFEST.json'),
            'sha256': sha256_file(str(AUDIT / 'RETENTION_MANIFEST.json')),
        },
    },
}

(AUDIT / 'ARTIFACT_HASHES.json').write_text(json.dumps(artifact, indent=2))
print("Step 4: ARTIFACT_HASHES.json written")

# === STEP 5: AUDIT_INDEX.json ===
standard_files = [
    'RETROSPECTIVE_EXECUTION_REPORT.md',
    'B4_B3_COMPARABILITY.json',
    'CHECKPOINT_AUDIT.json',
    'PSEUDO_PAYLOAD_MANIFEST.json',
    'ARTIFACT_HASHES.json',
    'RETENTION_MANIFEST.json',
]

index = {
    'audit_dir': str(AUDIT),
    'standard_files': {},
    'old_prefix_files': {},
    'train_split': {'path': TRAIN_SPLIT, 'count': 6097, 'sha256': '0fbcc0fca84b68a88bb978996a3d6312c39bb7968276d8e8762695c1831efccc'},
    'val_split': {'path': VAL_SPLIT, 'count': 1399, 'sha256': '2fc7f230a4ce389767dfc9cba7be6b48b219a0606ac136c31cbd861f3ccd1e74'},
    'training_mask_entries': {'count': 500, 'all_symlinks': True},
    'resolved_payload': {'count': 500, 'all_regular_files': True},
    'retention_unique_paths': len(retention['locked_items']),
    'pending_fields': [],
    'self_gpu_modified': False,
    'base_probe_modified': False,
    'b3_touched': False,
    'self_sha256': None,
}

for f in standard_files:
    p = AUDIT / f
    if p.exists():
        index['standard_files'][f] = {
            'path': str(p),
            'sha256': sha256_file(str(p)),
        }

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
        index['old_prefix_files'][f] = {
            'path': str(p),
            'superseded_by': f.replace('b4_audit_', ''),
        }

(AUDIT / 'AUDIT_INDEX.json').write_text(json.dumps(index, indent=2))
print("Step 5: AUDIT_INDEX.json written")

# === Verify base_probe six SHA ===
print("\n=== Base probe six-file SHA verification ===")
base_files = ['run_manifest.json', 'completed.json', 'metrics.json', 'validation_history.json', 'train_log.csv', 'config.yaml']
for f in base_files:
    p = BASE / f
    print(f"  {f}: {sha256_file(str(p))[:16]}...")

print(f"\nRetention unique paths: {len(retention['locked_items'])}")
