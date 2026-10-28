#!/usr/bin/env python3
"""Enumerate 500 pseudo masks for PSEUDO_PAYLOAD_MANIFEST."""
import os, json, hashlib
from pathlib import Path
from PIL import Image

PSEUDO_DIR = Path('/home/jhy/pseudo_labels_v2_multi/masks_official')
OUT = Path('/home/jhy/rare_diag_20260915/b4_migration_6296/audit_20260927/PSEUDO_PAYLOAD_MANIFEST.json')

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

entries = []
files = sorted(PSEUDO_DIR.glob('*.png'))
print(f"Found {len(files)} files")

for f in files:
    rp = f.resolve()
    stat = rp.stat()
    with Image.open(rp) as img:
        mode = img.mode
        shape = img.size  # (W, H)
        import numpy as np
        arr = np.array(img)
        labels = sorted(set(arr.flatten().tolist()))
    entries.append({
        'filename': f.name,
        'symlink_path': str(f),
        'is_symlink': f.is_symlink(),
        'resolved_absolute_path': str(rp),
        'resolved_sha256': sha256_file(rp),
        'size_bytes': stat.st_size,
        'image_mode': mode,
        'image_size_wh': list(shape),
        'unique_label_range': [min(labels), max(labels)],
        'num_unique_labels': len(labels),
    })

manifest = {
    'count': len(entries),
    'all_symlink': all(e['is_symlink'] for e in entries),
    'all_resolve_under': '/home/jhy/pseudo_labels_v2_multi/masks_official',
    'source_period': 'B4-era (2026-09-26)',
    'caveat': 'This freezes the payload as readable during B4 era. It does NOT prove B3 (Sep 5) read the same payload; symlinks were redirected Sep 8.',
    'entries': entries,
}

# Self hash
content = json.dumps(manifest, sort_keys=True).encode()
manifest_sha = hashlib.sha256(content).hexdigest()
manifest['manifest_sha256'] = manifest_sha

OUT.write_text(json.dumps(manifest, indent=2))
print(f"Saved {OUT}")
print(f"count={len(entries)}, all_symlink={manifest['all_symlink']}")
print(f"manifest_sha256={manifest_sha}")
print(f"first entry: {entries[0]['filename']}, labels={entries[0]['unique_label_range']}")
print(f"last entry: {entries[-1]['filename']}, labels={entries[-1]['unique_label_range']}")
