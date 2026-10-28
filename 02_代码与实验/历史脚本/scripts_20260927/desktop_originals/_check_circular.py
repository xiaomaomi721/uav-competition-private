#!/usr/bin/env python3
import json, hashlib, os
from pathlib import Path

AUDIT = Path('/home/jhy/rare_diag_20260915/b4_migration_6296/audit_20260927')

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

# Check current state
ah = json.load(open(AUDIT/'ARTIFACT_HASHES.json'))
rm = json.load(open(AUDIT/'RETENTION_MANIFEST.json'))
ai = json.load(open(AUDIT/'AUDIT_INDEX.json'))

print("=== ARTIFACT_HASHES ===")
print("self_excluded:", ah.get('self_excluded'))
print("files keys:", list(ah['files'].keys()))
print("includes retention_manifest:", 'retention_manifest' in ah['files'])
print("includes itself:", 'ARTIFACT_HASHES' in str(ah.get('files',{}).get('retention_manifest',{})))

print("\n=== RETENTION ===")
for i in rm['locked_items']:
    print(f"  {i['path'].split('/')[-1]}: locked={i['locked']}")
ret_paths = [i['path'] for i in rm['locked_items']]
print("includes RETENTION_MANIFEST itself:", any('RETENTION_MANIFEST' in p for p in ret_paths))

print("\n=== AUDIT_INDEX ===")
print("standard_files:", list(ai['standard_files'].keys()))
print("self_sha256:", ai.get('self_sha256'))
