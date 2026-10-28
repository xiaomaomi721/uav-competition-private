import json, hashlib
from pathlib import Path
AUDIT=Path('/home/jhy/rare_diag_20260915/b4_migration_6296/audit_20260927')
rm=json.load(open(AUDIT/'RETENTION_MANIFEST.json'))
ok=0; bad=0
for i in rm['locked_items']:
    p=Path(i['path'])
    if not p.exists():
        print(f'MISSING: {i["path"]}'); bad+=1; continue
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    if h!=i['sha256']:
        print(f'MISMATCH: {p.name}'); bad+=1
    else:
        ok+=1
print(f'Verified: {ok} OK, {bad} bad, total {len(rm["locked_items"])}')
