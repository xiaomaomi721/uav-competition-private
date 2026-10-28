import json, subprocess
from pathlib import Path
BASE = Path(__file__).resolve().parent
REPO = BASE.parent / '_github_private' / 'repo'
tree = json.loads((BASE / 'remote_tree.json').read_text(encoding='utf-8'))
verified = json.loads((BASE / 'repository_verified.json').read_text(encoding='utf-8'))
head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
assert head == verified['head'], 'Remote HEAD does not match local commit.'
out = subprocess.check_output(['git','ls-tree','-r','-z','HEAD'],cwd=REPO)
local = {}
for raw in out.split(b'\0'):
    if not raw: continue
    meta, name = raw.split(b'\t',1)
    mode, kind, sha = meta.decode().split()
    if kind=='blob': local[name.decode('utf-8')] = sha
remote = {x['path']:x['sha'] for x in tree['tree'] if x['type']=='blob'}
assert local == remote, 'Remote file paths or blob content differ.'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO).strip(), 'Uncommitted content.'
result={'status':'PASS','commit':head,'verified_blob_count':len(local),'visibility':verified['visibility'],'all_remote_paths_and_content_match':True}
(BASE/'file_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
