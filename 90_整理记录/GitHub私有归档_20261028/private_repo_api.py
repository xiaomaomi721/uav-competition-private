"""Use Windows Git Credential Manager without displaying/persisting credentials."""
import os, sys, json, subprocess, urllib.request, urllib.error
from pathlib import Path
BASE = Path(__file__).resolve().parent
OWNER, NAME = 'xiaomaomi721', 'uav-competition-private'
env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='never')
proc = subprocess.run(['git', 'credential', 'fill'], input=f'protocol=https\nhost=github.com\nusername={OWNER}\n\n', text=True, encoding='utf-8', capture_output=True, env=env)
if proc.returncode:
    raise SystemExit('Git Credential Manager could not supply authentication; no credential output displayed.')
cred = dict(line.split('=',1) for line in proc.stdout.splitlines() if '=' in line)
token = cred.get('password')
if not token: raise SystemExit('Git credential missing; no repository action performed.')

def api(method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request('https://api.github.com'+path, data=data, method=method,
        headers={'Authorization':'Bearer '+token, 'Accept':'application/vnd.github+json', 'User-Agent':'UAV-private-project-archive', 'X-GitHub-Api-Version':'2022-11-28', 'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=60) as response: return json.load(response)
    except urllib.error.HTTPError as e:
        if e.code==404: return None
        raise RuntimeError(f'GitHub API HTTP {e.code} for {method} {path}; credentials not logged.') from None

user = api('GET','/user')
assert user['login']==OWNER, 'Unexpected authenticated account; stopped.'
mode = sys.argv[1]
if mode == 'check':
    repo = api('GET',f'/repos/{OWNER}/{NAME}')
    print(json.dumps({'login':user['login'], 'repository_exists':repo is not None, 'private':None if repo is None else repo['private']}))
elif mode == 'create':
    assert (BASE.parent/'_github_private'/'repo'/'README.md').exists(), 'Prepare content before creating repo.'
    assert api('GET',f'/repos/{OWNER}/{NAME}') is None, 'Repository already exists; do not overwrite.'
    repo = api('POST','/user/repos',{'name':NAME,'description':'无人机语义分割比赛：方案、代码、实验记录与历史资料（私有归档）','private':True,'auto_init':False,'has_issues':True,'has_wiki':False,'has_projects':False})
    assert repo['private'] and repo['owner']['login']==OWNER
    result={k:repo.get(k) for k in ['full_name','html_url','clone_url','private','visibility','created_at','default_branch']}
    (BASE/'repository_created.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
elif mode == 'verify':
    repo = api('GET',f'/repos/{OWNER}/{NAME}')
    assert repo and repo['private'] and repo['visibility']=='private'
    branch = api('GET',f'/repos/{OWNER}/{NAME}/branches/'+repo['default_branch'])
    commit = branch['commit']['sha']
    tree=api('GET',f'/repos/{OWNER}/{NAME}/git/trees/{commit}?recursive=1')
    assert not tree.get('truncated'), 'Truncated tree cannot verify.'
    result={'full_name':repo['full_name'],'html_url':repo['html_url'],'private':repo['private'],'visibility':repo['visibility'],'default_branch':repo['default_branch'],'head':commit,'remote_files':sum(x['type']=='blob' for x in tree['tree'])}
    (BASE/'repository_verified.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (BASE/'remote_tree.json').write_text(json.dumps(tree,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
else: raise SystemExit('Unknown mode')
