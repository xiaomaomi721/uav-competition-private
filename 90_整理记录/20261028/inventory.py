"""Read-only project inventory; never descends into VCS or assistant settings."""
import os,json,hashlib,collections
from pathlib import Path
ROOTS={"desktop":Path(r"C:\Users\Lenovo\OneDrive\Documents\桌面\无人机"),
       "workspace":Path(r"C:\Users\Lenovo\OneDrive\Documents\ChatGPT\无人机")}
OUT=ROOTS["workspace"]/"文件整理_20261028"
OUT.mkdir(exist_ok=True)
PROTECTED={".git",".claude",".codex",".agents",".aws"}
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
all_results={}
for label,root in ROOTS.items():
 files=[];dirs=[];errors=[];skipped=[]
 for base,sub,names in os.walk(root,followlinks=False):
  base=Path(base)
  for name in list(sub):
   p=base/name
   if name in PROTECTED or p.is_symlink() or p==OUT:
    sub.remove(name);skipped.append(str(p.relative_to(root)))
   else:dirs.append(str(p.relative_to(root)))
  for name in names:
   p=base/name
   try:
    st=p.stat();files.append({"path":str(p.relative_to(root)),"bytes":st.st_size,"mtime_ns":st.st_mtime_ns})
   except Exception as e:errors.append({"path":str(p),"error":str(e)})
 groups=collections.defaultdict(list)
 for f in files:
  if 0<f["bytes"]<64*1024**2:groups[(f["bytes"],Path(f["path"]).suffix.lower())].append(f)
 duplicates=[]
 for same in groups.values():
  if len(same)<2:continue
  byhash=collections.defaultdict(list)
  for f in same:
   try:byhash[sha(root/f["path"])].append(f["path"])
   except Exception as e:errors.append({"path":f["path"],"error":str(e)})
  duplicates.extend({"sha256":h,"bytes_each":same[0]["bytes"],"paths":pp} for h,pp in byhash.items() if len(pp)>1)
 top=collections.defaultdict(lambda:{"files":0,"bytes":0})
 for f in files:
  t=Path(f["path"]).parts[0];top[t]["files"]+=1;top[t]["bytes"]+=f["bytes"]
 caches=[f for f in files if any(x in {"__pycache__",".ruff_cache",".pytest_cache",".mypy_cache"} for x in Path(f["path"]).parts) or Path(f["path"]).suffix==".pyc"]
 empty=[d for d in dirs if not any((root/d).iterdir())]
 result={"root":str(root),"files":files,"directories":dirs,"summary":{"files":len(files),"bytes":sum(f["bytes"] for f in files),
    "top":dict(top),"cache_files":len(caches),"cache_bytes":sum(f["bytes"] for f in caches),"empty_directories":empty,
    "duplicate_groups":len(duplicates)},"duplicates":duplicates,"cache_candidates":caches,"skipped":skipped,"errors":errors}
 all_results[label]=result
(OUT/"inventory_before.json").write_text(json.dumps(all_results,ensure_ascii=False,indent=2),encoding="utf-8")
for label,result in all_results.items():
 print(json.dumps({"root":label,"summary":result["summary"],"largest":sorted(result["files"],key=lambda f:-f["bytes"])[:8],
    "duplicate_examples":result["duplicates"][:12],"errors":result["errors"]},ensure_ascii=False))

