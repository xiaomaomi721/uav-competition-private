"""Post-organization validation without altering user files."""
import json,re,hashlib
from pathlib import Path
from urllib.parse import unquote
W=Path(r"C:\Users\Lenovo\OneDrive\Documents\ChatGPT\无人机")
O=W/"文件整理_20261028"
plan=json.loads((O/"organization_plan.json").read_text(encoding="utf-8"))
before=json.loads((O/"inventory_before.json").read_text(encoding="utf-8"))["desktop"]
R=Path(plan["root"]); records=R/"90_整理记录/20261028"
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def mapped(old):
 for m in plan["moves"]:
  try:return Path(m["to"])/Path(old).relative_to(m["from"])
  except ValueError:pass
 return Path(old)
deleted={c["original"] for c in plan["cache_delete"]}|{d["original"] for d in plan["duplicate_delete"]}
known={f["original"]:f["sha256"] for f in plan["verify_moved_files"]}
problems=[];kept=0
for f in before["files"]:
 if f["path"] in deleted:continue
 rel=mapped(f["path"])
 if f["path"] in {"README_从这里开始.md","00_Codex工作同步索引.md"}:rel=Path("90_整理记录/20261028/整理前入口文件")/f["path"]
 p=R/rel
 if not p.is_file():problems.append({"missing":str(rel)});continue
 if p.stat().st_size!=f["bytes"]:problems.append({"size_changed":str(rel)});continue
 if f["path"] in known and sha(p)!=known[f["path"]]:problems.append({"hash_changed":str(rel)});continue
 kept+=1
for c in plan["cache_delete"]:
 if (R/c["path"]).exists():problems.append({"cache_not_removed":c["path"]})
for d in plan["duplicate_delete"]:
 if (R/d["path"]).exists():problems.append({"duplicate_not_removed":d["path"]})
 if sha(R/d["keep"])!=d["sha256"]:problems.append({"canonical_zip_changed":d["keep"]})
links=[]
for name in ("README_从这里开始.md","00_Codex工作同步索引.md"):
 text=(R/name).read_text(encoding="utf-8-sig")
 for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)",text):
  if target.startswith(("https://","http://","#")):continue
  dest=R/unquote(target.strip("<>"))
  links.append({"document":name,"target":target,"exists":dest.exists()})
  if not dest.exists():problems.append({"broken_navigation":target})
for src in (W/"BLDA_Soft_执行_20261027").iterdir():
 if src.is_file():
  dest=R/plan["copy_experiment"]["to"]/src.name
  if not dest.exists() or sha(src)!=sha(dest):problems.append({"experiment_copy_mismatch":src.name})
result={"status":"PASS" if not problems else "FAILED","date":"2026-10-28",
 "original_files":len(before["files"]),"original_files_accounted_for":kept+len(deleted),
 "original_retained_files":kept,"intentional_removed_originals":len(deleted),
 "moved_file_sha256_checks":len(known),"navigation_links_checked":len(links),
 "root_visible_entries":sorted(p.name for p in R.iterdir() if not p.name.startswith(".")),
 "problems":problems,"navigation":links}
(O/"verification_result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in result.items() if k!="navigation"},ensure_ascii=False))
assert not problems

