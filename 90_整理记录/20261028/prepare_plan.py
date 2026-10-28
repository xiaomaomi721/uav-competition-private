"""Produce a reviewable plan and content hashes, without mutating the desktop tree."""
import json,hashlib
from pathlib import Path
ROOT=Path(r"C:\Users\Lenovo\OneDrive\Documents\桌面\无人机")
WORK=Path(r"C:\Users\Lenovo\OneDrive\Documents\ChatGPT\无人机")
OUT=WORK/"文件整理_20261028"
inv=json.loads((OUT/"inventory_before.json").read_text(encoding="utf-8"))
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
moves=[]
def add(name,destination):
 p=ROOT/name;assert p.exists(),str(p)
 moves.append({"from":name,"to":destination})
for p in sorted(ROOT.iterdir()):
 name=p.name
 if name in {".claude",".ruff_cache","提交包","README_从这里开始.md","00_Codex工作同步索引.md"}:continue
 if name in {"半决赛_全部方案总表_20261027.md","线上成绩台账_20261027.md","上一阶段冠军与提分贡献核对_20261027.md"}:
  add(name,"01_当前比赛/"+name)
 elif name.startswith("半决赛_") or name=="Claude助手_工作同步_20261027.md":
  add(name,"01_当前比赛/队友方案与诊断/"+name)
 elif name in {"3594张无标签测试图_利用方法与实施边界_20261027.md","标注有限与测试域泛化_近期论文与项目判断_20261027.md",
               "DINOv3-L_PMT_实现方案与实验结果_给队友_20261027.md","H75_BLDA-Soft_论文依据与执行方案_20261027.md","BLDA_Soft_方案依据_20261027"}:
  add(name,"01_当前比赛/Codex方案与研究/"+name)
 elif name=="队友方案审核交接_20261027":add(name,"01_当前比赛/审核交接/"+name)
 elif name=="scripts_20261027":add(name,"02_代码与实验/本轮实验/"+name)
 elif name.startswith("scripts_") or name in {"experiments","脚本"}:add(name,"02_代码与实验/历史脚本/"+name)
 elif name in {"文档归档","analysis_reports_20260925"}:add(name,"03_历史文档/"+name)
 elif name.startswith("PROJECT_HANDOVER_"):add(name,"03_历史文档/历史交接/"+name)
 elif name in {"审计记录","实验治理模板_20260926"}:add(name,"04_审计与模板/"+name)
 elif name in {"_归档整理_20260915","归档_整理记录"}:add(name,"90_整理记录/以往整理记录/"+name)
 else:raise RuntimeError("Unclassified root item: "+name)
def final_path(rel):
 for m in moves:
  old=Path(m["from"]);p=Path(rel)
  try:return str(Path(m["to"])/p.relative_to(old))
  except ValueError:pass
 return str(rel)
caches=[]
for f in inv["desktop"]["cache_candidates"]:
 # Cache copies inside signed historical handovers are retained.
 if "队友方案审核交接_20261027" in Path(f["path"]).parts:continue
 p=ROOT/f["path"]
 caches.append({"original":f["path"],"path":final_path(f["path"]),"bytes":p.stat().st_size,"sha256":sha(p)})
duplicate_group=next(g for g in inv["desktop"]["duplicates"] if g["sha256"]=="9069b2475cd7b0c8060d4f9105f7a9f9a54cb4488d0632ce2aafacb6bb3d2277")
remove_dup=Path(next(p for p in duplicate_group["paths"] if Path(p).parts[0]=="归档_整理记录"))
keep_dup=Path(next(p for p in duplicate_group["paths"] if Path(p).parts[0]=="提交包"))
assert (ROOT/remove_dup).exists() and (ROOT/keep_dup).exists()
dup_sha=sha(ROOT/remove_dup);assert dup_sha==sha(ROOT/keep_dup)=="9069b2475cd7b0c8060d4f9105f7a9f9a54cb4488d0632ce2aafacb6bb3d2277"
duplicates=[{"original":str(remove_dup),"path":final_path(remove_dup),"keep":str(keep_dup),"sha256":dup_sha,"bytes":(ROOT/remove_dup).stat().st_size}]
deleted={c["original"] for c in caches}|{str(remove_dup)}
receipts=[]
for f in inv["desktop"]["files"]:
 if f["path"] in deleted:continue
 new=final_path(f["path"])
 if new!=f["path"]:
  receipts.append({"original":f["path"],"final":new,"bytes":f["bytes"],"sha256":sha(ROOT/f["path"])})
plan={"root":str(ROOT),"workspace":str(WORK),"date":"2026-10-28","moves":moves,"cache_delete":caches,
 "duplicate_delete":duplicates,"verify_moved_files":receipts,"protected":[".claude","提交包（整体位置和内容保留）","Git仓库","模型权重","数据","原始审核快照"],
 "copy_experiment":{"from":str(WORK/"BLDA_Soft_执行_20261027"),"to":"02_代码与实验/本轮实验/BLDA_Soft_执行_20261027"}}
(OUT/"organization_plan.json").write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"moves":len(moves),"moved_files_to_hash_verify":len(receipts),"cache_files":len(caches),
 "cache_bytes":sum(c["bytes"] for c in caches),"duplicate_files":1,"duplicate_bytes":duplicates[0]["bytes"]},ensure_ascii=False))

