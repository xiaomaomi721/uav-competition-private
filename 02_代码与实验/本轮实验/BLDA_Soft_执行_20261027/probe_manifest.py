import json, pathlib, csv, subprocess, shutil
R=pathlib.Path('/data/uav'); P=R/'reproduction/H75_713294_REPRO/11_H76_ACDD_3594_20261026'
for f in [P/'manifests/semifinal_3594.csv',P/'manifests/semifinal_3594_summary.json',P/'code/package_semifinal.py']:
 if f.suffix=='.csv':
  rows=list(csv.DictReader(f.open())); print(json.dumps({'file':str(f),'count':len(rows),'first':rows[:2]},ensure_ascii=False))
 else:
  t=f.read_text(); print(json.dumps({'file':str(f),'content':t},ensure_ascii=False))
for d in [R/'analysis_codex_20261026',R/'reproduction/H75_713294_REPRO',R/'datasets/训练集/train',R/'code']:
 print(json.dumps({'directory':str(d),'files':[x.name for x in d.iterdir()]} ,ensure_ascii=False))
print(json.dumps({'disk_free_gib':shutil.disk_usage(R).free/1024**3}))

