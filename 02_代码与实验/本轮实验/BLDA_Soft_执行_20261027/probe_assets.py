import json, pathlib, hashlib, subprocess, sys, importlib.metadata
R=pathlib.Path('/data/uav')
P=R/'reproduction/H75_713294_REPRO/11_H76_ACDD_3594_20261026'
out={'python':sys.executable,'torch':importlib.metadata.version('torch'),
 'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used','--format=csv,noheader'],text=True)}
for d in [P/'code',P/'manifests',P/'p0c_delivery_20261026',R/'reproduction/H75_713294_REPRO/10_semifinal_testdata_20261025',R/'code/splits',R/'datasets']:
 out[str(d)]=[x.name for x in d.iterdir()] if d.exists() else None
for f in [R/'code/config.yaml',R/'code/src/models.py',R/'code/src/transforms.py',R/'code/src/dataset.py',R/'code/src/rare_crop.py']:
 out[str(f)]=f.read_text() if f.exists() else 'missing'
h=R/'runs/ASTRA_H75_X110_20261005/ckpt/H75.pt'
out['h75']={'path':str(h),'exists':h.exists(),'sha256':hashlib.sha256(h.read_bytes()).hexdigest() if h.exists() else None}
print(json.dumps(out,ensure_ascii=False))

