import json,pathlib,hashlib,shutil,sys,csv,collections,subprocess,os
R=pathlib.Path('/data/uav'); W=R/'runs/CODEX_BLDA_SOFT_20261027'
assert not W.exists(),f'Own experiment directory already exists: {W}'
W.mkdir(mode=0o700)
(W/'evidence').mkdir();(W/'snapshot').mkdir();(W/'logs').mkdir()
shutil.copytree(R/'code/src',W/'snapshot/src',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(R/'code/config.yaml',W/'snapshot/config.yaml')
P=R/'reproduction/H75_713294_REPRO/11_H76_ACDD_3594_20261026'
shutil.copy2(P/'manifests/semifinal_3594.csv',W/'evidence/semifinal_3594.csv')
shutil.copy2(R/'code/splits/train.txt',W/'evidence/source_train.txt')
shutil.copy2(P/'code/package_semifinal.py',W/'evidence/reference_package_semifinal.py')
train=(W/'evidence/source_train.txt').read_text().split()
D=R/'datasets/训练集'
allids=sorted(p.stem for p in (D/'train/images').glob('*.png'))
val=sorted(set(allids)-set(train))
assert len(set(train))==len(train)==5597
assert len(allids)==6996 and len(val)==1399,(len(allids),len(val))
assert all((D/'train/masks'/f'{s}.png').is_file() for s in allids)
(W/'evidence/source_val.txt').write_text('\n'.join(val)+'\n')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
H=R/'runs/ASTRA_H75_X110_20261005/ckpt/H75.pt'
assert sha(H)=='b5eea2795d8a6d90c803a15b2ca66cf0abcf62322ef1aacd0d4d0ec82098a28a'
rows=list(csv.DictReader((W/'evidence/semifinal_3594.csv').open()))
assert len(rows)==len({r['input_name'] for r in rows})==3594
I=R/'reproduction/H75_713294_REPRO/10_semifinal_testdata_20261025/images'
assert set(p.name for p in I.glob('*.png'))==set(r['input_name'] for r in rows)
assets={'root':str(W),'model':str(H),'model_sha256':sha(H),'source_data':str(D),
 'source_train':str(W/'evidence/source_train.txt'),'source_val':str(W/'evidence/source_val.txt'),
 'target_images':str(I),'target_manifest':str(W/'evidence/semifinal_3594.csv'),
 'pretrained':str(R/'models/segformer-b3-ade512'),'source_count':5597,'val_count':1399,
 'target_count':3594,'target_groups':dict(collections.Counter(r['subset'] for r in rows)),
 'physical_gpu_index':1,'gpu_uuid':'GPU-9f3707b0-4a7d-0ba5-6c3b-463906c93bd7',
 'python':sys.executable,'snapshot_sha256':{str(p.relative_to(W)):sha(p) for p in (W/'snapshot').rglob('*') if p.is_file()},
 'split_note':'source_val is complement of 5597 training IDs among all 6996 images; verify against prior val manifest before metric comparisons'}
(W/'assets.json').write_text(json.dumps(assets,indent=2,ensure_ascii=False))
print(json.dumps(assets,ensure_ascii=False))
for d in [R/'reproduction/H75_713294_REPRO/06_source1399_20261024', R/'analysis_codex_20261026/exact_overlap_v1']:
 print(json.dumps({'dir':str(d),'files':[p.name for p in d.iterdir()]}))

