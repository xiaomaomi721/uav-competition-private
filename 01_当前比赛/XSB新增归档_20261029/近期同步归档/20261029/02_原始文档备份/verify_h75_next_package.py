"""Independent numerical checks on the uploaded H75_NEXT_RESEARCH package.
Uses only the ZIP JSON, no private original images or weights.
"""
from pathlib import Path
import zipfile,json,numpy as np
Z=Path('/mnt/data/H75_NEXT_RESEARCH_20261029.zip')
base='H75_NEXT_RESEARCH_20261029/'
with zipfile.ZipFile(Z) as f:
 def j(path):return json.loads(f.read(base+path).decode('utf-8'))
 r0=j('02_H75错误机制/R0_A3_BSTAR_EFFECT.json')
 r1=j('02_H75错误机制/R1_REVERIFY.json')
 fu=j('01_复赛两模型融合/FUSION_PROBE.json')
 cm0=np.array(r0['variants']['raw']['confusion_matrix'],dtype=float)
 cm1=np.array(r0['variants']['bstar']['confusion_matrix'],dtype=float)
 trans=np.array(r0['bstar_effect']['flip_matrix_raw_to_bstar'])
 def miou(c):
  d=np.diag(c);return (d/(c.sum(0)+c.sum(1)-d)).mean()*100
 def reweight(c,p):
  conditional=c/c.sum(axis=1,keepdims=True)
  p=np.array(p,dtype=float); return conditional*(p/p.sum())[:,None]
 subsets=r1['target'].values();total=sum(s['pixels'] for s in subsets)
 targetraw=sum(np.array(s['zero_share_pct'])*s['pixels'] for s in subsets)/total
 targetbstar=sum(np.array(s['bstar_share_pct'])*s['pixels'] for s in subsets)/total
 print('Source',miou(cm0),miou(cm1))
 for name,p in [('train5597',r1['source_gt_prior_full']['train_full_5597']['share_pct']),('target_raw_proxy',targetraw),('target_bstar_proxy',targetbstar)]:
  a=miou(reweight(cm0,p));b=miou(reweight(cm1,p))
  print(name,round(a,4),round(b,4),'Bstar_Advantage_pp',round(b-a,4))
 print('BG->Barren',int(trans[0,4]),'BG->Agricultural',int(trans[0,6]))
 print('test3_new Barren Zero, Bstar',r1['target']['test3_new']['zero_share_pct'][4],r1['target']['test3_new']['bstar_share_pct'][4])
 c0=np.array(fu['results']['0.00']['confusion'],dtype=np.int64)
 c1=np.array(fu['results']['1.00']['confusion'],dtype=np.int64)
 num=fu['vs_alpha0']['1.00']['complementary_px_in_disagreement'];delta=int(c1.trace()-c0.trace())
 print('400 image net correct delta',delta,'correct-by-CTRL-only',(num-delta)//2,'correct-by-TREAT-only',(num+delta)//2)