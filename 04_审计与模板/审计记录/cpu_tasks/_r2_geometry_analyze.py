#!/usr/bin/env python3
"""Analyze fixed H300 geometry screen and enforce the preregistered gate."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from _r2_contracts import OUT, write_json
RNG=np.random.default_rng(20260922);BOOT=5000

def iou(cm):
 cm=np.asarray(cm,np.float64);d=cm.sum(1)+cm.sum(0)-np.diag(cm);x=np.divide(np.diag(cm),d,out=np.zeros(8),where=d>0);return x,float(x.mean())
def main():
 root=OUT/'04_geometry/H300';d=json.loads((root/'META.json').read_text());rec=d['records'];A={k:np.asarray([r['confusion'][k] for r in rec]) for k in ('G0','G1','G2')};out={}
 base_i,base_m=iou(A['G0'].sum(0))
 for tag in ('G1','G2'):
  cur_i,cur_m=iou(A[tag].sum(0));md=[];cd=[]
  for _ in range(BOOT):
   z=RNG.integers(0,len(rec),len(rec));x,m=iou(A[tag][z].sum(0));y,n=iou(A['G0'][z].sum(0));md.append(m-n);cd.append(x-y)
  md=np.asarray(md);cd=np.asarray(cd);mci=np.quantile(md,[.025,.975]);cci=np.quantile(cd,[.025,.975],axis=0)
  # User's gate says no class wholly negative: an upper CI below zero fails.
  passed=bool(mci[0]>0 and np.all(cci[:,1]>=0))
  out[tag]={'miou':cur_m,'delta_miou':cur_m-base_m,'delta_ci95':mci.tolist(),'per_class_iou':cur_i.tolist(),'per_class_delta':(cur_i-base_i).tolist(),'per_class_delta_ci95':cci.tolist(),'mean_flip_vs_G0':float(np.mean([r[f'flip_{tag}_vs_G0'] for r in rec])),'gate_pass':passed}
 passing=[k for k in out if out[k]['gate_pass']];winner=max(passing,key=lambda k:out[k]['delta_ci95'][0]) if passing else None
 report={'G0':{'miou':base_m,'per_class_iou':base_i.tolist()},'candidates':out,'winner_for_full1399':winner,'gate':'expand only winner; no test2 package','bootstrap_unit':'image','bootstrap_repeats':BOOT,'test2_ground_truth_used':False,'platform_submission_performed':False}
 write_json(root/'GEOMETRY_SCREEN_REPORT.json',report)
 lines=['# Geometry H300 Screen','',f'- G0 mIoU: `{base_m:.6f}`',f'- full-1399 winner: **{winner or "NONE"}**','', '| candidate | mIoU | delta | 95% CI | flip | gate |','|---|---:|---:|---:|---:|---|']
 for k,v in out.items():lines.append(f"| {k} | {v['miou']:.6f} | {v['delta_miou']:+.6f} | [{v['delta_ci95'][0]:+.6f}, {v['delta_ci95'][1]:+.6f}] | {v['mean_flip_vs_G0']:.4%} | {'PASS' if v['gate_pass'] else 'FAIL'} |")
 (root/'GEOMETRY_SCREEN_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
