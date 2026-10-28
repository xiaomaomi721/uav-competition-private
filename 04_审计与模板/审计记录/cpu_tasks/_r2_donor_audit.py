#!/usr/bin/env python3
"""Preliminary heterogeneous donor audit using existing Barren/Agri cache.

The donor cache is single-view full-resolution while theta278 is D4 deployment.
This mismatch is explicit; a positive result nominates, but does not certify, a
future exact donor cache. No target GT and no ensemble/package is produced.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from _r2_contracts import BSTAR,OUT,deployed_logits,write_json
RNG=np.random.default_rng(20260922);BOOT=2000
B2=Path('/home/jhy/rare_diag_20260915/b2_val_20260917')
def met(cm):
 d=cm.sum(1)+cm.sum(0)-np.diag(cm);x=np.divide(np.diag(cm),d,out=np.zeros(8),where=d>0);return x,float(x.mean())
def main():
 out=OUT/'05_donors';out.mkdir(parents=True,exist_ok=True);split=json.loads((OUT/'00_inventory/SOURCE_AUDIT_SPLIT.json').read_text());names=split['R1099']+split['H300'];theta=OUT/'01_holdout_cache/theta278/scores';don=B2/'logits_barrenAgri';gt=B2/'gt';n=len(names)
 cmT=np.zeros((8,8),np.int64);cmD=np.zeros_like(cmT);repair=np.zeros((n,8),np.int64);damage=np.zeros_like(repair);diffwrong=np.zeros_like(repair);validn=np.zeros(n,np.int64);corrT=np.zeros(n,np.int64);corrD=np.zeros(n,np.int64);oracle=np.zeros(n,np.int64);confbins=np.zeros((10,3),np.int64)
 for i,s in enumerate(names):
  g=np.load(gt/f'{s}.npy',allow_pickle=False);v=g!=255;zt=deployed_logits(np.load(theta/f'{s}.npy',allow_pickle=False).astype(np.float32));pt=zt.argmax(0).numpy();zd=torch.from_numpy(np.load(don/f'{s}.npy',allow_pickle=False).astype(np.float32))+torch.from_numpy(BSTAR)[:,None,None];pd=zd.argmax(0).numpy();cf=zd.softmax(0).max(0).values.numpy();x=g[v].astype(np.int64)*8+pt[v];cmT+=np.bincount(x,minlength=64).reshape(8,8);x=g[v].astype(np.int64)*8+pd[v];cmD+=np.bincount(x,minlength=64).reshape(8,8);ct=pt==g;cd=pd==g;rep=(~ct)&cd&v;dam=ct&(~cd)&v;dw=(~ct)&(~cd)&(pt!=pd)&v
  repair[i]=np.bincount(g[rep],minlength=8);damage[i]=np.bincount(g[dam],minlength=8);diffwrong[i]=np.bincount(g[dw],minlength=8);validn[i]=v.sum();corrT[i]=ct[v].sum();corrD[i]=cd[v].sum();oracle[i]=(ct|cd)[v].sum();bi=np.minimum((cf*10).astype(int),9)
  confbins[:,0]+=np.bincount(bi[rep],minlength=10);confbins[:,1]+=np.bincount(bi[dam],minlength=10);confbins[:,2]+=np.bincount(bi[v],minlength=10)
  if (i+1)%100==0:print(i+1,n,flush=True)
 iT,mT=met(cmT);iD,mD=met(cmD);net=repair-damage;vals=[];cls=[]
 for _ in range(BOOT):
  z=RNG.integers(0,n,n);vals.append(net[z].sum()/validn[z].sum());cls.append(net[z].sum(0)/np.maximum((repair[z]+damage[z]+diffwrong[z]).sum(0),1))
 vals=np.array(vals);cls=np.array(cls);pixel_oracle=(oracle.sum()/validn.sum());image_oracle=np.sum(np.maximum(corrT,corrD))/validn.sum()
 report={'donor':'Barren/Agricultural-specialized SegFormer','donor_checkpoint_sha256':'83ee03a1f2faee3e90923fb3049cb0e8746f2c27d51a435c9c68f73f6daf1915','protocol_warning':'donor existing cache is single-view full-resolution; theta278 uses D4 deployment. Preliminary nomination only.','theta278':{'miou':mT,'per_class_iou':iT.tolist()},'donor_result':{'miou':mD,'per_class_iou':iD.tolist()},'repair_pixels':repair.sum(0).tolist(),'damage_pixels':damage.sum(0).tolist(),'different_wrong_pixels':diffwrong.sum(0).tolist(),'net_repair_minus_damage':net.sum(0).tolist(),'net_fraction_point':float(net.sum()/validn.sum()),'net_fraction_ci95':np.quantile(vals,[.025,.975]).tolist(),'per_class_net_ci95':np.quantile(cls,[.025,.975],axis=0).tolist(),'confidence_bins_repair_damage_total':confbins.tolist(),'pixel_oracle_accuracy':float(pixel_oracle),'image_oracle_accuracy':float(image_oracle),'retain_for_exact_cache':bool(np.quantile(vals,.025)>0),'test2_ground_truth_used':False,'platform_submission_performed':False}
 write_json(out/'BARREN_AGRI_PRELIM_AUDIT.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
