#!/usr/bin/env python3
"""Evaluate theta278 and six fixed DART gates on frozen source H300 only.

Single-view source-safety evaluation; no target GT and no packaging. Parameter
and Bstar behavior audits are computed for every checkpoint.
"""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
from _r2_contracts import BSTAR,CONFIG,DATA,OUT,SNAP,THETA278,load_state,sha256_file,write_json
TAGS=['BASE','D0_1k','D0_2k','D1_1k','D1_2k','D2_1k','D2_2k']
def build(path,device):
 from src.models import build_model as make_model
 from src.utils import load_config
 cfg=load_config(str(CONFIG));cfg['cwfa']={'enabled':False,'probability':.3,'epsilon':4.,'norm_eps':1e-12,'start_step':1};cfg.pop('mixstyle',None);s,_=load_state(path);m=make_model(cfg);m.load_state_dict(s,strict=True);return m.to(device).eval()
def cm_add(cm,p,g):
 v=g!=255;cm+=np.bincount(g[v].astype(np.int64)*8+p[v],minlength=64).reshape(8,8)
def metric(cm):
 d=cm.sum(1)+cm.sum(0)-np.diag(cm);x=np.divide(np.diag(cm),d,out=np.zeros(8),where=d>0);return x,float(x.mean())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
 if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('2','3'):raise SystemExit('GPU2/3 only')
 for p in (str(SNAP),'/home/jhy/gpu2_day_queue'):
  if p not in sys.path:sys.path.insert(0,p)
 import torch,torch.nn.functional as F
 from PIL import Image
 from src.transforms import normalize_to_tensor
 split=json.loads((OUT/'00_inventory/SOURCE_AUDIT_SPLIT.json').read_text());names=split['H300'];R=OUT/'03_dart_fixed';paths={'BASE':THETA278}
 for v in ('D0','D1','D2'):
  for s in (1000,2000):paths[f'{v}_{s//1000}k']=R/v/f'checkpoint_step{s}.pt'
 out=OUT/'03_dart_eval/H300';out.mkdir(parents=True,exist_ok=False);base_state,_=load_state(THETA278);results={};preds={};t0=time.time();bstar=torch.from_numpy(BSTAR).to(a.device)[None,:,None,None]
 for tag,path in paths.items():
  model=build(path,a.device);cm=np.zeros((8,8),np.int64);hist=np.zeros(8,np.int64);prepost=0;pixels=0;plist=[]
  with torch.inference_mode():
   for i,stem in enumerate(names,1):
    with Image.open(DATA/f'train/images/{stem}.png') as f:im=np.asarray(f.convert('RGB')).copy()
    gt=np.load(f'/home/jhy/rare_diag_20260915/b2_val_20260917/gt/{stem}.npy',allow_pickle=False);px=normalize_to_tensor(im).unsqueeze(0).to(a.device)
    with torch.amp.autocast('cuda',enabled=False):z=model(pixel_values=px).logits.float();z=F.interpolate(z,size=gt.shape,mode='bilinear',align_corners=False)
    raw=z.argmax(1)[0];p=(z+bstar).argmax(1)[0];prepost+=int((raw!=p).sum());pixels+=p.numel();pn=p.cpu().numpy().astype(np.uint8);cm_add(cm,pn,gt);hist+=np.bincount(pn.ravel(),minlength=8);plist.append(pn)
  cur,_=load_state(path);rms={};
  for scope,prefix in [('stage0','segformer.stages.0.'),('stage1','segformer.stages.1.'),('stage2','segformer.stages.2.'),('stage3','segformer.stages.3.'),('head','decode_head.')]:
   ss=nn=0
   for k in base_state:
    if k.startswith(prefix) and base_state[k].is_floating_point():d=(cur[k].float()-base_state[k].float());ss+=float((d*d).sum());nn+=d.numel()
   rms[scope]=(ss/max(nn,1))**.5
  iou,miou=metric(cm);bias=(cur['decode_head.classifier.bias'].float()-base_state['decode_head.classifier.bias'].float()).tolist()
  results[tag]={'path':str(path),'sha256':sha256_file(path),'confusion':cm.tolist(),'per_class_iou':iou.tolist(),'miou':miou,'class_share':(hist/hist.sum()).tolist(),'bstar_prepost_flip':prepost/pixels,'stage_rms_delta':rms,'head_bias_drift':bias};preds[tag]=plist;del model;torch.cuda.empty_cache();print(tag,miou,flush=True)
 base=preds['BASE'];
 for tag in TAGS[1:]:results[tag]['flip_vs_theta278']=sum(int((a!=b).sum()) for a,b in zip(preds[tag],base))/(len(base)*1024*1024)
 report={'scope':'DART_FIXED_H300_SOURCE_SAFETY_SINGLE_VIEW','H300_sha256':split['H300_sha256'],'results':results,'runtime_sec':round(time.time()-t0,1),'test2_ground_truth_used':False,'platform_submission_performed':False,'automatic_continuation':False};write_json(out/'DART_H300_REPORT.json',report);print(json.dumps({k:{'miou':v['miou'],'flip':v.get('flip_vs_theta278')} for k,v in results.items()},indent=2))
if __name__=='__main__':main()
