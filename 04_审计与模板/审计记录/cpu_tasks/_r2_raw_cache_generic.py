#!/usr/bin/env python3
"""Generic raw native-256 D4 cache; used once for 278->test1.

Current local strict loader, exact inverse, FP32 view mean, no Bstar. Saves
full D4 disagreement and full-run FP16 quantization argmax error without a
second pass, then fixed-8 live-vs-cache sanity.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
SNAP=Path('/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1'); CONFIG=SNAP/'stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml'
sys.path.insert(0,str(SNAP));sys.path.insert(0,'/home/jhy/rare_diag_20260915/round2_20260921/scripts')
from _r2_test2_cache import build_model_local,views_and_mean,sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--tag',required=True);ap.add_argument('--dataset-dir',required=True);ap.add_argument('--prefix',required=True);ap.add_argument('--count',type=int,required=True);ap.add_argument('--outdir',required=True);ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
 if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('2','3'):raise SystemExit('GPU2/3 only')
 import torch
 from PIL import Image
 from src.transforms import normalize_to_tensor
 test=Path(a.dataset_dir);out=Path(a.outdir);ck=Path(a.checkpoint)
 if out.exists():raise SystemExit(f'outdir exists {out}')
 (out/'scores').mkdir(parents=True)
 names=sorted((p.name for p in test.glob('*.png')),key=lambda n:int(Path(n).stem.split('_')[-1]));assert names==[f'{a.prefix}_{i}.png' for i in range(1,a.count+1)]
 model=build_model_local(CONFIG,ck,a.device);rows=[];qd=0;qp=0;t0=time.time()
 for j,n in enumerate(names,1):
  with Image.open(test/n) as im:img=np.asarray(im.convert('RGB')).copy()
  px=normalize_to_tensor(img).unsqueeze(0).to(a.device);views,mean=views_and_mean(model,px);half=mean.half()
  qd+=int((mean.argmax(0)!=half.float().argmax(0)).sum());qp+=mean.shape[-2]*mean.shape[-1]
  np.save(out/'scores'/f'{Path(n).stem}.npy',half.cpu().numpy())
  pred=mean.argmax(0);dg=(views.argmax(1)!=pred.unsqueeze(0)).float().mean(0)
  rows.append({'image':n,'d4_disagreement_mean':float(dg.mean()),'d4_disagreement_p95':float(torch.quantile(dg,.95)),'d4_any_disagree_frac':float((dg>0).float().mean())})
  if j%100==0:print(f'{j}/{a.count}',flush=True)
 sanity=[]; fixed=sorted(set([1,2,3,max(1,a.count//5),max(1,a.count//2),max(1,4*a.count//5),a.count-1,a.count]))
 for idx in fixed:
  n=f'{a.prefix}_{idx}.png'
  with Image.open(test/n) as im:img=np.asarray(im.convert('RGB')).copy()
  px=normalize_to_tensor(img).unsqueeze(0).to(a.device);_,live=views_and_mean(model,px);cached=torch.from_numpy(np.load(out/'scores'/f'{a.prefix}_{idx}.npy').astype(np.float32)).to(a.device)
  sanity.append({'image':n,'max_abs':float((live-cached).abs().max()),'argmax_disagree_frac':float((live.argmax(0)!=cached.argmax(0)).float().mean())})
 meta={'created':datetime.now(timezone.utc).isoformat(),'tag':a.tag,'checkpoint':str(ck),'checkpoint_sha256':sha(ck),'dataset_dir':str(test),'count':a.count,'protocol':'D4 TTA8; exact inverse; native [8,256,256]; FP32 view mean; no Bstar; autocast off','physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'runtime_sec':round(time.time()-t0,1),'code_sha256':sha(Path(__file__)),'fp16_quantization':{'pixels':qp,'argmax_disagree_pixels':qd,'argmax_disagree_frac':qd/qp},'d4_disagreement':rows,'live_vs_cache_sanity':sanity}
 (out/'META.json').write_text(json.dumps(meta,indent=2));print(json.dumps({k:v for k,v in meta.items() if k!='d4_disagreement'},indent=2))
if __name__=='__main__':main()
