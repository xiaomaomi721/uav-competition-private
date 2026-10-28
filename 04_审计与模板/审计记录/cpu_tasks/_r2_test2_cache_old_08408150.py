#!/usr/bin/env python3
"""Round2: raw D4 native-256 cache for test2 (one model, one pass).

Saves fp16 [8 classes,256,256] canonical FP32 mean logits for test2_1..1300.
No Bstar. Also records per-image D4 disagreement from the eight canonical
native views before discarding them. Finally re-forwards 8 fixed images and
checks live-vs-fp16-cache consistency.

Physical GPU2 only. No training.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

SNAP=Path('/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1')
CONFIG=SNAP/'stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml'
TEST=Path('/home/jhy/datasets/AIC2026_UAV_semifinal/test_2/images')
INTEGRITY=Path('/home/jhy/rare_diag_20260915/round2_20260921/TEST2_INTEGRITY.json')
VIEWS=[(k,hf) for k in range(4) for hf in (False,True)]

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def d4f(t,i):
 import torch
 k,hf=VIEWS[i]
 if k:t=torch.rot90(t,k,(-2,-1))
 if hf:t=t.flip((-1,))
 return t

def d4i(t,i):
 import torch
 k,hf=VIEWS[i]
 if hf:t=t.flip((-1,))
 if k:t=torch.rot90(t,-k,(-2,-1))
 return t.contiguous()

def views_and_mean(model,px):
 import torch
 vs=[]
 with torch.inference_mode():
  for i in range(8):
   with torch.amp.autocast('cuda',enabled=False): z=model(pixel_values=d4f(px,i)).logits.float()
   z=d4i(z,i)[0]
   if tuple(z.shape)!=(8,256,256): raise RuntimeError(f'bad native shape {z.shape}')
   vs.append(z)
 v=torch.stack(vs,0) # views,classes,h,w
 return v,v.mean(0,dtype=torch.float32)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--tag',required=True); ap.add_argument('--outdir',required=True); ap.add_argument('--device',default='cuda:0'); a=ap.parse_args()
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='2' or os.environ.get('CUDA_DEVICE_ORDER')!='PCI_BUS_ID': raise SystemExit('physical GPU2 only')
 for p in ('/home/jhy/gpu2_day_queue',str(SNAP)):
  if p not in sys.path:sys.path.insert(0,p)
 import torch
 from PIL import Image
 from src.transforms import normalize_to_tensor
 from tta_anatomy_v2 import build_model
 ck=Path(a.checkpoint); out=Path(a.outdir)
 if out.exists():raise SystemExit(f'outdir exists: {out}')
 (out/'scores').mkdir(parents=True)
 names=sorted((p.name for p in TEST.glob('*.png')),key=lambda n:int(Path(n).stem.split('_')[-1]))
 assert names==[f'test2_{i}.png' for i in range(1,1301)]
 cksha=sha(ck); codesha=sha(Path(__file__))
 integ=json.loads(INTEGRITY.read_text())
 model=build_model(CONFIG,ck,a.device).eval()
 rows=[]; t0=time.time()
 for j,n in enumerate(names,1):
  with Image.open(TEST/n) as im: img=np.asarray(im.convert('RGB')).copy()
  px=normalize_to_tensor(img).unsqueeze(0).to(a.device)
  views,mean=views_and_mean(model,px)
  np.save(out/'scores'/f'{Path(n).stem}.npy',mean.half().cpu().numpy())
  meanpred=mean.argmax(0)
  disagree=(views.argmax(1)!=meanpred.unsqueeze(0)).float().mean(0)
  rows.append({'image':n,'d4_disagreement_mean':float(disagree.mean()),
               'd4_disagreement_p95':float(torch.quantile(disagree,0.95)),
               'd4_any_disagree_frac':float((disagree>0).float().mean())})
  del px,views,mean,disagree
  if j%100==0:
   e=time.time()-t0; print(f'{j}/1300 {e:.0f}s {e/j:.3f}s/img',flush=True)
 # sanity: fixed 8 images re-forward; compare against fp16 cache tolerance and argmax
 sanity=[]
 for idx in [1,2,3,100,500,900,1299,1300]:
  n=f'test2_{idx}.png'
  with Image.open(TEST/n) as im: img=np.asarray(im.convert('RGB')).copy()
  px=normalize_to_tensor(img).unsqueeze(0).to(a.device)
  _,live=views_and_mean(model,px); cached=torch.from_numpy(np.load(out/'scores'/f'test2_{idx}.npy').astype(np.float32)).to(a.device)
  sanity.append({'image':n,'max_abs':float((live-cached).abs().max()),
                 'argmax_disagree_frac':float((live.argmax(0)!=cached.argmax(0)).float().mean())})
 meta={'created':datetime.now(timezone.utc).isoformat(),'scope':'ROUND2_TEST2_RAW_NATIVE256_D4_TTA8','tag':a.tag,'checkpoint':str(ck),'checkpoint_sha256':cksha,'dataset_dir':str(TEST),'dataset_archive_sha256':integ['archive_sha256'],'dataset_file_manifest':'test2_files_sha256.json','count':1300,'protocol':'D4 TTA8; exact inverse; native [8,256,256]; FP32 view mean; no Bstar; autocast off','stored':'fp16 [8,256,256] raw mean logits','code':str(Path(__file__)),'code_sha256':codesha,'runtime_sec':round(time.time()-t0,1),'d4_disagreement':rows,'live_vs_cache_sanity':sanity,'torch':torch.__version__,'cuda':torch.version.cuda}
 (out/'META.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
 print(json.dumps({'tag':a.tag,'count':1300,'runtime_sec':meta['runtime_sec'],'sanity':sanity},indent=2))
if __name__=='__main__':main()
