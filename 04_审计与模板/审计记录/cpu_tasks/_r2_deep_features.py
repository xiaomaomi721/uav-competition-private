#!/usr/bin/env python3
"""Round2 1B: source-only Exp18 stage2/stage3 pooled feature extraction.

Single-view only, identical ImageNet normalization for source6996/test1/test2.
Loads the explicitly-provenanced Exp18 source-only checkpoint strictly. Calls
only the SegFormer encoder with output_hidden_states; no decode head is needed.
For every image saves spatial-mean pooled stage2 and stage3 features.

Physical GPU2 or GPU3 only. No training.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

SNAP=Path('/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1')
CKPT=SNAP/'stage18_runs/stage18_w1_exp18_b3_1024_full_ce_formal_seed42/best_miou.pt'
CKSHA='66667268d824d559a68b4b80872fe64d729bed7c6361d1a9939d6a3f92de88fd'
CFG=SNAP/'stage18_runs/stage18_w1_exp18_b3_1024_full_ce_formal_seed42/config.yaml'
SRC=Path('/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/train/images')
T1=Path('/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/images')
T2=Path('/home/jhy/datasets/AIC2026_UAV_semifinal/test_2/images')

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(device):
 import torch
 from src.utils import load_config
 from src.models import build_model
 cfg=load_config(str(CFG)); cfg['cwfa']={'enabled':False,'probability':0.3,'epsilon':4.,'norm_eps':1e-12,'start_step':1}
 m=build_model(cfg); ck=torch.load(CKPT,map_location='cpu',weights_only=False); st=ck.get('model',ck); m.load_state_dict(st,strict=True); return m.to(device).eval()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--outdir',required=True);ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
 if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('2','3'):raise SystemExit('GPU2/3 only')
 for p in (str(SNAP),):
  if p not in sys.path:sys.path.insert(0,p)
 import torch
 from PIL import Image
 from src.transforms import normalize_to_tensor
 assert sha(CKPT)==CKSHA
 out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
 groups=[('source6996',sorted(SRC.glob('*.png'))),('test1',sorted(T1.glob('*.png'),key=lambda p:int(p.stem.split('_')[-1]))),('test2',sorted(T2.glob('*.png'),key=lambda p:int(p.stem.split('_')[-1])))]
 assert [len(x[1]) for x in groups]==[6996,500,1300]
 model=load(a.device); t0=time.time(); shapes=None
 with torch.inference_mode():
  for group,paths in groups:
   names=[]; f2=[]; f3=[]
   for i,p in enumerate(paths,1):
    with Image.open(p) as im:img=np.asarray(im.convert('RGB')).copy()
    px=normalize_to_tensor(img).unsqueeze(0).to(a.device)
    with torch.amp.autocast('cuda',enabled=False): enc=model.segformer(pixel_values=px,output_hidden_states=True,return_dict=True)
    hs=enc.hidden_states
    a2=hs[2].float().mean(dim=(-2,-1))[0].cpu().numpy();a3=hs[3].float().mean(dim=(-2,-1))[0].cpu().numpy()
    if shapes is None:shapes={'stage2_hidden':list(hs[2].shape),'stage3_hidden':list(hs[3].shape),'stage2_pooled_dim':len(a2),'stage3_pooled_dim':len(a3)}
    names.append(p.name);f2.append(a2);f3.append(a3)
    if i%200==0:print(f'{group} {i}/{len(paths)}',flush=True)
   np.savez_compressed(out/f'{group}.npz',names=np.array(names),stage2=np.stack(f2),stage3=np.stack(f3))
 meta={'created':datetime.now(timezone.utc).isoformat(),'scope':'SOURCE_ONLY_EXP18_POOLED_FEATURES','checkpoint':str(CKPT),'checkpoint_sha256':CKSHA,'config':str(CFG),'ancestry':'published ADE B3 -> official source-only Exp18 CE; no test1/pseudo/UDA','preprocess':'src.transforms.normalize_to_tensor; RGB; single view; 1024x1024','groups':{g:len(p) for g,p in groups},'features':shapes,'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'runtime_sec':round(time.time()-t0,1),'code_sha256':sha(Path(__file__))}
 (out/'META.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
