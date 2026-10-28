#!/usr/bin/env python3
"""Round2 DART D0/D1/D2 bounded pilot. Exactly 2k steps; gates 1k/2k.

D0 never opens test2 or target caches. D1 learns hard high-confidence 261==278
agreement. D2 adds source-reliability-weighted soft theta278 disagreement KL.
No EMA, no CutMix/MixStyle/AdaIN, no resume beyond 2k, no packaging/submission.
"""
from __future__ import annotations
import argparse,hashlib,json,os,random,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from _r2_contracts import (BSTAR,CONFIG,DATA,EXPECTED_FILE_SHA,IGNORE_INDEX,OUT,ROUND2,SNAP,TEST2,THETA278,
 canonical_state_sha,d4_forward,d4_inverse,load_state,sha256_file,write_json)
TRAIN_PREFIX=('segformer.stages.2.','segformer.stages.3.')
HEAD_PREFIX='decode_head.'; MAX_STEPS=2000;GATES=(1000,2000);BASE_LR=1e-5;HEAD_MULT=.05;WD=.01;RAMP=500
LAMBDA_HARD=.5;LAMBDA_SOFT=.25;CONF_AGREE=.98

def is_train(name):return name.startswith(TRAIN_PREFIX) or name.startswith(HEAD_PREFIX)
def mode_map(model):
 model.eval();model.get_submodule('segformer.stages.2').train();model.get_submodule('segformer.stages.3').train()
 # head params train through autograd while module remains eval (buffers frozen)
def snap_frozen(model):return {n:p.detach().clone() for n,p in model.named_parameters() if not is_train(n)}|{'buf::'+n:b.detach().clone() for n,b in model.named_buffers()}
def frozen_diff(model,snap):
 import torch
 p=dict(model.named_parameters());b=dict(model.named_buffers());bad=[]
 for n,v in snap.items():cur=b[n[5:]] if n.startswith('buf::') else p[n];bad.append(n) if not torch.equal(cur.detach(),v) else None
 return bad
def state_sha_names(model,names):
 h=hashlib.sha256();p=dict(model.named_parameters())
 for n in sorted(names):h.update(n.encode()+b'\0'+p[n].detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def build_model(device):
 from src.models import build_model as make_model
 from src.utils import load_config
 cfg=load_config(str(CONFIG));cfg['cwfa']={'enabled':False,'probability':.3,'epsilon':4.,'norm_eps':1e-12,'start_step':1};cfg.pop('mixstyle',None)
 st,_=load_state(THETA278);m=make_model(cfg);m.load_state_dict(st,strict=True);return m.to(device)
class Target:
 def __init__(self):
  from PIL import Image
  self.names=[f'test2_{i}.png' for i in range(1,1301)];assert all((TEST2/n).is_file() for n in self.names)
 def __len__(self):return len(self.names)
 def __getitem__(self,i):
  from PIL import Image
  from src.transforms import normalize_to_tensor
  n=self.names[i]
  with Image.open(TEST2/n) as f:x=np.asarray(f.convert('RGB')).copy()
  return {'pixel_values':normalize_to_tensor(x),'name':n}
def collate(b):
 import torch
 return {'pixel_values':torch.stack([x['pixel_values'] for x in b]),'name':[x['name'] for x in b]}
def strong_photo(x):
 import torch,torch.nn.functional as F
 # Work in RGB [0,1], then restore ImageNet normalization. Frozen ranges/order.
 mean=torch.tensor([.485,.456,.406],device=x.device,dtype=x.dtype)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=x.device,dtype=x.dtype)[None,:,None,None]
 x=(x*std+mean).clamp(0,1);br=1+random.uniform(-.15,.15);ct=1+random.uniform(-.15,.15);sat=1+random.uniform(-.12,.12);gam=random.uniform(.9,1.1)
 x=x*br;mu=x.mean((-2,-1),keepdim=True);x=(x-mu)*ct+mu;gray=x.mean(1,keepdim=True);x=(x-gray)*sat+gray
 x=x.clamp(0,1).pow(gam);x=x+torch.randn_like(x)*random.uniform(0,.015)
 if random.random()<.25:x=F.avg_pool2d(x,3,1,1)
 return (x.clamp(0,1)-mean)/std

def save_ckpt(step,model,opt,scaler,out,variant,frozen,train_names,history):
 bad=frozen_diff(model,frozen)
 if bad:raise RuntimeError(f'frozen drift: {bad[:5]}')
 import torch
 path=out/f'checkpoint_step{step}.pt'
 torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'step':step,'variant':variant,
  'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},
  'DART':{'init':str(THETA278),'init_sha256':EXPECTED_FILE_SHA['theta278'],'max_steps':MAX_STEPS,'gates':GATES,
  'trainable':train_names,'old_bstar':BSTAR.tolist(),'test2_ground_truth_used':False,'platform_submission_performed':False}},path)
 write_json(out/f'CHECKPOINT_step{step}.json',{'path':str(path),'sha256':sha256_file(path),'step':step,'frozen_identical':True,
  'trainable_scope_sha256':state_sha_names(model,train_names),'last_record':history[-1]})

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--variant',required=True,choices=('D0','D1','D2'));ap.add_argument('--output-dir',required=True);ap.add_argument('--max-steps',type=int,default=2000);ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
 if a.max_steps!=MAX_STEPS:raise SystemExit('DART hard stop requires max_steps exactly 2000')
 if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('2','3') or os.environ.get('CUDA_DEVICE_ORDER')!='PCI_BUS_ID':raise SystemExit('physical GPU2/3 only')
 if a.variant=='D2':
  gate=json.loads((OUT/'02_reliability/D2_GATE.json').read_text());
  if gate['status']!='PASS':raise SystemExit('D2 reliability gate did not pass')
 for p in (str(SNAP),'/home/jhy/gpu2_day_queue'):
  if p not in sys.path:sys.path.insert(0,p)
 import torch,torch.nn.functional as F
 from torch.utils.data import DataLoader
 from src.dataset import UAVSegmentationDataset,segmentation_collate
 from src.rare_crop import RareClassCropper
 from src.transforms import TrainTransform
 from src.utils import load_config
 if sha256_file(THETA278)!=EXPECTED_FILE_SHA['theta278']:raise SystemExit('theta278 SHA mismatch')
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False)
 seed=20260922;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 cfg=load_config('/home/jhy/rare_diag_20260915/atda_20260918/atda_s2s3.yaml');ids=[x.strip() for x in (SNAP/'splits/train.txt').read_text().split() if x.strip()];assert len(ids)==5597
 crop=RareClassCropper(crop_size=1024,rare_probability=.5,priority_ids=(7,4,6),max_offset_fraction=.25)
 src=UAVSegmentationDataset(DATA,ids,TrainTransform(crop_size=1024,config=cfg['transforms'],rare_cropper=crop,ignore_index=255),maximum=None,seed=seed,random_subset=False)
 sl=DataLoader(src,batch_size=1,shuffle=True,num_workers=4,pin_memory=True,drop_last=True,collate_fn=segmentation_collate,generator=torch.Generator().manual_seed(seed+10001),persistent_workers=True)
 tl=None
 if a.variant!='D0':tl=DataLoader(Target(),batch_size=1,shuffle=True,num_workers=4,pin_memory=True,drop_last=True,collate_fn=collate,generator=torch.Generator().manual_seed(seed+10002),persistent_workers=True)
 model=build_model(a.device);train=[];base=[];head=[]
 for n,p in model.named_parameters():
  p.requires_grad_(is_train(n))
  if n.startswith(TRAIN_PREFIX):base.append(p);train.append((n,p))
  elif n.startswith(HEAD_PREFIX):head.append(p);train.append((n,p))
 frozen=snap_frozen(model);mode_map(model);train_names=[n for n,_ in train]
 opt=torch.optim.AdamW([{'params':base,'lr':BASE_LR},{'params':head,'lr':BASE_LR*HEAD_MULT}],weight_decay=WD);scaler=torch.amp.GradScaler('cuda',enabled=True);bstar=torch.from_numpy(BSTAR).to(a.device)[None,:,None,None]
 weights=None
 if a.variant=='D2':weights=np.load(OUT/'02_reliability/D2_WEIGHTS.npz');W=torch.from_numpy(weights['weights'].astype(np.float32)).to(a.device);ce=torch.from_numpy(weights['conf_edges'].astype(np.float32)).to(a.device);de=torch.from_numpy(weights['d4_edges'].astype(np.float32)).to(a.device);d4mean={Path(x['image']).stem:x['d4_disagreement_mean'] for x in json.loads((ROUND2/'cache_278_test2_raw/META.json').read_text())['d4_disagreement']}
 i_src=iter(sl);i_t=iter(tl) if tl else None;hist=[];start=time.time();skips=[]
 for step in range(1,MAX_STEPS+1):
  try:s=next(i_src)
  except StopIteration:i_src=iter(sl);s=next(i_src)
  for gi,g in enumerate(opt.param_groups):g['lr']=(BASE_LR if gi==0 else BASE_LR*HEAD_MULT)*(1-(step-1)/10000)
  opt.zero_grad(set_to_none=True);sx=s['pixel_values'].to(a.device);sy=s['labels'].to(a.device)
  with torch.amp.autocast('cuda',enabled=True):sz=model(pixel_values=sx).logits
  lsrc=F.cross_entropy(F.interpolate(sz.float(),size=sy.shape[-2:],mode='bilinear',align_corners=False),sy,ignore_index=255);lh=lsrc.new_zeros(());ls=lsrc.new_zeros(());covh=covs=0.
  if a.variant!='D0':
   try:t=next(i_t)
   except StopIteration:i_t=iter(tl);t=next(i_t)
   if set(t)!=set(('pixel_values','name')):raise RuntimeError('target returned unexpected key')
   stem=Path(t['name'][0]).stem;tx=t['pixel_values'].to(a.device);view=random.randrange(8);aug=strong_photo(d4_forward(tx,view))
   with torch.amp.autocast('cuda',enabled=True):uz=model(pixel_values=aug).logits
   z=d4_inverse(uz.float(),view)+bstar
   z261=torch.from_numpy(np.load(ROUND2/f'cache_261_test2_raw/scores/{stem}.npy').astype(np.float32)).to(a.device)[None]+bstar
   z278=torch.from_numpy(np.load(ROUND2/f'cache_278_test2_raw/scores/{stem}.npy').astype(np.float32)).to(a.device)[None]+bstar
   p278=z278.softmax(1);c278,y278=p278.max(1);y261=z261.argmax(1);agree=y261==y278;mh=agree&(c278>=CONF_AGREE)
   if mh.any():lh=F.cross_entropy(z,y278,reduction='none')[mh].mean();covh=float(mh.float().mean())
   if a.variant=='D2':
    cb=torch.bucketize(c278,ce)-1;cb=cb.clamp(0,len(ce)-2);db=int(np.clip(np.digitize(d4mean[stem],weights['d4_edges'])-1,0,len(weights['d4_edges'])-2));trans=y261*8+y278;w=W[trans,cb,db]*(~agree)
    if w.sum()>0:
     kl=F.kl_div(z.log_softmax(1),p278,reduction='none').sum(1);ls=(w*kl).sum()/w.sum();covs=float((w>0).float().mean())
  ramp=min(1.,step/RAMP);loss=lsrc+ramp*(LAMBDA_HARD*lh+LAMBDA_SOFT*ls)
  scaler.scale(loss).backward();scaler.unscale_(opt);scale=scaler.get_scale();scaler.step(opt);scaler.update();stepped=scaler.get_scale()>=scale
  if not stepped:skips.append(step)
  mode_map(model);bad=frozen_diff(model,frozen)
  if bad:raise RuntimeError(f'frozen drift step {step}: {bad[:3]}')
  row={'step':step,'source':float(lsrc),'hard':float(lh),'soft':float(ls),'loss':float(loss),'hard_coverage':covh,'soft_coverage':covs,'stepped':stepped,'elapsed':round(time.time()-start,1)};hist.append(row)
  if step%25==0 or step==1:print(json.dumps(row),flush=True)
  if step in GATES:save_ckpt(step,model,opt,scaler,out,a.variant,frozen,train_names,hist)
 if (out/'checkpoint_step2001.pt').exists():raise RuntimeError('hard stop violation')
 summary={'variant':a.variant,'steps':MAX_STEPS,'gates':GATES,'elapsed_sec':round(time.time()-start,1),'amp_skip_steps':skips,'history_tail':hist[-20:],'trainable_names':train_names,'stage23_params':sum(p.numel() for p in base),'head_params':sum(p.numel() for p in head),'test2_ground_truth_used':False,'platform_submission_performed':False,'automatic_continuation':False}
 write_json(out/'DART_RUN_SUMMARY.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
