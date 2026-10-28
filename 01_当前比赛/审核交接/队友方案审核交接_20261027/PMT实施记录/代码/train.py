import os
os.environ.setdefault('CUDA_DEVICE_ORDER','PCI_BUS_ID')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','GPU-bc304d79-c807-968a-eaad-d2c2ecc5a984')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys,json,time,math,subprocess,random,argparse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/home/jhy/audits/codex_pmt_feasibility_20261026/vendor')
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,Subset
from model import UAVPMT
from loss import mask_loss
from backbone import sha256

DATA=Path('/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/train')
SPLIT=Path('/home/jhy/uav_semantic_segmentation/splits')
SEED=20261026
def guard():
 out=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
 for line in out.splitlines():
  uuid,pid=[x.strip() for x in line.split(',')]
  if uuid==os.environ['CUDA_VISIBLE_DEVICES'] and int(pid)!=os.getpid():
   raise RuntimeError('GPU0 now has another process; stopping our own job: '+line)
def seed():
 torch.set_num_threads(2);torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
class Source(Dataset):
 def __init__(self,names,count=None,augment=False):
  self.names=names;self.count=count or len(names);self.augment=augment;self.orders={}
 def __len__(self):return self.count
 def __getitem__(self,i):
  if self.augment:
   epoch,offset=divmod(i,len(self.names))
   if epoch not in self.orders:self.orders[epoch]=np.random.default_rng(SEED+epoch).permutation(len(self.names))
   name=self.names[self.orders[epoch][offset]]
  else:name=self.names[i]
  with Image.open(DATA/'images'/f'{name}.png') as im:x=np.array(im.convert('RGB'))
  with Image.open(DATA/'masks'/f'{name}.png') as im:y=np.array(im,dtype=np.int64)
  assert x.shape==(1024,1024,3) and y.shape==(1024,1024),(name,x.shape,y.shape)
  assert y.min()>=0 and y.max()<=8,(name,np.unique(y))
  assert (y>0).any(),name
  if self.augment:
   rng=np.random.default_rng(SEED+1_000_000+i);k=int(rng.integers(4));flip=bool(rng.integers(2))
   x=np.rot90(x,k);y=np.rot90(y,k)
   if flip:x=np.flip(x,1);y=np.flip(y,1)
  return torch.from_numpy(x.copy()).permute(2,0,1).float()/255,torch.from_numpy(y.copy()),name
def report(obj,path=None):
 print(json.dumps(obj,ensure_ascii=False),flush=True)
 if path:
  with open(path,'a') as f:f.write(json.dumps(obj,ensure_ascii=False)+'\n')
def anneal(net,step,total):
 starts=np.array([0,5384,7403,9422,11442,13461])/16826*total
 ends=np.array([3000,8749,10769,12788,14807,16826])/16826*total
 p=np.clip((ends-step)/(ends-starts),0,1)**.9
 net.decoder.attn_mask_probs.copy_(torch.tensor(p,device='cuda'))
def lr_factor(step,total):
 if step<400:return (step+1)/400
 return .5*(1+math.cos(math.pi*min(1,(step-400)/max(1,total-400))))
def metrics(cm):
 tp=np.diag(cm);u=cm.sum(0)+cm.sum(1)-tp
 iou=np.divide(tp,u,out=np.full(8,np.nan),where=u>0)
 return {'miou':float(np.nanmean(iou)*100),'iou':(iou*100).tolist(),'cm':cm.tolist()}
@torch.no_grad()
def evaluate(net,names,out):
 cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
 net.eval();saved=net.decoder.attn_mask_probs.clone();net.decoder.attn_mask_probs.zero_()
 cms=np.zeros((8,8),np.int64);subset_cm=cms.copy()
 subset=set((ROOT/'no_detected_train_overlap.txt').read_text().split()) if (ROOT/'no_detected_train_overlap.txt').exists() else set()
 loader=DataLoader(Source(names),batch_size=1,num_workers=2,pin_memory=True)
 nsub=0;start=time.time()
 for i,(x,y,name) in enumerate(loader):
  if i%100==0:guard()
  with torch.autocast('cuda',dtype=torch.bfloat16):
   masks,cls=net(x.cuda(non_blocking=True))
  logits=F.interpolate(masks[-1].float(),size=(1024,1024),mode='bilinear',align_corners=False)
  score=torch.einsum('bqc,bqhw->bchw',cls[-1].float().softmax(-1)[...,:8],logits.sigmoid())
  pred=score.argmax(1).cpu().numpy()[0];gt=y.numpy()[0]-1;valid=(gt>=0)&(gt<8)
  cm=np.bincount(gt[valid]*8+pred[valid],minlength=64).reshape(8,8);cms+=cm
  if name[0] in subset:subset_cm+=cm;nsub+=1
  if (i+1)%200==0:report({'eval_done':i+1,'elapsed_s':time.time()-start})
 result={'protocol':'native1024 single-view; no Bstar; ignore0; attention masks disabled at inference','all':metrics(cms),'n':len(names),'no_detected_train_overlap':metrics(subset_cm) if nsub else None,'n_no_detected_train_overlap':nsub,'elapsed_s':time.time()-start}
 out.write_text(json.dumps(result,indent=2));report(result)
 net.decoder.attn_mask_probs.copy_(saved);net.train()
 torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
 return result
def save(net,opt,step,out,args):
 # Backbone immutable, separately SHA-verified and explicitly required for restore.
 state={k:v for k,v in net.state_dict().items() if not k.startswith('encoder.')}
 data={'model':state,'optimizer':opt.state_dict(),'step':step,'config':vars(args),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}
 tmp=out.with_suffix('.tmp');torch.save(data,tmp);tmp.replace(out)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['smoke','train'],required=True);ap.add_argument('--steps',type=int,default=8000);ap.add_argument('--resume');args=ap.parse_args()
 seed();guard()
 assert (ROOT/'conversion_verified.json').exists(),'First verify converted backbone outputs.'
 integrity=json.loads((ROOT/'source_integrity.json').read_text())
 assert all(not v['bad'] for v in integrity.values())
 if args.mode=='train':assert (ROOT/'smoke_passed.json').exists(),'Real-data smoke test must pass first.'
 for path,expected in [('dino.safetensors','45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941'),('pmt_decoder.bin','5039b0090a15a836c6217fc6d3ebced96fc217b1fd6908f1fbb4a274fce92566')]:
  assert sha256(ROOT/path)==expected,path
 out=ROOT/('smoke' if args.mode=='smoke' else 'formal');out.mkdir(exist_ok=True)
 if args.mode=='train' and (out/'config.json').exists() and not args.resume:raise RuntimeError('Existing formal run: use explicit resume.')
 names=(SPLIT/'train.txt').read_text().split();val=(SPLIT/'val.txt').read_text().split()
 assert len(names)==5597 and len(val)==1399 and not set(names)&set(val)
 net=UAVPMT(ROOT/'dino.safetensors',ROOT/'pmt_decoder.bin').cuda().train()
 params=[p for n,p in net.named_parameters() if p.requires_grad and not n.startswith('decoder.class_head.')]
 head=list(net.decoder.class_head.parameters())
 opt=torch.optim.AdamW([{'params':params,'lr':2e-5,'base_lr':2e-5},{'params':head,'lr':2e-4,'base_lr':2e-4}],weight_decay=.05)
 startstep=0
 if args.resume:
  cp=torch.load(args.resume,map_location='cpu',weights_only=False) # only our own local checkpoint
  missing=net.load_state_dict(cp['model'],strict=False)
  assert all(k.startswith('encoder.') for k in missing.missing_keys) and not missing.unexpected_keys
  opt.load_state_dict(cp['optimizer']);startstep=cp['step']
  torch.set_rng_state(cp['torch_rng']);torch.cuda.set_rng_state_all(cp['cuda_rng']);np.random.set_state(cp['numpy_rng']);random.setstate(cp['python_rng'])
 config={**vars(args),'seed':SEED,'side':1024,'batch':1,'accum':4,'train_n':len(names),'val_n':len(val),'train_split_sha256':sha256(SPLIT/'train.txt'),'val_split_sha256':sha256(SPLIT/'val.txt'),'trainable':sum(p.numel() for p in net.parameters() if p.requires_grad),'load_report':net.load_report,'loss':{'class':2,'mask_bce':5,'dice':5,'no_object':.1,'points':12544,'ignore':0},'encoder_frozen':True,'torch':torch.__version__}
 (out/'config.json').write_text(json.dumps(config,indent=2));report(config)
 steps=4 if args.mode=='smoke' else args.steps
 data=Source(names,steps*4,augment=True);loader=DataLoader(Subset(data,range(startstep*4,steps*4)),batch_size=1,num_workers=4,pin_memory=True,generator=torch.Generator().manual_seed(SEED))
 opt.zero_grad(set_to_none=True);start=time.time();losses=[];torch.cuda.reset_peak_memory_stats()
 for micro,(x,y,name) in enumerate(loader,start=startstep*4):
  step=micro//4;guard() if micro%40==0 else None
  anneal(net,step,args.steps)
  for pg in opt.param_groups:pg['lr']=pg['base_lr']*lr_factor(step,args.steps)
  x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):ms,cs=net(x)
  loss=sum(mask_loss(m[0],c[0],y[0]) for m,c in zip(ms,cs))
  assert torch.isfinite(loss),f'nonfinite loss step={step}'
  (loss/4).backward();losses.append(float(loss.detach()))
  assert not net.encoder.training and all(p.grad is None for p in net.encoder.parameters())
  if micro%4==3:
   gn=torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],1.0,error_if_nonfinite=True)
   opt.step();opt.zero_grad(set_to_none=True)
   if (step+1)%20==0 or step<4:
    report({'update':step+1,'loss_avg':sum(losses)/len(losses),'grad_norm':float(gn),'elapsed_s':time.time()-start,'peak_MiB':torch.cuda.max_memory_allocated()/2**20,'lr':opt.param_groups[0]['lr']},out/'train.jsonl');losses=[]
   if args.mode=='train' and (step+1 in [100,2000,4000,8000] or (step+1)%1000==0):
    save(net,opt,step+1,out/f'step_{step+1:05d}.pt',args)
    if step+1 in [2000,4000,8000]:evaluate(net,val,out/f'val_{step+1:05d}.json')
  del loss,ms,cs
 if args.mode=='smoke':
  evaluate(net,val[:4],out/'eval_smoke.json')
  (ROOT/'smoke_passed.json').write_text(json.dumps({'updates':steps,'peak_MiB':torch.cuda.max_memory_allocated()/2**20,'elapsed_s':time.time()-start}))
 else:
  if args.steps%1000:save(net,opt,args.steps,out/f'step_{args.steps:05d}.pt',args)
 report({'completed':args.mode,'updates':steps})
if __name__=='__main__':main()

