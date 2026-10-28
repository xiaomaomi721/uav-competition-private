"""Convert official timm DINOv3 weights to installed HF model, with strict accounting."""
import os,sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'/home/jhy/audits/codex_pmt_feasibility_20261026/vendor')
import torch
from torch import nn
from transformers import DINOv3ViTConfig,DINOv3ViTModel
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parent
def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def backbone(path):
 cfg=DINOv3ViTConfig(hidden_size=1024,intermediate_size=4096,num_hidden_layers=24,num_attention_heads=16,patch_size=16,num_register_tokens=4)
 cfg._attn_implementation='sdpa'
 net=DINOv3ViTModel(cfg)
 s=load_file(str(path)); d={}; used=set()
 def put(dst,src):
  d[dst]=s[src];used.add(src)
 for a,b in [('embeddings.cls_token','cls_token'),('embeddings.register_tokens','reg_token'),('embeddings.patch_embeddings.weight','patch_embed.proj.weight'),('embeddings.patch_embeddings.bias','patch_embed.proj.bias'),('norm.weight','norm.weight'),('norm.bias','norm.bias')]:put(a,b)
 d['embeddings.mask_token']=torch.zeros_like(net.embeddings.mask_token)
 for i in range(24):
  a=f'model.layer.{i}.';b=f'blocks.{i}.'
  for n in ['norm1.weight','norm1.bias','norm2.weight','norm2.bias']:put(a+n,b+n)
  for n,m in [('attention.o_proj','attn.proj'),('mlp.up_proj','mlp.fc1'),('mlp.down_proj','mlp.fc2')]:
   for p in ['weight','bias']:put(a+n+'.'+p,b+m+'.'+p)
  put(a+'layer_scale1.lambda1',b+'gamma_1');put(a+'layer_scale2.lambda1',b+'gamma_2')
  q,k,v=s[b+'attn.qkv.weight'].chunk(3);used.add(b+'attn.qkv.weight')
  for n,x in [('q',q),('k',k),('v',v)]:
   d[a+'attention.'+n+'_proj.weight']=x
   if n!='k':d[a+'attention.'+n+'_proj.bias']=torch.zeros(1024)
 assert used==set(s), sorted(set(s)-used)
 net.load_state_dict(d,strict=True)
 return net.eval().requires_grad_(False)
def verify(path):
 import timm
 import subprocess
 gpu=os.environ['CUDA_VISIBLE_DEVICES']
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 assert not any(gpu in line for line in apps.splitlines()),'GPU occupied before conversion verification'
 assert sha256(path)=='45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941'
 torch.set_num_threads(2);torch.manual_seed(42)
 h=backbone(path).cuda()
 t=timm.create_model('vit_large_patch16_dinov3',pretrained=False).eval().cuda()
 t.load_state_dict(load_file(str(path)),strict=True)
 rows=[]
 for side in [256,512]:
  x=torch.randn(1,3,side,side,device='cuda')
  with torch.no_grad():
   ho=h(x,output_hidden_states=True)
   hf=[h.norm(ho.hidden_states[n]) for n in [6,12,18,24]]
   z,rope=t._pos_embed(t.patch_embed(x));z=t.norm_pre(z);tf=[]
   for i,blk in enumerate(t.blocks):
    z=blk(z,rope=rope)
    if i in [5,11,17,23]:tf.append(t.norm(z))
  for i,(a,b) in enumerate(zip(hf,tf)):
   rel=((a-b).square().mean()/a.square().mean()).sqrt().item()
   rows.append(dict(side=side,layer=[5,11,17,23][i],relative_rmse=rel,max_abs=(a-b).abs().max().item()))
   assert rel<0.001,rows[-1]
 print(json.dumps(rows),flush=True)
 (ROOT/'conversion_verified.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':verify(Path(sys.argv[1]))

