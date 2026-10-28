import torch
from torch import nn
from pathlib import Path
from backbone import backbone
from models.pmd import PlainMaskDecoder
class UAVPMT(nn.Module):
 def __init__(self,backbone_path,decoder_path,side=1024):
  super().__init__()
  self.encoder=backbone(backbone_path)
  self.bn=nn.ModuleList([nn.SyncBatchNorm(1024) for _ in range(4)])
  self.decoder=PlainMaskDecoder(embed_dim=1024,hidden_dim=1024,num_prefix_tokens=5,grid_size=(side//16,side//16),patch_size=(16,16),num_classes=8,num_q=100,num_blocks=6,masked_attn_enabled=True,interaction_indices=[5,11,17,23],num_heads=16)
  self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
  self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
  state=torch.load(decoder_path,map_location='cpu',weights_only=True)
  if 'state_dict' in state:state=state['state_dict']
  expected={k:v for k,v in self.state_dict().items() if k.startswith(('bn.','decoder.')) and not k.startswith('decoder.class_head.') and k!='decoder.attn_mask_probs'}
  mapped={};ignored=[]
  for k,v in state.items():
   n=k.removeprefix('network.')
   if n in expected:
    assert expected[n].shape==v.shape,(n,expected[n].shape,v.shape)
    mapped[n]=v
   else:ignored.append(k)
  assert set(mapped)==set(expected),sorted(set(expected)-set(mapped))
  # Only known non-task layers may be dropped.
  assert all(k.removeprefix('network.').startswith(('decoder.class_head.','encoder.','criterion.')) or k.removeprefix('network.')=='decoder.attn_mask_probs' for k in ignored),ignored
  self.load_state_dict(mapped,strict=False)
  self.load_report={'loaded_tensors':len(mapped),'ignored_keys':ignored}
 def train(self,mode=True):
  super().train(mode);self.encoder.eval();return self
 def forward(self,x):
  x=(x-self.mean)/self.std
  with torch.no_grad():
   z=self.encoder(x,output_hidden_states=True)
   features=[self.encoder.norm(z.hidden_states[i]) for i in [6,12,18,24]]
   rope=self.encoder.rope_embeddings(x)
  features=[bn(v.transpose(1,2)).transpose(1,2) for bn,v in zip(self.bn,features)]
  return self.decoder(features,rope)

