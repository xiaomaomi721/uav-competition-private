import sys, json
sys.path.insert(0,'/home/jhy/audits/codex_pmt_feasibility_20261026/vendor')
import torch,timm
from transformers import DINOv3ViTConfig,DINOv3ViTModel
torch.set_num_threads(2)
t=timm.create_model('vit_large_patch16_dinov3',pretrained=False)
h=DINOv3ViTModel(DINOv3ViTConfig(hidden_size=1024,intermediate_size=4096,num_hidden_layers=24,num_attention_heads=16,patch_size=16,num_register_tokens=4))
for name,m in [('TIMM',t),('HF',h)]:
 print(name,[(k,list(v.shape)) for k,v in m.state_dict().items() if not any('.'+str(i)+'.' in k for i in range(1,24))],flush=True)
print('HFCONFIG',h.config,flush=True)
print('ROPE',t.rope,flush=True)
