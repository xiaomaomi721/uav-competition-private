from pathlib import Path
import torch,numpy as np,json
from PIL import Image
r=Path(__file__).resolve().parent
s=torch.load(r/'pmt_decoder.bin',map_location='cpu',weights_only=True)
print('checkpoint_type',type(s))
if 'state_dict' in s:s=s['state_dict']
print('nkeys',len(s),'keys',list(s)[:15],list(s)[-15:],flush=True)
print('heads',[(k,list(v.shape)) for k,v in s.items() if 'class_head' in k],flush=True)
data=next(Path('/home/jhy/datasets').glob('*训练集/train/masks'))
print('labels',[(p.name,np.unique(np.asarray(Image.open(p))).tolist()) for p in sorted(data.glob('*.png'))[:10]],flush=True)

