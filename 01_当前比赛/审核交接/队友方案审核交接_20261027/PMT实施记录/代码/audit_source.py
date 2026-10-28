"""CPU-only source integrity audit; official train/val remain unchanged."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,time
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parent
D=Path('/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/train')
S=Path('/home/jhy/uav_semantic_segmentation/splits')
def check(name):
 with Image.open(D/'images'/f'{name}.png') as im:shape,mode=im.size,im.mode
 with Image.open(D/'masks'/f'{name}.png') as im:y=np.array(im)
 bad=[]
 if shape!=(1024,1024) or mode!='RGB':bad.append('image shape/mode')
 if y.shape!=(1024,1024) or y.min()<0 or y.max()>8:bad.append('label shape/range')
 c=np.bincount(y.flatten(),minlength=9)
 if c[1:].sum()==0:bad.append('all ignore')
 return name,c.tolist(),bad
def main():
 start=time.time();out={}
 for split in ['train','val']:
  names=(S/f'{split}.txt').read_text().split();rows=[];tot=np.zeros(9,np.int64);presence=tot.copy()
  with ThreadPoolExecutor(max_workers=4) as pool:
   for i,(name,c,bad) in enumerate(pool.map(check,names)):
    tot+=c;presence+=np.array(c)>0
    if bad:rows.append({'name':name,'bad':bad})
  out[split]={'n':len(names),'pixel_counts_ignore_then8':tot.tolist(),'images_containing_ignore_then8':presence.tolist(),'bad':rows}
  print(json.dumps({split:out[split],'elapsed_s':time.time()-start}),flush=True)
 (R/'source_integrity.json').write_text(json.dumps(out,indent=2))
 assert all(not v['bad'] for v in out.values())
if __name__=='__main__':main()

