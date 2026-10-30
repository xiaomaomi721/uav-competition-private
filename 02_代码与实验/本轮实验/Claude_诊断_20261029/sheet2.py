import csv, io, zipfile, os
import numpy as np
from PIL import Image, ImageDraw
SRC="/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/train"
PAL=np.array([[200,200,200],[230,60,60],[250,170,40],[50,110,240],[160,110,50],[30,150,40],[200,90,220],[0,240,240],[0,0,0]],np.uint8)
ids=sorted(os.listdir(SRC+"/images"))
rng=np.random.default_rng(1); sel=rng.choice(len(ids),20,replace=False)
T=192; sheet=Image.new("RGB",(T*4,T*10))
for k,i in enumerate(sel):
    im=Image.open(f"{SRC}/images/{ids[i]}").convert("RGB").resize((T,T))
    g=np.asarray(Image.open(f"{SRC}/masks/{ids[i]}")).astype(np.int64); g=np.where(g==0,9,g)-1
    r,c=divmod(k,2); sheet.paste(im,(c*2*T,r*T)); sheet.paste(Image.fromarray(PAL[g]).resize((T,T),Image.NEAREST),(c*2*T+T,r*T))
    ImageDraw.Draw(sheet).text((c*2*T+3,r*T+3),ids[i],fill=(255,255,0))
sheet.save("/home/jhy/aic2026_claude_diag_20261029/src_sheet.jpg",quality=85)
print(len(ids), ids[:3], ids[-3:])
