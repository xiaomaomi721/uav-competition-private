import csv, io, zipfile
import numpy as np
from PIL import Image, ImageDraw
MAN="/home/jhy/aic2026_ziponly_20261028/manifests/semifinal_3594.csv"
CH="/home/jhy/aic2026_ctrl3594_teacher_20261027/reference/H75_BSTAR_20261026/H75_BSTAR_semifinal.zip"
M2F="/home/jhy/aic2026_ziponly_20261028/07_candidate_zips/M2F_E02_B1_20261028_V1/predictions.zip"
IMG="/home/jhy/aic2026_h76_20261026/data/test3/images/"
rows=list(csv.DictReader(open(MAN)))
C=np.load("/home/jhy/aic2026_claude_diag_20261029/champ_frac.npy")
PAL=np.array([[200,200,200],[230,60,60],[250,170,40],[50,110,240],[160,110,50],[30,150,40],[200,90,220],[0,240,240]],np.uint8)
rng=np.random.default_rng(0)
sel_noveg=rng.choice(np.where((C[:,5]==0)&(C[:,0]>0.3))[0],6,replace=False)
sel_noroad=rng.choice(np.where((C[:,2]==0)&(C[:,0]>0.3)&(C[:,5]>0))[0],6,replace=False)
T=256
sheet=Image.new("RGB",(T*3,T*12),(0,0,0))
zc,zm=zipfile.ZipFile(CH),zipfile.ZipFile(M2F)
for r,i in enumerate(list(sel_noveg)+list(sel_noroad)):
    n=rows[i]["output_name"]
    im=Image.open(IMG+rows[i]["input_name"]).convert("RGB").resize((T,T))
    a=np.asarray(Image.open(io.BytesIO(zc.read(n))))-1; b=np.asarray(Image.open(io.BytesIO(zm.read(n))))-1
    sheet.paste(im,(0,r*T)); sheet.paste(Image.fromarray(PAL[a]).resize((T,T),Image.NEAREST),(T,r*T)); sheet.paste(Image.fromarray(PAL[b]).resize((T,T),Image.NEAREST),(2*T,r*T))
    ImageDraw.Draw(sheet).text((4,r*T+4),f"{n} {rows[i]['subset']}",fill=(255,255,0))
sheet.save("/home/jhy/aic2026_claude_diag_20261029/fail_sheet.jpg",quality=88)
print("noveg",[rows[i]["output_name"] for i in sel_noveg]); print("noroad",[rows[i]["output_name"] for i in sel_noroad])
