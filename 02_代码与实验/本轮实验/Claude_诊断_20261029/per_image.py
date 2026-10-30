import csv, io, zipfile, json
from multiprocessing import Pool
import numpy as np
from PIL import Image
MAN="/home/jhy/aic2026_ziponly_20261028/manifests/semifinal_3594.csv"
CH="/home/jhy/aic2026_ctrl3594_teacher_20261027/reference/H75_BSTAR_20261026/H75_BSTAR_semifinal.zip"
M2F="/home/jhy/aic2026_ziponly_20261028/07_candidate_zips/M2F_E02_B1_20261028_V1/predictions.zip"
rows=list(csv.DictReader(open(MAN)))
def f(i):
    out=[]
    for z in (CH,M2F):
        a=np.asarray(Image.open(io.BytesIO(zipfile.ZipFile(z).read(rows[i]["output_name"])))).astype(np.int64)-1
        out.append(np.bincount(a.ravel(),minlength=8)/a.size)
    return i,out[0],out[1]
with Pool(48) as p: R=p.map(f,range(len(rows)),chunksize=40)
C=np.array([r[1] for r in R]); M=np.array([r[2] for r in R])
np.save("/home/jhy/aic2026_claude_diag_20261029/champ_frac.npy",C); np.save("/home/jhy/aic2026_claude_diag_20261029/m2f_frac.npy",M)
sub=np.array([r["subset"] for r in rows])
N=["Bg","Bld","Road","Water","Barren","Veg","Agri","Veh"]
for cls in (2,5):
    miss=C[:,cls]==0
    print(f"== champion predicts no {N[cls]}: {miss.sum()} images; by subset",{s:int((miss&(sub==s)).sum()) for s in np.unique(sub)})
    print("   mean class frac in those images:",dict(zip(N,(100*C[miss].mean(0)).round(2))))
    print("   M2F predicts that class there in",int((M[miss,cls]>0).sum()),"images; M2F mean frac",round(100*M[miss,cls].mean(),3))
both=(C[:,2]==0)&(C[:,5]==0)
print("no road AND no veg:",both.sum(),"mean frac",dict(zip(N,(100*C[both].mean(0)).round(2))))
dom=C.argmax(1); print("dominant-class histogram (champ):",dict(zip(N,np.bincount(dom,minlength=8))))
# images whose prediction is >90% one class
for k in range(8):
    m=C[:,k]>0.9
    if m.sum(): print(f"  >90% {N[k]}: {m.sum()} imgs, subsets",{s:int((m&(sub==s)).sum()) for s in np.unique(sub)})
idx=np.where(C[:,5]==0)[0]
print("sample no-veg names:",[rows[i]["output_name"]+":"+rows[i]["subset"] for i in idx[:12]])
