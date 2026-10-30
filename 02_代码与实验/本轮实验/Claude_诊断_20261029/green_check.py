import csv, io, zipfile, os, json
from multiprocessing import Pool
import numpy as np
from PIL import Image
D="/home/jhy/aic2026_claude_diag_20261029/"
MAN="/home/jhy/aic2026_ziponly_20261028/manifests/semifinal_3594.csv"
CH="/home/jhy/aic2026_ctrl3594_teacher_20261027/reference/H75_BSTAR_20261026/H75_BSTAR_semifinal.zip"
TGT="/home/jhy/aic2026_h76_20261026/data/test3/images/"; TD="/home/jhy/aic2026_h76_20261026/code/reliable_kd_20261028/teacher_d4/"
SRC="/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/train"
rows=list(csv.DictReader(open(MAN))); ids=sorted(os.listdir(SRC+"/images"))
lt=np.load(D+"cluster_K3_tgt.npy"); ls=np.load(D+"cluster_K3_src.npy")
def tj(i):
    r=rows[i]; im=np.asarray(Image.open(TGT+r["input_name"]).convert("RGB").resize((256,256))).reshape(-1,3).astype(np.float64)
    p=(np.asarray(Image.open(io.BytesIO(zipfile.ZipFile(CH).read(r["output_name"]))))[::4,::4]-1).reshape(-1)
    d=np.load(TD+os.path.splitext(r["input_name"])[0]+".npz"); w=((d["votes"].astype(np.float32)/8)**2*(1-d["hn"].astype(np.float32)).clip(0,1)).reshape(-1)
    s=np.zeros((8,3)); n=np.bincount(p,minlength=8); ws=np.bincount(p,weights=w,minlength=8)
    for k in range(8): s[k]=im[p==k].sum(0)
    return s,n,ws
def sj(i):
    im=np.asarray(Image.open(f"{SRC}/images/{ids[i]}").convert("RGB").resize((256,256))).reshape(-1,3).astype(np.float64)
    g=np.asarray(Image.open(f"{SRC}/masks/{ids[i]}"))[::4,::4].reshape(-1).astype(np.int64)
    s=np.zeros((8,3)); n=np.zeros(8)
    for k in range(8): m=g==k+1; s[k]=im[m].sum(0); n[k]=m.sum()
    return s,n
N=["Bg","Bld","Road","Water","Barren","Veg","Agri","Veh"]
with Pool(16) as P:
    T=P.map(tj,np.where(lt==1)[0].tolist(),chunksize=10); S=P.map(sj,np.where(ls==1)[0].tolist(),chunksize=10)
ts=sum(t[0] for t in T); tn=sum(t[1] for t in T); tw=sum(t[2] for t in T)
ss=sum(s[0] for s in S); sn=sum(s[1] for s in S)
print("TARGET green-cluster, champion-predicted class: mean RGB / pixel share / mean teacher weight a^2(1-H)")
for k in range(8): print(f"  {N[k]:6s} RGB={np.round(ts[k]/max(tn[k],1),1)}  share={100*tn[k]/tn.sum():.1f}%  w={tw[k]/max(tn[k],1):.3f}")
print("SOURCE green-cluster, GT class: mean RGB / share")
for k in range(8): print(f"  {N[k]:6s} RGB={np.round(ss[k]/max(sn[k],1),1)}  share={100*sn[k]/sn.sum():.1f}%")
