import csv, os, json
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageFilter
import torch, torch.nn.functional as F
torch.set_num_threads(1)
SRC="/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/train"
TGT="/home/jhy/aic2026_h76_20261026/data/test3/images/"
SC="/home/jhy/aic2026_ziponly_20261028/01_h75_cache/source1399_fp32/scores/"
TC="/home/jhy/aic2026_ziponly_20261028/01_h75_cache/target3594_fp32/scores/"
MAN="/home/jhy/aic2026_ziponly_20261028/manifests/semifinal_3594.csv"
VAL=[x.strip() for x in open("/home/jhy/uav_semantic_segmentation/splits/val.txt").read().split() if x.strip()]
VAL=[v if v.endswith(".png") else v+".png" for v in VAL]
BSTAR=torch.tensor([-0.2,0.1,0.0709,0.4,3.6,0.4,2.1215,-0.5])[:,None,None]
def feat(path):
    im=Image.open(path).convert("RGB").resize((256,256),Image.BILINEAR)
    a=np.asarray(im).astype(np.float32)
    g=a.mean(2); lap=g[1:-1,1:-1]*4-g[:-2,1:-1]-g[2:,1:-1]-g[1:-1,:-2]-g[1:-1,2:]
    mx,mn=a.max(2),a.min(2); sat=np.where(mx>0,(mx-mn)/np.maximum(mx,1),0)
    return [float(np.log(lap.var()+1e-6)),float(sat.mean()),float(g.mean()),float(g.std()),float(a[...,1].mean()-a[...,0].mean())]
def src_job(name):
    f=feat(f"{SRC}/images/{name}")
    gt=np.asarray(Image.open(f"{SRC}/masks/{name}")).astype(np.int64)
    out={"name":name,"feat":f,"gtfrac":np.bincount(np.where(gt>0,gt-1,8).ravel(),minlength=9)[:8].tolist()}
    if name in VS:
        z=torch.from_numpy(np.load(SC+name+".npy").astype(np.float32))
        v=gt>=1; y=gt[v]-1
        for tag,b in (("zero",0*BSTAR),("bstar",BSTAR)):
            p=F.interpolate((z+b)[None],size=gt.shape,mode="bilinear",align_corners=False)[0].argmax(0).numpy()[v]
            out[tag]=np.bincount(y*8+p,minlength=64).tolist()
    return out
def tgt_job(r):
    f=feat(TGT+r["input_name"])
    z=torch.from_numpy(np.load(TC+r["output_name"]+".npy").astype(np.float32)).reshape(8,-1)
    pz=np.bincount(z.argmax(0).numpy(),minlength=8); pb=np.bincount((z+BSTAR[:,:,0]).argmax(0).numpy(),minlength=8)
    return {"name":r["output_name"],"subset":r["subset"],"feat":f,"zero":pz.tolist(),"bstar":pb.tolist()}
VS=set(VAL)
if __name__=="__main__":
    ids=sorted(os.listdir(SRC+"/images"))
    with Pool(64) as p:
        S=p.map(src_job,ids,chunksize=20)
        T=p.map(tgt_job,list(csv.DictReader(open(MAN))),chunksize=20)
    json.dump({"source":S,"target":T},open("/home/jhy/aic2026_claude_diag_20261029/domain_feat.json","w"))
    print("done",len(S),len(T),sum("zero" in s for s in S))
