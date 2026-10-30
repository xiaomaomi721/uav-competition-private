import json, numpy as np
d=json.load(open("/home/jhy/aic2026_claude_diag_20261029/domain_feat.json"))
S=[s for s in d["source"] if "zero" in s]
lab=np.load("/home/jhy/aic2026_claude_diag_20261029/cluster_K3_src.npy")
allS=d["source"]; idx={s["name"]:i for i,s in enumerate(allS)}
N=["Bg","Bld","Road","Water","Barren","Veg","Agri","Veh"]
def decomp(cm):
    tp=np.diag(cm); U=cm.sum(0)+cm.sum(1)-tp; iou=tp/np.maximum(U,1)
    L=np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            if i!=j: L[i,j]=cm[i,j]*(1/max(U[i],1)+1/max(U[j],1))/8*100
    return iou*100, L
for tag in ("zero","bstar"):
    cm=sum(np.array(s[tag]).reshape(8,8) for s in S)
    iou,L=decomp(cm)
    print(f"=== Source1399 H75 {tag}: mIoU={iou.mean():.3f}  total loss={100-iou.mean():.2f}pp")
    print("  per-class IoU:",dict(zip(N,iou.round(1))))
    cls_loss=[(L[k,:].sum()+L[:,k].sum())/2 for k in range(8)]
    print("  loss touching each class (pp, pair loss split half/half):",dict(zip(N,np.round(cls_loss,2))))
    pairs=sorted([(L[i,j]+L[j,i],N[i],N[j],L[i,j],L[j,i]) for i in range(8) for j in range(i+1,8)],reverse=True)[:8]
    for v,a,b,ab,ba in pairs: print(f"  {a:6s}<->{b:6s} {v:5.2f}pp  ({a}->{b} {ab:.2f}, {b}->{a} {ba:.2f})")
    tri=sum(L[i,j] for i in (0,4,6) for j in (0,4,6) if i!=j)
    print(f"  Bg/Barren/Agri triangle internal: {tri:.2f}pp of {100-iou.mean():.2f}pp")
# per appearance cluster (zero)
for k in range(3):
    sel=[s for s in S if lab[idx[s["name"]]]==k]
    cm=sum(np.array(s["zero"]).reshape(8,8) for s in sel); iou,L=decomp(cm)
    print(f"cluster{k} n={len(sel)} zero mIoU={iou.mean():.2f}  Barren IoU={iou[4]:.1f} Bg={iou[0]:.1f} Agri={iou[6]:.1f}")
