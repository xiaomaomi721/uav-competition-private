import json, numpy as np
d=json.load(open("/home/jhy/aic2026_claude_diag_20261029/domain_feat.json"))
S,T=d["source"],d["target"]
fs=np.array([s["feat"] for s in S]); ft=np.array([t["feat"] for t in T])
names=["loglapvar","sat","gray_mean","gray_std","G-R"]
for k,n in enumerate(names):
    print(f"{n:10s} src q10/50/90 {np.percentile(fs[:,k],[10,50,90]).round(3)}  tgt {np.percentile(ft[:,k],[10,50,90]).round(3)}")
# histogram of loglapvar
bins=np.linspace(min(fs[:,0].min(),ft[:,0].min()),max(fs[:,0].max(),ft[:,0].max()),21)
hs,_=np.histogram(fs[:,0],bins); ht,_=np.histogram(ft[:,0],bins)
print("loglapvar bins",bins.round(2)); print("src",(hs/hs.sum()*100).round(1)); print("tgt",(ht/ht.sum()*100).round(1))
# kmeans 2 clusters on standardized [loglapvar, sat, gray_std]
X=np.vstack([fs,ft])[:,[0,1,3]]; mu,sd=X.mean(0),X.std(0); Z=(X-mu)/sd
rng=np.random.default_rng(0)
for K in (2,3):
    c=Z[rng.choice(len(Z),K,replace=False)]
    for _ in range(100):
        lab=((Z[:,None]-c[None])**2).sum(2).argmin(1); c=np.array([Z[lab==k].mean(0) for k in range(K)])
    ls,lt=lab[:len(fs)],lab[len(fs):]
    print(f"== K={K} centers(raw):",(c*sd+mu).round(3).tolist())
    for k in range(K):
        gt=np.array([s["gtfrac"] for s in S])[ls==k].sum(0); gt=gt/gt.sum()*100
        tz=np.array([t["zero"] for t in T])[lt==k].sum(0); tb=np.array([t["bstar"] for t in T])[lt==k].sum(0)
        print(f" cluster{k}: src n={int((ls==k).sum())} ({(ls==k).mean()*100:.1f}%)  tgt n={int((lt==k).sum())} ({(lt==k).mean()*100:.1f}%)  tgt subsets",{s:int(((lt==k)&(np.array([t['subset'] for t in T])==s)).sum()) for s in ('test1','test2','test3_new')})
        print("   src GT%  ",gt.round(2).tolist())
        print("   tgt zero%",(tz/tz.sum()*100).round(2).tolist()); print("   tgt bstar%",(tb/tb.sum()*100).round(2).tolist())
        vi=[i for i,s in enumerate(S) if "zero" in s and ls[i]==k]
        for tag in ("zero","bstar"):
            cm=np.array([S[i][tag] for i in vi]).sum(0).reshape(8,8) if vi else np.zeros((8,8))
            iou=np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1)-np.diag(cm),1)
            pres=(cm.sum(1)>0)
            print(f"   src-val n={len(vi)} {tag} mIoU={100*iou[pres].mean():.2f} perclass",(100*iou).round(1).tolist())
    np.save(f"/home/jhy/aic2026_claude_diag_20261029/cluster_K{K}_src.npy",ls); np.save(f"/home/jhy/aic2026_claude_diag_20261029/cluster_K{K}_tgt.npy",lt)
