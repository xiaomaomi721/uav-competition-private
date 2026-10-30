import json, csv, numpy as np
from multiprocessing import Pool
D="/home/jhy/aic2026_claude_diag_20261029/"; PC="/home/jhy/aic2026_presence_calib_20261029/out/"
N=["Bg","Bld","Road","Water","Barren","Veg","Agri","Veh"]
OFF_T=np.array([3523,2329,2288,1754,721,3021,1281,752]); OFF_S=np.array([6777,4592,4367,2537,1098,6094,1192,1846])
S=json.load(open(D+"domain_feat.json"))["source"]
G=np.array([s["gtfrac"] for s in S],float)          # per-image GT pixel counts (valid classes)
share=G/np.maximum(G.sum(1,keepdims=True),1)
pres=(G>0)
assert (pres.sum(0)==OFF_S).all(), pres.sum(0)
a=np.array([share[pres[:,k],k].mean() for k in range(8)])   # mean coverage when present
raw=a*OFF_T/3594; est=raw/raw.sum()
print("Source presence matches official train column: True")
print("mean coverage when present:",dict(zip(N,(100*a).round(2))))
print(f"Target est share (normalized, raw sum={raw.sum():.3f}):",dict(zip(N,(100*est).round(2))))
src_share=G.sum(0)/G.sum(); print("Source pixel share:",dict(zip(N,(100*src_share).round(2))))
def L(path,*keys):
    d=json.load(open(path))
    for k in keys: d=d[k]
    return np.array(d,float)/100
areas={}
zc=json.load(open(D+"zip_compare_out.json"))
areas["CHAMP(70.999)"]=np.array(zc["CHAMP_self"]["area_pct_all"])/100
areas["C1(70.908)"]=np.array(zc["C1"]["area_pct_all"])/100
areas["CTRL(70.319)"]=np.array(zc["CTRL"]["area_pct_all"])/100
areas["ZERO(67.980)"]=L(D+"region_gate_eval.json","target","zero","area_pct")
areas["M2F(67.624)"]=np.array(zc["M2F_E02"]["area_pct_all"])/100
areas["GATE4096"]=np.array(json.load(open(D+"zip_compare_gate.json"))["GATE4096"]["area_pct_all"])/100
r=json.load(open(PC+"EVAL_VAL_RESULT.json"))["candidates"]
for k in r:
    if k not in ("bstar","zero_bias"): areas[k]=np.array(r[k]["target_area_pct"]["all3594"])/100
for f in ("EVAL_VAL_TEAMMATE.json","EVAL_VAL_BARREN_2P6.json"):
    r=json.load(open(PC+f))["candidates"]
    for k in r:
        if k!="bstar": areas[k]=np.array(r[k]["target_area_pct"]["all3594"])/100
# AGRI_ONLY: Bstar with Agri 2.1215->2.8215, native 256 grid, from H75 FP32 cache
BST=np.array([-0.2,0.1,0.0709,0.4,3.6,0.4,2.1215,-0.5],np.float32); AG=BST.copy(); AG[6]=2.8215
TC="/home/jhy/aic2026_ziponly_20261028/01_h75_cache/target3594_fp32/scores/"
names=[x["output_name"] for x in csv.DictReader(open("/home/jhy/aic2026_ziponly_20261028/manifests/semifinal_3594.csv"))]
def job(n):
    z=np.load(TC+n+".npy").reshape(8,-1)
    return np.bincount((z+AG[:,None]).argmax(0),minlength=8), np.bincount((z+BST[:,None]).argmax(0),minlength=8)
with Pool(48) as p: R=p.map(job,names,chunksize=30)
ag=sum(x[0] for x in R); bs=sum(x[1] for x in R)
areas["AGRI_ONLY(native)"]=ag/ag.sum(); areas["BSTAR(native)"]=bs/bs.sum()
print("\nname                     L1(pp)  sum|log(pred/est)|   per-class pred-est (pp)")
for k,v in areas.items():
    l1=100*np.abs(v-est).sum(); lr=np.abs(np.log(np.maximum(v,1e-6)/est)).sum()
    print(f"{k:24s} {l1:6.2f}   {lr:6.3f}   ",(100*(v-est)).round(2).tolist())
