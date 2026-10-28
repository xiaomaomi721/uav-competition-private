#!/usr/bin/env python3
"""Round2 1A: CPU traditional image-space domain comparison.

Groups: source6996, source5597, source holdout1399, test1 500, test2 1300.
Critical rule: StandardScaler, PCA, and KMeans are fitted ONLY on source6996.
test1/test2 are transform/assign only. No label prediction.
"""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage, linalg

DATA=Path('/home/jhy/datasets/2026-低空图像语义分割赛道-训练集')
TEST2=Path('/home/jhy/datasets/AIC2026_UAV_semifinal/test_2/images')
ATDA=Path('/home/jhy/rare_diag_20260915/atda_20260918')
OUT=Path('/home/jhy/rare_diag_20260915/round2_20260921/image_space')
RS=192

def feature(path):
 im=np.asarray(Image.open(path).convert('RGB').resize((RS,RS))).astype(np.float32)/255
 r,g,b=im[...,0],im[...,1],im[...,2]; mx=im.max(-1); mn=im.min(-1); de=mx-mn
 s=np.where(mx>0,de/(mx+1e-8),0); v=mx
 h=np.zeros_like(r); h=np.where(mx==r,((g-b)/(de+1e-8))%6,h); h=np.where(mx==g,(b-r)/(de+1e-8)+2,h); h=np.where(mx==b,(r-g)/(de+1e-8)+4,h); h/=6
 gray=.299*r+.587*g+.114*b
 sob=np.hypot(ndimage.sobel(gray,0),ndimage.sobel(gray,1)); lap=ndimage.laplace(gray)
 hist=[]
 for c in (r,g,b): hist.extend(np.histogram(c,bins=16,range=(0,1))[0]/c.size)
 gh=np.histogram(gray,bins=64,range=(0,1))[0].astype(float); gh/=gh.sum(); ent=-(gh[gh>0]*np.log(gh[gh>0])).sum()
 ff=np.abs(np.fft.fftshift(np.fft.fft2(gray-gray.mean()))); y,x=np.mgrid[:RS,:RS]; rr=np.sqrt((y-RS/2)**2+(x-RS/2)**2); edges=np.linspace(0,RS/2,7); rings=np.array([ff[(rr>=edges[i])&(rr<edges[i+1])].mean() for i in range(6)]); rings/=rings.sum()+1e-9
 return np.r_[r.mean(),r.std(),g.mean(),g.std(),b.mean(),b.std(),h.mean(),h.std(),s.mean(),s.std(),v.mean(),v.std(),hist,sob.mean(),lap.var(),ent,rings].astype(np.float32)
def work(x): return x[0],feature(x[1])
def frechet(A,B):
 ma,mb=A.mean(0),B.mean(0); ca=np.cov(A,rowvar=False); cb=np.cov(B,rowvar=False); root=linalg.sqrtm(ca@cb); root=np.real(root); return float(np.sum((ma-mb)**2)+np.trace(ca+cb-2*root))
def centroid(A,B): return float(np.linalg.norm(A.mean(0)-B.mean(0)))
def cosine_centroid(A,B):
 a,b=A.mean(0),B.mean(0); return float(1-(a@b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 train=sorted((p.stem for p in (DATA/'train/images').glob('*.png')))
 ids5597=set(x.strip() for x in (ATDA/'source_ids_5597.txt').read_text().splitlines() if x.strip())
 hold=[x for x in train if x not in ids5597]
 t1=sorted((p.stem for p in (DATA/'images').glob('*.png')),key=lambda n:int(n.split('_')[-1])); t2=sorted((p.stem for p in TEST2.glob('*.png')),key=lambda n:int(n.split('_')[-1]))
 tasks=[]
 for x in train:tasks.append(('s_'+x,DATA/'train/images'/f'{x}.png'))
 for x in t1:tasks.append(('a_'+x,DATA/'images'/f'{x}.png'))
 for x in t2:tasks.append(('b_'+x,TEST2/f'{x}.png'))
 F={}
 with ProcessPoolExecutor(max_workers=48) as ex:
  for i,(k,v) in enumerate(ex.map(work,tasks,chunksize=12),1):
   F[k]=v
   if i%1000==0:print(f'{i}/{len(tasks)}',flush=True)
 Xs=np.stack([F['s_'+x] for x in train]); Xtr=np.stack([F['s_'+x] for x in sorted(ids5597)]); Xh=np.stack([F['s_'+x] for x in hold]); X1=np.stack([F['a_'+x] for x in t1]); X2=np.stack([F['b_'+x] for x in t2])
 from sklearn.preprocessing import StandardScaler
 from sklearn.decomposition import PCA
 from sklearn.cluster import KMeans
 from sklearn.neighbors import NearestNeighbors
 sc=StandardScaler().fit(Xs); Zs=sc.transform(Xs); Ztr=sc.transform(Xtr); Zh=sc.transform(Xh); Z1=sc.transform(X1); Z2=sc.transform(X2)
 pca=PCA(n_components=30,random_state=42).fit(Zs); P={k:pca.transform(v) for k,v in {'source6996':Zs,'source5597':Ztr,'holdout1399':Zh,'test1':Z1,'test2':Z2}.items()}
 km=KMeans(n_clusters=12,n_init=20,random_state=42).fit(P['source6996']); assign={k:km.predict(v) for k,v in P.items()}
 nn=NearestNeighbors(n_neighbors=5).fit(P['source6996']); knn={k:nn.kneighbors(v)[0] for k,v in P.items() if k!='source6996'}
 pairs={}
 for a,b in [('source6996','test1'),('source6996','test2'),('test1','test2'),('source5597','test2'),('holdout1399','test2')]:
  pairs[f'{a}__{b}']={'centroid_euclidean':centroid(P[a],P[b]),'centroid_cosine':cosine_centroid(P[a],P[b]),'frechet30':frechet(P[a],P[b])}
 occupancy={}
 for k,l in assign.items(): occupancy[k]=(np.bincount(l,minlength=12)/len(l)).round(6).tolist()
 source_occ=np.array(occupancy['source6996']); new_like=[]
 for c,x in enumerate(occupancy['test2']):
  if x>=.03 and source_occ[c]<.01:new_like.append({'cluster':c,'test2_share':x,'source_share':float(source_occ[c])})
 active=[i for i,x in enumerate(occupancy['test2']) if x>=.05]
 cent=km.cluster_centers_; sep=[]
 for i in active:
  for j in active:
   if i<j:sep.append(float(np.linalg.norm(cent[i]-cent[j])))
 report={'fit_policy':'StandardScaler/PCA/KMeans fit ONLY on source6996; targets transform/assign only','counts':{k:len(v) for k,v in P.items()},'pca_variance_30':float(pca.explained_variance_ratio_.sum()),'pair_distances':pairs,'knn_to_source':{k:{'mean_top1':float(v[:,0].mean()),'median_top1':float(np.median(v[:,0])),'p90_top1':float(np.quantile(v[:,0],.9)),'mean_top5':float(v.mean())} for k,v in knn.items()},'cluster_occupancy':occupancy,'test2_new_or_rare_source_clusters':new_like,'test2_internal_multidomain_evidence':{'clusters_ge_5pct':active,'n_clusters_ge_5pct':len(active),'mean_pairwise_source_cluster_centroid_separation':float(np.mean(sep)) if sep else 0.0,'occupancy_entropy':float(-sum(x*np.log(x) for x in occupancy['test2'] if x>0))},'feature_dim':int(Xs.shape[1])}
 (OUT/'IMAGE_SPACE_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
