#!/usr/bin/env python3
"""Matched-size bootstrap for Round2 image-space features.

Recomputes the same 69-D traditional features, fits scaler/PCA only on source
6996, saves feature matrices, then:
- test2 random 500, 100 repeats: source->test2_sub500 distance CI vs test1(500)
- source internal random 500 and 1300 subset baselines, 100 repeats each
- unbiased MMD^2 estimates on 500-sized samples (RBF, source-derived bandwidth)
"""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage,linalg
DATA=Path('/home/jhy/datasets/2026-低空图像语义分割赛道-训练集');T2=Path('/home/jhy/datasets/AIC2026_UAV_semifinal/test_2/images');OUT=Path('/home/jhy/rare_diag_20260915/round2_20260921/image_space_bootstrap');RS=192;NREP=100;RNG=np.random.default_rng(20260921)
def ft(p):
 im=np.asarray(Image.open(p).convert('RGB').resize((RS,RS))).astype(np.float32)/255;r,g,b=im[...,0],im[...,1],im[...,2];mx=im.max(-1);mn=im.min(-1);de=mx-mn;s=np.where(mx>0,de/(mx+1e-8),0);v=mx;h=np.zeros_like(r);h=np.where(mx==r,((g-b)/(de+1e-8))%6,h);h=np.where(mx==g,(b-r)/(de+1e-8)+2,h);h=np.where(mx==b,(r-g)/(de+1e-8)+4,h);h/=6;gray=.299*r+.587*g+.114*b;sob=np.hypot(ndimage.sobel(gray,0),ndimage.sobel(gray,1));lap=ndimage.laplace(gray);hist=np.r_[[np.histogram(c,bins=16,range=(0,1))[0]/c.size for c in (r,g,b)]].ravel();gh=np.histogram(gray,bins=64,range=(0,1))[0].astype(float);gh/=gh.sum();ent=-(gh[gh>0]*np.log(gh[gh>0])).sum();ff=np.abs(np.fft.fftshift(np.fft.fft2(gray-gray.mean())));y,x=np.mgrid[:RS,:RS];rr=np.sqrt((y-RS/2)**2+(x-RS/2)**2);ed=np.linspace(0,RS/2,7);rings=np.array([ff[(rr>=ed[i])&(rr<ed[i+1])].mean() for i in range(6)]);rings/=rings.sum()+1e-9;return np.r_[r.mean(),r.std(),g.mean(),g.std(),b.mean(),b.std(),h.mean(),h.std(),s.mean(),s.std(),v.mean(),v.std(),hist,sob.mean(),lap.var(),ent,rings].astype(np.float32)
def work(x):return x[0],ft(x[1])
def fr(A,B):
 ma,mb=A.mean(0),B.mean(0);ca=np.cov(A,rowvar=False);cb=np.cov(B,rowvar=False);q=np.real(linalg.sqrtm(ca@cb));return float(np.sum((ma-mb)**2)+np.trace(ca+cb-2*q))
def cd(A,B):return float(np.linalg.norm(A.mean(0)-B.mean(0)))
def mmd2u(A,B,gamma):
 # unbiased; sizes 500, vectorized matrices acceptable
 aa=np.sum(A*A,1)[:,None];bb=np.sum(B*B,1)[:,None];K=np.exp(-gamma*np.maximum(aa+aa.T-2*A@A.T,0));L=np.exp(-gamma*np.maximum(bb+bb.T-2*B@B.T,0));M=np.exp(-gamma*np.maximum(aa+bb.T-2*A@B.T,0));n=len(A);m=len(B);return float((K.sum()-np.trace(K))/(n*(n-1))+(L.sum()-np.trace(L))/(m*(m-1))-2*M.mean())
def summ(v):return {'mean':float(np.mean(v)),'std':float(np.std(v,ddof=1)),'ci95':[float(np.quantile(v,.025)),float(np.quantile(v,.975))]}
def main():
 OUT.mkdir(parents=True,exist_ok=True);src=sorted((DATA/'train/images').glob('*.png'));t1=sorted((DATA/'images').glob('*.png'),key=lambda p:int(p.stem.split('_')[-1]));t2=sorted(T2.glob('*.png'),key=lambda p:int(p.stem.split('_')[-1]));tasks=[('s'+p.stem,p) for p in src]+[('a'+p.stem,p) for p in t1]+[('b'+p.stem,p) for p in t2];F={}
 with ProcessPoolExecutor(max_workers=48) as ex:
  for i,(k,v) in enumerate(ex.map(work,tasks,chunksize=12),1):F[k]=v;print(f'{i}/{len(tasks)}',flush=True) if i%1000==0 else None
 Xs=np.stack([F['s'+p.stem] for p in src]);X1=np.stack([F['a'+p.stem] for p in t1]);X2=np.stack([F['b'+p.stem] for p in t2]);from sklearn.preprocessing import StandardScaler;from sklearn.decomposition import PCA;sc=StandardScaler().fit(Xs);pca=PCA(n_components=30,random_state=42).fit(sc.transform(Xs));S=pca.transform(sc.transform(Xs));A=pca.transform(sc.transform(X1));B=pca.transform(sc.transform(X2));np.savez_compressed(OUT/'features_pca30.npz',source=S,test1=A,test2=B)
 # source-derived RBF gamma via 2000 random pair squared distance median
 ii=RNG.integers(0,len(S),2000);jj=RNG.integers(0,len(S),2000);med=np.median(np.sum((S[ii]-S[jj])**2,1));gamma=1/(2*med+1e-12)
 test1={'frechet':fr(S,A),'centroid':cd(S,A),'mmd2_unbiased':mmd2u(S[RNG.choice(len(S),500,False)],A,gamma)}
 bfr=[];bcd=[];bm=[];s500fr=[];s1300fr=[];s500cd=[];s1300cd=[];sm=[]
 for _ in range(NREP):
  b=B[RNG.choice(len(B),500,False)];x500=S[RNG.choice(len(S),500,False)];y500=S[RNG.choice(len(S),500,False)];x1300=S[RNG.choice(len(S),1300,False)];y1300=S[RNG.choice(len(S),1300,False)];bfr.append(fr(S,b));bcd.append(cd(S,b));bm.append(mmd2u(S[RNG.choice(len(S),500,False)],b,gamma));s500fr.append(fr(x500,y500));s500cd.append(cd(x500,y500));sm.append(mmd2u(x500,y500,gamma));s1300fr.append(fr(x1300,y1300));s1300cd.append(cd(x1300,y1300))
 rep={'fit_policy':'source6996 only','n_repeats':NREP,'gamma_source_median_heuristic':float(gamma),'test1_500':test1,'source_to_test2_sub500':{'frechet':summ(bfr),'centroid':summ(bcd),'mmd2_unbiased':summ(bm)},'source_internal_500_vs_500':{'frechet':summ(s500fr),'centroid':summ(s500cd),'mmd2_unbiased':summ(sm)},'source_internal_1300_vs_1300':{'frechet':summ(s1300fr),'centroid':summ(s1300cd)},'test1_vs_test2_sub500_comparison':{'test1_frechet_percentile_among_test2_sub500':float(np.mean(np.array(bfr)<=test1['frechet'])),'test1_centroid_percentile_among_test2_sub500':float(np.mean(np.array(bcd)<=test1['centroid']))}}
 (OUT/'IMAGE_MATCHED_BOOTSTRAP.json').write_text(json.dumps(rep,indent=2));print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
