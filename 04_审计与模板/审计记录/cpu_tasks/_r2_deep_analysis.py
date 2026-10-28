#!/usr/bin/env python3
"""Round2 source-only deep-feature distances + matched-size bootstrap.

Input: Exp18 source-only pooled stage2/stage3 features. For each stage, fit
StandardScaler/PCA/KMeans ONLY on source6996. Report centroid cosine,
Fréchet-style distance, unbiased MMD, kNN-to-source, cluster occupancy, and
100x test2-sub500 matched-size bootstrap plus source-internal 500/1300 baselines.
"""
from pathlib import Path
import json,numpy as np
from scipy import linalg
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
R=Path('/home/jhy/rare_diag_20260915/round2_20260921');D=R/'deep_features_exp18';OUT=R/'feature_space';OUT.mkdir(exist_ok=True);rng=np.random.default_rng(20260921)
def fr(A,B):
 ma,mb=A.mean(0),B.mean(0);ca=np.cov(A,rowvar=False);cb=np.cov(B,rowvar=False);q=np.real(linalg.sqrtm(ca@cb));return float(np.sum((ma-mb)**2)+np.trace(ca+cb-2*q))
def cc(A,B):a,b=A.mean(0),B.mean(0);return float(1-(a@b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def cd(A,B):return float(np.linalg.norm(A.mean(0)-B.mean(0)))
def mmd(A,B,g):
 aa=np.sum(A*A,1)[:,None];bb=np.sum(B*B,1)[:,None];K=np.exp(-g*np.maximum(aa+aa.T-2*A@A.T,0));L=np.exp(-g*np.maximum(bb+bb.T-2*B@B.T,0));M=np.exp(-g*np.maximum(aa+bb.T-2*A@B.T,0));n=len(A);m=len(B);return float((K.sum()-np.trace(K))/(n*(n-1))+(L.sum()-np.trace(L))/(m*(m-1))-2*M.mean())
def su(x):return {'mean':float(np.mean(x)),'std':float(np.std(x,ddof=1)),'ci95':[float(np.quantile(x,.025)),float(np.quantile(x,.975))]}
def one(stage):
 S=np.load(D/'source6996.npz')[stage];A=np.load(D/'test1.npz')[stage];B=np.load(D/'test2.npz')[stage];sc=StandardScaler().fit(S);Z=sc.transform(S);ZA=sc.transform(A);ZB=sc.transform(B);nc=min(64,Z.shape[1]);pca=PCA(n_components=nc,random_state=42).fit(Z);P=pca.transform(Z);PA=pca.transform(ZA);PB=pca.transform(ZB);km=KMeans(n_clusters=12,n_init=20,random_state=42).fit(P);occ=lambda X:(np.bincount(km.predict(X),minlength=12)/len(X)).round(6).tolist();nn=NearestNeighbors(n_neighbors=5).fit(P);da=nn.kneighbors(PA)[0];db=nn.kneighbors(PB)[0];ii=rng.integers(0,len(P),3000);jj=rng.integers(0,len(P),3000);med=np.median(np.sum((P[ii]-P[jj])**2,1));g=1/(2*med+1e-12);sref=P[rng.choice(len(P),500,False)];test1={'centroid':cd(P,PA),'cosine_centroid':cc(P,PA),'frechet':fr(P,PA),'mmd2_unbiased':mmd(sref,PA,g),'knn_top1_mean':float(da[:,0].mean()),'knn_top1_p90':float(np.quantile(da[:,0],.9))};bf=[];bc=[];bm=[];bk=[];si500=[];si1300=[]
 for _ in range(100):
  b=PB[rng.choice(len(PB),500,False)];x=P[rng.choice(len(P),500,False)];y=P[rng.choice(len(P),500,False)];x13=P[rng.choice(len(P),1300,False)];y13=P[rng.choice(len(P),1300,False)];bf.append(fr(P,b));bc.append(cd(P,b));bm.append(mmd(P[rng.choice(len(P),500,False)],b,g));bk.append(float(nn.kneighbors(b)[0][:,0].mean()));si500.append(fr(x,y));si1300.append(fr(x13,y13))
 return {'fit_policy':'scaler/PCA/KMeans source6996 only','dims':{'raw':S.shape[1],'pca':nc},'pca_variance':float(pca.explained_variance_ratio_.sum()),'test1_500':test1,'test2_full':{'centroid':cd(P,PB),'cosine_centroid':cc(P,PB),'frechet':fr(P,PB),'knn_top1_mean':float(db[:,0].mean()),'knn_top1_p90':float(np.quantile(db[:,0],.9))},'test1_test2':{'centroid':cd(PA,PB),'cosine_centroid':cc(PA,PB),'frechet':fr(PA,PB)},'test2_sub500_bootstrap':{'centroid':su(bc),'frechet':su(bf),'mmd2_unbiased':su(bm),'knn_top1_mean':su(bk)},'source_internal':{'500_vs_500_frechet':su(si500),'1300_vs_1300_frechet':su(si1300)},'cluster_occupancy':{'source':occ(P),'test1':occ(PA),'test2':occ(PB)},'test1_frechet_percentile_within_test2_sub500':float(np.mean(np.array(bf)<=test1['frechet']))}
def main():
 r={s:one(s) for s in ('stage2','stage3')};(OUT/'FEATURE_SPACE_REPORT.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':main()
