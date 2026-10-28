#!/usr/bin/env python3
"""CPU Fourier evidence and deterministic FDA transform contract; no training."""
from __future__ import annotations
import hashlib,json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from _r2_contracts import DATA,OUT,ROUND2,SEED,TEST2,sha256_file,write_json
RS=256;NR=64;RNG=np.random.default_rng(SEED)
def radial(path):
 x=np.asarray(Image.open(path).convert('RGB').resize((RS,RS),Image.Resampling.BILINEAR),np.float32)/255
 y,xg=np.mgrid[:RS,:RS];r=np.sqrt((y-RS/2)**2+(xg-RS/2)**2);edges=np.linspace(0,RS/2,NR+1);v=[]
 for c in range(3):
  a=np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(x[...,c]-x[...,c].mean()))));v.extend([a[(r>=edges[i])&(r<edges[i+1])].mean() for i in range(NR)])
 return np.asarray(v,np.float32)
def work(p):return radial(p)
def fda(source,target,beta):
 s=np.asarray(Image.open(source).convert('RGB'),np.float32)/255;t=np.asarray(Image.open(target).convert('RGB').resize((s.shape[1],s.shape[0]),Image.Resampling.BILINEAR),np.float32)/255
 out=np.empty_like(s);h,w=s.shape[:2];rad=max(1,int(round(beta*min(h,w))));cy,cx=h//2,w//2
 for c in range(3):
  fs=np.fft.fftshift(np.fft.fft2(s[...,c]));ft=np.fft.fftshift(np.fft.fft2(t[...,c]));amp=np.abs(fs);at=np.abs(ft);amp[cy-rad:cy+rad+1,cx-rad:cx+rad+1]=at[cy-rad:cy+rad+1,cx-rad:cx+rad+1];out[...,c]=np.fft.ifft2(np.fft.ifftshift(amp*np.exp(1j*np.angle(fs)))).real
 return np.clip(np.rint(np.clip(out,0,1)*255),0,255).astype(np.uint8)
def main():
 out=OUT/'06_fourier';out.mkdir(parents=True,exist_ok=False);src=sorted((DATA/'train/images').glob('*.png'));t1=sorted((DATA/'images').glob('*.png'),key=lambda p:int(p.stem.split('_')[-1]));t2=sorted(TEST2.glob('*.png'),key=lambda p:int(p.stem.split('_')[-1]));groups={'source':src,'test1':t1,'test2':t2};F={}
 with ProcessPoolExecutor(max_workers=32) as ex:
  for k,paths in groups.items():F[k]=np.stack(list(ex.map(work,paths,chunksize=16)));print(k,len(paths),flush=True)
 means={k:v.mean(0) for k,v in F.items()};dist={k:float(np.linalg.norm(means[k]-means['source'])) for k in ('test1','test2')};gap=np.abs(means['test2']-means['source']).reshape(3,NR).mean(0);cum=np.cumsum(gap);cum/=cum[-1];i1=int(np.searchsorted(cum,.25))+1;i2=int(np.searchsorted(cum,.50))+1;beta1=round(i1/(2*NR),4);beta2=round(i2/(2*NR),4)
 sample_src=src[0];sample_t=t2[int(hashlib.sha256((str(SEED)+sample_src.stem).encode()).hexdigest()[:8],16)%len(t2)];sha=[]
 for b in (beta1,beta2):
  a=fda(sample_src,sample_t,b);p=out/f'FDA_SAMPLE_beta{b}.png';Image.fromarray(a).save(p);q=fda(sample_src,sample_t,b);sha.append({'beta':b,'path':str(p),'sha256':sha256_file(p),'deterministic':bool(np.array_equal(a,q)),'source':sample_src.name,'target':sample_t.name})
 report={'scope':'FOURIER_STATS_AND_FDA_PREP_ONLY','counts':{k:len(v) for k,v in groups.items()},'profile':'per-channel log-amplitude radial 64 bins at 256','source_distance':dist,'radial_gap':gap.tolist(),'recommended_beta':{'conservative':beta1,'medium':beta2,'rule':'25%/50% cumulative source-test2 radial spectral gap'},'samples':sha,'compliance':'BLOCKED_PENDING_RULE_REVIEW','FDA_training_performed':False,'test2_ground_truth_used':False,'platform_submission_performed':False}
 np.savez_compressed(out/'FOURIER_PROFILES.npz',**F);write_json(out/'FOURIER_FDA_REPORT.json',report);write_json(out/'FDA_COMPLIANCE.json',{'status':'BLOCKED_PENDING_RULE_REVIEW','training_allowed':False});print(json.dumps(report,indent=2))
if __name__=='__main__':main()
