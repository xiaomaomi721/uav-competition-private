"""Soft class-conditioned OLA with explicit domains, detached bounded correction, and resumable statistics.
This is a project adaptation, not an exact reproduction of BLDA.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn.functional as F

class Reservoir:
    def __init__(self, capacity=4096):
        self.capacity=capacity
        self.values=None; self.keys=None; self.image_ids=None; self.seen=0
    def offer(self,values,keys,image_id,observed_count):
        values=np.asarray(values,np.float32)
        if len(values)==0: return
        ids=np.full(len(values),image_id,np.int64)
        if self.values is not None:
            values=np.concatenate([self.values,values])
            keys=np.concatenate([self.keys,keys])
            ids=np.concatenate([self.image_ids,ids])
        order=np.argsort(-keys,kind="stable")[:self.capacity]
        self.values=values[order].copy(); self.keys=keys[order].copy(); self.image_ids=ids[order].copy()
        self.seen+=int(observed_count)
    def ready(self,min_pixels=500,min_images=5):
        return self.values is not None and len(self.values)>=min_pixels and len(np.unique(self.image_ids))>=min_images
    def state(self): return copy.deepcopy(self.__dict__)
    def load(self,state): self.__dict__.update(copy.deepcopy(state))

def fit_gmm(values,previous=None,k=5,iterations=3,momentum=.99):
    x=np.asarray(values,np.float64)
    if x.ndim==1: x=x[:,None]
    if previous is None:
        means=np.quantile(x,np.linspace(.1,.9,k),axis=0).T
        var=np.maximum(x.var(axis=0),1e-4)[:,None].repeat(k,axis=1)
        weights=np.ones_like(means)/k
    else:
        weights,means,var=[v.astype(np.float64).copy() for v in previous]
    data=x.T[:,:,None]
    for _ in range(iterations):
        logp=np.log(np.maximum(weights[:,None,:],1e-12))-.5*np.log(2*np.pi*var[:,None,:])-.5*(data-means[:,None,:])**2/var[:,None,:]
        resp=np.exp(logp-logp.max(axis=-1,keepdims=True))
        resp/=np.maximum(resp.sum(-1,keepdims=True),1e-30)
        mass=resp.sum(axis=1).clip(1e-8)
        means=(resp*data).sum(axis=1)/mass
        var=((data-means[:,None,:])**2*resp).sum(axis=1)/mass
        var=var.clip(1e-4)
        weights=mass/mass.sum(axis=-1,keepdims=True)
    order=np.argsort(means,axis=-1)
    new=tuple(np.take_along_axis(a,order,axis=-1) for a in (weights,means,var))
    if previous is not None:
        new=tuple(momentum*old+(1-momentum)*n for old,n in zip(previous,new))
    new=(new[0]/new[0].sum(-1,keepdims=True),new[1],new[2].clip(1e-4))
    assert all(np.isfinite(v).all() for v in new)
    return tuple(v.astype(np.float32) for v in new)

def cdf_numpy(x,params):
    weights,means,var=[torch.as_tensor(v,dtype=torch.float64) for v in params]
    t=torch.as_tensor(x,dtype=torch.float64)
    result=(weights[:,None,:]*.5*(1+torch.erf((t[:,:,None]-means[:,None,:])/torch.sqrt(2*var[:,None,:])))).sum(-1)
    return result.numpy()

def quantiles(params, q):
    w,m,v=params
    lo=(m-10*np.sqrt(v)).min(-1)[:,None].repeat(len(q),axis=1).astype(np.float64)
    hi=(m+10*np.sqrt(v)).max(-1)[:,None].repeat(len(q),axis=1).astype(np.float64)
    for _ in range(48):
        mid=(lo+hi)/2; left=cdf_numpy(mid,params)<q[None,:]
        lo=np.where(left,mid,lo); hi=np.where(left,hi,mid)
    return (lo+hi)/2

class OnlineStats:
    def __init__(self, device="cuda:0", seed=20261027, bins=1025):
        self.device=torch.device(device); self.bins=bins
        self.rows={d:[Reservoir() for _ in range(8)] for d in ("source","target")}
        self.anchors=[Reservoir(),Reservoir()]
        self.params={d:[None]*8 for d in ("source","target")}
        self.anchor_params=[None,None]
        self.tables={}
        self.generator=torch.Generator(device=self.device).manual_seed(seed+40000)
        self.updates=0
    @torch.no_grad()
    def collect(self,domain,raw,label,mask,image_id):
        assert domain in self.rows
        z=raw.detach().float().permute(0,2,3,1).reshape(-1,8)
        y=label.reshape(-1)
        valid=mask.reshape(-1)&(y>=0)&(y<8)
        assert torch.isfinite(z).all()
        indices=torch.nonzero(valid,as_tuple=False).flatten()
        if not len(indices): return
        keys=torch.rand(len(y),device=z.device,generator=self.generator,dtype=torch.float64)
        for c in range(8):
            ids=torch.nonzero(valid&(y==c),as_tuple=False).flatten()
            if not len(ids): continue
            picked=torch.topk(keys[ids],min(4096,len(ids)),sorted=True).indices
            picked=ids[picked]
            self.rows[domain][c].offer(z[picked].cpu().numpy(),keys[picked].cpu().numpy(),image_id,len(ids))
        if domain=="source":
            picked=indices[torch.topk(keys[indices],min(4096,len(indices)),sorted=True).indices]
            yy=y[picked]
            pos=z[picked,yy][:,None]
            self.anchors[0].offer(pos.cpu().numpy(),keys[picked].cpu().numpy(),image_id,len(indices))
            # Unbiased marginal negative-score estimator: one random non-GT channel per sampled source pixel.
            keys2=torch.rand(len(indices),device=z.device,generator=self.generator,dtype=torch.float64)
            sel=torch.topk(keys2,min(4096,len(indices)),sorted=True).indices
            p2=indices[sel]; yy=y[p2]
            other=torch.randint(0,7,(len(p2),),device=z.device,generator=self.generator)
            other=other+(other>=yy).long()
            neg=z[p2,other][:,None]
            self.anchors[1].offer(neg.cpu().numpy(),keys2[sel].cpu().numpy(),image_id,len(indices))
    def fit(self):
        for i,r in enumerate(self.anchors):
            if r.ready(): self.anchor_params[i]=fit_gmm(r.values,self.anchor_params[i])
        for d in self.rows:
            for c,r in enumerate(self.rows[d]):
                if r.ready(): self.params[d][c]=fit_gmm(r.values,self.params[d][c])
        self.updates+=1
        self.build_tables()
    def build_tables(self):
        self.tables={}
        if any(p is None for p in self.anchor_params): return
        q=np.linspace(.01,.99,2049)
        aq=[quantiles(p,q)[0] for p in self.anchor_params]
        for d in self.params:
            mins=np.zeros((8,8),np.float32); maxs=np.ones((8,8),np.float32)
            lut=np.zeros((8,8,self.bins),np.float32); valid=np.zeros(8,bool)
            for c,p in enumerate(self.params[d]):
                if p is None: continue
                w,m,v=p
                low=(m-6*np.sqrt(v)).min(-1); high=(m+6*np.sqrt(v)).max(-1)
                high=np.maximum(high,low+1e-3)
                grid=low[:,None]+(high-low)[:,None]*np.linspace(0,1,self.bins)[None,:]
                f=cdf_numpy(grid,p).clip(.01,.99)
                for l in range(8): lut[c,l]=np.interp(f[l],q,aq[0 if c==l else 1])
                mins[c]=low; maxs[c]=high; valid[c]=True
            self.tables[d]={"lo":torch.from_numpy(mins).to(self.device),"hi":torch.from_numpy(maxs).to(self.device),
                "lut":torch.from_numpy(lut).to(self.device),"valid":torch.from_numpy(valid).to(self.device)}
    @torch.no_grad()
    def correction(self,domain,raw_flat,classes,alpha):
        if alpha==0 or domain not in self.tables: return torch.zeros_like(raw_flat)
        tab=self.tables[domain]; n=len(raw_flat); out=torch.empty_like(raw_flat)
        channels=torch.arange(8,device=raw_flat.device)[None,:]
        for start in range(0,n,16384):
            z=raw_flat[start:start+16384].detach()
            cc=classes[start:start+16384]
            low=tab["lo"][cc]; high=tab["hi"][cc]
            u=((z-low)/(high-low)*(self.bins-1)).clamp(0,self.bins-1)
            ix=u.long().clamp(max=self.bins-2)
            frac=u-ix
            a=tab["lut"][cc[:,None],channels,ix]
            b=tab["lut"][cc[:,None],channels,ix+1]
            mapped=a+(b-a)*frac
            val=((mapped-z)*alpha).clamp(-.5,.5)
            out[start:start+len(z)]=val*tab["valid"][cc,None]
        return out
    def summary(self):
        return {"fit_updates":self.updates,"ready_rows":{d:[p is not None for p in self.params[d]] for d in self.rows},
          "reservoir":{d:[{"pixels":0 if r.values is None else len(r.values),"images":0 if r.image_ids is None else len(np.unique(r.image_ids)),
                           "seen_pixels":r.seen} for r in rr] for d,rr in self.rows.items()}}
    def state_dict(self):
        return {"rows":{d:[r.state() for r in rr] for d,rr in self.rows.items()},
          "anchors":[r.state() for r in self.anchors],"params":self.params,"anchor_params":self.anchor_params,
          "generator":self.generator.get_state(),"updates":self.updates,"bins":self.bins}
    def load_state_dict(self,state):
        assert self.bins==state["bins"]
        for d in self.rows:
            for r,s in zip(self.rows[d],state["rows"][d]): r.load(s)
        for r,s in zip(self.anchors,state["anchors"]): r.load(s)
        self.params=copy.deepcopy(state["params"]);self.anchor_params=copy.deepcopy(state["anchor_params"])
        self.generator.set_state(state["generator"].cpu());self.updates=state["updates"]
        self.build_tables()

def source_loss(raw, y, stats, alpha):
    z=raw.permute(0,2,3,1).reshape(-1,8); yy=y.reshape(-1)
    valid=(yy>=0)&(yy<8)
    if not valid.any(): return raw.sum()*0, {"mean_abs":0.0,"clip_fraction":0.0}
    z=z[valid]; yy=yy[valid]
    a=stats.correction("source",z.detach(),yy,alpha)
    loss=F.cross_entropy(z-a,yy)
    return loss,{"mean_abs":float(a.abs().mean()),"clip_fraction":float((a.abs()>=.499999).float().mean())}

def target_loss(raw,q,mask,bstar,stats,alpha):
    z=raw.permute(0,2,3,1).reshape(-1,8)
    prob=q.permute(0,2,3,1).reshape(-1,8).detach()
    valid=mask.reshape(-1)
    if not valid.any(): return raw.sum()*0,{"ordinary_kl":0.0,"mean_abs":0.0,"clip_fraction":0.0}
    z=z[valid];prob=prob[valid]; x=z+bstar.reshape(1,8)
    with torch.no_grad():
        kl=(prob*(prob.clamp_min(1e-30).log()-x.detach().log_softmax(-1))).sum(-1).mean()
    if alpha==0:
        return -(prob*x.log_softmax(-1)).sum(-1).mean(),{"ordinary_kl":float(kl),"mean_abs":0.0,"clip_fraction":0.0}
    terms=[]; abs_sum=0.; clip_sum=0.
    for c in range(8):
        labels=torch.full((len(z),),c,device=z.device,dtype=torch.long)
        a=stats.correction("target",z.detach(),labels,alpha)
        terms.append(-prob[:,c]*(x-a).log_softmax(-1)[:,c])
        abs_sum+=float((prob[:,c]*a.abs().mean(-1)).mean())
        clip_sum+=float((prob[:,c]*(a.abs()>=.499999).float().mean(-1)).mean())
    return torch.stack(terms).sum(0).mean(),{"ordinary_kl":float(kl),"mean_abs":abs_sum,"clip_fraction":clip_sum}

