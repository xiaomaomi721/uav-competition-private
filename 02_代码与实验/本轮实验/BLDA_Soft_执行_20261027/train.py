"""Matched C0/C1 H75 fine-tuning. All ordering and augmentations are step-deterministic."""
from common import *
from ola import OnlineStats,source_loss,target_loss
import argparse, copy
from torch.utils.data import Dataset,DataLoader
from src.transforms import TrainTransform
from src.rare_crop import RareClassCropper
from src.label_mapping import official_to_train

class Pairs(Dataset):
    def __init__(self,limit):
        self.sources=Path(ASSETS["source_train"]).read_text().split()
        self.rows=target_rows()
        sr=np.random.default_rng(SEED+10001).permutation(len(self.sources))
        tr=np.random.default_rng(SEED+10002).permutation(len(self.rows))
        self.order=[(self.sources[int(si)],self.rows[int(ti)]) for si,ti in zip(sr[:limit],tr[:limit])]
        assert len(self.order)==limit
        self.transform=TrainTransform(crop_size=1024,config=get_cfg()["transforms"],
          rare_cropper=RareClassCropper(1024,.5,(7,4,6),.25),ignore_index=255)
    def __len__(self): return len(self.order)
    def __getitem__(self,i):
        stem,row=self.order[i]; d=Path(ASSETS["source_data"])/"train"
        rgb=read_rgb(d/"images"/(stem+".png"))
        with Image.open(d/"masks"/(stem+".png")) as f: label=official_to_train(np.asarray(f).copy())
        sx,sy=self.transform(rgb,label,rng=random.Random(SEED+100000+i))
        tx=normalize_to_tensor(read_rgb(Path(ASSETS["target_images"])/row["input_name"]))
        raw=np.load(ROOT/"teacher_raw"/(Path(row["input_name"]).stem+".npy"),allow_pickle=False).astype(np.float32)
        assert raw.shape==(8,256,256) and np.isfinite(raw).all()
        return {"sx":sx,"sy":sy,"tx":tx,"teacher":torch.from_numpy(raw),"source":stem,"target":row["input_name"],
                "subset":row["subset"],"pair_index":i}

def model_digest(model):
    h=hashlib.sha256()
    for n,p in model.state_dict().items():
        h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def rng_state():
    return {"python":random.getstate(),"numpy":np.random.get_state(),
            "torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all()}

def set_rng(d):
    random.setstate(d["python"]);np.random.set_state(d["numpy"])
    torch.set_rng_state(d["torch"].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in d["cuda"]])

def save_training(out,name,model,opt,scaler,stats,step,arm,order_sha):
    path=out/name
    obj={"model":model.state_dict(),"optimizer":opt.state_dict(),"scaler":scaler.state_dict(),
         "statistics":stats.state_dict(),"rng":rng_state(),"step":step,"arm":arm,
         "h75_sha256":ASSETS["model_sha256"],"pair_order_sha256":order_sha,"seed":SEED}
    temp=path.with_suffix(".tmp");torch.save(obj,temp);temp.replace(path)
    return str(path)

def train():
    ap=argparse.ArgumentParser()
    ap.add_argument("--arm",choices=["C0","C1"],required=True)
    ap.add_argument("--steps",type=int,default=3594)
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--resume",default=None)
    args=ap.parse_args()
    guard();seed()
    if not (ROOT/"teacher_raw/COMPLETE.json").exists(): raise RuntimeError("Teacher cache incomplete")
    out=ROOT/("smoke" if args.smoke else args.arm)
    if not args.resume: out.mkdir(exist_ok=False)
    else: assert out.exists()
    ds=Pairs(args.steps)
    order=[{"step":i+1,"source":s,"target":r["input_name"],"subset":r["subset"]} for i,(s,r) in enumerate(ds.order)]
    order_sha=hashlib.sha256(json.dumps(order,sort_keys=True).encode()).hexdigest()
    dump(out/"pair_order.json",order)
    model=model_from()
    model.eval()
    model.get_submodule("segformer.stages.2").train()
    model.get_submodule("segformer.stages.3").train()
    for n,p in model.named_parameters(): p.requires_grad_(n.startswith(PREFIX))
    base=[p for n,p in model.named_parameters() if n.startswith(PREFIX[:2])]
    head=[p for n,p in model.named_parameters() if n.startswith("decode_head.")]
    opt=torch.optim.AdamW([{"params":base,"lr":1e-5},{"params":head,"lr":5e-7}],weight_decay=.01)
    scaler=torch.amp.GradScaler("cuda",enabled=True)
    stats=OnlineStats("cuda:0",SEED)
    start_step=0;attempts=0;skips=[]
    if args.resume:
        cp=torch.load(args.resume,map_location="cpu",weights_only=False)
        assert cp["arm"]==args.arm and cp["pair_order_sha256"]==order_sha and cp["h75_sha256"]==ASSETS["model_sha256"]
        model.load_state_dict(cp["model"],strict=True);opt.load_state_dict(cp["optimizer"])
        scaler.load_state_dict(cp["scaler"]);stats.load_state_dict(cp["statistics"])
        start_step=cp["step"];set_rng(cp["rng"]);del cp
    loader=DataLoader(ds,batch_size=1,sampler=list(range(start_step,len(ds))),num_workers=4,pin_memory=True,
          generator=torch.Generator().manual_seed(SEED+999),persistent_workers=True)
    b=torch.tensor(BSTAR,device="cuda")[None,:,None,None]
    metadata={"arm":args.arm,"smoke":args.smoke,"steps":args.steps,"gpu_uuid":GPU_UUID,
       "h75_sha256":ASSETS["model_sha256"],"teacher_identity":json.loads((ROOT/"teacher_raw/identity.json").read_text()),
       "pair_order_sha256":order_sha,"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
       "frozen_head_batchnorm":True,"torch":torch.__version__,"started_unix":time.time(),
       "implementation_notes":["source_transform_rng_fixed_per_pair",
          "target_strong_photo_matches_historical_ranges_with_independent_step_rng",
          "uniform_pixel_priority_reservoir; source negative anchor samples one uniformly chosen wrong channel per selected pixel",
          "source and target backward performed sequentially for memory; same scaled total objective",
          "retry AMP overflow on same pair, same model RNG, no statistics update until successful optimizer step"]}
    dump(out/"RUN_CONFIG.json",metadata)
    group_counts={g:np.zeros(8,dtype=np.int64) for g in ("test1","test2","test3_new")}
    source_counts=np.zeros(8,dtype=np.int64)
    t0=time.time();last_step=start_step
    hist=open(out/"history.jsonl","a" if args.resume else "w",buffering=1)
    try:
        for batch in loader:
            step=int(batch["pair_index"][0])+1
            if step%20==1:guard()
            alpha=0.0 if args.arm=="C0" else .1*max(0,min(1,(step-500)/500))
            if args.smoke:alpha=0.1 if step>20 else 0.
            sx=batch["sx"].cuda(non_blocking=True);sy=batch["sy"].cuda(non_blocking=True)
            tx=batch["tx"].cuda(non_blocking=True)
            q=(batch["teacher"].cuda(non_blocking=True)+b).softmax(1)
            conf,yl=q.max(1);blind=(yl==3)|(yl==4)|(yl==6)
            mask=(conf>=.98)|(blind&(conf>=.50))
            before_rng=rng_state()
            for retry in range(8):
                attempts+=1;set_rng(before_rng)
                opt.zero_grad(set_to_none=True)
                for gi,g in enumerate(opt.param_groups):g["lr"]=(1e-5 if gi==0 else 5e-7)*(1-(step-1)/10000)
                with torch.amp.autocast("cuda"):
                    sz=model(pixel_values=sx).logits
                sz=F.interpolate(sz.float(),size=sy.shape[-2:],mode="bilinear",align_corners=False)
                ls,ss=source_loss(sz,sy,stats,alpha)
                scaler.scale(.1*ls).backward()
                sraw=sz.detach()
                rng=random.Random(SEED+200000+step)
                view=rng.randrange(8)
                generator=torch.Generator(device="cuda").manual_seed(SEED+300000+step)
                aug=strong_photo(d4(tx,view),rng,generator)
                with torch.amp.autocast("cuda"):
                    tz=model(pixel_values=aug).logits
                tz=inv_d4(tz.float(),view)
                lt,ts=target_loss(tz,q,mask,b,stats,alpha)
                scaler.scale(min(1.,step/500)*lt).backward()
                traw=tz.detach()
                scaler.unscale_(opt)
                scale_before=scaler.get_scale()
                scaler.step(opt);scaler.update()
                if scaler.get_scale()>=scale_before: break
                skips.append({"step":step,"retry":retry,"scale":scale_before})
                emit(stage="AMP_RETRY",**skips[-1])
            else: raise RuntimeError("Repeated non-finite gradients")
            if not torch.isfinite(ls+lt): raise RuntimeError("Nonfinite objective")
            stats.collect("source",sraw,sy,sy!=255,step)
            stats.collect("target",traw,yl,mask,step)
            if step%20==0:stats.fit()
            cls=torch.bincount(yl[mask],minlength=8).cpu().numpy()
            src_cls=torch.bincount(sy[sy!=255],minlength=8).cpu().numpy()
            group=batch["subset"][0];group_counts[group]+=cls;source_counts+=src_cls
            last_step=step
            row={"step":step,"alpha":alpha,"source":batch["source"][0],"target":batch["target"][0],"subset":group,
              "source_loss":float(ls),"target_soft_ce":float(lt),"target_ordinary_kl":ts["ordinary_kl"],
              "source_correction":ss,"target_correction":{k:v for k,v in ts.items() if k!="ordinary_kl"},
              "mask_fraction":float(mask.float().mean()),"class_valid_pixels":cls.tolist(),
              "elapsed_sec":round(time.time()-t0,3),"scaler":scaler.get_scale(),"view":view,
              "gpu_peak_allocated_mib":torch.cuda.max_memory_allocated()/1024**2}
            if step in (1,20,50,500):
                row["model_state_digest"]=model_digest(model)
                row["pair_tensor_digest"]=hashlib.sha256(sx.cpu().numpy().tobytes()+sy.cpu().numpy().tobytes()+aug.cpu().numpy().tobytes()).hexdigest()
            hist.write(json.dumps(row)+"\n")
            if step%50==0 or step in (1,20,21) or step==args.steps:
                emit(stage="train",arm=args.arm,smoke=args.smoke,**row)
                dump(out/"progress.json",{**row,"statistics":stats.summary()})
            if not args.smoke and (step in (1000,2000) or step==args.steps):
                save_training(out,f"checkpoint_step{step}.pt",model,opt,scaler,stats,step,args.arm,order_sha)
                dump(out/f"statistics_step{step}.json",stats.summary())
        if args.smoke:
            path=save_training(out,"smoke_checkpoint.pt",model,opt,scaler,stats,last_step,args.arm,order_sha)
        else:
            path=str(out/f"checkpoint_step{last_step}.pt")
            torch.save({"model":model.state_dict(),"step":last_step,"arm":args.arm,
              "h75_sha256":ASSETS["model_sha256"],"teacher":"fixed_H75","bstar":BSTAR},out/"inference.pt")
        summary={**metadata,"status":"COMPLETED","effective_updates":last_step,
           "attempts_this_invocation":attempts,"amp_retries":skips,"runtime_sec":time.time()-t0,
           "peak_allocated_mib":torch.cuda.max_memory_allocated()/1024**2,"checkpoint":path,
           "target_effective_pixels_by_group":{g:v.tolist() for g,v in group_counts.items()},
           "source_effective_pixels":source_counts.tolist(),"statistics":stats.summary()}
        dump(out/"SUMMARY.json",summary);emit(stage="train_completed",arm=args.arm,smoke=args.smoke,runtime_sec=summary["runtime_sec"])
    except BaseException as ex:
        # Preserve resumable state only after a fully completed step, without claiming success.
        # In-loop interruption may occur after uncommitted forwards/statistics; last regular checkpoint remains authoritative.
        dump(out/"FAILED.json",{"error":repr(ex),"last_completed_step":last_step,"unix_time":time.time()})
        raise
    finally:hist.close()

if __name__=="__main__":train()

