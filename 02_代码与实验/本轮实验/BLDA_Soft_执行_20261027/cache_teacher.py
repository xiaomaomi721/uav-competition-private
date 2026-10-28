"""H75 weak-view raw native logits, with source-image SHA and resume verification."""
from common import *
import argparse
from torch.utils.data import DataLoader, Dataset

class Images(Dataset):
    def __init__(self, rows): self.rows=rows
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]; path=Path(ASSETS["target_images"])/row["input_name"]
        assert sha256(path)==row["file_sha256"], "Target image hash mismatch: "+str(path)
        return normalize_to_tensor(read_rgb(path)), row["input_name"]

def main():
    guard(); seed()
    rows=target_rows(); dest=ROOT/"teacher_raw"; dest.mkdir(exist_ok=True)
    identity={"model_sha256":ASSETS["model_sha256"],"target_manifest_sha256":sha256(ASSETS["target_manifest"]),
              "shape":[8,256,256],"dtype":"float16","bstar_absent":True,"teacher_view":"single_weak","forward_precision":"float32"}
    ip=dest/"identity.json"
    if ip.exists(): assert json.loads(ip.read_text())==identity
    else: dump(ip,identity)
    records={}
    old=dest/"files.json"
    if old.exists(): records=json.loads(old.read_text())
    pending=[]
    for row in rows:
        out=dest/(Path(row["input_name"]).stem+".npy")
        if out.exists() and row["input_name"] in records:
            assert sha256(out)==records[row["input_name"]]["sha256"]
            z=np.load(out,allow_pickle=False)
            assert z.shape==(8,256,256) and np.isfinite(z).all()
        else: pending.append(row)
    model=model_from(); start=time.time()
    loader=DataLoader(Images(pending),batch_size=1,num_workers=4,pin_memory=True,shuffle=False)
    with torch.inference_mode():
        for i,(x,names) in enumerate(loader,1):
            if i%50==1: guard()
            raw=predict_raw(model,x.cuda(non_blocking=True),tta=False)[0]
            assert tuple(raw.shape)==(8,256,256) and torch.isfinite(raw).all()
            name=names[0]; out=dest/(Path(name).stem+".npy")
            np.save(out,raw.cpu().numpy().astype(np.float16),allow_pickle=False)
            records[name]={"sha256":sha256(out),"bytes":out.stat().st_size}
            if i%100==0 or i==len(pending):
                dump(old,records); emit(stage="teacher_cache",done=len(records),total=3594,elapsed=round(time.time()-start,1))
    assert len(records)==3594
    dump(dest/"COMPLETE.json",{**identity,"count":len(records),"runtime_sec":time.time()-start,
         "files_manifest_sha256":sha256(old),"gpu_uuid":GPU_UUID})
if __name__=="__main__": main()

