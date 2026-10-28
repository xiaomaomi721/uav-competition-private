"""Diagnostic only: evaluate step2000 with its saved attention-mask schedule."""
import json,time
from train import ROOT,SPLIT,Source,DataLoader,guard,seed,UAVPMT,torch,F,np,metrics
from backbone import sha256
seed();guard()
p=ROOT/'formal/step_02000.pt'
assert sha256(p)=='8a36c910dbf608bd5d78bf03603a32d926b31fc2f377a68f8056361ef0f2730f'
out=ROOT/'formal/val_02000_saved_mask_20261027.json';assert not out.exists()
net=UAVPMT(ROOT/'dino.safetensors',ROOT/'pmt_decoder.bin').cuda()
cp=torch.load(p,map_location='cpu',weights_only=False)
missing=net.load_state_dict(cp['model'],strict=False)
assert not missing.unexpected_keys and all(k.startswith('encoder.') for k in missing.missing_keys)
del cp
net.eval();probs=net.decoder.attn_mask_probs.tolist()
assert all(v in [0.,1.] for v in probs),probs
print(json.dumps({'saved_attn_mask_probs':probs}),flush=True)
names=(SPLIT/'val.txt').read_text().split()
subset=set((ROOT/'no_detected_train_overlap.txt').read_text().split())
cms=np.zeros((8,8),np.int64);sub=cms.copy();nsub=0;start=time.time()
with torch.no_grad():
 for i,(x,y,name) in enumerate(DataLoader(Source(names),batch_size=1,num_workers=2,pin_memory=True)):
  if i%100==0:guard()
  with torch.autocast('cuda',dtype=torch.bfloat16):m,c=net(x.cuda(non_blocking=True))
  z=F.interpolate(m[-1].float(),size=(1024,1024),mode='bilinear',align_corners=False)
  score=torch.einsum('bqc,bqhw->bchw',c[-1].float().softmax(-1)[...,:8],z.sigmoid())
  pred=score.argmax(1).cpu().numpy()[0];gt=y.numpy()[0]-1;valid=(gt>=0)&(gt<8)
  cm=np.bincount(gt[valid]*8+pred[valid],minlength=64).reshape(8,8);cms+=cm
  if name[0] in subset:sub+=cm;nsub+=1
  if (i+1)%200==0:print(json.dumps({'eval_done':i+1,'elapsed_s':time.time()-start}),flush=True)
result={'protocol':'native1024 single-view; no Bstar; ignore0; saved checkpoint attention-mask schedule','saved_attn_mask_probs':probs,'n':len(names),'all':metrics(cms),'n_no_detected_train_overlap':nsub,'no_detected_train_overlap':metrics(sub),'elapsed_s':time.time()-start}
out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

