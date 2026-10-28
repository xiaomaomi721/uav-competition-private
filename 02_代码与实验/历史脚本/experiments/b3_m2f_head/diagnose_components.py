"""Six loss-component backwards on the SAME graph; diagnosis only."""
import argparse,hashlib,json,os,sys,traceback
from pathlib import Path

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['fp32','autocast'],required=True); ap.add_argument('--outdir',required=True); a=ap.parse_args()
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='2' or os.environ.get('CUDA_DEVICE_ORDER')!='PCI_BUS_ID': raise RuntimeError('GPU2 only')
 root=Path(a.outdir); root.mkdir(parents=True,exist_ok=False)
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
 import torch,numpy as np
 from b3_m2f_head import loss as L
 from b3_m2f_head.head import B3M2FStyleHead
 from b3_m2f_head.train_entry import build_encoder,build_batch,to_device_set_targets,build_weight_dict,DATA_ROOT
 torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42); torch.backends.cudnn.benchmark=False
 result={'mode':a.mode,'rows':[],'bce_calls':[],'no_optimizer':True,'seed':42}
 try:
  encoder=build_encoder('cuda:0').train(); head=B3M2FStyleHead().cuda().train()
  params=list(encoder.parameters())+list(head.parameters())
  h=hashlib.sha256()
  for prefix,model in [('encoder.',encoder),('head.',head)]:
   for n,p in model.named_parameters(): h.update((prefix+n).encode()); h.update(p.detach().cpu().numpy().tobytes())
  result['init_sha256']=h.hexdigest()
  criterion=L.SetCriterion(weight_dict=build_weight_dict(6,1.)).cuda()
  pixel,label=build_batch(DATA_ROOT,['0001'],1024,'cuda:0',42)
  result['image_sha256']=hashlib.sha256(pixel.cpu().numpy().tobytes()).hexdigest()
  result['label_sha256']=hashlib.sha256(label.numpy().tobytes()).hexdigest()
  torch.manual_seed(142); torch.cuda.manual_seed_all(142)
  with torch.autocast('cuda',dtype=torch.float16,enabled=a.mode=='autocast'):
   out=head(encoder(pixel_values=pixel,output_hidden_states=True).hidden_states)
  targets,pts,w,vm=L.build_set_targets(label,8)
  result['P']=pts.shape[1]; result['num_masks']=criterion._num_masks(targets)
  targets,pts,w,vm=to_device_set_targets(targets,pts,w,vm,'cuda:0')
  # Wrap for observation only; return the original scalar and retain its graph.
  original=L.sigmoid_ce_sum
  def observe(logits,labels,weight=None):
   value=original(logits,labels,weight)
   with torch.no_grad():
    unreduced=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='none')
    rec={'logits_shape':list(logits.shape),'target_shape':list(labels.shape),'unreduced_shape':list(unreduced.shape),'weight_shape':list(weight.shape) if weight is not None else None,
         'sum_returned':float(value.detach()),'unweighted_point_mean':float(unreduced.mean()),'weight_sum_per_mask':weight.sum(-1).flatten().tolist() if weight is not None else None,
         'weighted_mean':float((unreduced*weight).sum()/weight.expand_as(unreduced).sum()) if weight is not None else float(unreduced.mean())}
    result['bce_calls'].append(rec)
   return value
  L.sigmoid_ce_sum=observe
  aux=tuple({k:v.float() for k,v in x.items() if k in ('class_logits','mask_logits')} for x in out['aux_outputs'])
  losses=criterion(out['class_logits'].float(),out['mask_logits'].float(),targets,pts,w,aux,vm)
  L.sigmoid_ce_sum=original
  result['losses']={k:float(v.detach()) for k,v in losses.items()}
  result['loss_mask_div_P']=result['losses']['mask']/result['P']
  if not all(bool(torch.isfinite(v)) for v in losses.values()): raise RuntimeError('nonfinite loss before backward')
  focus=head.mask_embed[2].bias
  for family in ('main','aux'):
   for name in ('class','mask','dice'):
    keys=[name] if family=='main' else [k for k in losses if k.startswith(name+'_')]
    objective=sum(losses[k]*criterion.weight_dict[k] for k in keys)
    for p in params: p.grad=None
    objective.backward(retain_graph=True)
    grad=focus.grad
    rec={'component':family+'_'+name,'keys':keys,'coefficients':{k:criterion.weight_dict[k] for k in keys},'objective':float(objective.detach()),'focus_grad_none':grad is None}
    if grad is not None:
     f=grad.detach(); good=torch.isfinite(f)
     rec.update(dtype=str(f.dtype),finite=bool(good.all()),nan=int(torch.isnan(f).sum()),inf=int(torch.isinf(f).sum()),max_abs=float(f.abs().max()) if bool(good.all()) else 'nonfinite',finite_max_abs=float(f[good].abs().max()) if bool(good.any()) else None)
    rec['all_parameter_nonfinite_count']=sum(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params)
    result['rows'].append(rec)
  # Scope is exactly the six component backwards above. The joint-objective
  # backward is deliberately NOT reproduced here: it is outside this round's
  # authorisation, and the earlier diagnose_nan run already recorded it.
  result['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
  result['status']='COMPLETE'
 except Exception:result['error']=traceback.format_exc();result['status']='ERROR'
 (root/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(result,allow_nan=False),flush=True)
 return 0 if result['status']=='COMPLETE' else 1
if __name__=='__main__':raise SystemExit(main())
