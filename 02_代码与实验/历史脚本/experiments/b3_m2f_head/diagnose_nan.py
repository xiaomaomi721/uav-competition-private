"""Bounded GPU2 numerical diagnostics. Frozen model/loss files are not edited.
Same real batch, initial seed and sampling RNG in all precision modes.
No hyperparameter search. Results are diagnostic, not a training release.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['fp32', 'autocast', 'scaled'], required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--steps', type=int, default=1)
    ap.add_argument('--component', default='total', choices=['total', 'class', 'mask', 'dice', 'aux'])
    ap.add_argument('--anomaly', action='store_true')
    a = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
        raise RuntimeError('GPU2 mapping required before any output is created')
    root = Path(a.outdir)
    root.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    import numpy as np
    from b3_m2f_head import loss as L
    from b3_m2f_head.head import B3M2FStyleHead
    from b3_m2f_head.train_entry import build_encoder, build_batch, to_device_set_targets, build_weight_dict, DATA_ROOT, S62_CKPT
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = False
    events = (root / 'events.jsonl').open('x', encoding='utf-8', buffering=1)
    def emit(kind, **data):
        events.write(json.dumps({'kind': kind, **data}, ensure_ascii=False, allow_nan=False)+'\n')
    def stats(x):
        x = x.detach()
        f = x.float()
        good = torch.isfinite(f)
        v = f[good]
        return dict(shape=list(x.shape), dtype=str(x.dtype), finite=bool(good.all()),
                    nan=int(torch.isnan(f).sum()), inf=int(torch.isinf(f).sum()),
                    min=float(v.min()) if v.numel() else None,
                    max=float(v.max()) if v.numel() else None,
                    mean=float(v.mean()) if v.numel() else None,
                    std=float(v.std(unbiased=False)) if v.numel() else None,
                    max_abs=float(v.abs().max()) if v.numel() else None)
    state = {'step': 0, 'first_parameter': None, 'first_activation': None, 'backward': False}
    def check(name, x):
        s = stats(x)
        emit('forward', step=state['step'], name=name, **s)
        if not s['finite']:
            raise RuntimeError('first nonfinite forward: '+name)
        if x.requires_grad:
            def grad_hook(g):
                if not bool(torch.isfinite(g).all()) and state['first_activation'] is None:
                    state['first_activation'] = name
                    emit('first_bad_activation_gradient', step=state['step'], name=name, **stats(g))
                return g
            x.register_hook(grad_hook)
    def recurse(name, obj):
        if isinstance(obj, torch.Tensor): check(name, obj)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj): recurse(f'{name}.{i}', v)
        elif isinstance(obj, dict):
            for k,v in obj.items(): recurse(f'{name}.{k}',v)
    report = {'mode': a.mode, 'component': a.component, 'steps': [], 'status': 'ERROR'}
    try:
        source = Path(__file__).parent
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')}
        emit('manifest', sources=hashes, checkpoint=S62_CKPT, checkpoint_sha256=hashlib.sha256(Path(S62_CKPT).read_bytes()).hexdigest(),
             torch=torch.__version__, gpu=torch.cuda.get_device_name(0), seed=42,
             mode=a.mode, component=a.component, batch_size=1, stem='0001', points=12544,
             cudnn_benchmark=False, note='CUDA grid_sample backward may be nondeterministic; RNG inputs are fixed')
        encoder = build_encoder('cuda:0').train()
        head = B3M2FStyleHead().to('cuda:0').train()
        params = [('encoder.'+n,p) for n,p in encoder.named_parameters()] + [('head.'+n,p) for n,p in head.named_parameters()]
        init_hash = hashlib.sha256()
        for n,p in params:
            init_hash.update(n.encode()); init_hash.update(p.detach().cpu().contiguous().numpy().tobytes())
        emit('initial_parameters', sha256=init_hash.hexdigest())
        for n,p in params:
            def hook(g, name=n):
                if not bool(torch.isfinite(g).all()) and state['first_parameter'] is None:
                    state['first_parameter'] = name
                    emit('first_bad_parameter_gradient', step=state['step'], name=name, **stats(g))
                return g
            p.register_hook(hook)
        encoder.register_forward_hook(lambda m,i,o: recurse('encoder.features',o.hidden_states))
        head.pixel_decoder.register_forward_hook(lambda m,i,o: recurse('pixel_decoder',o))
        for i,layer in enumerate(head.layers):
            layer.register_forward_hook(lambda m,x,o,idx=i: recurse(f'query_layer.{idx}',o))
            def attn_pre(m, args, kwargs, idx=i):
                mask = kwargs.get('attn_mask')
                emit('attention_mask', step=state['step'], layer=idx,
                     all_blocked_rows=int(mask.all(-1).sum()) if mask is not None else 0,
                     shape=list(mask.shape) if mask is not None else None)
                if mask is not None and bool(mask.all(-1).any()):
                    raise RuntimeError('all-blocked row passed to attention')
            layer.cross_attn.register_forward_pre_hook(attn_pre, with_kwargs=True)
        head.register_forward_hook(lambda m,i,o: recurse('head',o))
        criterion=L.SetCriterion(weight_dict=build_weight_dict(6,1.0)).cuda()
        # Audit costs BEFORE the frozen matcher can nan_to_num them. Reuse its
        # exact point functions; actual matching still runs in the frozen code.
        @torch.no_grad()
        def matcher_pre(module,args):
            cl,ml,targets,pts,pw,vm=args
            for i,t in enumerate(targets):
                st,w=L.sample_target_masks(t['mask_labels'][None],vm[i:i+1],pts[i:i+1])
                so=L.sample_point(ml[i:i+1],pts[i:i+1])[0]
                cc=-cl[i].softmax(-1)[:,t['class_labels']]
                cm=L.pair_wise_sigmoid_ce(so,st[0],w)
                cd=L.pair_wise_dice(so,st[0],w)
                for name,v in [('sampled_logits',so),('class_cost',cc),('bce_cost',cm),('dice_cost',cd),('combined_cost',2*cc+5*cm+5*cd)]:
                    check('matcher.'+name,v)
        criterion.matcher.register_forward_pre_hook(matcher_pre)
        criterion.matcher.register_forward_hook(lambda m,i,o: emit('matching',step=state['step'],indices=[(r.tolist(),c.tolist()) for r,c in o]))
        optimizer=torch.optim.AdamW([{'params':encoder.parameters(),'lr':1e-5}, {'params':head.parameters(),'lr':1e-3}],weight_decay=.01)
        scaler=torch.amp.GradScaler('cuda',enabled=a.mode=='scaled')
        calls=[]
        optimizer.register_step_post_hook(lambda *args,**kw: calls.append(state['step']))
        pixel,label=build_batch(DATA_ROOT,['0001'],1024,'cuda:0',42)
        assert pixel.shape==(1,3,1024,1024) and label.shape==(1,1024,1024)
        emit('batch', image_sha256=hashlib.sha256(pixel.cpu().numpy().tobytes()).hexdigest(),
             label_sha256=hashlib.sha256(label.numpy().tobytes()).hexdigest(), ignore_pixels=int((label==255).sum()),
             labels=torch.unique(label).tolist())
        report['init_sha256']=init_hash.hexdigest()
        for step in range(1,a.steps+1):
            state.update(step=step,first_parameter=None,first_activation=None)
            torch.manual_seed(142); torch.cuda.manual_seed_all(142)
            optimizer.zero_grad(set_to_none=True)
            for group,base in zip(optimizer.param_groups,[1e-5,1e-3]): group['lr']=base*min(step/50,1)
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0=time.perf_counter()
            with torch.autograd.set_detect_anomaly(a.anomaly):
                with torch.autocast('cuda',dtype=torch.float16,enabled=a.mode!='fp32'):
                    outputs=head(encoder(pixel_values=pixel,output_hidden_states=True).hidden_states)
                targets,pts,w,vm=L.build_set_targets(label,8)
                targets,pts,w,vm=to_device_set_targets(targets,pts,w,vm,'cuda:0')
                aux=tuple({k:v.float() for k,v in x.items() if k in ('class_logits','mask_logits')} for x in outputs['aux_outputs'])
                losses=criterion(outputs['class_logits'].float(),outputs['mask_logits'].float(),targets,pts,w,aux,vm)
                for n,v in losses.items(): check('loss.'+n,v)
                total=criterion.total(losses)
                check('total',total)
                objective=total if a.component=='total' else (sum(v*criterion.weight_dict[k] for k,v in losses.items() if '_' in k) if a.component=='aux' else losses[a.component]*criterion.weight_dict[a.component])
                pre_scale=scaler.get_scale()
                check('scaled_objective',scaler.scale(objective))
                scaler.scale(objective).backward()
            scaler.unscale_(optimizer)
            bad=[]; missing=[]
            for n,p in params:
                if p.grad is None: missing.append(n)
                elif not bool(torch.isfinite(p.grad).all()):
                    bad.append(n)
                    emit('bad_gradient',step=step,name=n,**stats(p.grad))
            before=[(n,p,p.detach().clone()) for n,p in params if n in ('head.class_embed.weight','head.mask_embed.2.weight') or n==params[0][0]]
            previous_calls=len(calls)
            if not bad:
                norm=torch.nn.utils.clip_grad_norm_([p for _,p in params],.01,error_if_nonfinite=True)
                emit('unscaled_gradient_norm',step=step,value=float(norm))
            # Nonfinite gradients are NEVER clipped or applied. For scaled mode
            # GradScaler itself must skip and lower its scale; observe that path.
            if not bad or a.mode=='scaled':
                scaler.step(optimizer); scaler.update()
            changed={n:float((p.detach()-old).abs().max()) for n,p,old in before}
            torch.cuda.synchronize()
            rec=dict(step=step,losses={k:float(v) for k,v in losses.items()},total=float(total),
                     bad_gradients=len(bad),missing_gradients=missing,first_bad_parameter=state['first_parameter'],
                     first_bad_activation=state['first_activation'],scale_before=pre_scale,scale_after=scaler.get_scale(),
                     optimizer_called=len(calls)>previous_calls,parameter_max_delta=changed,
                     allocated_MiB=torch.cuda.max_memory_allocated()/2**20,reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
                     diagnostic_seconds=time.perf_counter()-t0)
            report['steps'].append(rec); emit('step_result',**rec)
            print(json.dumps(rec),flush=True)
            if bad and a.mode!='scaled': break
            del outputs,aux,losses,total,objective,targets,pts,w,vm,before
        report['status']='COMPLETE_DIAGNOSTIC'
    except Exception:
        report['error']=traceback.format_exc()
        emit('exception',traceback=report['error']); print(report['error'],flush=True)
    finally:
        (root/'RESULT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        events.close()
    return 0 if report['status']=='COMPLETE_DIAGNOSTIC' else 1


if __name__=='__main__':
    raise SystemExit(main())
