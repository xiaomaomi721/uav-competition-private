#!/usr/bin/env python3
"""Phase 1 DART reliability audit in exact old-Bstar deployment space.

Source GT is used only on the official 1399 holdout. Test2 is transform-only:
prevalence/support statistics are reported, never correctness.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from _r2_contracts import BSTAR, OUT, ROUND2, deployed_logits, sha256_lines, write_json

NCLASS = 8
BOOT = 1000
RNG = np.random.default_rng(20260922)
MIN_IMAGES = 25
MIN_PIXELS = 50000
CONF_EDGES = np.array([0., .70, .80, .90, .95, .98, 1.000001])
D4_EDGES = np.array([0., .005, .01, .02, .04, 1.000001])
CURVE_BINS = 200
COVERAGES = [.05, .10, .20, .40, .60, .80, 1.0]


def native(path: Path):
    return np.load(path, allow_pickle=False).astype(np.float32)


def deploy(path: Path):
    z = deployed_logits(native(path))
    p = z.softmax(0)
    top = torch.topk(p, 2, dim=0)
    pred = top.indices[0].numpy().astype(np.uint8)
    conf = top.values[0].numpy()
    margin = (top.values[0] - top.values[1]).numpy()
    ent = (-(p * p.clamp_min(1e-12).log()).sum(0) / np.log(8.)).numpy()
    return pred, conf, margin, ent


def prediction(path: Path):
    return deployed_logits(native(path)).argmax(0).numpy().astype(np.uint8)


def upsample_d4(path: Path):
    x = torch.from_numpy(np.load(path, allow_pickle=False).astype(np.float32) / 8.0)
    return F.interpolate(x[None, None], size=(1024, 1024), mode="bilinear",
                         align_corners=False)[0, 0].numpy()


def qci(values):
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def bootstrap_ratio(num_by_image, den_by_image):
    n = len(num_by_image); out = np.empty(BOOT, np.float64)
    for b in range(BOOT):
        idx = RNG.integers(0, n, n)
        den = den_by_image[idx].sum()
        out[b] = num_by_image[idx].sum() / den if den else np.nan
    return out[np.isfinite(out)]


def transition_bootstrap(num, den):
    # [images,64] -> [64,2] percentile CI
    out = np.full((64, 2), np.nan)
    n = len(num)
    vals = np.empty((BOOT, 64), np.float64)
    for b in range(BOOT):
        idx = RNG.integers(0, n, n)
        d = den[idx].sum(0); x = num[idx].sum(0)
        vals[b] = np.divide(x, d, out=np.full(64, np.nan), where=d > 0)
    for j in range(64):
        good = vals[:, j][np.isfinite(vals[:, j])]
        if len(good): out[j] = np.quantile(good, [.025, .975])
    return out


def hist_add(store, score, correct, valid):
    s = np.clip(score[valid], 0, 1)
    c = correct[valid]
    idx = np.minimum((s * CURVE_BINS).astype(np.int64), CURVE_BINS - 1)
    store[0] += np.bincount(idx, minlength=CURVE_BINS)
    store[1] += np.bincount(idx, weights=c.astype(np.float64), minlength=CURVE_BINS)


def curve_from_hist(store):
    den = np.cumsum(store[0][::-1]); num = np.cumsum(store[1][::-1])
    total = den[-1]
    cov = den / max(total, 1); prec = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    rows = []
    for target in COVERAGES:
        i = int(np.argmin(np.abs(cov - target)))
        rows.append({"coverage": float(cov[i]), "precision": float(prec[i]),
                     "score_threshold": float(1 - (i + 1) / CURVE_BINS)})
    return {"total": int(total), "points": rows}


def fold_of(stem):
    return int(sha256_lines([stem])[:8], 16) % 5


def read_d4_means(meta_path):
    meta = json.loads(meta_path.read_text())
    return {Path(x["image"]).stem: x["d4_disagreement_mean"] for x in meta["d4_disagreement"]}


def source_pass(stems, cache261, cache278, gt_dir, d4_dir, d4_image, collect_features=False,
                priors=None):
    n = len(stems)
    support=np.zeros((n,64),np.int64); c278=np.zeros_like(support);c261=np.zeros_like(support);neither=np.zeros_like(support)
    per_correct=np.zeros(n,np.int64);per_valid=np.zeros(n,np.int64)
    hists={k:np.zeros((2,CURVE_BINS),np.float64) for k in
           ("confidence","prob_margin","negative_entropy","d4_stability","confidence_x_transition","confidence_x_transition_x_d4")}
    # cell aggregates transition x confidence-bin x image-D4-bin
    shape=(n,64,len(CONF_EDGES)-1,len(D4_EDGES)-1)
    cell_sup=np.zeros(shape,np.int64);cell_util=np.zeros(shape,np.int64)
    for ii,stem in enumerate(stems):
        gt=np.load(gt_dir/f"{stem}.npy",allow_pickle=False)
        p261=prediction(cache261/f"{stem}.npy")
        p278,conf,margin,ent=deploy(cache278/f"{stem}.npy")
        valid=gt!=255; trans=(p261.astype(np.int16)*8+p278).astype(np.int16)
        corr278=p278==gt;corr261=p261==gt
        support[ii]=np.bincount(trans[valid],minlength=64)
        c278[ii]=np.bincount(trans[valid],weights=corr278[valid],minlength=64).astype(np.int64)
        c261[ii]=np.bincount(trans[valid],weights=corr261[valid],minlength=64).astype(np.int64)
        neither[ii]=np.bincount(trans[valid],weights=(~corr278&~corr261)[valid],minlength=64).astype(np.int64)
        per_correct[ii]=corr278[valid].sum();per_valid[ii]=valid.sum()
        if collect_features:
            d4=upsample_d4(d4_dir/f"{stem}.npy");stability=1-d4
            hist_add(hists["confidence"],conf,corr278,valid)
            hist_add(hists["prob_margin"],margin,corr278,valid)
            hist_add(hists["negative_entropy"],1-ent,corr278,valid)
            hist_add(hists["d4_stability"],stability,corr278,valid)
            prior=np.ones(64,np.float32) if priors is None else priors[fold_of(stem)]
            tp=prior[trans]
            hist_add(hists["confidence_x_transition"],conf*tp,corr278,valid)
            hist_add(hists["confidence_x_transition_x_d4"],conf*tp*(.5+.5*stability),corr278,valid)
            cb=np.clip(np.digitize(conf,CONF_EDGES)-1,0,len(CONF_EDGES)-2)
            db=int(np.clip(np.digitize(d4_image[stem],D4_EDGES)-1,0,len(D4_EDGES)-2))
            flat=(trans*(len(CONF_EDGES)-1)+cb)*(len(D4_EDGES)-1)+db
            vv=flat[valid].ravel();util=(corr278.astype(np.int8)-corr261.astype(np.int8))[valid].ravel()
            cell_sup[ii].reshape(-1)[:]+=np.bincount(vv,minlength=np.prod(shape[1:]))
            cell_util[ii].reshape(-1)[:]+=np.bincount(vv,weights=util,minlength=np.prod(shape[1:])).astype(np.int64)
        if (ii+1)%100==0: print(f"source audit {ii+1}/{n}",flush=True)
    return {"support":support,"correct278":c278,"correct261":c261,"neither":neither,
            "per_correct":per_correct,"per_valid":per_valid,"hists":hists,"cell_sup":cell_sup,"cell_util":cell_util}


def make_oof_priors(stems, support, correct):
    global_p=correct.sum()/support.sum();result={}
    folds=np.array([fold_of(s) for s in stems])
    for f in range(5):
        use=folds!=f;s=support[use].sum(0);c=correct[use].sum(0)
        result[f]=((c+global_p*1000)/(s+1000)).astype(np.float32)
    return result


def cell_weights(cell_sup,cell_util):
    flat_s=cell_sup.reshape(len(cell_sup),-1);flat_u=cell_util.reshape(len(cell_util),-1)
    total_s=flat_s.sum(0);total_u=flat_u.sum(0);images=(flat_s>0).sum(0)
    lo=np.full(len(total_s),-1.,np.float64);point=np.divide(total_u,total_s,out=np.zeros_like(total_u,dtype=float),where=total_s>0)
    n=len(flat_s)
    vals=np.empty((BOOT,len(total_s)),np.float32)
    for b in range(BOOT):
        idx=RNG.integers(0,n,n);s=flat_s[idx].sum(0);u=flat_u[idx].sum(0)
        vals[b]=np.divide(u,s,out=np.full(len(s),np.nan,dtype=np.float32),where=s>0)
    for j in np.where((total_s>=MIN_PIXELS)&(images>=MIN_IMAGES))[0]:
        good=vals[:,j][np.isfinite(vals[:,j])]
        if len(good):lo[j]=np.quantile(good,.025)
    weight=np.maximum(lo,0).reshape(cell_sup.shape[1:])
    return weight,{"support":total_s.reshape(cell_sup.shape[1:]),"images":images.reshape(cell_sup.shape[1:]),
                   "point":point.reshape(cell_sup.shape[1:]),"lower95":lo.reshape(cell_sup.shape[1:])}


def test2_prevalence(cache261,cache278,d4_image,source_prev,weights):
    trans=np.zeros(64,np.int64);accepted=0;total=0;outside=0
    class_share261=np.zeros(8,np.int64);class_share278=np.zeros(8,np.int64)
    for i in range(1,1301):
        stem=f"test2_{i}";p261=prediction(cache261/f"{stem}.npy");p278,conf,_,_=deploy(cache278/f"{stem}.npy")
        t=p261.astype(np.int16)*8+p278;trans+=np.bincount(t.ravel(),minlength=64)
        class_share261+=np.bincount(p261.ravel(),minlength=8);class_share278+=np.bincount(p278.ravel(),minlength=8)
        cb=np.clip(np.digitize(conf,CONF_EDGES)-1,0,len(CONF_EDGES)-2)
        db=int(np.clip(np.digitize(d4_image[stem],D4_EDGES)-1,0,len(D4_EDGES)-2))
        w=weights[t,cb,db];accepted+=int((w>0).sum());total+=w.size
        outside+=int((source_prev[t]==0).sum())
        if i%100==0:print(f"test2 prevalence {i}/1300",flush=True)
    return {"transition_pixels":trans.tolist(),"transition_prevalence":(trans/trans.sum()).tolist(),
            "class_share261":(class_share261/class_share261.sum()).tolist(),
            "class_share278":(class_share278/class_share278.sum()).tolist(),
            "D2_positive_weight_coverage":accepted/total,"outside_source_transition_support":outside/total}


def main():
    inv=OUT/"00_inventory";out=OUT/"02_reliability";out.mkdir(parents=True,exist_ok=False)
    split=json.loads((inv/"SOURCE_AUDIT_SPLIT.json").read_text());r=split["R1099"];h=split["H300"]
    c261=OUT/"01_holdout_cache/theta261/scores";c278=OUT/"01_holdout_cache/theta278/scores";d4dir=OUT/"01_holdout_cache/theta278/d4_disagreement_count"
    gt=Path('/home/jhy/rare_diag_20260915/b2_val_20260917/gt')
    src_d4=read_d4_means(OUT/"01_holdout_cache/theta278/META.json")
    print("R1099 transition pass",flush=True);base=source_pass(r,c261,c278,gt,d4dir,src_d4)
    priors=make_oof_priors(r,base["support"],base["correct278"])
    print("R1099 feature pass",flush=True);fit=source_pass(r,c261,c278,gt,d4dir,src_d4,True,priors)
    weights,cells=cell_weights(fit["cell_sup"],fit["cell_util"])
    print("H300 confirmation pass",flush=True);confirm=source_pass(h,c261,c278,gt,d4dir,src_d4,True,{i:base["correct278"].sum(0)/np.maximum(base["support"].sum(0),1) for i in range(5)})
    # Confirm net utility only within source-selected positive cells.
    flat_w=weights.reshape(-1);hs=confirm["cell_sup"].reshape(len(h),-1);hu=confirm["cell_util"].reshape(len(h),-1)
    chosen=flat_w>0;selected_s=hs[:,chosen].sum(1);selected_u=hu[:,chosen].sum(1)
    boot=bootstrap_ratio(selected_u,selected_s) if selected_s.sum() else np.array([])
    h_point=float(selected_u.sum()/selected_s.sum()) if selected_s.sum() else None
    h_ci=qci(boot) if len(boot) else [None,None]

    t2_d4=read_d4_means(ROUND2/"cache_278_test2_raw/META.json")
    print("test2 prevalence pass",flush=True)
    t2=test2_prevalence(ROUND2/"cache_261_test2_raw/scores",ROUND2/"cache_278_test2_raw/scores",t2_d4,base["support"].sum(0),weights)

    s=base["support"];a=base["correct278"];b=base["correct261"];n=base["neither"]
    adv=a-b;adv_ci=transition_bootstrap(adv,s);a_ci=transition_bootstrap(a,s);b_ci=transition_bootstrap(b,s)
    t2counts=np.asarray(t2["transition_pixels"]);sourcecounts=s.sum(0);rows=[]
    for x in range(64):
        aa,bb=divmod(x,8);sup=int(sourcecounts[x]);img=int((s[:,x]>0).sum())
        rows.append({"from":aa,"to":bb,"image_support":img,"pixel_support":sup,
                     "p278_correct":float(a[:,x].sum()/sup) if sup else None,"p278_correct_ci95":a_ci[x].tolist(),
                     "p261_correct":float(b[:,x].sum()/sup) if sup else None,"p261_correct_ci95":b_ci[x].tolist(),
                     "p_neither_correct":float(n[:,x].sum()/sup) if sup else None,
                     "net_advantage":float(adv[:,x].sum()/sup) if sup else None,"net_advantage_ci95":adv_ci[x].tolist(),
                     "source_prevalence":float(sup/sourcecounts.sum()),"test2_prevalence":float(t2counts[x]/t2counts.sum()),
                     "test2_source_prevalence_ratio":float((t2counts[x]/t2counts.sum())/(sup/sourcecounts.sum())) if sup else None,
                     "supported_for_D2":bool(img>=MIN_IMAGES and sup>=MIN_PIXELS)})
    table={"source_fit":"R1099 only","bootstrap_unit":"image","bootstrap_repeats":BOOT,"rows":rows,
           "warning":"source reliability is a prior, never test2 truth","test2_ground_truth_used":False}
    write_json(out/"DART_TRANSITION_TABLE.json",table)
    curves={k:curve_from_hist(v) for k,v in fit["hists"].items()}
    gate_pass=bool(len(boot) and h_ci[0] is not None and h_ci[0]>0 and t2["D2_positive_weight_coverage"]>0)
    gate={"status":"PASS" if gate_pass else "SKIPPED_BY_RELIABILITY_GATE","requirements":{"H300_selected_net_advantage_point":h_point,
          "H300_selected_net_advantage_ci95":h_ci,"selected_H300_pixels":int(selected_s.sum()),
          "test2_positive_weight_coverage":t2["D2_positive_weight_coverage"]},
          "weights_path":str(out/"D2_WEIGHTS.npz") if gate_pass else None,
          "test2_ground_truth_used":False,"platform_submission_performed":False}
    np.savez_compressed(out/"D2_WEIGHTS.npz",weights=weights,conf_edges=CONF_EDGES,d4_edges=D4_EDGES,
                        support=cells["support"],images=cells["images"],point=cells["point"],lower95=cells["lower95"])
    write_json(out/"D2_GATE.json",gate)
    report={"curves":curves,"gate":gate,"test2_prevalence":t2,
            "prototype_audit":{"status":"SKIPPED_COST_NOT_JUSTIFIED","reason":"existing Exp18 features are image-pooled, not spatial pixel features"},
            "method_notes":["all source decisions use raw+Bstar then bilinear logits 1024",
                            "OOF transition priors use five image groups",
                            "pixel D4 adds source evidence; test2 D4 is only per-image because existing cache discarded views",
                            "no test2 correctness claim"],"test2_ground_truth_used":False,"platform_submission_performed":False}
    write_json(out/"DART_RELIABILITY_AUDIT.json",report)
    md=["# DART Reliability Audit","",f"- D2 gate: **{gate['status']}**",
        f"- H300 selected-region net advantage: `{h_point}`; image-bootstrap 95% CI `{h_ci}`",
        f"- test2 supported positive-weight coverage: `{t2['D2_positive_weight_coverage']:.6f}`","",
        "## Precision at matched coverage","", "| signal | 5% | 10% | 20% | 40% | 60% | 80% | 100% |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key,val in curves.items():md.append("| "+key+" | "+" | ".join(f"{x['precision']:.6f}" for x in val["points"])+" |")
    md += ["","## Answers","","1. Transition information is judged only by the matched-coverage table above; it is retained only through OOF source priors.",
           "2. D4 incremental information is the difference between `confidence_x_transition` and `confidence_x_transition_x_d4`; test2 uses per-image D4 only.",
           f"3. Transitions with fewer than {MIN_IMAGES} images or {MIN_PIXELS} pixels receive zero D2 weight.",
           "4. Test2/source prevalence ratios are recorded in `DART_TRANSITION_TABLE.json`; unseen transitions receive zero weight.","",
           "Source transition reliability is only a prior. No test2 flip is labelled correct or incorrect."]
    (out/"DART_RELIABILITY_AUDIT.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    print(json.dumps({"gate":gate,"curves":curves},indent=2))

if __name__=="__main__":main()
