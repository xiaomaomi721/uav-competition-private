#!/usr/bin/env python3
"""Round2 paired model-space comparison: test1 vs test2.

Uses raw native caches only; no forward. Reports:
A) theta278 test1 vs test2 confidence/entropy/D4 disagreement/raw class share
B) theta261->theta278 flip structure on test1 vs test2
C) theta278 raw->old-Bstar action on test1 vs test2
All statements are prediction behavior, never GT accuracy/error.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import ndimage

ROOT=Path('/home/jhy/rare_diag_20260915');R2=ROOT/'round2_20260921';A=ROOT/'atda_20260918'
C={'test1':{'261':A/'ref261_cache/scores','278':R2/'cache_278_test1_raw/scores','meta278':R2/'cache_278_test1_raw/META.json','n':500,'prefix':'test1'},'test2':{'261':R2/'cache_261_test2_raw/scores','278':R2/'cache_278_test2_raw/scores','meta278':R2/'cache_278_test2_raw/META.json','meta261':R2/'cache_261_test2_raw/META.json','n':1300,'prefix':'test2'}}
BSTAR=np.array([-.2,.1,.0709,.4,3.6,.4,2.1215,-.5],np.float32)[:,None,None];CN=['Background','Building','Road','Water','Barren','Vegetation','Agricultural','Vehicle']
def softmax(z):z=z-z.max(0,keepdims=True);e=np.exp(z);return e/e.sum(0,keepdims=True)
def q(x):return {'mean':float(np.mean(x)),'q05':float(np.quantile(x,.05)),'q25':float(np.quantile(x,.25)),'median':float(np.median(x)),'q75':float(np.quantile(x,.75)),'q95':float(np.quantile(x,.95))}
def hist(x):h,e=np.histogram(x,bins=20,range=(0,1));return {'counts':h.tolist(),'edges':e.round(3).tolist()}
def ccbin(m):
 lab,n=ndimage.label(m)
 if not n:return {'components':0}
 s=np.bincount(lab.ravel())[1:];edges=[1,2,4,8,16,32,64,128,256,512,1024,4096,10000000];h,_=np.histogram(s,bins=edges);return {'components':int(n),'mean':float(s.mean()),'max':int(s.max()),'bins':{f'{edges[i]}-{edges[i+1]}':int(h[i]) for i in range(len(h))}}
def analyze(name,cfg):
 trans=np.zeros((8,8),np.int64);sh261=np.zeros(8,np.int64);sh278=np.zeros(8,np.int64);rawsh=np.zeros(8,np.int64);bsh=np.zeros(8,np.int64);flip=[];bstarflip=[];conf278=[];ent278=[];chg=[];unch=[];high=0;chgn=0;per=[];cc=[]
 for i in range(1,cfg['n']+1):
  z261=np.load(cfg['261']/f"{cfg['prefix']}_{i}.npy").astype(np.float32);z278=np.load(cfg['278']/f"{cfg['prefix']}_{i}.npy").astype(np.float32)
  p261=softmax(z261+BSTAR);p278=softmax(z278);c261=p261.max(0);c278=p278.max(0);e278=-(p278*np.log(np.clip(p278,1e-12,None))).sum(0)
  a261=(z261+BSTAR).argmax(0);a278=(z278+BSTAR).argmax(0);raw=z278.argmax(0);bst=(z278+BSTAR).argmax(0);m=a261!=a278;mb=raw!=bst
  np.add.at(trans,(a261,a278),1);sh261+=np.bincount(a261.ravel(),minlength=8);sh278+=np.bincount(a278.ravel(),minlength=8);rawsh+=np.bincount(raw.ravel(),minlength=8);bsh+=np.bincount(bst.ravel(),minlength=8)
  flip.append(m.mean());bstarflip.append(mb.mean());chg.append(c261[m]);unch.append(c261[~m]);high+=int((m&(c261>=.9)).sum());chgn+=int(m.sum());conf278.append(c278.ravel());ent278.append(e278.ravel());per.append({'image':f"{cfg['prefix']}_{i}.png",'261_to_278_flip':float(m.mean()),'raw_to_bstar_flip':float(mb.mean()),'mean_conf278':float(c278.mean()),'mean_entropy278':float(e278.mean())})
  if m.any():cc.append(ccbin(m))
 total=cfg['n']*256*256;conf278=np.concatenate(conf278);ent278=np.concatenate(ent278);chg=np.concatenate(chg);unch=np.concatenate(unch);meta=json.load(open(cfg['meta278']));d4=[x['d4_disagreement_mean'] for x in meta['d4_disagreement']]
 return {'n_images':cfg['n'],'grid_pixels':total,'theta278_raw':{'confidence':q(conf278),'entropy':q(ent278),'class_share_pct':(100*rawsh/total).round(5).tolist(),'per_image_confidence':q([x['mean_conf278'] for x in per]),'per_image_entropy':q([x['mean_entropy278'] for x in per]),'d4_disagreement_per_image':q(d4)},'261_to_278_bstar_space':{'overall_flip_pct':100*sum(x['261_to_278_flip'] for x in per)/cfg['n'],'per_image_flip_pct':q(100*np.array(flip)),'transition_matrix':trans.tolist(),'changed_261_confidence':q(chg),'unchanged_261_confidence':q(unch),'changed_confidence_hist':hist(chg),'unchanged_confidence_hist':hist(unch),'high_conf_ge_0p9_flip_fraction_of_changed':high/max(1,chgn),'net_area_change_pp':{CN[k]:float(100*(sh278[k]-sh261[k])/total) for k in range(8)},'cc_summary_aggregate_note':'per-image component summaries available','per_image_cc':cc},'old_bstar_action_on_278':{'overall_flip_pct':100*sum(bstarflip)/cfg['n'],'per_image_flip_pct':q(100*np.array(bstarflip)),'raw_share_pct':(100*rawsh/total).round(5).tolist(),'bstar_share_pct':(100*bsh/total).round(5).tolist(),'share_delta_pp':(100*(bsh-rawsh)/total).round(5).tolist()},'per_image':per}
def main():
 out={k:analyze(k,v) for k,v in C.items()};a=out['test1'];b=out['test2'];out['paired_summary']={'confidence_mean_delta_test2_minus_test1':b['theta278_raw']['confidence']['mean']-a['theta278_raw']['confidence']['mean'],'entropy_mean_delta_test2_minus_test1':b['theta278_raw']['entropy']['mean']-a['theta278_raw']['entropy']['mean'],'d4_mean_delta_test2_minus_test1':b['theta278_raw']['d4_disagreement_per_image']['mean']-a['theta278_raw']['d4_disagreement_per_image']['mean'],'261_to_278_flip_pct_delta':b['261_to_278_bstar_space']['overall_flip_pct']-a['261_to_278_bstar_space']['overall_flip_pct'],'high_conf_flip_fraction_delta':b['261_to_278_bstar_space']['high_conf_ge_0p9_flip_fraction_of_changed']-a['261_to_278_bstar_space']['high_conf_ge_0p9_flip_fraction_of_changed'],'old_bstar_flip_pct_delta':b['old_bstar_action_on_278']['overall_flip_pct']-a['old_bstar_action_on_278']['overall_flip_pct']}
 (R2/'MODEL_SPACE_PAIRED.json').write_text(json.dumps(out,indent=2));print(json.dumps(out['paired_summary'],indent=2))
if __name__=='__main__':main()
