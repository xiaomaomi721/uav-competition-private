#!/usr/bin/env python3
"""8-image loader parity: current local strict loader vs existing 278 test2 cache.

The original 278-test2 process was launched with code SHA
084081508ea4282c7f9e7243011f70ff126aeee2fc0a5ea5bedc23ffe74f8cf0
and used tta_anatomy_v2.build_model. The source was later overwritten. This
script uses the current local strict loader and exact same D4 math, comparing
fixed images to the immutable existing cache plus its original D4 summary.
"""
from pathlib import Path
import json,os,sys,numpy as np
SNAP=Path('/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1')
sys.path.insert(0,str(SNAP))
sys.path.insert(0,'/home/jhy/rare_diag_20260915/round2_20260921/scripts')
from _r2_test2_cache import build_model_local,views_and_mean,CONFIG
import torch
from PIL import Image
from src.transforms import normalize_to_tensor
CK=Path('/home/jhy/rare_diag_20260915/atda_20260918/export/composed_atda_8k_a1p25.pt')
TEST=Path('/home/jhy/datasets/AIC2026_UAV_semifinal/test_2/images')
CACHE=Path('/home/jhy/rare_diag_20260915/round2_20260921/cache_278_test2_raw')
OUT=Path('/home/jhy/rare_diag_20260915/round2_20260921/LOADER_PARITY_278.json')
idxs=[1,2,3,100,500,900,1299,1300]
model=build_model_local(CONFIG,CK,'cuda:0')
old=json.load(open(CACHE/'META.json')); by={x['image']:x for x in old['d4_disagreement']}; rows=[]
with torch.inference_mode():
 for idx in idxs:
  n=f'test2_{idx}.png'
  with Image.open(TEST/n) as im:img=np.asarray(im.convert('RGB')).copy()
  px=normalize_to_tensor(img).unsqueeze(0).cuda(); views,mean=views_and_mean(model,px)
  cached=torch.from_numpy(np.load(CACHE/'scores'/f'test2_{idx}.npy').astype(np.float32)).cuda()
  pred=mean.argmax(0); dg=(views.argmax(1)!=pred.unsqueeze(0)).float().mean(0)
  rows.append({'image':n,'fp32_vs_cache_max_abs':float((mean-cached).abs().max()),'argmax_disagreement':float((mean.argmax(0)!=cached.argmax(0)).float().mean()),'current_d4_mean':float(dg.mean()),'old_d4_mean':by[n]['d4_disagreement_mean'],'d4_mean_abs_delta':abs(float(dg.mean())-by[n]['d4_disagreement_mean'])})
report={'original_code_sha256':'084081508ea4282c7f9e7243011f70ff126aeee2fc0a5ea5bedc23ffe74f8cf0','original_loader':'tta_anatomy_v2.build_model strict; GPU2','original_command':'_r2_test2_cache.py --checkpoint composed_atda_8k_a1p25.pt --tag theta278_prefold --outdir cache_278_test2_raw --device cuda:0','current_loader':'local src.models.build_model + cwfa off + strict state load','rows':rows,'max_argmax_disagreement':max(x['argmax_disagreement'] for x in rows),'max_d4_mean_abs_delta':max(x['d4_mean_abs_delta'] for x in rows)}
report['loader_parity_pass']=report['max_argmax_disagreement']<=0.0002 and report['max_d4_mean_abs_delta']<=1e-7
OUT.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
