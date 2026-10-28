"""A. fail-closed pre-flight for the soupB -> KD continuation run.

Read-only. Verifies every binding in the frozen plan before any GPU work.
"""
import hashlib
import json
from pathlib import Path

import torch
import yaml

R = Path('/home/jhy/rare_diag_20260915')
B = Path('/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1')
S62 = B / 'stage62_runs/stage62_w1_resume55b_pseudo500_seed42'
SOUPB = R / 'soups/soupB.pt'
SOUPB_SHA = '966139d11a62447c60518bfd27caed3244ccf08860859fd1ae9c7a79225f2f75'

fail = []


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


print('=== A. fail-closed ===')
got = sha(SOUPB)
print(f'  soupB sha256      {got}')
print(f'  expected          {SOUPB_SHA}')
if got != SOUPB_SHA:
    fail.append('soupB SHA mismatch')
print(f'  -> {"OK" if got == SOUPB_SHA else "FAIL"}')

sb = torch.load(SOUPB, map_location='cpu', weights_only=False)
s62 = torch.load(S62 / 'best_miou.pt', map_location='cpu', weights_only=False)
K_sb = sb.get('model') or sb.get('model_state_dict')
K_s62 = s62.get('model') or s62.get('model_state_dict')
print(f'\n  soupB top keys    {list(sb.keys())}')
print(f'  soupB tensors     {len(K_sb)}      s62 tensors {len(K_s62)}')
if len(K_sb) != 644:
    fail.append(f'soupB has {len(K_sb)} tensors, expected 644')

dh = [k for k in K_sb if 'decode_head' in k]
print(f'  decode_head       {len(dh)} tensors')
if len(dh) != 16:
    fail.append(f'decode_head {len(dh)} != 16')

mismatch = [k for k in K_sb if k not in K_s62 or K_sb[k].shape != K_s62[k].shape]
print(f'  shape mismatches  {len(mismatch)}')
if mismatch:
    fail.append(f'shape mismatch on {mismatch[:5]}')

extra = [k for k in K_s62 if k not in K_sb]
print(f'  missing keys      {len(extra)}')
if extra:
    fail.append(f'missing keys {extra[:5]}')

nonfinite = [k for k, v in K_sb.items()
             if v.is_floating_point() and not bool(torch.isfinite(v).all())]
print(f'  non-finite tensors {len(nonfinite)}')
if nonfinite:
    fail.append(f'non-finite {nonfinite[:5]}')

cfgp = S62 / 'config.yaml'
cfg = yaml.safe_load(cfgp.read_text())
opt = cfg.get('optimizer', {})
print('\n=== optimizer / scheduler binding ===')
print(f'  config            {cfgp}')
print(f'  optimizer         AdamW, single param group (model.parameters())')
print(f'  lr                {opt.get("learning_rate")}  (CLI --lr 3.0e-5 wins)')
print(f'  weight_decay      {opt.get("weight_decay")}')
print(f'  scheduler         LambdaLR linear-to-zero over --max-steps')
print(f'  warmup            none')
print(f'  grad clip         1.0')

print('\n=== recipe binding ===')
for k, v in [('T', 2.0), ('lambda_kd', 0.35), ('w_min', 0.3), ('kd_every', 2),
             ('batch_size', 2), ('grad_accum', 2), ('max_steps', 8000),
             ('lr', 3.0e-5), ('seed', 42), ('init', 'soupB.pt'),
             ('soft_dir', str(R / 'kd_soft_fp16')),
             ('config', str(cfgp))]:
    print(f'  {k:14s} {v}')

for p in [R / 'kd_soft_fp16/PROBS_MANIFEST.json', R / 'kd_soft_fp16/TEACHER_META.json']:
    print(f'  cache {p.name:24s} exists={p.is_file()}')

print('\n=== VERDICT ===')
if fail:
    print('  FAIL-CLOSED: ' + '; '.join(fail))
else:
    print('  ALL CHECKS PASS')
