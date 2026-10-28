"""Evaluate our saved step2000; no optimizer, updates, or submission."""
import json
from pathlib import Path
from train import ROOT,SPLIT,guard,seed,evaluate,UAVPMT,torch
from backbone import sha256

seed();guard()
checkpoint=ROOT/'formal/step_02000.pt'
assert sha256(checkpoint)=='8a36c910dbf608bd5d78bf03603a32d926b31fc2f377a68f8056361ef0f2730f'
assert sha256(ROOT/'dino.safetensors')=='45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941'
out=ROOT/'formal/val_02000_recovered_20261027.json'
assert not out.exists(),'Keep existing evaluation intact'
net=UAVPMT(ROOT/'dino.safetensors',ROOT/'pmt_decoder.bin').cuda()
cp=torch.load(checkpoint,map_location='cpu',weights_only=False) # own local checkpoint
assert cp['step']==2000
missing=net.load_state_dict(cp['model'],strict=False)
assert not missing.unexpected_keys and all(k.startswith('encoder.') for k in missing.missing_keys)
del cp
evaluate(net,(SPLIT/'val.txt').read_text().split(),out)
