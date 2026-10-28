#!/usr/bin/env python3
"""One-pass semifinal test2 integrity + SHA manifest (CPU only)."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from PIL import Image

ROOT=Path('/home/jhy/datasets/AIC2026_UAV_semifinal')
IMG=ROOT/'test_2/images'
ARCH=ROOT/'test_2.zip'
OUT=Path('/home/jhy/rare_diag_20260915/round2_20260921')

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 names=sorted((p.name for p in IMG.glob('*.png')),key=lambda n:int(Path(n).stem.split('_')[-1]))
 exp=[f'test2_{i}.png' for i in range(1,1301)]
 assert IMG.is_dir() and ARCH.is_file() and names==exp
 rows=[]
 for i,n in enumerate(names,1):
  p=IMG/n
  rows.append({'filename':n,'bytes':p.stat().st_size,'sha256':sha(p)})
  if i%200==0: print(f'{i}/1300',flush=True)
 samples={}
 for i in [1,2,100,650,1299,1300]:
  p=IMG/f'test2_{i}.png'
  with Image.open(p) as im:
   samples[p.name]={'size':list(im.size),'mode':im.mode,'format':im.format}
   im.verify()
 report={'image_dir':str(IMG),'archive':str(ARCH),'archive_bytes':ARCH.stat().st_size,
         'archive_sha256':sha(ARCH),'png_count':len(names),'continuous_1_1300':names==exp,
         'samples':samples,'file_manifest':'test2_files_sha256.json'}
 (OUT/'test2_files_sha256.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
 (OUT/'TEST2_INTEGRITY.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report,indent=2))
if __name__=='__main__': main()
