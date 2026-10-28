#!/usr/bin/env python3
"""Pack multi-scale TTA logits into test2 submission ZIP."""
import json, sys, time
from pathlib import Path
import numpy as np

SCORES = Path('/home/jhy/rare_diag_20260915/mstta_alpha050_20260925/scores')
OUTDIR = Path('/home/jhy/rare_diag_20260915/mstta_alpha050_20260925/package')
BSTAR = np.array([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], dtype=np.float32)

def main():
    import torch, torch.nn.functional as F
    from PIL import Image
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pred_dir = OUTDIR / 'preds'
    pred_dir.mkdir(exist_ok=True)

    names = [f'test2_{i}.png' for i in range(1, 1301)]
    t0 = time.time()
    for i, name in enumerate(names, 1):
        stem = name.replace('.png', '')
        logits = np.load(SCORES / f'{stem}.npy').astype(np.float32)  # [8,256,256]
        z = torch.from_numpy(logits).unsqueeze(0)  # [1,8,256,256]
        z = z + torch.from_numpy(BSTAR).view(1, 8, 1, 1)
        z = F.interpolate(z, size=(1024, 1024), mode='bilinear', align_corners=False)
        pred = z.argmax(1)[0].cpu().numpy().astype(np.uint8) + 1  # 1..8
        Image.fromarray(pred, mode='L').save(pred_dir / name)
        if i % 200 == 0:
            print(f'{i}/1300 {time.time()-t0:.0f}s', flush=True)

    # ZIP
    import zipfile
    zip_path = OUTDIR / 'MSTTA_alpha050_test2.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
        for name in names:
            zf.write(pred_dir / name, name)

    # SHA256
    import hashlib
    h = hashlib.sha256()
    with open(zip_path, 'rb') as f:
        for b in iter(lambda: f.read(1<<20), b''):
            h.update(b)

    # Validate
    with zipfile.ZipFile(zip_path, 'r') as zf:
        flist = zf.namelist()
    assert len(flist) == 1300, f'expected 1300, got {len(flist)}'
    assert flist == names, 'name mismatch'

    print(f'ZIP: {zip_path}')
    print(f'SHA256: {h.hexdigest()}')
    print(f'Files: {len(flist)}')
    print(f'Time: {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
