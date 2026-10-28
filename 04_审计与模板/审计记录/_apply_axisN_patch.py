#!/usr/bin/env python3
"""Axis N patch applier: unsupervised target-domain self-consistency.

Creates, from the read-only P1 surgical trainer:

    src/_axisN_trainer.py       (all-or-nothing anchored insertions)
    scripts/_axisN_train.py     (entry, same monkey-patch shim pattern)

WHAT AXIS N IS
--------------
The whole project has never done unsupervised target-domain adaptation
(verified 2026-09-18: 熵最小化 / 一致性正则 / AdaBN / MMD / adversarial /
FixMatch / 自监督 all have ZERO occurrences in the repository). Axis N adds
exactly one information channel:

    L = CE(f(x_source), y_source)  +  lambda * consistency(f, x_target)

where x_target are the 500 official test1 images, used WITHOUT labels and
WITHOUT pseudo-labels. The consistency term is D4 self-consistency: predict the
image, predict its 180-degree rotation, rotate the second prediction back, and
require the two softmax fields to agree (symmetric KL).

This copies NO teacher decision, so unlike pseudo-labelling it cannot inherit
the champion's systematic errors. The semantic anchor is the source CE, which
also prevents the degenerate all-one-class solution that plain entropy
minimisation suffers from.

NOT touched: loss (CE), criterion construction, augmentation, CWFA, AMP,
gradient clipping, optimizer params, LR schedule, the P1 surgical machinery.

Insertions (each must match exactly once):
    N1  helper class + consistency fn            before `def train_once(`
    N2  unlabelled target DataLoader             before `val_loader = DataLoader(`
    N3  target iterator                          after  `iterator = iter(train_loader)`
    N4  consistency term in the step             before the non-finite guard
"""
from __future__ import annotations

from pathlib import Path

E = Path("/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1")
SRC = E / "src/_p1_surgical_trainer.py"
DST = E / "src/_axisN_trainer.py"
SRC_ENTRY = E / "scripts/_p1_surgical_train.py"
DST_ENTRY = E / "scripts/_axisN_train.py"


def read(p: Path) -> str:
    with p.open("r", encoding="utf-8", newline="") as fh:
        return fh.read()


def write(p: Path, t: str) -> None:
    with p.open("w", encoding="utf-8", newline="") as fh:
        fh.write(t)


N1 = '''# ===================== AXIS N HELPERS (unsupervised target consistency) =====================
class _AxisNTargetDataset:
    """Unlabelled target images. Yields the normalised image tensor only.

    No labels and no pseudo-labels are read anywhere in this path: the only
    signal taken from the target set is self-consistency of the model's own
    predictions under a D4 transform.
    """

    def __init__(self, images_dir, max_images: int = 0):
        import numpy as _np
        from PIL import Image as _Image

        self._np = _np
        self._Image = _Image
        self.files = sorted(images_dir.glob("*.png"),
                            key=lambda p: int(p.stem.split("_")[-1]))
        if max_images > 0:
            self.files = self.files[:max_images]
        if not self.files:
            raise SystemExit(f"AxisN: no target images in {images_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        from .transforms import normalize_to_tensor
        with self._Image.open(self.files[index]) as im:
            arr = self._np.asarray(im.convert("RGB")).copy()
        return normalize_to_tensor(arr)


def _axisN_consistency(logits_a: "torch.Tensor", logits_b: "torch.Tensor") -> "torch.Tensor":
    """Symmetric KL between two softmax fields, averaged over pixels.

    Purely unsupervised: no labels, no teacher, no pseudo-labels. The source CE
    is what keeps this from collapsing to a constant prediction.
    """
    pa = F.softmax(logits_a.float(), dim=1).clamp_min(1e-6)
    pb = F.softmax(logits_b.float(), dim=1).clamp_min(1e-6)
    kl_ab = (pa * (pa.log() - pb.log())).sum(dim=1)
    kl_ba = (pb * (pb.log() - pa.log())).sum(dim=1)
    return 0.5 * (kl_ab.mean() + kl_ba.mean())
# ==========================================================================================


'''


N2 = '''    # ---- AXIS N: unlabelled target loader (consistency only, no labels) ----
    _axn_cfg = config.get("target_consistency", {})
    axisN_on = bool(_axn_cfg.get("enabled", False))
    axisN_lambda = float(_axn_cfg.get("lambda", 1.0))
    axisN_every = max(1, int(_axn_cfg.get("every", 1)))
    axisN_last = None
    target_loader = None
    if axisN_on:
        _axn_dir = Path(str(_axn_cfg.get("images_dir", "")))
        if not _axn_dir.is_dir():
            raise SystemExit(f"AxisN: target images_dir not found: {_axn_dir}")
        _axn_ds = _AxisNTargetDataset(_axn_dir,
                                      max_images=int(_axn_cfg.get("max_images", 0)))
        target_loader = DataLoader(
            _axn_ds,
            batch_size=int(_axn_cfg.get("batch_size", batch_size)),
            shuffle=True,
            num_workers=0,
            drop_last=False,
            generator=torch.Generator().manual_seed(seed + 20_001),
        )
        print(f"AXIS N enabled: target_images={len(_axn_ds)} lambda={axisN_lambda} "
              f"every={axisN_every} images_dir={_axn_dir}", flush=True)
    # -----------------------------------------------------------------------
'''

N3 = '''        iterator = iter(train_loader)
    if axisN_on:
        target_iterator = iter(target_loader)
'''

N4 = '''                if axisN_on and (step % axisN_every == 0):
                    # Unsupervised target self-consistency: rotate 180, predict,
                    # rotate the prediction back, and require agreement. No
                    # labels or pseudo-labels are involved anywhere here.
                    try:
                        _t = next(target_iterator)
                    except StopIteration:
                        target_iterator = iter(target_loader)
                        _t = next(target_iterator)
                    _t = _t.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        _la = model(pixel_values=_t).logits
                        _lb = model(pixel_values=torch.rot90(_t, 2, dims=(2, 3))).logits
                    _lb = torch.rot90(_lb, 2, dims=(2, 3))
                    if _la.shape[-2:] != _lb.shape[-2:]:
                        _lb = F.interpolate(_lb, size=_la.shape[-2:], mode="bilinear",
                                            align_corners=False)
                    _cons = _axisN_consistency(_la, _lb)
                    loss = loss + (axisN_lambda * _cons) / accumulation
                    axisN_last = float(_cons.detach())
'''

SUBS = [
    ("helpers", "def train_once(", N1 + "def train_once(", 1),
    ("target_loader", "    val_loader = DataLoader(\n", N2 + "    val_loader = DataLoader(\n", 1),
    # `iterator = iter(train_loader)` also appears in the epoch-refill branch, so
    # anchor on the initialisation site only (it is followed by step_times=[]).
    ("target_iter", "        iterator = iter(train_loader)\n    step_times: list[float] = []\n",
     N3.replace("        iterator = iter(train_loader)\n",
                "        iterator = iter(train_loader)\n    step_times: list[float] = []\n"), 1),
    ("consistency", "                if not torch.isfinite(loss):\n", N4 + "                if not torch.isfinite(loss):\n", 1),
]


def main() -> int:
    for p in (SRC, SRC_ENTRY):
        if not p.is_file():
            raise SystemExit(f"missing source: {p}")
    if DST.exists() or DST_ENTRY.exists():
        raise SystemExit(f"refusing to overwrite {DST} / {DST_ENTRY}")

    text = read(SRC)
    report = []
    for name, old, new, want in SUBS:
        got = text.count(old)
        if got != want:
            raise SystemExit(f"ANCHOR MISMATCH [{name}]: want {want}, found {got}. Nothing written.")
        text = text.replace(old, new, want)
        report.append((name, got))
    write(DST, text)
    print(f"wrote {DST} ({len(text)} bytes)")

    entry = read(SRC_ENTRY)
    mk = 'importlib.import_module("src._p1_surgical_trainer")'
    if entry.count(mk) != 1:
        raise SystemExit("entry anchor mismatch")
    write(DST_ENTRY, entry.replace(mk, 'importlib.import_module("src._axisN_trainer")'))
    print(f"wrote {DST_ENTRY}")

    for n, c in report:
        print(f"  {n:16s} matched x{c}")
    print("ALL ANCHORS OK, originals untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
