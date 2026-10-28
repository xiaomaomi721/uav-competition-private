#!/usr/bin/env python3
"""P1 SURGICAL patch applier.

Creates, from READ-ONLY originals, two additive artifacts:

    src/_p1_surgical_trainer.py     = trainer_formal_semeda_cwfa.py + P1 patches
    scripts/_p1_surgical_train.py   = probe/train_formal_variant.py  + P1 handoff

The originals are opened read-only and never modified.  All-or-nothing: every
anchor must match EXACTLY ONCE before anything is written.

WHY THESE FIVE PATCHES
----------------------
1. init-from        -- the historical trainer has no way to start from an
                       existing checkpoint (only --resume, which restores
                       optimizer/scheduler too).  P1 must start from exact
                       weights with a FRESH optimizer.
2. trainable scope  -- freeze everything outside segformer.stages.2.* and build
                       the optimizer only over the trainable set.
3. frozen-module eval -- scoped-out leaf modules go to eval() so their BN uses
                       frozen running stats; plus a hard per-step buffer restore
                       so nothing can move them behind our back.
4. gate checkpoints -- the historical `if step in (44000,48000,52000,56000)`
                       save writes WEIGHT-ONLY checkpoints (no optimizer, no
                       scheduler), so they cannot resume the trajectory.
                       P1 requires seamless 4k->8k->16k->32k continuation, so the
                       gate save must carry optimizer/scheduler/scaler state.
5. gate validation  -- run validation exactly at the gate steps.

NOT touched: loss, criterion, data pipeline, augmentation, CWFA, AMP,
gradient clipping, optimizer construction parameters, or the LR schedule.
"""
from __future__ import annotations

from pathlib import Path

E = Path("/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1")
SRC_TRAINER = E / "src/trainer_formal_semeda_cwfa.py"
DST_TRAINER = E / "src/_p1_surgical_trainer.py"
SRC_ENTRY = E / "scripts/probe/train_formal_variant.py"
DST_ENTRY = E / "scripts/_p1_surgical_train.py"


def read(p: Path) -> str:
    with p.open("r", encoding="utf-8", newline="") as fh:
        return fh.read()


def write(p: Path, text: str) -> None:
    with p.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)


HELPERS = '''# ===================== P1 SURGICAL HELPERS (additive) =====================
def _p1_freeze_scope(model, scope_prefixes):
    """requires_grad only inside scope_prefixes; return the trainable params."""
    keep, frozen = [], []
    for name, param in model.named_parameters():
        in_scope = any(name.startswith(p) for p in scope_prefixes)
        param.requires_grad_(in_scope)
        (keep if in_scope else frozen).append(name)
    if not keep:
        raise RuntimeError(f"P1: trainable scope matched no parameters: {scope_prefixes}")
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_t = sum(p.numel() for p in trainable)
    n_f = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"P1 SURGICAL scope={scope_prefixes} trainable_tensors={len(keep)} "
          f"trainable_params={n_t} frozen_params={n_f} frozen_tensors={len(frozen)}",
          flush=True)
    return trainable


def _p1_frozen_buffer_snapshot(model, scope_prefixes):
    """Every buffer outside the scope is frozen: snapshot so it can be restored."""
    snap = {}
    for name, buf in model.named_buffers():
        if not any(name.startswith(p) for p in scope_prefixes):
            snap[name] = buf.detach().clone()
    n_rt = sum(1 for k in snap if k.endswith("num_batches_tracked"))
    print(f"P1 SURGICAL frozen buffers snapshotted: {len(snap)} "
          f"(num_batches_tracked={n_rt})", flush=True)
    return snap


def _p1_restore_frozen_buffers(model, snap):
    """Hard guarantee: a frozen module's state can never move, whatever the
    forward pass or the mode bookkeeping does."""
    if not snap:
        return
    with torch.no_grad():
        for name, buf in model.named_buffers():
            ref = snap.get(name)
            if ref is not None and buf.shape == ref.shape and not torch.equal(buf, ref):
                buf.copy_(ref)


def _p1_apply_frozen_eval(model, scope_prefixes):
    """Leaf modules outside the scope -> eval(), so their BN normalises with the
    frozen running stats instead of batch statistics.  In-scope modules stay in
    train()."""
    for name, module in model.named_modules():
        if name == "" or list(module.children()):
            continue
        if not any(name.startswith(p) for p in scope_prefixes):
            module.eval()
# ==========================================================================


'''


A_BUILD_OLD = '''    model = build_model(config).to(device)
'''
A_BUILD_NEW = '''    model = build_model(config).to(device)
    # ===================== P1 SURGICAL: init-from (weights only) =====================
    _p1_init_from = config.get("p1_init_from")
    if _p1_init_from and resume is None:
        import hashlib as _p1_hashlib
        _p1_init_path = Path(str(_p1_init_from))
        if not _p1_init_path.is_file():
            raise FileNotFoundError(f"P1 p1_init_from not found: {_p1_init_path}")
        _p1_ckpt = torch.load(_p1_init_path, map_location="cpu", weights_only=False)
        _p1_state = None
        if isinstance(_p1_ckpt, dict):
            for _k in ("model", "model_state_dict", "state_dict"):
                _c = _p1_ckpt.get(_k)
                if (isinstance(_c, dict) and _c
                        and all(torch.is_tensor(v) for v in _c.values())):
                    _p1_state = _c
                    break
        if _p1_state is None:
            raise RuntimeError(f"P1 p1_init_from carries no model state: {_p1_init_path}")
        model.load_state_dict(_p1_state, strict=True)
        _p1_h = _p1_hashlib.sha256()
        with _p1_init_path.open("rb") as _p1_f:
            for _p1_b in iter(lambda: _p1_f.read(1 << 20), b""):
                _p1_h.update(_p1_b)
        print(f"P1 SURGICAL init_from={_p1_init_path} "
              f"sha256={_p1_h.hexdigest()[:16]} tensors={len(_p1_state)} "
              f"ckpt_step={_p1_ckpt.get('step') if isinstance(_p1_ckpt, dict) else None} "
              f"fresh_optimizer=True", flush=True)
        config["p1_init_from_resolved"] = {
            "path": str(_p1_init_path),
            "sha256": _p1_h.hexdigest(),
            "ckpt_step": (int(_p1_ckpt.get("step", -1))
                          if isinstance(_p1_ckpt, dict) else -1),
        }
        del _p1_ckpt, _p1_state
    elif _p1_init_from and resume is not None:
        print("P1 SURGICAL init_from SKIPPED because resume is set: continuing the "
              "trained trajectory rather than re-initialising", flush=True)
    # ================================================================================
'''

A_OPT_OLD = '''    optimizer_cfg = config.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg.get("learning_rate", 6.0e-5)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.01)),
    )
'''
A_OPT_NEW = '''    optimizer_cfg = config.get("optimizer", {})
    # ============ P1 SURGICAL: trainable scope + frozen-module protection ============
    p1_scope = [str(_s) for _s in (config.get("p1_trainable_scope") or [])]
    p1_frozen_buf = {}
    if p1_scope:
        _p1_trainable = _p1_freeze_scope(model, p1_scope)
        p1_frozen_buf = _p1_frozen_buffer_snapshot(model, p1_scope)
        _p1_apply_frozen_eval(model, p1_scope)
        optimizer = torch.optim.AdamW(
            _p1_trainable,
            lr=float(optimizer_cfg.get("learning_rate", 6.0e-5)),
            weight_decay=float(optimizer_cfg.get("weight_decay", 0.01)),
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimizer_cfg.get("learning_rate", 6.0e-5)),
            weight_decay=float(optimizer_cfg.get("weight_decay", 0.01)),
        )
    # =================================================================================
'''

A_TRAIN_OLD = '''    try:
        model.train()
        _ce_last = None
        _semeda_last = None
'''
A_TRAIN_NEW = '''    try:
        model.train()
        if p1_scope:
            _p1_apply_frozen_eval(model, p1_scope)
        _ce_last = None
        _semeda_last = None
'''

A_ZG_OLD = '''            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
'''
A_ZG_NEW = '''            optimizer.zero_grad(set_to_none=True)
            if p1_frozen_buf:
                _p1_restore_frozen_buffers(model, p1_frozen_buf)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
'''

A_GATE_OLD = '''            if step in (44000, 48000, 52000, 56000):
                torch.save(
                    _checkpoint_payload(
                        model, step, best_miou, config,
                        multi_resolution_selector_state=resolution_selector_state(),
                    ),
                    output_dir / f"checkpoint_step{step}.pt",
                )
'''
A_GATE_NEW = '''            if step in tuple(config.get("gate_steps", (44000, 48000, 52000, 56000))):
                torch.save(
                    _checkpoint_payload(
                        model, step, best_miou, config,
                        optimizer, scheduler, scaler,
                        optimizer_update_count=optimizer_update_count,
                        skipped_optimizer_steps=skipped_optimizer_steps,
                        multi_resolution_selector_state=resolution_selector_state(),
                    ),
                    output_dir / f"checkpoint_step{step}.pt",
                )
'''

A_VAL_OLD = '''            if validation_interval > 0 and (step % validation_interval == 0 or step == max_steps):
'''
A_VAL_NEW = '''            if ((validation_interval > 0 and step % validation_interval == 0)
                    or step == max_steps
                    or step in tuple(config.get("gate_steps", ()))):
'''

SUBSTITUTIONS = [
    ("helpers", "def train_once(", HELPERS + "def train_once(", 1),
    ("init_from", A_BUILD_OLD, A_BUILD_NEW, 1),
    ("optimizer_scope", A_OPT_OLD, A_OPT_NEW, 1),
    ("loop_train", A_TRAIN_OLD, A_TRAIN_NEW, 1),
    ("buffer_restore", A_ZG_OLD, A_ZG_NEW, 1),
    ("gate_ckpt", A_GATE_OLD, A_GATE_NEW, 1),
    ("gate_val", A_VAL_OLD, A_VAL_NEW, 1),
]


def main() -> int:
    for p in (SRC_TRAINER, SRC_ENTRY):
        if not p.is_file():
            raise SystemExit(f"missing read-only source: {p}")
    if DST_TRAINER.exists() or DST_ENTRY.exists():
        raise SystemExit(f"refusing to overwrite: {DST_TRAINER} / {DST_ENTRY}")

    text = read(SRC_TRAINER)
    report = []
    for name, old, new, want in SUBSTITUTIONS:
        got = text.count(old)
        if got != want:
            raise SystemExit(
                f"ANCHOR MISMATCH [{name}]: expected {want} occurrence(s), found {got}. "
                f"Nothing written.")
        text = text.replace(old, new, want)
        report.append((name, want))
    write(DST_TRAINER, text)
    print(f"wrote {DST_TRAINER} ({len(text)} bytes)")

    entry_src = read(SRC_ENTRY)
    marker = "import src.trainer as base  # noqa: E402\nvariant = importlib.import_module(\"src.trainer_formal_semeda_cwfa\")"
    if entry_src.count(marker) != 1:
        raise SystemExit("ANCHOR MISMATCH [entry variant import]. Nothing written.")
    entry = entry_src.replace(
        marker,
        "import src.trainer as base  # noqa: E402\n"
        "variant = importlib.import_module(\"src._p1_surgical_trainer\")")
    if "src._p1_surgical_trainer" not in entry:
        raise SystemExit("entry rewrite failed")
    write(DST_ENTRY, entry)
    print(f"wrote {DST_ENTRY} ({len(entry)} bytes)")

    print("\nanchor report:")
    for name, want in report:
        print(f"  {name:16s} matched x{want}")
    print("ALL ANCHORS OK, original files untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
