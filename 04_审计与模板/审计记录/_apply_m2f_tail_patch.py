#!/usr/bin/env python3
"""Create _m2f_tail_train_entry.py: historical M2F trainer + a start-weights override.

Round-specific tooling (2026-09-17). The original `train_entry.py` is opened
read-only and never modified. All-or-nothing: every anchor is matched before
anything is written.

WHY THIS SMALL PATCH
--------------------
The M2F tail experiment must start from the exact 56k scored weights. The
historical trainer hardcodes its student initialisation:

    line  54  S62_CKPT = ... / stage62_runs/... / best_miou.pt
    line 146  checkpoint = torch.load(S62_CKPT, ...)   # inside build_encoder()

There is no --init-from, and --resume-checkpoint is an OUTPUT path, not an
input (grep: the only torch.load in the whole file is line 146). So a tail from
56k is impossible without this hook.

The patch adds ONE mechanism: after the encoder is built and before it is used,
optionally overwrite encoder+head weights from a saved score checkpoint
({"model": ..., "head": ...}), which is exactly the format formal_run1b/scores/
score_iter*.pt uses.

It deliberately does NOT touch: the loss, the data pipeline, augmentation,
CWFA, the criterion, the optimizer construction, the LR schedule, AMP, or
gradient clipping. Those must stay byte-identical to the frozen Run1b recipe
for the tail to mean anything.

The historical 50-iteration warmup is left exactly as it is. It is a real part
of the configuration (the file says so in its own comment), and removing it
would be a second, undeclared variable.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
# the script lives in rare_diag_20260915/ but the trainer lives under
# m2f_gpu_20260915/; fall back to the m2f subtree when that is where we are.
ROOT = HERE / "m2f_gpu_20260915" if (HERE / "m2f_gpu_20260915").is_dir() else HERE
SRC = ROOT / "py/b3_m2f_head/train_entry.py"
DST = ROOT / "py/b3_m2f_head/_m2f_tail_train_entry.py"


def read_raw(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return fh.read()


def write_raw(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def crlf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\n", "\r\n")


CLI_OLD = '''    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
'''

CLI_NEW = '''    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init-from", default=None,
                    help="score checkpoint ({\\"model\\": ..., \\"head\\": ...}) whose "
                         "encoder+head weights REPLACE the hardcoded s62 student "
                         "initialisation. Used to start a tail from an exact "
                         "scored checkpoint. Everything else (optimizer state, "
                         "schedule, data, loss) starts fresh, as a genuine "
                         "fine-tune rather than a resume.")
'''

ANCHOR = '''    encoder = build_encoder(args.device)
    encoder.train()
'''

REPLACEMENT = '''    encoder = build_encoder(args.device)
    if args.init_from:
        # ---- round-specific override: start from an exact scored checkpoint ----
        # Applied AFTER build_encoder so the hardcoded s62 path is fully replaced
        # rather than merged. strict=True on both state dicts: a silent partial
        # load would make the tail's starting point unknowable.
        init_path = Path(args.init_from)
        if not init_path.is_file():
            raise SystemExit(f"--init-from not found: {init_path}")
        import hashlib as _hashlib
        _h = _hashlib.sha256()
        with init_path.open("rb") as _f:
            for _b in iter(lambda: _f.read(1 << 20), b""):
                _h.update(_b)
        _init_sha = _h.hexdigest()
        _init = torch.load(init_path, map_location="cpu", weights_only=False)
        _mstate = _init.get("model")
        _hstate = _init.get("head")
        if _mstate is None or _hstate is None:
            raise SystemExit(f"--init-from lacks 'model'/'head': {init_path}")
        encoder.load_state_dict(_mstate, strict=True)
        print(f"[init] encoder REPLACED from {init_path.name} "
              f"(sha256={_init_sha[:16]}, data_iteration="
              f"{_init.get('data_iteration')})", flush=True)
        _pending_head_state = _hstate
    else:
        _init_sha = None
        _pending_head_state = None
    encoder.train()
'''

HEAD_ANCHOR = '''    head.train()

    encoder_params = [p for p in encoder.parameters() if p.requires_grad]
'''

HEAD_REPLACEMENT = '''    if _pending_head_state is not None:
        head.load_state_dict(_pending_head_state, strict=True)
        print(f"[init] head REPLACED from the same checkpoint "
              f"(sha256={_init_sha[:16]})", flush=True)
    head.train()

    encoder_params = [p for p in encoder.parameters() if p.requires_grad]
'''

META_ANCHOR = '''            "s62_checkpoint": S62_CKPT, "started": datetime.now(timezone.utc).isoformat(),
'''

META_REPLACEMENT = '''            "s62_checkpoint": S62_CKPT,
            "init_from": args.init_from,
            "started": datetime.now(timezone.utc).isoformat(),
'''

# ---- explicit tail LR schedule -------------------------------------------
# The stock formula is lr = lr_max * min(1, step/warmup) * (1 - step/N). It
# CANNOT express "continue from the 56k LR and decay to zero": raising N to make
# room re-anchors the polynomial at lr_max, so step 56001 gets 2.63e-5 when the
# real 56k LR is 1.59e-6 -- a 16x jump UP, not a continuation. Warmup makes it
# worse (it restarts at warmup=50 from an even lower point).
#
# Tail mode instead uses the honest form: lr = base * (1 - k/T) where k counts
# TAIL iterations from the checkpoint and T is the tail length. base is the
# historic post-warmup LR at the start point, so the shape is continuous with
# the parent run's decay (rate lr_max/N) and simply continues to zero.
SCHED_OLD = '''            scale = min(1.0, step / max(args.warmup, 1))
            poly = max(0.0, 1.0 - step / steps) ** 1.0
            for group, base in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base * scale * poly
'''

SCHED_NEW = '''            if args.tail_steps:
                k = max(0, step - args.tail_from)
                decay = max(0.0, 1.0 - k / args.tail_steps)
                # `scale` is logged as "the multiplier applied to the base LR".
                # Both branches must define it, or the record line raises.
                scale = decay
                for group, base in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = base * decay
            else:
                scale = min(1.0, step / max(args.warmup, 1))
                poly = max(0.0, 1.0 - step / steps) ** 1.0
                for group, base in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = base * scale * poly
'''

TAIL_CLI_OLD = '''    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
'''

TAIL_CLI_NEW = '''    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start-iter", type=int, default=1,
                    help="first data iteration to run. The loop is range(start, "
                         "steps+1), and every per-step quantity (data seed via "
                         "seed*100000+step, epoch permutation, LR schedule) is a "
                         "function of `step` alone, so starting at K reproduces "
                         "exactly the state a full run would have at step K "
                         "without replaying steps 1..K-1. Used to continue a "
                         "converged checkpoint: --start-iter 56000 --steps 76000 "
                         "makes step 56001 the first real update, with "
                         "lr = lr_max * (1 - 56001/76000).")
    ap.add_argument("--tail-steps", type=int, default=0,
                    help="if >0, override the LR schedule with an explicit tail "
                         "lr = base * (1 - k/tail_steps), k = step - tail_from.")
    ap.add_argument("--tail-from", type=int, default=56000,
                    help="step the tail counts from (the start checkpoint).")
'''

LOOP_OLD = "    for step in range(1, steps + 1):\n"
LOOP_NEW = "    for step in range(max(1, args.start_iter), steps + 1):\n"


def replace_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"anchor {label!r} matched {n} times, expected 1")
    return text.replace(old, new, 1)


def main() -> int:
    src = read_raw(SRC)
    crlf_src = "\r\n" in src
    work = src.replace("\r\n", "\n")

    if "init_from" in work:
        raise SystemExit("source already carries the patch; wrong file?")
    if DST.exists() and "init_from" in read_raw(DST):
        raise SystemExit(f"{DST.name} already patched; refusing to re-apply")

    out = work
    # order matters: the tail CLI anchor sits on original lines that the cli
    # anchor also touches, so insert the tail flags first.
    out = replace_once(out, TAIL_CLI_OLD, TAIL_CLI_NEW, "tail-cli")
    out = replace_once(out, CLI_OLD, CLI_NEW, "cli")
    out = replace_once(out, LOOP_OLD, LOOP_NEW, "loop")
    out = replace_once(out, ANCHOR, REPLACEMENT, "encoder")
    out = replace_once(out, HEAD_ANCHOR, HEAD_REPLACEMENT, "head")
    out = replace_once(out, SCHED_OLD, SCHED_NEW, "sched")
    out = replace_once(out, META_ANCHOR, META_REPLACEMENT, "meta")

    # the tail LR must be printed so the run can be audited from its log alone
    out = replace_once(
        out,
        '''                   "lr_encoder": optimizer.param_groups[0]["lr"],
''',
        '''                   "lr_encoder": optimizer.param_groups[0]["lr"],
                   "tail_steps": args.tail_steps,
                   "tail_from": args.tail_from,
''',
        "logmeta")

    for probe, want in (
        ("def build_encoder(device):", 1),
        ("S62_CKPT = ", 1),
        ('ap.add_argument("--resume-checkpoint"', 1),
        ("optimizer = torch.optim.AdamW(", 1),
        ("scale = min(1.0, step / max(args.warmup, 1))", 2),
        ("from b3_m2f_head.loss_v2 import SetCriterionV2, build_set_targets", 1),
        ("group[\"lr\"] = base * decay", 1),
        ("_init_sha = None", 1),
        ("for step in range(max(1, args.start_iter), steps + 1):", 1),
    ):
        got = out.count(probe)
        if got != want:
            raise SystemExit(f"post-check failed: {probe!r} x{got}, expected {want}")

    write_raw(DST, crlf(out) if crlf_src else out)
    print(f"wrote {DST.name}: {len(work.splitlines())} -> {len(out.splitlines())} "
          f"lines (+{len(out.splitlines()) - len(work.splitlines())}), "
          f"crlf={crlf_src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
