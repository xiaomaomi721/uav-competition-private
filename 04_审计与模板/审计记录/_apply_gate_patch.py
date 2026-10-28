#!/usr/bin/env python3
"""Insert the minimal GPU-offload gate into a copy of the historical _kd_train.py.

Round-specific tooling. Writes _kd_train_soup_gate.py from _kd_train.py; the
original file is opened read-only and never modified. All-or-nothing: every
anchor is matched before anything is written to disk.

NOTE ON LINE ENDINGS: _kd_train.py is UTF-8 with CRLF line endings (525 CRLF,
0 bare LF). It is read and written with newline="" so the bytes survive
untouched; every pattern below is CRLF-normalised via crlf() before matching.
Without that, a pattern containing a bare "\n" matches zero times and the
patch silently does nothing.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "_kd_train.py"
DST = HERE / "_kd_train_soup_gate.py"


def crlf(text: str) -> str:
    """Normalise newlines to CRLF (idempotent)."""
    return text.replace("\r\n", "\n").replace("\n", "\r\n")


HELPERS = '''
# --------------------------------------------------------------------------
# GPU-offload gate (round-specific addition; see _kd_train_soup_gate.py)
# --------------------------------------------------------------------------
# The 3200-step gate has to run D4-TTA8 on the SAME physical GPU2 while this
# training process stays alive (same process => same DataLoader / RNG stream).
# Keeping the model and the AdamW moments resident would not fit alongside the
# TTA8 process, so at the gate both are parked in host memory, the caching
# allocator is flushed, and this process polls for a sentinel file.
#
# Deliberately NOT implemented: resume-from-checkpoint, loader replay,
# generator replay, any checkpoint-restore framework. The gate is only a
# temporary device migration inside one uninterrupted process.
def _optimizer_state_to_device(optimizer, device) -> int:
    """Move every tensor in the AdamW state (exp_avg / exp_avg_sq / tensor step).

    `step` is a 0-dim tensor under AdamW, so it is migrated too; nothing is
    re-created, no param group is touched, no new optimizer is built.

    torch is imported HERE on purpose: _kd_train.py defers its torch imports
    into main() (see assert_gpu), so there is no module-level `torch` to lean on.
    """
    import torch
    moved = 0
    for group in optimizer.param_groups:
        for prm in group["params"]:
            st = optimizer.state.get(prm)
            if not st:
                continue
            for k, v in st.items():
                if isinstance(v, torch.Tensor):
                    st[k] = v.to(device)
                    moved += 1
    return moved


def _wait_for_gate(flag_dir, flag_stem, poll_seconds=10.0) -> str:
    """Block until <flag_dir>/<flag_stem>_{CONTINUE,STOP} appears.

    No loader is advanced and no optimizer/scheduler call is made while
    waiting: the process is parked exactly where it was after the 3200th
    optimizer update.
    """
    go = flag_dir / f"{flag_stem}_CONTINUE"
    stop = flag_dir / f"{flag_stem}_STOP"
    while True:
        if stop.is_file():
            return "STOP"
        if go.is_file():
            return "CONTINUE"
        time.sleep(poll_seconds)

'''

CLI_OLD = '''    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
'''

CLI_NEW = '''    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--offload-at", type=int, default=0,
                    help="after this many optimizer updates: save the gate "
                         "checkpoint, park the model and the AdamW state in host "
                         "memory, and poll for <outdir>/gate_{CONTINUE,STOP}. "
                         "0 disables the gate entirely.")
    args = ap.parse_args()
'''

META_OLD = '        "grad_accum": args.grad_accum,\n'

META_NEW = ('        "grad_accum": args.grad_accum,\n'
            '        "offload_at": args.offload_at,\n'
            '        "gate_flag": "outdir/gate_{CONTINUE,STOP}",\n')

LOOP_OLD = "    for step in range(max_steps):\n"

LOOP_NEW = ("    gate_stopped = False\n"
            "    for step in range(max_steps):\n")

GATE_ANCHOR = "        if step % 50 == 0 or step == max_steps - 1:\n"

GATE = '''        if args.offload_at and (step + 1) == int(args.offload_at):
            # ---- gate: park this process off the GPU, keep the process alive --
            # Reached only after the 3200th optimizer update, the scaler update
            # and the LR step have all returned: optimizer, scheduler, scaler,
            # both loaders and every RNG stream stay exactly as they are. The
            # only thing that changes is where the parameters and the AdamW
            # moments live.
            gp = outdir / f"checkpoint_step{step+1}_gate.pt"
            torch.save({"model": model.state_dict(), "config": config,
                        "step": step + 1, "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()}, gp)
            gate_lr = scheduler.get_last_lr()[0]
            gate_scale = scaler.get_scale()
            gate_alloc = torch.cuda.memory_allocated() / 2**20
            gate_reserved = torch.cuda.memory_reserved() / 2**20
            print(f"[gate] step={step+1} saved {gp.name} "
                  f"lr={gate_lr:.6e} scaler_scale={gate_scale:.1f} "
                  f"allocated={gate_alloc:.1f}MiB reserved={gate_reserved:.1f}MiB",
                  flush=True)

            model.to("cpu")
            n_moved = _optimizer_state_to_device(optimizer, torch.device("cpu"))
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            gate_free = torch.cuda.mem_get_info()[0] / 2**20
            print(f"[gate] OFFLOADED moved {n_moved} optimizer tensors to cpu; "
                  f"cuda allocated={torch.cuda.memory_allocated()/2**20:.1f}MiB "
                  f"reserved={torch.cuda.memory_reserved()/2**20:.1f}MiB "
                  f"device_free={gate_free:.1f}MiB", flush=True)
            print(f"[gate] waiting for {outdir}/gate_CONTINUE or gate_STOP",
                  flush=True)

            verdict = _wait_for_gate(outdir, "gate")
            print(f"[gate] verdict={verdict}", flush=True)

            if verdict == "STOP":
                # Clean exit. The optimizer state is already inside the gate
                # checkpoint, and no optimizer.step() runs after step 3200.
                gate_stopped = True
                break

            import gc
            gc.collect()
            model.to(device)
            n_back = _optimizer_state_to_device(optimizer, device)
            print(f"[gate] RELOADED moved {n_back} optimizer tensors to {device}; "
                  f"lr={scheduler.get_last_lr()[0]:.6e} "
                  f"scaler_scale={scaler.get_scale():.1f} -> resuming at "
                  f"step {step+2}", flush=True)
            continue

'''

COMPLETED_OLD = '''    (outdir / "COMPLETED.json").write_text(json.dumps({
        "status": "completed", "steps": max_steps, "history": history,
        "finished": datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
'''

COMPLETED_NEW = '''    (outdir / "COMPLETED.json").write_text(json.dumps({
        "status": "gate_stop" if gate_stopped else "completed",
        "steps": max_steps, "history": history,
        "finished": datetime.now(timezone.utc).isoformat(),
        "offload_at": args.offload_at,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"anchor {label!r} matched {n} times, expected 1")
    return text.replace(old, new, 1)


def read_raw(path: Path) -> str:
    """Read text without newline translation (Path.read_text has no `newline`)."""
    with path.open("r", encoding="utf-8", newline="") as fh:
        return fh.read()


def write_raw(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def main() -> int:
    # newline="" keeps the CRLF bytes exactly as they are on disk
    src = read_raw(SRC)
    if "\r\n" not in src:
        raise SystemExit("expected CRLF source; refusing to guess")
    src = src.replace("\r\n", "\n")   # work in LF, re-expand at the end

    if "offload_at" in src or "_wait_for_gate" in src:
        raise SystemExit("_kd_train.py already carries the gate; wrong source?")
    if DST.exists() and "_wait_for_gate" in read_raw(DST):
        raise SystemExit(f"{DST.name} already patched; refusing to re-apply")

    out = src

    # 1. module-level helpers, directly above the "# data" banner. Both the
    #    banner dash line and its preceding newline are sliced out of the file
    #    itself, so the dash count is never assumed.
    marker = "\n# data\n"
    if out.count(marker) != 1:
        raise SystemExit(f'"# data" banner matched {out.count(marker)} times')
    i = out.index(marker)
    j = out.rindex("\n", 0, i)
    data_anchor = out[j + 1:i + len(marker)]          # "# ------...\n# data\n"
    out = out[:j + 1] + HELPERS.lstrip("\n") + data_anchor + out[i + len(marker):]

    # 2. --offload-at CLI flag
    out = replace_once(out, CLI_OLD, CLI_NEW, "cli")

    # 3. record it in KD_META.json
    out = replace_once(out, META_OLD, META_NEW, "meta")

    # 4. gate_stopped flag next to the step loop
    out = replace_once(out, LOOP_OLD, LOOP_NEW, "loopflag")

    # 5. the gate block itself, after the LR step and before the 50-step log
    out = replace_once(out, GATE_ANCHOR, GATE + GATE_ANCHOR, "gate")

    # 6. COMPLETED.json reports how the process ended
    out = replace_once(out, COMPLETED_OLD, COMPLETED_NEW, "completed")

    # Post-checks: the training logic these lines belong to must be untouched,
    # and the gate must sit after the optimizer/scheduler update, not before.
    for probe, want in (('ap.add_argument("--device", default="cuda:0")', 1),
                        ("\n        scheduler.step()", 1),
                        ("\n        scaler.update()", 1),
                        # 1 creation + 1 epoch-boundary recreation, both original
                        ("real_iter = iter(real_loader)", 2),
                        ("test_iter = iter(test_loader)", 2),
                        ("scaler.scale(total / accum).backward()", 1),
                        ("_wait_for_gate(outdir, \"gate\")", 1)):
        got = out.count(probe)
        if got != want:
            raise SystemExit(f"post-check failed: {probe!r} x{got}, expected {want}")

    if (out.index("\n        scheduler.step()")
            > out.index("if args.offload_at and (step + 1)")):
        raise SystemExit("gate block landed before the LR step; refusing")

    # restore the original CRLF endings
    write_raw(DST, crlf(out))
    print(f"wrote {DST.name}: {len(src.splitlines())} -> {len(out.splitlines())} "
          f"lines (+{len(out.splitlines()) - len(src.splitlines())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
