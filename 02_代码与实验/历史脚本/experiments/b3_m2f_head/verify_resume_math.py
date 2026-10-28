"""Verify the resume patch reproduces the UNINTERRUPTED LR schedule bit-for-bit.

The claim being tested: parking at step P and resuming must give exactly the
same lr at every subsequent optimizer step as a single uninterrupted run would.
If this fails, --save-full-at/--resume-from is not usable for the gate plan.
"""
import torch

LR = 3.0e-5
TOTAL = 8000
PARK = 3200


def make():
    p = torch.nn.Parameter(torch.zeros(1))
    o = torch.optim.AdamW([p], lr=LR, weight_decay=0.01)
    s = torch.optim.lr_scheduler.LambdaLR(
        o, lr_lambda=lambda st: max(0.0, 1.0 - min(st, TOTAL) / TOTAL))
    return p, o, s


# ---- reference: one uninterrupted run, record lr AFTER each optimizer step ----
p, o, s = make()
ref = []
for step in range(6400):
    o.step()
    s.step()
    ref.append(s.get_last_lr()[0])

# ---- interrupted: run to PARK, save, rebuild, load, continue ----
p, o, s = make()
for step in range(PARK):
    o.step()
    s.step()
saved = {"optimizer": o.state_dict(), "scheduler": s.state_dict()}

p2, o2, s2 = make()
o2.load_state_dict(saved["optimizer"])
s2.load_state_dict(saved["scheduler"])

print(f"last_epoch after load      = {s2.last_epoch}   (expect {PARK})")
print(f"lr after load              = {s2.get_last_lr()[0]:.10e}")
print(f"lr at end of uninterrupted = {ref[PARK-1]:.10e}")
# _LRScheduler.step() bumps last_epoch THEN takes the lr, so after N calls
# last_epoch == N and the lr already reflects lambda(N).
assert s2.last_epoch == PARK, "last_epoch not restored"
assert s2.get_last_lr()[0] == ref[PARK - 1], "lr after load differs"

res = []
for step in range(PARK, 6400):
    o2.step()
    s2.step()
    res.append(s2.get_last_lr()[0])

bad = [(i + PARK, a, b) for i, (a, b) in enumerate(zip(res, ref[PARK:])) if a != b]
print(f"\nsteps compared             = {len(res)}")
print(f"lr mismatches              = {len(bad)}")
if bad:
    for i, a, b in bad[:5]:
        print(f"  step {i}: resumed {a:.10e} vs uninterrupted {b:.10e}")
    raise SystemExit("RESUME LR SCHEDULE MISMATCH")

print(f"lr at 3200 (first new step) = {res[0]:.10e}")
print(f"lr at 6400 (last step)      = {res[-1]:.10e}")
print(f"lr at 7999 (never reached)  = {LR * (1 - 7999 / TOTAL):.10e}")

# ---- GradScaler round-trip ----
# NOTE: GradScaler(enabled=False).state_dict() is {} BY DESIGN, so a disabled
# scaler proves nothing. Use the real enabled=True path (no GPU memory is
# allocated by constructing/dumping one).
sc = torch.amp.GradScaler("cuda", enabled=True)
sc._scale = torch.tensor(4096.0)          # set_scale() was removed in torch 2.13
sd = sc.state_dict()
sc2 = torch.amp.GradScaler("cuda", enabled=True)
sc2.load_state_dict(sd)
print(f"\nscaler state_dict keys     = {sorted(sd.keys())}")
print(f"scaler scale round-trip    = {float(sd['scale'])} -> "
      f"{float(sc2.state_dict()['scale'])}")
assert float(sc2.state_dict()["scale"]) == 4096.0, "scaler scale lost"

print("\n=== RESUME MATH OK ===")
