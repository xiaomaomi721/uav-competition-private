"""Structural invariants for the clean acceptance script. Run before trusting it.

Invariant 1: no bare `tie_checked += 1` (it is a dict now).
Invariant 2: every gate records at most once on any linear path, i.e. no gate has
             an unconditional record(True) that could overwrite an earlier
             record(False).
"""
import ast
import re
import sys
from pathlib import Path

P = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/home/jhy/rare_diag_20260915/m2f_gpu_20260915/c2_56k/c2_acceptance_clean.py")
src = P.read_text(encoding="utf-8")
tree = ast.parse(src)
lines = src.splitlines()

print(f"file: {P}")
print(f"lines: {len(lines)}")
print(f"bare 'tie_checked += 1' present: {'tie_checked += 1' in src}")

# ---- find every record("NAME", True/False, ...) call with its line number ----
calls = []
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id == "record" and node.args:
        a0 = node.args[0]
        if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
            val = None
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                val = node.args[1].value
            calls.append((node.lineno, a0.value, val))

# ---- for each gate, is every record() inside a conditional branch? ----
# A call is "guarded" if any enclosing If / IfExp / Try / assert-exit precedes it
# on the same function body. We approximate by checking the enclosing compound
# statements in the AST.
guard_of = {}


def walk(node, guards):
    for child in ast.iter_child_nodes(node):
        new_guards = guards
        if isinstance(child, ast.If):
            new_guards = guards + [child.lineno]
            walk(child, new_guards)
            continue
        if isinstance(child, (ast.For, ast.While, ast.Try, ast.With)):
            new_guards = guards + [child.lineno]
            walk(child, new_guards)
            continue
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) \
                and child.func.id == "record" and child.args:
            a0 = child.args[0]
            if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                guard_of.setdefault(child.lineno, list(guards))
        walk(child, new_guards)


walk(tree, [])

by_gate = {}
for ln, name, val in calls:
    by_gate.setdefault(name, []).append((ln, val, bool(guard_of.get(ln))))

print()
print("per-gate record() call sites (line, value, inside-a-conditional):")
bad = []
for name in sorted(by_gate):
    for ln, val, guarded in by_gate[name]:
        flag = ""
        if val is True and not guarded:
            flag = "  <== UNGUARDED PASS (can overwrite an earlier FAIL)"
            bad.append((name, ln))
        print(f"  {name:26s} line {ln:4d} value={str(val):5s} guarded={guarded}{flag}")

print()
# The real risk is narrow: a gate that HAS a FAIL site AND also an unguarded
# PASS site, because then a detected failure could be overwritten. A gate whose
# only record() is an unguarded PASS and which is guarded by bare `assert`
# statements is safe -- an assert failure aborts before the PASS is reached.
violations = []
for name in sorted(by_gate):
    has_fail = any(v is False for _, v, _ in by_gate[name])
    unguarded_pass = [ln for ln, v, g in by_gate[name] if v is True and not g]
    if has_fail and unguarded_pass:
        violations.append((name, unguarded_pass))

print(f"gates with BOTH a FAIL site and an unguarded PASS: {len(violations)}")
for name, lns in violations:
    print(f"  {name} unguarded PASS at lines {lns}")

# report which gates rely on bare asserts instead of a FAIL record
for name in sorted(by_gate):
    if not any(v is False for _, v, _ in by_gate[name]):
        print(f"  note: {name} has no record(FAIL); it is guarded by assert/raise")

print()
ok = True
if "tie_checked += 1" in src:
    print("INVARIANT FAIL: bare tie_checked += 1 still present")
    ok = False
if violations:
    print("INVARIANT FAIL: a gate could overwrite its own FAIL with a PASS")
    ok = False
print("INVARIANTS_OK" if ok else "INVARIANTS_FAILED")
