import sys
p = "/home/jhy/aic2026_h76_20261026/code/h76_3594/package_ctrl3594.py"
s = open(p).read()
if "--expected-sha256" in s:
    print("already patched")
    sys.exit(0)

anchor1 = '    ap.add_argument("--bias-variants", default="bstar,agri0",'
assert anchor1 in s, "anchor1"
s = s.replace(anchor1, '    ap.add_argument("--expected-sha256", default="")\n' + anchor1, 1)

anchor2 = "    ckpt_sha = sha256_file(ckpt)\n"
assert anchor2 in s, "anchor2"
s = s.replace(anchor2,
              anchor2
              + '    if a.expected_sha256 and ckpt_sha != a.expected_sha256:\n'
              + '        raise SystemExit(f"checkpoint SHA mismatch: {ckpt_sha} != {a.expected_sha256}")\n',
              1)
open(p, "w").write(s)
print("patched packager")
