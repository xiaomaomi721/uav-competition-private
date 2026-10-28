import hashlib, pathlib, shutil, sys, json, collections

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = pathlib.Path('.')
MIR = ROOT / '_非决策材料归档_20260914' / '原路径镜像'
OUT = ROOT / '_从归档救出_20260915'
if OUT.exists():
    shutil.rmtree(OUT)


def h(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


SKIP = ('_非决策材料归档', '_待删除_', '_从归档救出')

have = set()
for p in ROOT.rglob('*'):
    s = str(p)
    if not p.is_file():
        continue
    if any(s.startswith(x) for x in SKIP) or '__pycache__' in s:
        continue
    try:
        have.add(h(p))
    except Exception:
        pass

rescued = []

# 1) 提交包：工作区没有的 zip
for p in sorted((MIR / '提交包').rglob('*.zip')):
    if h(p) in have:
        continue
    d = OUT / '提交包_独有zip' / p.relative_to(MIR / '提交包')
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, d)
    rescued.append((d, p.relative_to(MIR), h(p), p.stat().st_size))

# 2) 用户工作产物：汇报材料 / 导师答疑配图 / 论文抓取残留
PAIRS = [
    ('文档归档/07_汇报与复盘', '汇报材料'),
    ('导师答疑配图', '导师答疑配图'),
    ('文档归档/06_论文抓取残留', '论文抓取残留'),
]
for top, dst in PAIRS:
    src = MIR / top
    if not src.exists():
        continue
    for p in sorted(src.rglob('*')):
        if not p.is_file():
            continue
        try:
            dg = h(p)
        except Exception:
            continue
        if dg in have:
            continue
        d = OUT / dst / p.relative_to(src)
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, d)
        rescued.append((d, p.relative_to(MIR), dg, p.stat().st_size))

tot = sum(r[3] for r in rescued)
print(f'救出 {len(rescued)} 个文件，{tot / 1e6:.1f} MB')

c = collections.Counter()
for d, _orig, _dg, _s in rescued:
    c[d.relative_to(OUT).parts[0]] += 1
for k, v in sorted(c.items()):
    print(f'   {k:24s} {v}')

OUT.mkdir(exist_ok=True)
(OUT / '救出清单.json').write_text(json.dumps({
    'schema': 'RESCUED_FROM_ARCHIVE_V1',
    'created': '2026-09-15',
    'source': str(MIR),
    'note': '从 _非决策材料归档_20260914 中救出的、工作区其余位置不存在的文件（按 SHA256 判定）。'
            '删除归档前必须先确认本目录内容已妥善保留。',
    'count': len(rescued),
    'total_bytes': tot,
    'items': [{'dest': str(d), 'original_relative': str(o), 'sha256': g, 'bytes': s}
              for d, o, g, s in rescued],
}, ensure_ascii=False, indent=1), encoding='utf-8')
print('已写 救出清单.json')
