import json
from pathlib import Path

D = Path('/home/jhy/rare_diag_20260915/champ237_test1_cache_20260918')
p = D / 'FORWARD_META.json'
m = json.loads(p.read_text(encoding='utf-8'))
stale = m.get('champion_score')

m['champion_score'] = 0.702574
m['champion_score_original_stale_field'] = stale
m['correction'] = (
    'champion_score 是复制 _champ_test1_cache.py 时遗留的样板字段(旧模型 0.698641)。'
    '本缓存的 champion_sha256=1485a66a14342932... 对应 '
    'donor_gate2_20260917/S_semeda_M.pt，平台分 0.702574(包 237)。'
    '2026-09-18 修正，原值保留。'
)
m['verified_against_submitted_package'] = {
    'package': '237_S_semeda_M.zip',
    'operator': 'Bstar',
    'max_abs_class_share_diff_pp': 0.0008,
    'method': 'cached logits + b* -> argmax+1 -> class histogram vs submitted zip',
}
m['zero_bias_class_share_pct'] = {
    'Background': 31.386, 'Building': 17.999, 'Road': 8.304, 'Water': 7.512,
    'Barren': 0.865, 'Vegetation': 23.958, 'Agricultural': 9.325, 'Vehicle': 0.652,
}
m['bstar_class_share_pct'] = {
    'Background': 28.153, 'Building': 18.169, 'Road': 8.382, 'Water': 7.584,
    'Barren': 1.790, 'Vegetation': 24.313, 'Agricultural': 10.985, 'Vehicle': 0.624,
}

(D / 'META_CORRECTION_20260918.json').write_text(json.dumps({
    'file': 'FORWARD_META.json',
    'field': 'champion_score',
    'was': stale,
    'now': 0.702574,
    'reason': '样板字段遗留，与 champion_sha256 不符',
    'corrected_at': '2026-09-18',
}, ensure_ascii=False, indent=1), encoding='utf-8')

p.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(m, ensure_ascii=False, indent=1))
