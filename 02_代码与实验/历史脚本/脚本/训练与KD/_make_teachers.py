import json, pathlib

B = '/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1'
t = {
    'stage23B': {
        'config': B + '/stage23_runs/stage23_w1_exp23_b3_1024_cwfa_ce_formal_seed42/config.yaml',
        'checkpoint': B + '/stage23_runs/stage23_w1_exp23_b3_1024_cwfa_ce_formal_seed42/best_miou.pt',
    },
    's62': {
        'config': B + '/stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml',
        'checkpoint': B + '/stage62_runs/stage62_w1_resume55b_pseudo500_seed42/best_miou.pt',
    },
    's58': {
        'config': B + '/stage58_runs/stage58_w1_exp58_b3_cwfa_ce_regionC_seed42/config.yaml',
        'checkpoint': B + '/stage58_runs/stage58_w1_exp58_b3_cwfa_ce_regionC_seed42/best_miou.pt',
    },
}
pathlib.Path('kd_teachers.json').write_text(json.dumps(t, indent=2), encoding='utf-8')
for k, v in t.items():
    ok = pathlib.Path(v['config']).is_file() and pathlib.Path(v['checkpoint']).is_file()
    print(f"{k:9s} {'OK' if ok else 'MISSING'}")
