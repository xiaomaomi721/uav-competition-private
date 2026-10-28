import json
d = json.load(open('/home/jhy/rare_diag_20260915/b4_migration_6296/base_probe/validation_history.json'))
for v in d:
    print(f"step={v['step']} mIoU={v['raw_mIoU']:.4f} best={v.get('raw_best_mIoU',0):.4f} best_step={v.get('raw_best_step','?')}")
