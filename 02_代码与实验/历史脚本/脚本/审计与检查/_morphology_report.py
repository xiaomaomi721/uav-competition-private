#!/usr/bin/env python3
"""Summarize completed CPU morphology audits and cross-check historical metrics."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/jhy/rare_diag_20260915')
BUNDLE = Path('/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1')
GT_SOURCE = BUNDLE / 'diagnostics/champion_loss_audit_01/full'


def load(p):
    return json.loads(p.read_text())


def pct(v):
    return '—' if v is None else f'{100*v:.2f}%'


def area_row(label, s):
    shares = s['pixel_share_by_min_component_area']
    quantiles = ['—' if s[k] is None else f'{s[k]:,.1f}' for k in ('p50', 'p90', 'p99')]
    return '| ' + ' | '.join([label, f"{s['n_components']:,}", *quantiles,
                             *[pct(shares[str(t)]) for t in (64, 256, 1024, 4096)]]) + ' |'


def table_header():
    return ['| 对象 | 连通域数 | P50面积 | P90面积 | P99面积 | ≥64像素占比 | ≥256像素占比 | ≥1024像素占比 | ≥4096像素占比 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']


def main():
    gt_doc = load(ROOT / 'morph_v2_gt_B/ERROR_MORPHOLOGY.json')
    test_doc = load(ROOT / 'morph_v2_test/ERROR_MORPHOLOGY.json')
    gt = gt_doc['gt_error']['B']
    old = load(GT_SOURCE / 'summary.json')
    miou = sum(v['iou'] for v in gt['per_class'].values()) / 8
    assert gt['images_used'] == old['n_images'] == 1399
    assert gt['valid_px'] == old['valid_pixels']
    assert abs(miou - old['mIoU']) < 1e-6
    assert abs(1 - gt['delta_rate'] - old['pixel_accuracy']) < 1e-6
    for k, v in gt['per_class'].items():
        assert abs(v['iou'] - old['per_class_iou'][k]) < 1e-6
    for s in test_doc['test_disagreement'].values():
        assert s['images_used'] == 500
        assert not s['coverage']['A_only'] and not s['coverage']['B_only']
    receipt = {
        'gt_model_identity': load(GT_SOURCE / 'model_identity.json'),
        'gt_validation_identity': load(GT_SOURCE / 'validation_identity.json'),
        'historical_metric_crosscheck': {'pass': True, 'valid_pixels_exact_match': True,
                                        'miou': miou, 'per_class_iou_max_abs_diff':
                                        max(abs(v['iou']-old['per_class_iou'][k]) for k,v in gt['per_class'].items())},
        'test_sources': test_doc['sources'],
        'limitations': ['s62/s58 full validation masks not confirmed; no GT audit for them',
                        'GT uses single-view zero bias; test uses TTA8 zero bias; unmatched protocols',
                        'test1 is not all semifinal test data; disagreement is not error',
                        'prediction PNG directory identity follows historical receipt, not fresh model reproduction',
                        '8-connected components may bridge thin paths or merge distinct class-pair errors',
                        'Ignore interfaces excluded; some regions near Ignore can be labelled interior'],
        'files': {}}
    for path in [ROOT/'_error_morphology_v2.py', ROOT/'_error_morphology_v2_gt.py',
                 ROOT/'morph_v2_gt_B/ERROR_MORPHOLOGY.json', ROOT/'morph_v2_test/ERROR_MORPHOLOGY.json']:
        receipt['files'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    lines = ['# GT 真实错误与测试域分歧形态审计（CPU）', '',
             '## 1. 覆盖、协议与验证', '',
             '- GT：Stage23 B，1399/1399；单视图、无 TTA、无偏置。',
             '- test1：B、s62、s58 的 07/27/23 零偏置提交包，各 500/500，TTA8。',
             '- 没有加载模型或运行新增推理；没有修改 KD 训练。',
             '- 未找到可确认来源的 s62/s58 全量验证预测；不能声称已完成三模型 GT 对比。',
             f'- GT 有效像素 {gt["valid_px"]:,}，与历史审计精确一致；mIoU={miou:.8f}，逐类 IoU 均在 1e-6 内复现历史记录。',
             '- Ignore=0 排除；预测0仅允许出现在GT Ignore位置。完全正确的图像、GT中某类缺失时的误报均纳入IoU。',
             '- 连通域为8连通；面积分位数汇总所有图像的全部连通域后计算，跨图不相连；阈值占比以像素面积加权，不是逐图均值。',
             '- 真实类别对逐对计算连通域，不使用“大区域主导类别”代替；无最小面积过滤、无每图top-N截断。',
             '- 边界：相邻有效类别变化两侧像素均作为种子，欧氏距离≤3/5像素；不计图像外边界或Ignore接口。测试边界是两模型预测边界的并集，不是GT。',
             '', '## 2. Stage23 B 的 GT 真实错误', '',
             f'错误率 **{pct(gt["delta_rate"])}**，错误像素 {gt["delta_px"]:,}。', '', *table_header(),
             area_row('全部GT错误', gt['components']),
             area_row('去3px边界后的内部错误', gt['boundary']['3']['interior_components']),
             area_row('去5px边界后的内部错误', gt['boundary']['5']['interior_components']), '',
             '| 边界宽度 | 边界占有效区域 | 错误中在边界的份额 | 边界内错误率 | 内部错误率 |',
             '|---|---:|---:|---:|---:|']
    for w,b in gt['boundary'].items():
        lines.append(f'| {w}px | {pct(b["share_of_valid_in_band"])} | {pct(b["share_of_delta_in_band"])} | {pct(b["delta_rate_in_band"])} | {pct(b["delta_rate_interior"])} |')
    lines += ['', '### 逐类指标（按IoU从低到高）', '', '| 类别 | IoU | 类内漏错率 | GT像素 | FP像素 | FN像素 |', '|---|---:|---:|---:|---:|---:|']
    for k,v in sorted(gt['per_class'].items(),key=lambda kv:kv[1]['iou']):
        lines.append(f'| {k} | {pct(v["iou"])} | {pct(v["gt_error_rate"])} | {v["gt_px"]:,} | {v["fp"]:,} | {v["fn"]:,} |')
    lines += ['', '### 全局错误像素最多的15个真实有向类别对', '',
              '下表仅为展示截取；JSON保留所有非零类别对。方向为 GT → 预测。', '',
              '| 类别对 | 错误像素 | 占全部错误 | 连通域P90 | 去5px边界后像素 | 内部≥1024占比 | 内部≥4096占比 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for k,v in sorted(gt['directed_pairs'].items(),key=lambda kv:-kv[1]['components']['total_px'])[:15]:
        a,inside=v['components'],v['interior5_components']; shares=inside['pixel_share_by_min_component_area']
        lines.append(f'| {k} | {a["total_px"]:,} | {pct(v["share_of_all_delta"])} | {a["p90"]:.1f} | {inside["total_px"]:,} | {pct(shares["1024"])} | {pct(shares["4096"])} |')
    lines += ['', '## 3. test1 模型分歧（不是错误率）', '', *table_header()]
    for k,s in test_doc['test_disagreement'].items():
        lines.append(area_row(k, s['components']))
    for k,s in test_doc['test_disagreement'].items():
        lines += ['', f'### {k}', '', f'分歧率 **{pct(s["delta_rate"])}**，{s["delta_px"]:,}像素。',
                  f'在双方预测边界并集5px带之外的分歧占 {pct(s["boundary"]["5"]["share_of_delta_interior"])}。', '',
                  '| A预测 → B预测 | 精确像素数 | 占本组全部分歧 | P90面积 | ≥4096像素占比 |', '|---|---:|---:|---:|---:|']
        for pair,v in sorted(s['directed_pairs'].items(),key=lambda kv:-kv[1]['components']['total_px'])[:10]:
            a=v['components']
            lines.append(f'| {pair} | {a["total_px"]:,} | {pct(v["share_of_all_delta"])} | {a["p90"]:.1f} | {pct(a["pixel_share_by_min_component_area"]["4096"])} |')
    lines += ['', '## 4. 解释边界与下一步使用方式', '',
              '- 形态只能定位要解决的现象，不能证明原因是当前head，也不能证明换Mask2Former必然提升。',
              '- 若内部大区域/特定类对占比显著，可将区域语义一致性、mask classification与跨尺度上下文列为试验靶点；同时保留类别校准、训练监督和域偏移等解释。',
              '- 边界带面积占比与边界错误率一起看：边界附近错误多，不等于总体收益必然来自边界优化。',
              '- 大区域像素占比不等于mIoU收益占比。8类等权，Barren/Vehicle等稀有类必须单独评价，不应被Background等大类淹没。',
              '- GT单视图与测试TTA8不匹配，且测试没有GT；不能从两组形态率直接推导域偏移大小或教师谁更正确。',
              '- 当前测试仅test1的500张，不能直接外推到全部复赛测试图。',
              '- 旧morph_v1的GT匹配为0、分位数/阈值占比是逐图均值、有向类别对计数不精确；该版本不作为决策依据。',
              '- 本报告不授权新训练、不改变KD/Run2队列；架构收益需要GPU2空闲后另行受控实验验证。', '']
    out=ROOT/'morph_v2_report'
    out.mkdir(exist_ok=False)
    (out/'AUDIT_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'CPU形态审计报告.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'crosscheck':receipt['historical_metric_crosscheck'],'report':str(out)},ensure_ascii=False))


if __name__=='__main__':
    main()
