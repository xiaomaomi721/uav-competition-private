"""CPU-only Stage23 optimizer audit and opt-in canonical paramwise backup.

Does not modify training code/configs or launch training. Audit output creation is
exclusive; existing outputs are never overwritten. Group construction itself
creates no scheduler and has no warmup. Run --audit on the authorized server.
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

import torch
from torch import nn

SOURCE = Path('/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1')
RUN = SOURCE / 'stage23_runs/stage23_w1_exp23_b3_1024_cwfa_ce_formal_seed42'
OUTPUT = Path('/home/jhy/gpu2_day_queue/next_experiment')
CONFIG_SHA = '68cf3a98b7bb17dc564fbbb31dced103543e86bd9e01b967c161c37f1120995c'
TRAINER_SHA = '3f3168fff21f498b273dd296d3b15184ed874c04397924a001cfdd45d76052f3'
MODELS_SHA = '7c003956ec2953c5cc6f63c031438d5189163d0e2ac5e2bab7d33edbdbf290be'
NORM_TYPES = (nn.LayerNorm, nn.modules.batchnorm._BatchNorm)
POSITION_TOKENS = {'pos_embed', 'pos_embedding', 'pos_embeddings', 'position_embedding',
                   'position_embeddings', 'positional_embedding', 'positional_embeddings', 'pos_block'}
REFERENCES = [
    {'organization': 'NVIDIA NVlabs',
     'url': 'https://github.com/NVlabs/SegFormer/blob/master/local_configs/segformer/B2/segformer.b2.512x512.ade.160k.py',
     'settings': 'AdamW lr=0.00006, wd=0.01, betas=(0.9,0.999); pos_block decay_mult=0, norm decay_mult=0, head lr_mult=10; poly power=1, min_lr=0, by_epoch=False; linear warmup_iters=1500, warmup_ratio=1e-6.'},
    {'organization': 'OpenMMLab',
     'url': 'https://github.com/open-mmlab/mmsegmentation/blob/main/configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py',
     'settings': 'AdamW lr=0.00006, wd=0.01; corresponding pos_block/norm/head custom_keys; LinearLR warmup ends at 1500, PolyLR power=1 starts at 1500.'},
]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_paramwise_groups(model, *, base_lr=6e-5, weight_decay=0.01,
                               head_path='decode_head', head_lr_mult=10.0,
                               positional_parameter_names=()):
    """Return torch optimizer groups + JSON metadata, without creating a scheduler.

    All registered parameters (including frozen, matching model.parameters()) are
    covered exactly once by identity. Head is an exact module subtree. Norms use
    module type, never text matching. Positional exemptions require explicitly
    reviewed full names; unknown position-like tokens fail closed. Ordinary
    transformer blocks, patch projections, DWConv and non-norm biases decay.
    Cross-rule composition: head sets LR, norm/explicit position sets WD. Shared
    parameters with conflicting alias classifications are rejected.
    """
    if not (math.isfinite(base_lr) and base_lr > 0 and math.isfinite(weight_decay)
            and weight_decay >= 0 and math.isfinite(head_lr_mult) and head_lr_mult > 0):
        raise ValueError('Invalid optimizer hyperparameters')
    head = model.get_submodule(head_path)
    if not any(True for _ in head.parameters()):
        raise ValueError('Head module has no parameters')
    aliases = list(model.named_parameters(remove_duplicate=False))
    by_name = dict(aliases)
    positional = set(positional_parameter_names)
    if positional - by_name.keys():
        raise ValueError('Unknown positional full names: ' + repr(sorted(positional - by_name.keys())))
    candidates = {n for n, _ in aliases if POSITION_TOKENS.intersection(n.lower().split('.'))}
    if candidates - positional:
        raise ValueError('Position-like names require explicit review: ' + repr(sorted(candidates - positional)))
    modules = dict(model.named_modules(remove_duplicate=False))
    seen = {}
    records = []
    buckets = collections.OrderedDict()
    norms = []
    for name, param in aliases:
        owner_path = name.rsplit('.', 1)[0] if '.' in name else ''
        owner = modules[owner_path]
        is_head = name == head_path or name.startswith(head_path + '.')
        is_norm = isinstance(owner, NORM_TYPES)
        is_position = name in positional
        signature = (is_head, is_norm, is_position)
        if id(param) in seen:
            previous = seen[id(param)]
            if previous['signature'] != signature:
                raise ValueError('Conflicting shared-parameter rules: ' + previous['record']['name'] + ' vs ' + name)
            previous['record']['aliases'].append(name)
            continue
        no_decay = is_norm or is_position
        group_name = ('head' if is_head else 'backbone') + ('_no_decay' if no_decay else '_decay')
        lr = base_lr * (head_lr_mult if is_head else 1.0)
        wd = 0.0 if no_decay else weight_decay
        group = buckets.setdefault(group_name, {'group_name': group_name, 'params': [], 'lr': lr, 'weight_decay': wd})
        group['params'].append(param)
        record = {'name': name, 'aliases': [name], 'shape': list(param.shape),
                  'numel': param.numel(), 'requires_grad': param.requires_grad,
                  'owner_module_type': type(owner).__module__ + '.' + type(owner).__name__,
                  'head': is_head, 'normalization': is_norm, 'positional': is_position,
                  'group_name': group_name, 'lr': lr, 'weight_decay': wd}
        records.append(record)
        seen[id(param)] = {'signature': signature, 'record': record}
        if is_norm:
            norms.append(name)
    # Deterministic readable order, independent of first no-decay occurrence.
    order = ('backbone_decay', 'backbone_no_decay', 'head_decay', 'head_no_decay')
    groups = [buckets[k] for k in order if k in buckets]
    actual_ids = [id(p) for g in groups for p in g['params']]
    expected_ids = {id(p) for p in model.parameters()}
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
        raise AssertionError('Non-exhaustive or duplicate parameter grouping')
    metadata = {'head_path': head_path, 'base_lr': base_lr, 'head_lr_mult': head_lr_mult,
                'base_weight_decay': weight_decay, 'parameter_tensor_count': len(records),
                'scalar_parameter_count': sum(r['numel'] for r in records),
                'positional_name_candidates': sorted(candidates),
                'positional_parameter_names': sorted(positional),
                'positional_parameter_count': len(positional),
                'normalization_parameter_names': norms, 'groups': [], 'parameters': records,
                'tests': {'all_parameters_exhaustive': True, 'unique_by_identity': True,
                          'conflicting_aliases': False, 'scheduler_created': False,
                          'hidden_warmup': False}}
    for group in groups:
        metadata['groups'].append({k: v for k, v in group.items() if k != 'params'} | {
            'tensor_count': len(group['params']), 'scalar_parameter_count': sum(p.numel() for p in group['params']),
            'parameter_names': [r['name'] for r in records if r['group_name'] == group['group_name']]})
    return groups, metadata


def selftest():
    # A misleading name is intentionally used to defeat substring heuristics.
    m = nn.Module()
    m.normalizer_not_a_norm = nn.Linear(4, 4)
    m.quiet = nn.LayerNorm(4)
    m.blocks = nn.ModuleList([nn.Linear(4, 4)])
    m.decode_head = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm2d(4))
    m.pos_embed = nn.Parameter(torch.ones(1, 4))
    try:
        canonical_paramwise_groups(m)
        raise AssertionError('Unreviewed position parameter was not rejected')
    except ValueError as exc:
        assert 'explicit review' in str(exc)
    groups, meta = canonical_paramwise_groups(m, positional_parameter_names=['pos_embed'])
    r = {x['name']: x for x in meta['parameters']}
    assert r['normalizer_not_a_norm.weight']['weight_decay'] == .01
    assert r['quiet.weight']['weight_decay'] == 0
    assert r['blocks.0.weight']['weight_decay'] == .01
    assert r['pos_embed']['weight_decay'] == 0
    assert r['decode_head.1.weight']['weight_decay'] == 0
    assert math.isclose(r['decode_head.1.weight']['lr'], 6e-4, rel_tol=1e-14)
    assert r['decode_head.0.bias']['weight_decay'] == .01
    opt = torch.optim.AdamW(groups, lr=6e-5, weight_decay=.01)
    assert all(math.isclose(g['lr'], lr, rel_tol=1e-14)
               for g, lr in zip(opt.param_groups, [6e-5, 6e-5, 6e-4, 6e-4]))
    # One scalar-algebra optimizer self-test on synthetic parameters, not a model training run.
    p = m.blocks[0].weight
    before = p.detach().clone()
    p.grad = torch.zeros_like(p)
    opt.step()
    torch.testing.assert_close(p, before * (1 - 6e-5 * .01), rtol=0, atol=0)
    assert not opt.state.get(m.quiet.weight)
    m.alias = m.blocks[0]  # Same-rule shared alias is included only once.
    _, shared_meta = canonical_paramwise_groups(m, positional_parameter_names=['pos_embed'])
    assert shared_meta['parameter_tensor_count'] == meta['parameter_tensor_count']
    m.decode_head.register_parameter('conflict', m.blocks[0].weight)
    try:
        canonical_paramwise_groups(m, positional_parameter_names=['pos_embed'])
        raise AssertionError('Conflicting shared parameter was not rejected')
    except ValueError as exc:
        assert 'Conflicting shared-parameter' in str(exc)
    return {'status': 'PASS', 'cases': ['module_type_not_substring', 'full_head_including_BatchNorm',
        'ordinary_blocks_keep_decay', 'non_norm_bias_keeps_decay', 'explicit_position_no_decay',
        'unreviewed_position_rejected', 'exhaustive_unique', 'consistent_alias_dedup',
        'conflicting_alias_rejected', 'synthetic_AdamW_zero_gradient_decay', 'no_scheduler_or_warmup']}


def extract_constructors(trainer_text):
    tree = ast.parse(trainer_text)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train_once')
    wanted = {'optimizer_cfg', 'optimizer', 'max_steps', 'scheduler_total_steps', 'power', 'scheduler'}
    nodes = [n for n in fn.body if isinstance(n, ast.Assign)
             and any(isinstance(x, ast.Name) and x.id in wanted for x in n.targets)]
    assert len(nodes) == 6
    return nodes, [{'start_line': n.lineno, 'end_line': n.end_lineno,
                    'source': ast.get_source_segment(trainer_text, n)} for n in nodes]


def audit():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be hidden before importing torch'
    assert os.environ.get('OMP_NUM_THREADS') == os.environ.get('MKL_NUM_THREADS') == '2'
    assert os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
    assert os.getpriority(os.PRIO_PROCESS, 0) == 19
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    tests = selftest()
    import yaml
    import transformers
    sys.path.insert(0, str(SOURCE))
    from src.models import build_model, resolve_model_source, get_cwfa_module
    cfg_path = RUN / 'config.yaml'
    assert sha256(cfg_path) == CONFIG_SHA
    assert sha256(SOURCE / 'src/trainer.py') == TRAINER_SHA
    assert sha256(SOURCE / 'src/models.py') == MODELS_SHA
    cfg = yaml.safe_load(cfg_path.read_text())
    assert not any(cfg.get(k, {}).get('enabled', False) for k in ['edge_branch', 'distill', 'mixstyle', 'ema'])
    assert cfg.get('decode_head', '') != 'attention_fusion'
    resolution = resolve_model_source(cfg)
    assert resolution['runtime_model_source'] == '/home/jhy/models/segformer-b3-ade512'
    assert resolution['local_files_only'] is True
    pretrained = Path(resolution['runtime_model_source'])
    source_paths = list((SOURCE / 'src').glob('*.py')) + [cfg_path, Path(cfg['source_config']),
        RUN / 'run_manifest.json', RUN / 'last.pt', RUN / 'train_log.csv',
        pretrained / 'config.json', pretrained / 'pytorch_model.bin', Path(__file__).resolve()]
    before = {str(p): sha256(p) for p in source_paths}
    torch.manual_seed(cfg['seed'])
    model = build_model(cfg)  # Actual production function; no .to(cuda), no train_once.
    assert all(p.device.type == 'cpu' for p in model.parameters())
    assert all(b.device.type == 'cpu' for b in model.buffers())
    assert not torch.cuda.is_initialized()
    cwfa = get_cwfa_module(model)
    assert cwfa is not None and len(list(cwfa.parameters())) == 0
    trainer_text = (SOURCE / 'src/trainer.py').read_text()
    nodes, excerpts = extract_constructors(trainer_text)
    env = {'torch': torch, 'config': cfg, 'model': model}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE / 'src/trainer.py'), 'exec'), env)
    assert env['scheduler_total_steps'] >= env['max_steps']
    opt, scheduler = env['optimizer'], env['scheduler']
    name_by_id = {id(p): n for n, p in model.named_parameters()}
    actual = []
    for index, group in enumerate(opt.param_groups):
        actual.append({'index': index, 'group_name': group.get('group_name'), 'display_name': '<unnamed>',
            'tensor_count': len(group['params']), 'scalar_parameter_count': sum(p.numel() for p in group['params']),
            'parameter_names': [name_by_id[id(p)] for p in group['params']],
            'hyperparameters': {k: v for k, v in group.items() if k != 'params'}})
    groups, recommended = canonical_paramwise_groups(model, base_lr=cfg['optimizer']['learning_rate'],
        weight_decay=cfg['optimizer']['weight_decay'], head_path='decode_head', positional_parameter_names=())
    # Exact same source constructor AST, only replace model.parameters() with audited groups.
    opt_node = next(n for n in nodes if isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'optimizer')
    import copy
    replacement = copy.deepcopy(opt_node)
    assert ast.unparse(replacement.value.args[0]) == 'model.parameters()'
    replacement.value.args[0] = ast.Name(id='canonical_groups', ctx=ast.Load())
    replacement = ast.fix_missing_locations(replacement)
    backup_env = dict(env, canonical_groups=groups)
    exec(compile(ast.Module(body=[replacement], type_ignores=[]), '<backup-one-argument-change>', 'exec'), backup_env)
    backup_opt = backup_env['optimizer']
    sched_node = next(n for n in nodes if n.targets[0].id == 'scheduler')
    exec(compile(ast.Module(body=[sched_node], type_ignores=[]), str(SOURCE / 'src/trainer.py'), 'exec'), backup_env)
    backup_scheduler = backup_env['scheduler']
    assert backup_opt.defaults == opt.defaults
    assert [g['lr'] for g in backup_opt.param_groups] == [g['lr'] for g in recommended['groups']]
    steps = [0, 1, 20, 1500, 28000, 28001, 55980, 56000, 56001]
    probes = [{'successful_updates': step, 'lambda': scheduler.lr_lambdas[0](step),
               'actual_lr': scheduler.base_lrs[0] * scheduler.lr_lambdas[0](step),
               'recommended_lrs': [lr * f(step) for lr, f in zip(backup_scheduler.base_lrs, backup_scheduler.lr_lambdas)]}
              for step in steps]
    assert probes[0]['lambda'] == 1.0 and probes[-1]['lambda'] == 0.0
    assert all(a['lambda'] >= b['lambda'] for a, b in zip(probes, probes[1:]))
    cp = torch.load(RUN / 'last.pt', map_location='cpu', weights_only=False, mmap=True)
    cpopt = cp.get('optimizer') or cp['optimizer_state_dict']
    cpsch = cp.get('scheduler') or cp['scheduler_state_dict']
    cpmodel = cp.get('model') or cp['model_state_dict']
    names = list(name_by_id.values())
    assert all(n in cpmodel and list(cpmodel[n].shape) == list(p.shape) for n, p in model.named_parameters())
    cpgroups = []
    for i, g in enumerate(cpopt['param_groups']):
        numel = sum(cpopt['state'][pid]['exp_avg'].numel() for pid in g['params'])
        cpgroups.append({'index': i, 'group_name': g.get('group_name'), 'tensor_count': len(g['params']),
                        'scalar_parameter_count_from_optimizer_moments': numel,
                        'hyperparameters': {k: v for k, v in g.items() if k != 'params'},
                        'parameter_ids': g['params']})
    assert len(actual) == len(cpgroups) == 1
    assert actual[0]['tensor_count'] == cpgroups[0]['tensor_count'] == 641
    assert actual[0]['scalar_parameter_count'] == cpgroups[0]['scalar_parameter_count_from_optimizer_moments'] == 47228616
    assert cpgroups[0]['hyperparameters']['weight_decay'] == .01
    assert cpsch['base_lrs'] == [6e-5]
    assert math.isclose(cpopt['param_groups'][0]['lr'], scheduler.lr_lambdas[0](cpsch['last_epoch']) * 6e-5, rel_tol=1e-14)
    trace = []
    all_trace_match = True
    rows_count = 0
    with (RUN / 'train_log.csv').open(encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            rows_count += 1
            updates = int(row['optimizer_update_count'])
            expected_lr = 6e-5 * scheduler.lr_lambdas[0](updates)
            if not math.isclose(float(row['learning_rate']), expected_lr, rel_tol=1e-9, abs_tol=1e-15):
                all_trace_match = False
            if rows_count <= 3 or int(row['step']) in [1500, 28000, 28001, 56000]:
                trace.append({k: row[k] for k in ['step', 'optimizer_update_count', 'skipped_optimizer_steps', 'learning_rate']})
    assert all_trace_match
    normalization_modules = [{'name': n, 'type': type(m).__name__,
        'direct_parameter_names': [n + '.' + pn for pn, _ in m.named_parameters(recurse=False)]}
        for n, m in model.named_modules() if isinstance(m, NORM_TYPES)]
    implementation_paths = [Path(inspect.getfile(type(model))), Path(inspect.getfile(torch.optim.AdamW)),
                            Path(inspect.getfile(torch.optim.lr_scheduler.LambdaLR))]
    before.update({str(p): sha256(p) for p in implementation_paths})
    # Confirm source/weights/checkpoint/log were only read and remained byte-identical.
    unchanged = all(sha256(p) == digest for p, digest in before.items())
    assert unchanged
    assert not torch.cuda.is_initialized()
    variant_text = (SOURCE / 'src/trainer_formal_semeda_cwfa.py').read_text()
    variant_nodes, variant_excerpts = extract_constructors(variant_text)
    variant_same = [ast.dump(n, include_attributes=False) for n in nodes] == [ast.dump(n, include_attributes=False) for n in variant_nodes]
    data = {'schema_version': 1, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'decision': 'UNIFORM_CONFIRMED_CANONICAL_PARAMWISE_BACKUP_ONLY',
        'paramwise_experiment_not_needed': False, 'backup_experiment_launched': False,
        'source_config_path': str(cfg_path), 'source_config_sha256': CONFIG_SHA,
        'input_configuration': cfg, 'source_sha256': before,
        'runtime': {'python_executable': sys.executable, 'python_version': sys.version,
            'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
            'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'cuda_initialized': torch.cuda.is_initialized(),
            'all_parameters_and_buffers_cpu': True, 'num_threads': torch.get_num_threads(),
            'num_interop_threads': torch.get_num_interop_threads(), 'nice': os.getpriority(os.PRIO_PROCESS, 0),
            'ionice': 'idle class 3 supplied by invocation', 'bytecode_writes_disabled': True},
        'model_class': type(model).__module__ + '.' + type(model).__name__, 'model_source_resolution': resolution,
        'model_num_labels': model.config.num_labels, 'cwfa_parameter_tensor_count': len(list(cwfa.parameters())),
        'actual_optimizer': {'class': type(opt).__module__ + '.' + type(opt).__name__,
            'constructor_method': 'Exact six assignment AST nodes from train_once, compiled with original filename; no trainer import or train_once execution.',
            'source_excerpts': excerpts, 'defaults_observed': opt.defaults, 'groups': actual,
            'head_lr_multiplier': 1.0, 'norm_weight_decay': .01, 'positional_matches': []},
        'variant_optimizer_scheduler_ast_identical': variant_same,
        'variant_source_excerpts': variant_excerpts,
        'actual_scheduler': {'class': type(scheduler).__name__, 'state_after_construction': scheduler.state_dict(),
            'total_steps': env['scheduler_total_steps'], 'power': env['power'], 'warmup_steps': 0,
            'step_unit': 'Successful optimizer updates; skip AMP-overflow updates, preserve original loop.',
            'lambda_probes': probes, 'all_train_log_rows_match_original_polynomial': all_trace_match,
            'train_log_rows_checked': rows_count, 'selected_train_log_rows': trace},
        'checkpoint': {'path': str(RUN / 'last.pt'), 'load_mode': 'CPU mmap read-only, no model load_state_dict or optimizer load_state_dict',
            'step': cp['step'], 'optimizer_update_count': cp['optimizer_update_count'],
            'skipped_optimizer_steps': cp['skipped_optimizer_steps'], 'groups': cpgroups,
            'scheduler_state': cpsch, 'all_current_parameter_names_and_shapes_present': True,
            'parameter_names_note': 'Checkpoint optimizer IDs contain no names; scalar count independently summed from exp_avg. Full names come from actual instantiated model, with checkpoint model name/shape validation.'},
        'normalization_modules': normalization_modules, 'recommended': recommended,
        'recommended_optimizer_defaults': backup_opt.defaults,
        'recommended_scheduler_state_after_construction': backup_scheduler.state_dict(),
        'selftests': tests, 'source_files_unchanged_after_audit': unchanged,
        'references': REFERENCES,
        'reference_verification': 'Official URLs/settings corroborated by WebSearch. WebFetch of github.com and raw.githubusercontent.com returned domain-safety/network verification errors; no alternate transport bypass used. References are illustrative configs, not an exact version-pinned reproduction.',
        'backup_design': {'status': 'DESIGN_ONLY_NOT_INTEGRATED_NOT_QUEUED',
            'single_factor': 'optimizer parameter-group policy (head LR and norm/position WD combined as one predeclared recipe, not separable causal effects)',
            'only_constructor_change': 'model.parameters() -> canonical_paramwise_groups(model, ...)[0]',
            'head': 'exact decode_head subtree, including linear projections, linear_fuse, batch_norm and classifier',
            'decay': 'LayerNorm and BatchNorm parameters zero; explicit reviewed positional names zero (actual zero matches); all other weights AND biases retain .01',
            'warmup_steps': 0, 'scheduler': 'Unchanged source LambdaLR, total_steps=56000, power=1; same successful-update stepping.',
            'initialization': 'Fresh published local pretrained via original build_model, seed42; do not resume Stage23 last.pt or import its optimizer state.',
            'other_settings': 'Keep entire audited Stage23 config unchanged: B3, 8 classes, crop1024, CE, batch2 accumulation2, AMP, seed42, 56000 steps, splits, transforms, RareCrop, CWFA p.3 epsilon4 start28001, validation/checkpoint rules.',
            'guardrail': 'No runnable experiment YAML, run script, queue entry, or trainer edit created. Do not combine with RareCrop experiment.'}}
    report = render_report(data)
    assert OUTPUT.parent.is_dir()
    OUTPUT.mkdir(exist_ok=True)
    json_path = OUTPUT / 'SEGFORMER_OPTIMIZER_GROUPS.json'
    report_path = OUTPUT / 'SEGFORMER_OPTIMIZER_AUDIT.md'
    assert not json_path.exists() and not report_path.exists(), 'Refuse to overwrite existing audit outputs'
    with json_path.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write('\n')
    with report_path.open('x', encoding='utf-8') as f:
        f.write(report)
    for label, items in [('ACTUAL', actual), ('RECOMMENDED', recommended['groups'])]:
        for g in items:
            hp = g.get('hyperparameters', g)
            print(label, g['group_name'], g['tensor_count'], g['scalar_parameter_count'], hp['lr'], hp['weight_decay'])
    print('SELFTEST', json.dumps(tests))
    print('CHECKPOINT', json.dumps(data['checkpoint'], default=str))
    print('TRACE_ROWS', rows_count, 'ALL_POLYNOMIAL_MATCH', all_trace_match)
    print('SOURCE_UNCHANGED', unchanged, 'CUDA_INITIALIZED', torch.cuda.is_initialized())
    for p in [json_path, report_path]:
        print('ARTIFACT', p, 'SHA256', sha256(p))


def render_report(d):
    a = d['actual_optimizer']; s = d['actual_scheduler']; c = d['checkpoint']; r = d['recommended']
    lines = ['# Stage23 actual optimizer audit and canonical-paramwise backup', '',
        '## Decision', '',
        '**UNIFORM_CONFIRMED_CANONICAL_PARAMWISE_BACKUP_ONLY**. Actual constructor and historical optimizer state are uniform, not equivalent to the requested paramwise recipe. `PARAMWISE_EXPERIMENT_NOT_NEEDED` does **not** apply. Backup is design-only, not integrated, not queued, not launched.', '',
        'Generated UTC: ' + d['generated_at_utc'], '',
        '## Source identity and method', '',
        '- Source run: `' + str(RUN) + '`.',
        '- Audited saved config SHA256: `' + CONFIG_SHA + '` (matches requested 68cf prefix and run_manifest artifact inventory).',
        '- The manifest top-level config hash is for the input `configs/stage23/wave1_exp23_b3_1024_cwfa_ce.yaml`, not the runtime-expanded saved config; these are different files, not a hash contradiction.',
        '- Used `/home/jhy/.venvs/uavseg/bin/python`, CUDA_VISIBLE_DEVICES empty, OMP/MKL=2, PyTorch intra/inter-op=2, nice19, ionice idle, offline local pretrained, disabled bytecode writes.',
        '- Called real `src.models.build_model(config)` on CPU. Compiled exact six optimizer/scheduler assignments extracted by AST from `src/trainer.py::train_once`; did not call the training function, load datasets, run inference/backprop on the real model, or query GPUs.',
        '- All real model parameters/buffers were CPU and CUDA was not initialized. Local ADE classifier 150 -> 8 mismatch/reinitialization is expected production build behavior; actual classifier is [8,768,1,1] plus [8] bias.',
        '- Current formal variant constructor/scheduler AST identical: `' + str(d['variant_optimizer_scheduler_ast_identical']) + '`. Current source is not asserted to be a frozen 2026-08 historical trainer; last.pt and the full LR log provide independent historical behavior evidence.', '',
        '## Exact production constructor evidence', '', '```python']
    for excerpt in a['source_excerpts']:
        lines += ['# src/trainer.py:' + str(excerpt['start_line']) + '-' + str(excerpt['end_line']), excerpt['source']]
    lines += ['```', '', 'The AST executes real constructor defaults; observed defaults (not guessed):', '',
              '```json', json.dumps(a['defaults_observed'], indent=2), '```', '',
              '## All actual groups', '', '| Group name | Tensors | Scalar parameters | Initial LR | WD |',
              '|---|---:|---:|---:|---:|']
    for g in a['groups']:
        h = g['hyperparameters']; lines.append(f"| <unnamed> (index {g['index']}) | {g['tensor_count']} | {g['scalar_parameter_count']:,} | {h['lr']:.12g} | {h['weight_decay']} |")
    lines += ['', '- Head multiplier = 1, normalization WD = .01. No paramwise grouping, bias exemption, or positional exemption exists in this actual optimizer.',
        '- JSON includes **every actual group member full name**, every recommended group member full name and a per-parameter manifest with shape, numel, owner type, flags, LR/WD.', '',
        '## Actual HF names, normalization and positional interpretation', '',
        '- Actual class: `' + d['model_class'] + '`.',
        '- Backbone names are `segformer.stages.<i>...`, not an assumed older `segformer.encoder.block...` namespace.',
        '- Head path is exact `decode_head`; whole subtree receives x10, not only `decode_head.classifier`.',
        '- Head includes `decode_head.linear_projections.{0,1,2,3}.proj.{weight,bias}`, `decode_head.linear_fuse.weight`, `decode_head.batch_norm.{weight,bias}`, `decode_head.classifier.{weight,bias}`.',
        '- Normalization matches are selected by `isinstance(module, (torch.nn.LayerNorm, torch.nn.modules.batchnorm._BatchNorm))`, direct parameters only; no blind substring heuristic.',
        '- Normalization modules: ' + str(len(d['normalization_modules'])) + '; parameter tensors: ' + str(len(r['normalization_parameter_names'])) + '.',
        '- Actual positional/pos_block parameter matches: **0**. Explicit reviewed positional allowlist is empty; no synthetic empty optimizer group is created.',
        '- The token `block`/`blocks` is NOT a positional rule. Attention, MLP, depthwise convolution and patch projection parameters inside ordinary blocks continue to decay. Non-normalization biases also continue to decay.',
        '- Head LR and no-decay WD rules compose on independent axes; head BatchNorm gets LR=6e-4 and WD=0, with no precedence conflict. This is a module-aware HF adaptation of the requested intent, not a claim of byte-identical legacy custom-key substring precedence.',
        '- CWFA contributes 0 parameter tensors.', '',
        '## Actual scheduler: no warmup', '',
        '- Source lambda: `max(0.0, (1.0 - min(step, scheduler_total_steps) / scheduler_total_steps) ** power)`.',
        '- Resolved total_steps=56000, power=1.0; absent total_steps resolves through the actual source to max_steps. No warmup branch exists; lambda(0)=1.',
        '- `src/trainer.py:660-671`: scheduler.step() only follows a successful scaler/optimizer update. Accumulation remains 2; failed AMP updates do not advance scheduler.',
        '- Historical last.pt: outer step=' + str(c['step']) + ', successful optimizer updates=' + str(c['optimizer_update_count']) + ', skipped=' + str(c['skipped_optimizer_steps']) + '.',
        '- Historical one-group optimizer has 641 tensors and 47,228,616 scalars (independently summed from exp_avg moments), WD=.01, initial_lr=6e-5; final LR=' + str(c['groups'][0]['hyperparameters']['lr']) + '.',
        '- Historical scheduler last_epoch=55980, base_lrs=[6e-5]. Nonzero final LR follows 20 skipped updates; do not force it to zero or change scheduler indexing.',
        '- Checked all ' + str(s['train_log_rows_checked']) + ' train_log.csv rows against the source lambda at each recorded optimizer_update_count: **all match**. This also checks early LR behavior, not merely final state.',
        '- Checkpoint optimizer IDs do not carry names; full names are observed on the instantiated model and every current parameter name/shape was verified present in checkpoint model state.', '',
        '| Successful updates | Lambda | Baseline LR | Backup backbone/head LR |', '|---:|---:|---:|---|']
    for p in s['lambda_probes']:
        lines.append(f"| {p['successful_updates']} | {p['lambda']:.12g} | {p['actual_lr']:.12g} | {p['recommended_lrs'][0]:.12g} / {p['recommended_lrs'][2]:.12g} |")
    lines += ['', '## Canonical paramwise backup: all recommended actual groups', '',
        '| Group name | Tensors | Scalar parameters | Initial LR | WD |', '|---|---:|---:|---:|---:|']
    for g in r['groups']:
        lines.append(f"| {g['group_name']} | {g['tensor_count']} | {g['scalar_parameter_count']:,} | {g['lr']:.12g} | {g['weight_decay']} |")
    lines += ['', '**Single predeclared factor = optimizer grouping policy** (head x10 + type-based norm/explicit-position no-decay). This bundle does not isolate the causal effects of head LR versus norm WD individually.', '',
        'Only proposed constructor change: replace `model.parameters()` by the groups returned from `canonical_paramwise_groups(model, base_lr=6e-5, weight_decay=.01, head_path="decode_head", positional_parameter_names=())[0]`. Existing AdamW constructor kwargs/defaults and exact scheduler source are preserved. Merely adding `paramwise_cfg` to YAML would do nothing because the current trainer does not consume it.', '',
        '**NO warmup.** Reference warmup1500 is documentation only, not an experiment change. Keep original polynomial power1/56000 and successful-update semantics. Fresh initialization from the same local published pretrained with seed42; do not resume the historical single-group optimizer into the new four-group optimizer.', '',
        'Keep all other Stage23 settings unchanged: B3, 8 labels, 1024 crop, CE, AMP, batch2/accumulation2, 56000 outer steps, 1400 validation interval, splits, all transforms and RareCrop settings, CWFA p=.3 epsilon4 norm_eps=1e-12 start_step28001, checkpoint/evaluation rules. Do not bundle another agent\'s RareCrop change. The full original configuration is embedded in JSON. No runnable experiment config, queue entry or integration change has been made.', '',
        '## CPU tests and safeguards', '',
        '- Synthetic self-test PASS: type-based normalization despite misleading names; full head/BatchNorm; ordinary blocks decay; ordinary bias decay; explicit position exemption; unknown position review rejection; exhaustive unique coverage; same-rule alias dedup; conflict alias rejection; one synthetic zero-gradient AdamW decay step; no helper scheduler/warmup.',
        '- Real CPU model PASS: all 641 parameters exhaustive and unique; 47,228,616 scalars; actual/recommended optimizers instantiated using same constructor with only its parameter argument changed; observed defaults identical.',
        '- Real CPU scheduler PASS: same AST scheduler instantiated on all backup groups; initial group LRs unchanged, x10 head/backbone ratio preserved until all reach zero; lambda probes monotonic, no hidden warmup.',
        '- Real SegFormer was not forward/backward trained and its optimizer was not stepped. The sole optimizer.step self-test used tiny synthetic tensors only.',
        '- Hashes of source files, pretrained files, checkpoint and log unchanged before/after audit. No existing model/trainer/config/run files modified; CUDA not initialized; no GPU/process queries or kills.', '',
        '## References (not imported experiment settings)', '']
    for ref in REFERENCES:
        lines += ['- [' + ref['organization'] + '](' + ref['url'] + '): ' + ref['settings']]
    lines += ['', d['reference_verification'], '',
        '## Source SHA256', '', '| Absolute path | SHA256 |', '|---|---|']
    for p, h in d['source_sha256'].items():
        lines.append('| `' + p + '` | `' + h + '` |')
    lines += ['', '## Artifacts', '',
        '- `' + str(OUTPUT / 'SEGFORMER_OPTIMIZER_AUDIT.md') + '`',
        '- `' + str(OUTPUT / 'SEGFORMER_OPTIMIZER_GROUPS.json') + '`',
        '- `' + str(Path(__file__).resolve()) + '` (new audit/backup helper; not imported by existing training code)', '',
        'Reproduction: run the helper with `--audit` under CUDA_VISIBLE_DEVICES empty, OMP_NUM_THREADS=2, MKL_NUM_THREADS=2, PYTHONDONTWRITEBYTECODE=1, HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, UAVSEG_MODEL_SOURCE=/home/jhy/models/segformer-b3-ade512, nice -n 19 ionice -c 3 and the stated Python. Existing output files intentionally cause refusal; use `--self-test` for a write-free helper retest.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.audit:
        audit()
    elif args.self_test:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        torch.set_num_threads(2)
        print(json.dumps(selftest(), indent=2))
    else:
        parser.error('Choose --audit or --self-test')
