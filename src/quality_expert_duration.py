"""Single-variable duration contracts; the SEM331 implementation stays untouched."""
import json
from pathlib import Path
from src.quality_protocol import sha256, verify_hashes


def validate_recipe(config, base):
    permitted = {'experiment_id', 'parent_experiment', 'partitions', 'allowed_rgb_parents', 'study', 'training'}
    if set(config) != set(base) | {'study'}:
        raise ValueError('Unexpected duration recipe fields')
    if any(config[k] != base[k] for k in base if k not in permitted):
        raise ValueError('Non-duration recipe changed')
    expected_training = {**base['training'], 'steps': 6000}
    if config['training'] != expected_training or base['training']['steps'] != 3000:
        raise ValueError('Only full3000 to6000 duration may change')
    if config['experiment_id'] != 'SEM-335-SCAFFOLD-EXPERT-DURATION' or config['partitions'] != ['oof-0', 'oof-1', 'oof-2']:
        raise ValueError('Unregistered experiment/partition')
    if config['allowed_rgb_parents'] != ['RGB-110-SCAFFOLD-CTRL'] or config['study']['expert'] != 'fiveclass':
        raise ValueError('Only Scaffold fiveclass duration is registered')
    if config['study']['automatic_promotion'] or config['study']['automatic_development_fit']:
        raise ValueError('Promotion/development fitting is not authorized in this study')


def open_control(root, config, partition, expert):
    if partition not in config['partitions'] or expert != 'fiveclass':
        raise ValueError('Duration experiment excludes development/binary fitting')
    study = config['study']
    verify_hashes(root, {study['baseline_config']: study['baseline_config_sha256'],
        study['baseline_trainer']: study['baseline_trainer_sha256'],
        study['baseline_fusion']+'/result.json': study['baseline_fusion_sha256']})
    validate_recipe(config, json.loads((root / study['baseline_config']).read_text()))
    spec = study['controls'][partition]
    verify_hashes(root, {spec['run']+'/result.json': spec['sha256'], spec['audit']+'/result.json': spec['audit_sha256']})
    result = json.loads((root / spec['run'] / 'result.json').read_text())
    audit = json.loads((root / spec['audit'] / 'result.json').read_text())
    if result['status'] != 'completed_expert_fit_pending_independent_audit' or result['trained_steps'] != 3000:
        raise ValueError('Incomplete baseline expert')
    if audit['status'] != 'completed_independent_expert_replay' or audit['run_result_sha256'] != spec['sha256'] or audit['binding'] != result['binding']:
        raise ValueError('Baseline independent acceptance differs')
    if result['binding']['expert'] != expert or result['access']['partition'] != partition:
        raise ValueError('Baseline branch/fold differs')
    verify_hashes(root / spec['run'], result['artifacts'])
    verify_hashes(root, result['code_sha256'])
    return result, root / spec['run']


def check_binding(record, control):
    for key in ('access', 'class_counts', 'class_weights', 'train_label_hashes', 'runtime'):
        if record[key] != control[key]:
            raise ValueError('Duration control differs: '+key)
    a, b = record['binding'], control['binding']
    if set(a) != set(b) or any(a[k] != b[k] for k in a if k != 'config_sha256'):
        raise ValueError('Duration control model/data/pixel binding differs')


def compare_states(actual, expected):
    import torch
    if actual.keys() != expected.keys():
        raise ValueError('Control checkpoint tensor names differ')
    mismatches = [k for k in actual if actual[k].shape != expected[k].shape or actual[k].dtype != expected[k].dtype
        or not torch.equal(actual[k].detach().cpu(), expected[k])]
    if mismatches:
        raise ValueError('Control reproduction failed; no6000-step inference: '+', '.join(mismatches[:6]))
    return {'exact': True, 'tensor_count': len(actual), 'tolerance': 0}


def check_prefix(model, control_path, control):
    import torch
    checkpoint = control_path / 'expert.pt'
    if sha256(checkpoint) != control['artifacts']['expert.pt']:
        raise ValueError('Baseline checkpoint changed')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload['binding'] != control['binding'] or payload['access'] != control['access'] or payload['steps'] != 3000:
        raise ValueError('Baseline checkpoint payload differs')
    return {**compare_states(model.state_dict(), payload['state_dict']),
        'checkpoint_sha256': sha256(checkpoint), 'steps': 3000}
