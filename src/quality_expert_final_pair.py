"""Same-trajectory paired budget comparison; historical models are references."""
import json
from pathlib import Path
from src.quality_protocol import sha256, verify_hashes


def validate_recipe(config, base):
    parent=json.loads((Path(__file__).resolve().parents[1]/config['study']['parent_config']).read_text())
    fixed=set(parent)-{'experiment_id','parent_experiment','partitions','study','git_commit'}
    if any(config[k]!=parent[k] for k in fixed):
        raise ValueError('Final fitting changed a SEM336 scientific recipe field')
    if config['experiment_id']!='SEM-362-B2-PAIRED-FINAL' or config['partitions']!=['development'] or config['study']['endpoints']!=[3000,6000]:
        raise ValueError('Wrong final paired role or endpoints')
    if config['study']['automatic_promotion']:
        raise ValueError('Automatic promotion forbidden')


def open_control(root, config, partition, expert):
    if partition not in config['partitions'] or expert != 'fiveclass':
        raise ValueError('Only registered final fiveclass fitting allowed')
    study = config['study']
    verify_hashes(root,{study['parent_config']:study['parent_config_sha256']})
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



def save_endpoint(model, binding, access, steps, path):
    """Save before further optimization; assert serialization leaves RNG intact."""
    import torch
    import hashlib
    cpu = torch.get_rng_state().clone()
    cuda = [v.clone() for v in torch.cuda.get_rng_state_all()]
    state = {n:v.detach().cpu() for n,v in model.state_dict().items()}
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise FloatingPointError("Nonfinite paired endpoint")
    torch.save({"state_dict":state,"binding":binding,"access":access,"steps":steps},path)
    if not torch.equal(cpu,torch.get_rng_state()) or any(not torch.equal(x,y) for x,y in zip(cuda,torch.cuda.get_rng_state_all())):
        raise ValueError("Checkpoint serialization changed training RNG")
    return {"steps":steps,"tensor_count":len(state),"checkpoint_sha256":sha256(path),
        "rng_preserved":True,"cpu_rng_sha256":hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
        "cuda_rng_sha256":[hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for v in cuda]}


def enforce_transfer_budget(root, config, pending_bytes=0):
    import shutil
    spec=config['study']['budget']
    folder=Path(root)/spec['aggregate_root']
    used=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
    maximum=int(spec['maximum_new_gib']*2**30)
    free=shutil.disk_usage(folder).free
    if pending_bytes<0 or used+pending_bytes>maximum or free<max(pending_bytes,maximum-used):
        raise OSError('Actual available space or registered aggregate output budget insufficient')
    return used
