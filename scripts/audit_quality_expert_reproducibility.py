#!/usr/bin/env python3
"""Read-only8-step repeat comparison plus isolated synthetic CUDA operator probe.

Does not rerun the full experiment, change its deterministic flags or relax gates.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json


def compare_repeats(a, b):
    import torch
    if a.keys() != b.keys(): raise ValueError('State tensor names differ')
    rows = []
    for name,value in a.items():
        other = b[name]
        if value.shape != other.shape or value.dtype != other.dtype: raise ValueError('State shape/dtype differs')
        if not torch.isfinite(value).all() or not torch.isfinite(other).all(): raise ValueError('Nonfinite state')
        if not torch.equal(value,other):
            delta = (value.double()-other.double()).abs()
            rows.append({'tensor':name,'elements_different':int((value!=other).sum()),'max_absolute_delta':float(delta.max())})
    return {'state_tensors':len(a),'different_tensors':len(rows),'different_elements':sum(r['elements_different'] for r in rows),
        'top_differences':sorted(rows,key=lambda r:r['max_absolute_delta'],reverse=True)[:12]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('first','repeat','artifact-root','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--operator-probe',action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    record = {'experiment_id':'SEM-335-SCAFFOLD-EXPERT-DURATION','kind':'engineering_repeatability_diagnosis',
        'status':'initializing','quality_metrics':None,'target_labels_decoded':False,'full_fit_retried':False,
        'code_sha256':sha256(Path(__file__)),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    try:
        import torch
        runs = [json.loads((path/'result.json').read_text()) for path in (a.first,a.repeat)]
        if any(r['status'] != 'passed_expert_smoke' or r['trained_steps'] != 8 for r in runs):
            raise ValueError('Requires two finite8-step engineering runs')
        equal = {k:runs[0][k] == runs[1][k] for k in ('binding','access','code_sha256','runtime','class_counts','class_weights','amp')}
        if not all(equal.values()): raise ValueError('Repeat changed recorded recipe or inputs')
        states = []
        source_proofs = {}
        for path,r in zip((a.first,a.repeat),runs):
            verify_hashes(path,r['artifacts'])
            payload = torch.load(path/'expert.pt',map_location='cpu',weights_only=True)
            if payload['binding'] != r['binding'] or payload['access'] != r['access'] or payload['steps'] != 8:
                raise ValueError('Smoke checkpoint payload differs')
            states.append(payload['state_dict'])
            source_proofs[str(path.relative_to(ROOT) if path.is_absolute() else path)] = sha256(path/'result.json')
        protocol = json.loads((ROOT/'experiments/runs/QUAL-001-FOUNDATION/config.yaml').read_text())
        frozen = {r['path']:r['sha256'] for r in json.loads((ROOT/protocol['frozen_manifest']).read_text())['artifacts']}
        verify_hashes(a.artifact_root,frozen)
        record.update(seed=runs[0]['binding']['seed'],recorded_inputs_equal=equal,source_result_hashes=source_proofs,
            runtime=runs[0]['runtime'],first_loss=[r['loss']['first'] for r in runs],last_loss=[r['loss']['last'] for r in runs],
            state_comparison=compare_repeats(*states),trained_rgb_parameters=False)
        if a.operator_probe:
            with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable for isolated operator probe')
                # Mirrors the fiveclass decoder interpolation, not the entire
                # model. Strict mode lives in this process only.
                import torch.nn.functional as F
                torch.manual_seed(20260805)
                x = torch.randn((1,5,192,192),device='cuda',dtype=torch.bfloat16,requires_grad=True)
                before = torch.are_deterministic_algorithms_enabled()
                strict = {'input_shape':list(x.shape),'output_shape':[1,5,768,768],'dtype':str(x.dtype),
                    'algorithm':'F.interpolate bilinear align_corners=False backward','synthetic_only':True}
                try:
                    torch.use_deterministic_algorithms(True)
                    F.interpolate(x,size=(768,768),mode='bilinear',align_corners=False).float().square().mean().backward()
                    strict['status'] = 'strict_backward_supported'
                except RuntimeError as exc:
                    strict.update(status='strict_backward_rejected',error=str(exc))
                finally:
                    torch.use_deterministic_algorithms(before)
                record['operator_probe'] = strict
        verify_hashes(a.artifact_root,frozen)
        for path,r in zip((a.first,a.repeat),runs): verify_hashes(path,r['artifacts'])
        if sha256(Path(__file__)) != record['code_sha256']: raise ValueError('Diagnosis code changed')
        record.update(status='completed_repeatability_diagnosis_not_quality_comparison',frozen_verified=len(frozen))
    except BaseException as exc:
        record.update(status='repeatability_diagnosis_failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['seconds'] = time.monotonic()-started
        write_new_json(a.output/'result.json',record)
        print(json.dumps(record),flush=True)


if __name__ == '__main__':
    main()
