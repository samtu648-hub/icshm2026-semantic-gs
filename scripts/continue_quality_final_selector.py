#!/usr/bin/env python3
"""Continue SEM363 with an explicit recovered B2 binding; never fit a head."""
import argparse
import copy
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_quality_final_selector as base
import numpy as np
from PIL import Image
from src.quality_protocol import sha256, verify_hashes, write_new_json

RECEIPT = 'outputs/quality/final_transfer/SEM-362-recovery-preparation-r1/authorization.json'
CODE = base.CODE + ['scripts/continue_quality_final_selector.py', 'scripts/audit_quality_final_selector_transfer.py']


def relocate(config, receipt):
    """Only execution locations change; the sealed scientific configuration stays."""
    c = copy.deepcopy(config)
    batch = Path(receipt['gpu_batch'])
    if batch.is_absolute() or '..' in batch.parts or not batch.is_relative_to('outputs/quality/final_transfer'):
        raise ValueError('Invalid recovery batch')
    if receipt['new_b2_run'] != str(batch/'fit') or receipt['new_b2_audit'] != str(batch/'audit'):
        raise ValueError('Recovery dependency is outside accepted batch')
    c['b2_run'] = receipt['new_b2_run']
    c['b2_audit'] = receipt['new_b2_audit']
    c['final_sources']['b2'] = receipt['new_b2_run']
    return c


def execution():
    receipt = base.read(ROOT/RECEIPT)
    original = base.read(ROOT/base.CONFIG)
    verify_hashes(ROOT, {base.CONFIG: receipt['config363_sha256'], original['b2_config']: receipt['config362_sha256']})
    c = relocate(original, receipt)
    saved = ROOT/receipt['saved_fit']
    verify_hashes(ROOT, {receipt['saved_fit']+'/result.json': receipt['saved_fit_result_sha256']})
    old = base.read(saved/'result.json')
    if old['status'] != 'completed_fit_only_pending_final_B2' or old['config_sha256'] != receipt['config363_sha256']:
        raise ValueError('Unaccepted saved fit')
    verify_hashes(saved, old['fit_artifacts'])
    verify_hashes(ROOT, old['code_sha256'])
    queue = base.read(ROOT/receipt['gpu_batch']/'result.json')
    b2 = base.read(ROOT/c['b2_run']/'result.json')
    audit = base.read(ROOT/c['b2_audit']/'result.json')
    if (queue['status'] != 'completed_final_paired_B2' or queue['config_sha256'] != receipt['config362_sha256']
            or b2['binding']['config_sha256'] != receipt['config362_sha256']
            or audit['run_result_sha256'] != sha256(ROOT/c['b2_run']/'result.json')
            or audit['status'] != 'completed_independent_expert_replay'
            or [r['steps'] for r in audit['fresh_reloads']] != [3000, 3000, 6000, 6000]):
        raise ValueError('Unaccepted recovered paired B2')
    verify_hashes(ROOT, queue['code_sha256'])
    if queue['seconds'] + receipt['previous_gpu_seconds'] > 1800:
        raise ValueError('Aggregate GPU budget exceeded')
    return c, receipt, saved, old


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(); a.output = a.output.resolve(); a.fit_only = False
    if not a.output.is_relative_to(ROOT/'outputs/quality/final_transfer'):
        parser.error('Isolated transfer output required')
    started = time.time()
    receipt = base.read(ROOT/RECEIPT)
    a.deadline = started + receipt['remaining_cpu_seconds']
    a.output.mkdir(parents=True, exist_ok=False)
    rec = dict(status='started', config_sha256=sha256(ROOT/base.CONFIG),
               execution_receipt_sha256=sha256(ROOT/RECEIPT), code_sha256={n:sha256(ROOT/n) for n in CODE},
               new_cpu_fits=0, new_gpu_training=False, automatic_promotion=False)
    write_new_json(a.output/'started.json', rec)
    try:
        c, receipt, saved, old = execution()
        rec.update({k:c[k] for k in ('experiment_id','git_commit','seed','dataset_manifest','split_id')})
        write_new_json(a.output/'effective_config.json', c)
        rec['effective_config_sha256'] = sha256(a.output/'effective_config.json')
        for n in [base.CONFIG, RECEIPT, *CODE]:
            dest=a.output/'code'/n; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/n,dest)
        proof = base.provenance(a,c)
        write_new_json(a.output/'provenance.json', proof)
        for n in ('model.npz','fit.json','audit_fit.json'): shutil.copyfile(saved/n,a.output/n)
        rec['reused_fit_result_sha256'] = sha256(saved/'result.json')
        with np.load(a.output/'model.npz',allow_pickle=False) as z: model={k:z[k] for k in z.files}
        (a.output/'predictions').mkdir(); sources=base.final_sources(c)
        for i,name in enumerate(c['prediction_names']):
            base.guard(a,c); pred=base.predict(base.load(sources,name),model)
            Image.fromarray(pred).save(a.output/'predictions'/name)
            if (i+1)%10==0: print('sealed prediction progress',i+1,flush=True)
        files={str(f.relative_to(a.output)):sha256(f) for f in a.output.rglob('*') if f.is_file() and
               (f.parent.name=='predictions' or f.name in ('model.npz','fit.json','audit_fit.json','effective_config.json'))}
        write_new_json(a.output/'predictions_sealed.json',dict(files=files,views=74,sealed_unix=time.time(),config_sha256=rec['config_sha256']))
        with (a.output/'audit.log').open('x') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/audit_quality_final_selector_transfer.py'),
                '--output',str(a.output),'--artifact-root',str(a.artifact_root),'--deadline',str(a.deadline)],
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,a.deadline-time.time()))
        result=base.score(a,c); base.freeze(c,a.artifact_root); base.guard(a,c)
        verify_hashes(ROOT,{base.CONFIG:rec['config_sha256'],RECEIPT:rec['execution_receipt_sha256'],**rec['code_sha256'],**c['input_hashes']})
        verify_hashes(a.output,files)
        rec.update(status='completed_fixed_final_transfer_development',**result,seconds=time.time()-started,
                   peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                   frozen_rgb_verified=293,frozen_artifacts_verified=18,new_74_predictions=74,new_74_scores=1,
                   predictions_seal_sha256=sha256(a.output/'predictions_sealed.json'),
                   audit_sha256=sha256(a.output/'audit.json'),per_view_sha256=sha256(a.output/'per_view.json'))
        write_new_json(a.output/'result.json',rec)
        print('SEM363 completed; delta raw pp',rec['pooled']['delta_miou_pp'],flush=True)
    except BaseException as exc:
        write_new_json(a.output/'failure.json',dict(type=type(exc).__name__,message=str(exc),seconds=time.time()-started)); raise


if __name__ == '__main__': main()
