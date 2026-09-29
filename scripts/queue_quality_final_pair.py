#!/usr/bin/env python3
"""SEM362 authorized one-fit queue; no retries and no fixed disk reserve."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,write_new_json,verify_hashes
from src.quality_expert_final_pair import enforce_transfer_budget
from scripts.run_quality_expert_final_pair import CONFIG,CODE


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--batch',type=Path,required=True)
    a=p.parse_args();a.batch=a.batch.resolve()
    if not a.batch.is_relative_to(ROOT/'outputs/quality/final_transfer'):
        p.error('Isolated final_transfer batch required')
    a.batch.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/CONFIG).read_text());start=time.time();deadline=start+1800
    code=sorted(set(CODE+('scripts/queue_quality_final_pair.py','scripts/audit_quality_expert_final_pair.py','tests/test_quality_expert_final_pair.py')))
    rec=dict(experiment_id=cfg['experiment_id'],status='started',started_unix=start,deadline=deadline,
             config_sha256=sha256(ROOT/CONFIG),code_sha256={n:sha256(ROOT/n) for n in code},events=[],automatic_promotion=False)
    receipt=ROOT/'outputs/quality/final_transfer/SEM-362-preparation-r1/authorization.json'
    approval=json.loads(receipt.read_text())
    if approval['config362_sha256']!=rec['config_sha256'] or approval['disk_reserve_gib']!=0 or not approval['user_confirmation']:
        raise ValueError('Missing authorized matching final-transfer scope')
    rec['authorization_sha256']=sha256(receipt)
    write_new_json(a.batch/'started.json',rec)
    common=[sys.executable,ROOT/'scripts/run_quality_expert_final_pair.py','--rgb-run',ROOT/cfg['study']['rgb_run'],
            '--rendered-root',ROOT/cfg['study']['rendered_root'],'--target-root',ROOT/cfg['study']['target_root'],
            '--artifact-root',a.artifact_root,'--partition','development','--expert','fiveclass','--deadline',str(deadline)]
    try:
        for stage in ('preflight','fit','audit'):
            verify_hashes(ROOT,{CONFIG:rec['config_sha256'],**rec['code_sha256']})
            enforce_transfer_budget(ROOT,cfg)
            if stage=='audit':
                cmd=[sys.executable,ROOT/'scripts/audit_quality_expert_final_pair.py','--run',a.batch/'fit',
                     '--rgb-run',ROOT/cfg['study']['rgb_run'],'--rendered-root',ROOT/cfg['study']['target_root'],
                     '--artifact-root',a.artifact_root,'--output',a.batch/stage,'--deadline',str(deadline)]
            else:
                cmd=common+['--mode','preflight' if stage=='preflight' else 'full','--output',a.batch/stage]
                if stage=='fit':cmd+=['--preflight',a.batch/'preflight']
            with (a.batch/(stage+'.log')).open('x') as log:
                subprocess.run(list(map(str,cmd)),cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                    MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,deadline-time.time()))
            result=json.loads((a.batch/stage/'result.json').read_text())
            expected={'preflight':'passed_expert_cpu_preflight','fit':'completed_expert_fit_pending_independent_audit',
                      'audit':'completed_independent_expert_replay'}[stage]
            if result['status']!=expected:raise ValueError('Incomplete stage '+stage)
            event=dict(stage=stage,result_sha256=sha256(a.batch/stage/'result.json'),seconds=time.time()-start)
            rec['events'].append(event);write_new_json(a.batch/(stage+'-accepted.json'),event)
            print(stage,'accepted',round(time.time()-start,2),flush=True)
        rec.update(status='completed_final_paired_B2',new_fits=1,endpoints=[3000,6000],prediction_views=74,
                   independent_replay_processes=4,aggregate_output_bytes=enforce_transfer_budget(ROOT,cfg))
    except BaseException as exc:
        rec.update(status='stopped_preserved_final_pair',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        rec['seconds']=time.time()-start;write_new_json(a.batch/'result.json',rec)


if __name__=='__main__':main()
