#!/usr/bin/env python3
"""Exactly one approved SEM349 final fit, then two fresh processes; no retries."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_mask2former_final import FinalContext, require_approval
from src.quality_protocol import sha256, verify_hashes, write_new_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['artifact-root','batch','preflight','approval']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.batch=a.batch.resolve()
    if not a.batch.is_relative_to(ROOT/'outputs/quality/mask2former'):p.error('Use an isolated mask2former batch')
    ctx=FinalContext(ROOT,a.artifact_root,a.batch)
    approval_hash=require_approval(a.approval,ctx,a.preflight)
    ready=json.loads(a.preflight.read_text())
    if ready['status']!='ready_cpu_roles_sampling_budget_no_gpu_execution' or ready['binding']!=ctx.binding:
        raise ValueError('Readiness differs')
    verify_hashes(a.preflight.parent,ready['artifacts'])
    if any((a.batch/n).exists() for n in ['fit','fresh-1','fresh-2']):raise FileExistsError('No retry or duplicate fit in approved batch')
    out=a.batch/'queue';out.mkdir(exist_ok=False)
    r=dict(experiment_id=ctx.cfg['experiment_id'],status='started_approved_final_queue',approval_sha256=approval_hash,
           config_sha256=ctx.binding['config_sha256'],preflight_sha256=sha256(a.preflight),stages=[],target_labels_decoded=False)
    start=time.monotonic()
    try:
        for name,mode in [('fit','train'),('fresh-1','replay'),('fresh-2','replay')]:
            ctx.guard()
            command=[sys.executable,str(ROOT/'scripts/run_quality_mask2former_final.py'),
                     '--artifact-root',str(a.artifact_root),'--batch',str(a.batch),'--preflight',str(a.preflight),
                     '--approval',str(a.approval),'--output',str(a.batch/name),'--mode',mode]
            if mode=='replay':command+=['--run',str(a.batch/'fit')]
            remaining=5400-(time.monotonic()-start)
            if remaining<=0:raise TimeoutError('90-minute whole batch hard limit')
            with (out/(name+'.log')).open('x') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=remaining,check=True)
            result=json.loads((a.batch/name/'result.json').read_text())
            if mode=='replay' and (result.get('probabilities_exact') is not True or result['compared_views']!=74
                                  or result['run_result_sha256']!=sha256(a.batch/'fit/result.json')):
                raise ValueError('Missing all74 exact fresh replay')
            r['stages'].append({'stage':name,'result_sha256':sha256(a.batch/name/'result.json')})
        ctx.finish();r.update(status='completed_final219_fit_and_two74_fresh_replays_no_quality_scoring',rgb_exports_verified=293,frozen_verified=18)
    except BaseException as exc:
        r.update(status='failed_final_queue_no_retry',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        r['seconds']=time.monotonic()-start;write_new_json(out/'result.json',r)


if __name__=='__main__':main()
