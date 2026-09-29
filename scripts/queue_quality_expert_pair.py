#!/usr/bin/env python3
"""One-shot SEM336 serial continuation; finishes at fixed OOF comparison."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json
from scripts.run_quality_expert_pair import CODE as TRAIN_CODE, CONFIG
from scripts.evaluate_quality_expert_pair import CODE as EVAL_CODE


def accepted_first(result, cfg_hash):
    if result['status'] != 'completed_expert_fit_pending_independent_audit' or result['trained_steps'] != 6000:
        raise ValueError('First full training did not complete6000 steps')
    if not result.get('paired_control_3000',{}).get('rng_preserved') or result.get('endpoints') != [3000,6000] or result['binding']['config_sha256'] != cfg_hash or result['access']['partition'] != 'oof-0':
        raise ValueError('First full model/control provenance differs')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('batch','artifact-root'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--wait-first',action='store_true',required=True,help='First full fit must already exist; wait on GPU lock, not a timer')
    a = p.parse_args()
    batch = a.batch.resolve()
    output = batch/'queue'
    with (ROOT/'outputs/quality/duration_queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        output.mkdir(parents=True,exist_ok=False)
        start = time.monotonic()
        cfg = json.loads((ROOT/CONFIG).read_text())
        code = sorted(set(TRAIN_CODE+EVAL_CODE+('scripts/audit_quality_expert_pair.py','scripts/queue_quality_expert_pair.py')))
        first = batch/'full-oof-0'
        record = {'experiment_id':cfg['experiment_id'],'status':'waiting_first_full_model','automatic_promotion':False,
            'seed':cfg['seed'],'config_sha256':sha256(ROOT/CONFIG),'code_sha256':{n:sha256(ROOT/n) for n in code},
            'first_started_sha256':sha256(first/'started.json'),'events':[],
            'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
        for name in (*code,CONFIG):
            dest = output/'code'/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,dest)
        write_new_json(output/'started.json',record)
        def event(name,**kw):
            row = {'index':len(record['events']),'event':name,'unix':time.time(),**kw}
            write_new_json(output/f"event-{row['index']:03d}.json",row)
            record['events'].append(row); print(json.dumps(row),flush=True)
        def unchanged():
            verify_hashes(ROOT,record['code_sha256'])
            if sha256(ROOT/CONFIG) != record['config_sha256'] or sha256(first/'started.json') != record['first_started_sha256']:
                raise ValueError('Active duration queue configuration/first run changed')
            if shutil.disk_usage(batch).free < cfg['study']['minimum_free_gib']*1024**3:
                raise OSError('Duration queue100GiB reserve unavailable')
        def run(stage,cmd,expected):
            unchanged(); event('started',stage=stage)
            with (output/(stage+'.log')).open('x') as log:
                subprocess.run(list(map(str,cmd)),cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
            unchanged()
            r = json.loads((batch/stage/'result.json').read_text())
            if r['status'] != expected: raise ValueError('Incomplete duration stage: '+stage)
            event('completed',stage=stage,result_sha256=sha256(batch/stage/'result.json'))
        try:
            unchanged(); event('waiting_first_full_gpu_lock')
            # A local process dependency, not a recurring timer or future task.
            with (ROOT/'outputs/quality/gpu.lock').open('a') as first_lock:
                fcntl.flock(first_lock,fcntl.LOCK_EX)
                unchanged()
                accepted_first(json.loads((first/'result.json').read_text()),record['config_sha256'])
                event('accepted_first',result_sha256=sha256(first/'result.json'))
            common = [sys.executable,ROOT/'scripts/run_quality_expert_pair.py','--rgb-run',ROOT/cfg['study']['rgb_run'],
                '--rendered-root',ROOT/cfg['study']['rendered_root'],'--artifact-root',a.artifact_root,'--expert','fiveclass']
            for part in cfg['partitions']:
                if part != 'oof-0':
                    for mode in ('preflight','smoke','full'):
                        stage = mode+'-'+part
                        cmd = common+['--partition',part,'--mode',mode,'--output',batch/stage]
                        if mode == 'full': cmd += ['--smoke',batch/('smoke-'+part)]
                        run(stage,cmd,{'preflight':'passed_expert_cpu_preflight','smoke':'passed_expert_smoke','full':'completed_expert_fit_pending_independent_audit'}[mode])
                stage = 'audit-'+part
                run(stage,[sys.executable,ROOT/'scripts/audit_quality_expert_pair.py','--run',batch/('full-'+part),
                    '--rgb-run',ROOT/cfg['study']['rgb_run'],'--rendered-root',ROOT/cfg['study']['rendered_root'],
                    '--artifact-root',a.artifact_root,'--output',batch/stage],'completed_independent_expert_replay')
            run('comparison',[sys.executable,ROOT/'scripts/evaluate_quality_expert_pair.py','--batch',batch,
                '--artifact-root',a.artifact_root,'--output',batch/'comparison'],'completed_fixed_paired_219_oof_duration_comparison_requires_review')
            record.update(status='completed_paired_duration_batch_requires_human_review',comparison_sha256=sha256(batch/'comparison/result.json'))
        except BaseException as exc:
            record.update(status='stopped_duration_batch_not_method_rejection',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            record['seconds'] = time.monotonic()-start
            write_new_json(output/'result.json',record)
            print(json.dumps({'status':record['status'],'output':str(output)}),flush=True)


if __name__ == '__main__':
    main()
