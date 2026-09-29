#!/usr/bin/env python3
"""Serial SEM344 probe/replay then fixed trajectory; no optimizer entrypoint."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes,write_new_json
CONFIG='experiments/runs/SEM-344-EXTERNAL-CAMERA-PREFLIGHT/config.yaml'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['artifact-root','data-root','probe','replay','trajectory','journal']:
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.journal.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/CONFIG).read_text());worker=ROOT/'scripts/run_quality_external_camera.py'
    code={n:sha256(ROOT/n) for n in ['scripts/run_quality_external_camera.py','src/quality_external_camera.py','scripts/queue_quality_external_camera.py']}
    r={'status':'running','config_sha256':sha256(ROOT/CONFIG),'code_sha256':code,'stages':[],'training':False}
    write_new_json(a.journal/'started.json',r)
    try:
        for phase,base,repeat in [('probe',a.probe,None),('probe',a.replay,a.probe),('trajectory',a.trajectory,None)]:
            for fold in range(3):
                for stage in ['render','infer']:
                    verify_hashes(ROOT,code)
                    dest=base/f'{stage}-oof-{fold}'
                    args=[sys.executable,str(worker),'--artifact-root',str(a.artifact_root),'--data-root',str(a.data_root),
                          '--output',str(dest),'--phase',phase,'--fold',str(fold),'--stage',stage]
                    if stage=='infer':args+=['--render-root',str(base/f'render-oof-{fold}')]
                    if repeat:args+=['--repeat-from',str(repeat/f'{stage}-oof-{fold}')]
                    if phase=='trajectory':args+=['--probe-root',str(a.replay)]
                    t=time.monotonic()
                    if dest.exists():
                        rr=json.loads((dest/'result.json').read_text())
                        if rr['status']!='passed_external_camera_stage' or rr['config_sha256']!=r['config_sha256'] or any(rr['code_sha256'].get(k)!=v for k,v in code.items() if k!=str(Path(__file__).relative_to(ROOT))):
                            raise ValueError('Existing stage cannot be reused')
                        verify_hashes(dest,rr['artifacts'])
                    else:
                        logfile=a.journal/f'{base.name}-{stage}-{fold}.log'
                        with logfile.open('xb') as log:
                            subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=cfg['resource']['maximum_seconds_per_stage'])
                    rr=json.loads((dest/'result.json').read_text())
                    if rr['status']!='passed_external_camera_stage':raise ValueError('Stage failed')
                    r['stages'].append({'directory':str(dest.relative_to(ROOT) if dest.is_absolute() else dest),'result_sha256':sha256(dest/'result.json'),'reused':time.monotonic()-t<1})
                    print(json.dumps({'completed':str(dest),'stages':len(r['stages'])}),flush=True)
        r['status']='completed_camera_probe_replay_and_fixed_trajectory'
    except BaseException as exc:
        r.update(status='failed_external_camera_queue',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:write_new_json(a.journal/'result.json',r)

if __name__=='__main__':main()
