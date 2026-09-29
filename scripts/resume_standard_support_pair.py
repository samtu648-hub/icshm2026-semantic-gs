#!/usr/bin/env python3
"""Recover an externally terminated candidate without repeating the control."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);a=p.parse_args()
    out=ROOT/'outputs/quality/rgb141';record_path=out/'pair_recovery-r1.json'
    assert json.loads((out/'control-r1/result.json').read_text())['status']=='completed_all400'
    assert json.loads((out/'support-r1/result.json').read_text())['status']=='failed_preserved_external_sigterm'
    if record_path.exists():raise FileExistsError(record_path)
    record={'status':'running','started_unix':time.time(),'steps':[],
        'preserved_failure':'support-r1','active_candidate':'support-r2','science_changed':False,'driver_pid':os.getpid()}
    env=os.environ.copy();env['TORCH_HOME']=str(a.artifact_root/'r1/weights/torch');env['PYTHONDONTWRITEBYTECODE']='1'
    def save():record_path.write_text(json.dumps(record,indent=2)+'\n')
    try:
        save();record['stage']='candidate_training';save()
        command=[sys.executable,str(ROOT/'scripts/run_standard_support.py'),'--artifact-root',str(a.artifact_root),
            '--experiment','RGB-143-STANDARD-SUPPORT-DENSIFY','--run-name','support-r2']
        tick=time.monotonic();subprocess.run(command,cwd=ROOT,env=env,check=True)
        record['steps'].append({'stage':'candidate_training','seconds':time.monotonic()-tick});save()
        for arm,eid in [('control-r1','RGB-142-STANDARD-BUDGET-CTRL'),('support-r2','RGB-143-STANDARD-SUPPORT-DENSIFY')]:
            record['stage']='evaluation_'+arm;save();tick=time.monotonic()
            command=[str(a.artifact_root/'r1/envs/r1-cuda128/bin/python'),str(ROOT/'scripts/evaluate_standard_support.py'),
                '--artifact-root',str(a.artifact_root),'--experiment',eid,'--run',str(out/arm)]
            subprocess.run(command,cwd=ROOT,env=env,check=True)
            record['steps'].append({'stage':record['stage'],'seconds':time.monotonic()-tick});save()
        record['status']='completed_train_and_evaluate'
    except BaseException as exc:record.update(status='failed_preserved',error=repr(exc));raise
    finally:record['finished_unix']=time.time();save()

if __name__=='__main__':main()
