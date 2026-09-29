#!/usr/bin/env python3
"""Sequential authorized 30k pair and evaluation; no automatic parameter search."""
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
    out=ROOT/'outputs/quality/rgb141';record={'status':'running','steps':[]};start=time.monotonic()
    resultpath=out/'pair_result.json'
    if resultpath.exists():raise FileExistsError(resultpath)
    env=os.environ.copy();env['TORCH_HOME']=str(a.artifact_root/'r1/weights/torch');env['PYTHONDONTWRITEBYTECODE']='1'
    try:
        for arm,eid in [('control','RGB-142-STANDARD-BUDGET-CTRL'),('support','RGB-143-STANDARD-SUPPORT-DENSIFY')]:
            command=[sys.executable,str(ROOT/'scripts/run_standard_support.py'),'--artifact-root',str(a.artifact_root),'--experiment',eid,'--run-name',arm+'-r1']
            tick=time.monotonic();subprocess.run(command,check=True,env=env,cwd=ROOT)
            record['steps'].append({'phase':'train','arm':arm,'seconds':time.monotonic()-tick})
            resultpath.write_text(json.dumps(record,indent=2)+'\n')
        for arm,eid in [('control','RGB-142-STANDARD-BUDGET-CTRL'),('support','RGB-143-STANDARD-SUPPORT-DENSIFY')]:
            command=[str(a.artifact_root/'r1/envs/r1-cuda128/bin/python'),str(ROOT/'scripts/evaluate_standard_support.py'),
                '--artifact-root',str(a.artifact_root),'--experiment',eid,'--run',str(out/(arm+'-r1'))]
            tick=time.monotonic();subprocess.run(command,check=True,env=env,cwd=ROOT)
            record['steps'].append({'phase':'evaluate','arm':arm,'seconds':time.monotonic()-tick})
            resultpath.write_text(json.dumps(record,indent=2)+'\n')
        record['status']='completed_train_and_evaluate'
    except BaseException as exc:record.update(status='failed_preserved',error=repr(exc));raise
    finally:
        record['seconds']=time.monotonic()-start;resultpath.write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':main()
