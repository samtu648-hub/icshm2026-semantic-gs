#!/usr/bin/env python3
"""Durable sequential driver for the already-authorized SEM379 experiment."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/quality/SEM-379/r1'
p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);a=p.parse_args()
r={'status':'running','started_unix':time.time(),'steps':[]}
try:
 for name,command in [('training',['scripts/run_sem379.py','--artifact-root',str(a.artifact_root),'--stage','mask2former']),('completion',['scripts/finish_sem379.py','--artifact-root',str(a.artifact_root)])]:
  r['stage']=name;(OUT/'driver.json').write_text(json.dumps(r,indent=2)+'\n')
  with (OUT/('train.log' if name=='training' else 'completion-queue.log')).open('x') as log:
   subprocess.run([sys.executable,*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  r['steps'].append(name)
 r['status']='artifacts_ready_pending_visual_review'
except BaseException as e:r.update(status='failed_preserved',error=repr(e));raise
finally:r['finished_unix']=time.time();(OUT/'driver.json').write_text(json.dumps(r,indent=2)+'\n')
