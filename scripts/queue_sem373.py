#!/usr/bin/env python3
"""Current authorized fixed SEM373 chain; failures stop and preserve each stage."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);a=p.parse_args();base=ROOT/'outputs/quality/SEM-373';out=base/'queue-r1';out.mkdir(exist_ok=False);r=dict(status='waiting_frozen_semantic',stages=[]);started=time.monotonic()
 def save(): (out/'progress.json').write_text(json.dumps(r,indent=2)+'\n')
 try:
  save()
  while not (base/'semantic-r1/result.json').exists():time.sleep(5)
  assert json.loads((base/'semantic-r1/result.json').read_text())['status']=='completed'
  calls=[('trained-gradient',[str(ROOT/'scripts/probe_sem373_trained.py'),'--artifact-root',str(a.artifact_root)])]
  calls += [(stage,[str(ROOT/'scripts/run_sem373_joint.py'),'--artifact-root',str(a.artifact_root),'--stage',stage]) for stage in ['control','joint','eval-baseline','eval-control','eval-joint']]
  env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
  for name,cmd in calls:
   r['status']='running_'+name;save()
   with (base/(name+'-r1.log')).open('x') as log:subprocess.run([sys.executable,*cmd],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   q=json.loads((base/(name+'-r1')/'result.json').read_text());assert q['status']=='completed';r['stages'].append(dict(stage=name,seconds=q['seconds']));save()
  r['status']='completed_training_and_evaluation'
 except BaseException as e:r.update(status='failed_preserved',error=repr(e));raise
 finally:r['seconds']=time.monotonic()-started;save();(out/'result.json').write_text(json.dumps(r,indent=2)+'\n')

if __name__=='__main__':main()
