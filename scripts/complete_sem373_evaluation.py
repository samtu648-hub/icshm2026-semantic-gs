#!/usr/bin/env python3
"""Continue fixed evaluation only after a saved endpoint's audit-only recovery."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);a=p.parse_args();base=ROOT/'outputs/quality/SEM-373';out=base/'evaluation-queue-r1';out.mkdir(exist_ok=False);r=dict(status='starting',stages=[]);start=time.monotonic()
 def save(): (out/'progress.json').write_text(json.dumps(r,indent=2)+'\n')
 try:
  att=json.loads((base/'joint-r1/completion-attestation.json').read_text());assert att['additional_training_steps']==0 and att['verified_matched_initial_model_both_optimizers_and_sampling']
  calls=[(name,[str(ROOT/'scripts/run_sem373_joint.py'),'--artifact-root',str(a.artifact_root),'--stage',name]) for name in ['eval-baseline','eval-control','eval-joint']]
  calls.append(('geometry-counterfactual',[str(ROOT/'scripts/diagnose_sem373_geometry.py'),'--artifact-root',str(a.artifact_root)]))
  env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
  for name,cmd in calls:
   r['status']='running_'+name;save()
   with (base/(name+'-r1.log')).open('x') as log:subprocess.run([sys.executable,*cmd],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   q=json.loads((base/(name+'-r1')/'result.json').read_text());assert q['status']=='completed';r['stages'].append(dict(stage=name,seconds=q['seconds']));save()
  r['status']='completed_fixed_evaluation'
 except BaseException as e:r.update(status='failed_preserved',error=repr(e));raise
 finally:r['seconds']=time.monotonic()-start;save();(out/'result.json').write_text(json.dumps(r,indent=2)+'\n')

if __name__=='__main__':main()
