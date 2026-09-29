#!/usr/bin/env python3
"""One fixed serial training and evaluation chain; failures preserved."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);a=p.parse_args();base=ROOT/'outputs/quality/SEM-374';out=base/'queue-r2';out.mkdir(exist_ok=False);r=dict(status='starting',stages=[]);start=time.monotonic()
 def save(): (out/'progress.json').write_text(json.dumps(r,indent=2)+'\n')
 try:
  pre=json.loads((base/'preflight-r4/result.json').read_text());assert pre['status']=='completed' and pre['no_parameter_updates']
  assert json.loads((base/'tracks-independent-audit.json').read_text())['status']=='passed'
  env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
  for stage in ['control','joint','eval-control','eval-joint','depth-baseline']:
   r['status']='running_'+stage;save()
   with (base/(stage+'-r2.log')).open('x') as log:subprocess.run([sys.executable,str(ROOT/'scripts/run_sem374_structure.py'),'--artifact-root',str(a.artifact_root),'--stage',stage,'--attempt','r2'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   q=json.loads((base/(stage+'-r2')/'result.json').read_text());assert q['status']=='completed';r['stages'].append(dict(stage=stage,seconds=q['seconds']));save()
  r['status']='completed'
 except BaseException as e:r.update(status='failed_preserved',error=repr(e));raise
 finally:r['seconds']=time.monotonic()-start;save();(out/'result.json').write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
