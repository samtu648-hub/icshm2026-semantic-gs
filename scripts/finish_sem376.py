#!/usr/bin/env python3
"""CPU-only completion after both authorized crossover inference arms."""
import argparse,json,os,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);a=p.parse_args();b=ROOT/'outputs/quality/SEM-376/r1';q=b/'finish';q.mkdir(exist_ok=False);r=dict(status='waiting_cross_inference',started_unix=time.time(),stages=[])
 def save():
  p=q/'status.tmp';p.write_text(json.dumps(r,indent=2)+'\n');p.replace(q/'status.json')
 env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
 try:
  save()
  while True:
   state=json.loads((b/'queue/status.json').read_text())['status']
   if state.startswith('failed'):raise RuntimeError('Inference failed, no retry')
   if state.startswith('completed'):break
   time.sleep(5)
  for stage,script,args in [('audit','audit_sem376.py',['--artifact-root',str(a.artifact_root)]),('comparison','build_sem376.py',[])]:
   r['status']=stage;save();shutil.copyfile(ROOT/'scripts'/script,b/'code/scripts'/script)
   with (q/(stage+'.log')).open('x') as log:subprocess.run([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   r['stages'].append(stage)
  r['status']='completed_cpu_audit_and_artifacts_pending_visual_report'
 except BaseException as e:r.update(status='failed_preserved',error=str(e));raise
 finally:save()
if __name__=='__main__':main()
