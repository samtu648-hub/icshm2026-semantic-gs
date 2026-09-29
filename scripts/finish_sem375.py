#!/usr/bin/env python3
"""Wait for the authorized queue, then close RGB and independent comparison audits."""
import argparse,json,os,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);a=p.parse_args();b=ROOT/'outputs/quality/SEM-375/r1';out=b/'finish';out.mkdir(exist_ok=False)
 env=os.environ.copy();cuda=a.artifact_root/'r1/toolchain/cuda-12.8';env.update(CUDA_HOME=str(cuda),PATH=str(cuda/'bin')+os.pathsep+env.get('PATH',''),LD_LIBRARY_PATH=str(cuda/'lib64')+os.pathsep+env.get('LD_LIBRARY_PATH',''),PYTHONPATH=str(ROOT),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
 r=dict(status='waiting_fit_inference',started_unix=time.time(),stages=[])
 def save():
  p=out/'status.tmp';p.write_text(json.dumps(r,indent=2)+'\n');p.replace(out/'status.json')
 try:
  save()
  while True:
   q=json.loads((b/'queue/status.json').read_text())
   if q['status'].startswith('failed'):raise RuntimeError('Training queue failed; no retry')
   if q['status'].startswith('completed'):break
   time.sleep(5)
  tasks=[('fresh_rgb','verify_sem375_rgb.py',['--artifact-root',str(a.artifact_root)]),('audit','audit_sem375.py',['--batch',str(b)]),('comparison','build_sem375_comparison.py',['--batch',str(b)])]
  for name,script,args in tasks:
   r['status']=name;save();shutil.copyfile(ROOT/'scripts'/script,b/'code/scripts'/script)
   with (out/(name+'.log')).open('x') as log:subprocess.run([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   r['stages'].append(name);save()
  r['status']='completed_audits_and_artifacts_pending_human_readable_report'
 except BaseException as e:r.update(status='failed_preserved',error=str(e));raise
 finally:save()
if __name__=='__main__':main()
