#!/usr/bin/env python3
"""One authorized matched fit followed by full prediction; no automatic retries."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);a=p.parse_args();q=ROOT/'outputs/quality/SEM-375/r1/queue';q.mkdir(exist_ok=False)
 cuda=a.artifact_root/'r1/toolchain/cuda-12.8';env=os.environ.copy();env.update(CUDA_HOME=str(cuda),PATH=str(cuda/'bin')+os.pathsep+env.get('PATH',''),LD_LIBRARY_PATH=str(cuda/'lib64')+os.pathsep+env.get('LD_LIBRARY_PATH',''),TORCH_HOME=str(a.artifact_root/'r1/weights/torch'),PYTHONPATH=str(ROOT),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',PYTHONDONTWRITEBYTECODE='1')
 r=dict(status='starting',started_unix=time.time(),stages=[])
 def save():
  tmp=q/'status.tmp';tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(q/'status.json')
 try:
  assert json.loads((q.parent/'prepare/result.json').read_text())['status']=='completed'
  for stage in ['mask2former','inference']:
   r['status']=stage;save()
   with (q/(stage+'.log')).open('x') as log:subprocess.run([sys.executable,str(ROOT/'scripts/run_sem375_matched.py'),'--artifact-root',str(a.artifact_root),'--stage',stage],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   rec=json.loads((q.parent/stage/'result.json').read_text());assert rec['status']=='completed';r['stages'].append(dict(stage=stage,seconds=rec['seconds']))
  r['status']='completed_fit_and_inference_pending_cpu_visual_audit'
 except BaseException as e:r.update(status='failed_preserved',error=str(e));raise
 finally:save()
if __name__=='__main__':main()
