#!/usr/bin/env python3
"""Serial continuation of the user's full-data comparison, no quality gates."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import write_new_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);a=p.parse_args();base=ROOT/'outputs/quality/backbone_comparison';q=base/'queue-r1';q.mkdir(exist_ok=False)
 cuda=a.artifact_root/'r1/toolchain/cuda-12.8';env=os.environ.copy();env.update(CUDA_HOME=str(cuda),PATH=str(cuda/'bin')+os.pathsep+env.get('PATH',''),LD_LIBRARY_PATH=str(cuda/'lib64')+os.pathsep+env.get('LD_LIBRARY_PATH',''),TORCH_HOME=str(a.artifact_root/'r1/weights/torch'),PYTHONPATH=str(ROOT),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',PYTHONDONTWRITEBYTECODE='1')
 r=dict(status='waiting_standard',stages=[],started_unix=time.time());start=time.monotonic()
 def status():
  tmp=q/'status.tmp';tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(q/'status.json')
 def run(name,cmd,result):
  r['status']=name;status()
  with (q/(name+'.log')).open('x') as log:subprocess.run([sys.executable,*cmd],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  rec=json.loads(result.read_text())
  if not rec['status'].startswith('completed'):raise RuntimeError('Stage incomplete '+name)
  r['stages'].append(dict(name=name,seconds=rec['seconds'],result=str(result.relative_to(ROOT))));status()
 try:
  status()
  while not (base/'standard-full-r1/result.json').exists():time.sleep(10)
  if json.loads((base/'standard-full-r1/result.json').read_text())['status']!='completed_all400':raise RuntimeError('Standard incomplete; preserved')
  rgb=ROOT/'scripts/run_backbone_all400.py'
  for attempt,smoke in [('mip-smoke-r1',True),('mip-full-r1',False)]:
   run(attempt,[str(rgb),'--artifact-root',str(a.artifact_root),'--method','mip-splatting','--attempt',attempt]+(['--smoke'] if smoke else []),base/attempt/'result.json')
  stage=ROOT/'scripts/backbone_comparison_stage.py'
  arms=[('standard','standard-3dgs',base/'standard-full-r1'),('mip','mip-splatting',base/'mip-full-r1'),('scaffold','scaffold-gs',ROOT/'outputs/quality/all_data/RGB-120-r2')]
  # Requested reconstruction order is complete before semantic fits.
  for name,method,model in arms:
   for task in ['export']+([] if name=='scaffold' else ['preflight','mask2former']):
    run(name+'-'+task,[str(stage),'--artifact-root',str(a.artifact_root),'--method',method,'--rgb-run',str(model),'--batch',str(base/name),'--stage',task],base/name/task/'result.json')
  for name,method,model in arms:
   for task in ['rgb-evaluation','inference']:
    run(name+'-'+task,[str(stage),'--artifact-root',str(a.artifact_root),'--method',method,'--rgb-run',str(model),'--batch',str(base/name),'--stage',task],base/name/task/'result.json')
  r['status']='completed_training_exports_and_metrics_pending_visual_review'
 except BaseException as e:r.update(status='failed_preserved',error=str(e));raise
 finally:r['seconds']=time.monotonic()-start;status();write_new_json(q/'result.json',r)
if __name__=='__main__':main()
