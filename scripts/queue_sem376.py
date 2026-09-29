#!/usr/bin/env python3
import argparse,json,os,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes,write_new_json
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);a=p.parse_args();b=ROOT/'outputs/quality/SEM-376/r1';q=b/'queue';q.mkdir(parents=True,exist_ok=False)
 cfg=json.loads((ROOT/'experiments/runs/SEM-376-CROSS-RGB-INFERENCE/config.yaml').read_text());input_hashes={};snap=b/'code/scripts';snap.mkdir(parents=True)
 for f in ['run_sem376_cross.py','queue_sem376.py']:shutil.copyfile(ROOT/'scripts'/f,snap/f)
 for arm in cfg['arms'].values():
  rgb=ROOT/arm['rgb'];sem=ROOT/arm['semantic'];fit=json.loads((sem/'mask2former/result.json').read_text());verify_hashes(ROOT,fit['code_sha256'])
  exp=json.loads((rgb/'export/result.json').read_text());verify_hashes(rgb/'export/images',exp['image_hashes']);verify_hashes(sem/'mask2former',fit['artifacts'])
  for p in [rgb/'export/result.json',sem/'mask2former/result.json',sem/'mask2former/model-6000.pt']:input_hashes[str(p.relative_to(ROOT))]=sha256(p)
 write_new_json(b/'input_hashes.json',input_hashes)
 cuda=a.artifact_root/'r1/toolchain/cuda-12.8';env=os.environ.copy();env.update(CUDA_HOME=str(cuda),PATH=str(cuda/'bin')+os.pathsep+env.get('PATH',''),LD_LIBRARY_PATH=str(cuda/'lib64')+os.pathsep+env.get('LD_LIBRARY_PATH',''),PYTHONPATH=str(ROOT),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
 r=dict(status='starting',training_steps=0,started_unix=time.time(),stages=[])
 def save():
  p=q/'status.tmp';p.write_text(json.dumps(r,indent=2)+'\n');p.replace(q/'status.json')
 try:
  save()
  for arm in cfg['arms']:
   for fresh in [False,True]:
    name=arm+('-fresh' if fresh else '-inference');r['status']=name;save()
    with (q/(name+'.log')).open('x') as log:subprocess.run([sys.executable,str(ROOT/'scripts/run_sem376_cross.py'),'--artifact-root',str(a.artifact_root),'--arm',arm]+(['--fresh'] if fresh else []),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    rec=json.loads((b/arm/('fresh' if fresh else 'inference')/'result.json').read_text());assert rec['status']=='completed';r['stages'].append(dict(name=name,seconds=rec['seconds']));save()
  verify_hashes(ROOT,input_hashes);r['status']='completed_cross_inference_pending_cpu_audit'
 except BaseException as e:r.update(status='failed_preserved',error=str(e));raise
 finally:save()
if __name__=='__main__':main()
