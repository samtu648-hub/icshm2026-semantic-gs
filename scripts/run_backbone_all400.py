#!/usr/bin/env python3
"""Isolated full-resolution all400 native run; preserve every attempt."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,write_new_json
from scripts.run_quality_rgb import source_manifest

def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--method',choices=['standard-3dgs','mip-splatting'],required=True);p.add_argument('--attempt',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 base=ROOT/'outputs/quality/backbone_comparison';base.mkdir(exist_ok=True)
 out=base/a.attempt;out.mkdir(exist_ok=False)
 source=a.artifact_root/'r1/third_party'/a.method;parent=ROOT/'outputs/quality/all_data/RGB-120-r2';cuda=a.artifact_root/'r1/toolchain/cuda-12.8'
 eid='RGB-130-STANDARD-ALL400' if a.method=='standard-3dgs' else 'RGB-131-MIP-ALL400';config=ROOT/'experiments/runs'/eid/'config.yaml'
 env=os.environ.copy();env.update(CUDA_HOME=str(cuda),PATH=str(cuda/'bin')+os.pathsep+env.get('PATH',''),LD_LIBRARY_PATH=str(cuda/'lib64')+os.pathsep+env.get('LD_LIBRARY_PATH',''),PYTHONPATH=os.pathsep.join(map(str,[source,source/'submodules/diff-gaussian-rasterization',source/'submodules/simple-knn',ROOT])),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TORCH_HOME=str(a.artifact_root/'r1/weights/torch'),WANDB_MODE='disabled',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
 steps=32 if a.smoke else 30000
 cmd=[sys.executable,str(ROOT/'scripts/backbone_all400_worker.py'),'--source',str(source),'--access',str(parent/'access.json'),'--evidence',str(out/'worker.json'),'--seed','0','--','--source_path',str(parent/'scene'),'--model_path',str(out/'model'),'--resolution','1','--data_device','cpu','--iterations',str(steps),'--test_iterations','-1','--save_iterations',str(steps),'--checkpoint_iterations',str(steps),'--quiet']
 r=dict(experiment_id=eid,smoke=a.smoke,status='starting',command=cmd,config_sha256=sha256(config),split_sha256=sha256(parent/'split.json'),source=source_manifest(source),code_sha256={n:sha256(ROOT/n) for n in ['scripts/backbone_all400_worker.py','scripts/run_backbone_all400.py']},started_unix=time.time(),free_bytes_start=__import__('shutil').disk_usage(out).free)
 start=time.monotonic()
 try:
  with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
   r['gpu_processes_before']=apps
   if any(line.strip() and 'ToDesk' not in line for line in apps.splitlines()):raise RuntimeError('Other GPU compute process active')
   write_new_json(out/'started.json',r)
   with (out/'train.log').open('x') as log:
    proc=subprocess.Popen(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
    with (out/'resources.jsonl').open('x') as monitor:
     while proc.poll() is None:
      gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,power.draw,clocks.sm','--format=csv,noheader,nounits'],capture_output=True,text=True)
      monitor.write(json.dumps(dict(seconds=time.monotonic()-start,gpu=gpu.stdout.strip(),free_bytes=__import__('shutil').disk_usage(out).free))+'\n');monitor.flush();time.sleep(10)
    if proc.returncode:raise RuntimeError(f'Native process exited {proc.returncode}; see train.log')
   worker=json.loads((out/'worker.json').read_text());assert len(worker['actual_train'])==400 and not worker['actual_evaluation']
   r.update(status='completed_smoke' if a.smoke else 'completed_all400',worker=worker,artifacts={str(x.relative_to(out)):sha256(x) for x in (out/'model').rglob('*') if x.is_file()})
 except BaseException as e:r.update(status='failed_preserved',error=str(e));raise
 finally:
  r['seconds']=time.monotonic()-start;write_new_json(out/'result.json',r)
  with (ROOT/'experiments/runs'/eid/'report.md').open('a') as f:f.write(f'\n实际批次 {out.relative_to(ROOT)}：{r["status"]}，{r["seconds"]:.2f}秒，smoke={a.smoke}。详见该目录result.json/train.log；无质量结论。\n')
if __name__=='__main__':main()
