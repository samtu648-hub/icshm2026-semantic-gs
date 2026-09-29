#!/usr/bin/env python3
"""Run the registered RGB137 Standard-3DGS candidate in an isolated directory."""
import argparse,fcntl,json,os,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,write_new_json
from scripts.run_quality_rgb import source_manifest
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--scene',type=Path,required=True);p.add_argument('--attempt',required=True);a=p.parse_args();a.scene=a.scene.resolve();out=ROOT/'outputs/quality/rgb137'/a.attempt;out.mkdir(parents=True,exist_ok=False)
 source=a.artifact_root/'r1/third_party/standard-3dgs';cuda=a.artifact_root/'r1/toolchain/cuda-12.8';access=ROOT/'outputs/quality/all_data/RGB-120-r2/access.json';env=os.environ.copy();env.update(CUDA_HOME=str(cuda),PATH=str(cuda/'bin')+os.pathsep+env.get('PATH',''),LD_LIBRARY_PATH=str(cuda/'lib64')+os.pathsep+env.get('LD_LIBRARY_PATH',''),PYTHONPATH=os.pathsep.join(map(str,[source,source/'submodules/diff-gaussian-rasterization',source/'submodules/simple-knn',ROOT])),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TORCH_HOME=str(a.artifact_root/'r1/weights/torch'),WANDB_MODE='disabled',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
 cmd=[sys.executable,str(ROOT/'scripts/backbone_all400_worker.py'),'--source',str(source),'--access',str(access),'--evidence',str(out/'worker.json'),'--seed','0','--expected-init-policy','all400 released-track DLT initialization; all400 photometric training','--','--source_path',str(a.scene),'--model_path',str(out/'model'),'--resolution','1','--data_device','cpu','--iterations','30000','--test_iterations','-1','--save_iterations','30000','--checkpoint_iterations','30000','--quiet']
 r=dict(experiment_id='RGB-137-STANDARD-ALL400-INIT400',status='starting',command=cmd,seed=0,config_sha256=sha256(ROOT/'experiments/runs/RGB-137-STANDARD-ALL400-INIT400/config.yaml'),scene_manifest_sha256=sha256(a.scene/'initialization_manifest.json'),source=source_manifest(source),code_sha256={x:sha256(ROOT/x) for x in ('scripts/backbone_all400_worker.py','scripts/run_rgb137_all400_init.py')},free_bytes_start=shutil.disk_usage(out).free);start=time.monotonic()
 try:
  with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);r['gpu_processes_before']=apps
   if any(x.strip() and 'ToDesk' not in x for x in apps.splitlines()):raise RuntimeError('Other GPU compute process active')
   write_new_json(out/'started.json',r)
   with (out/'train.log').open('x') as log:
    proc=subprocess.Popen(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
    with (out/'resources.jsonl').open('x') as mon:
     while proc.poll() is None:
      gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,power.draw,clocks.sm','--format=csv,noheader,nounits'],capture_output=True,text=True);mon.write(json.dumps(dict(seconds=time.monotonic()-start,gpu=gpu.stdout.strip(),free_bytes=shutil.disk_usage(out).free))+'\n');mon.flush();time.sleep(10)
    if proc.returncode:raise RuntimeError(f'Native process exited {proc.returncode}')
   w=json.loads((out/'worker.json').read_text());assert len(w['actual_train'])==400 and not w['actual_evaluation'];r.update(status='completed_all400',worker=w,artifacts={str(x.relative_to(out)):sha256(x) for x in (out/'model').rglob('*') if x.is_file()})
 except BaseException as e:r.update(status='failed_preserved',error=f'{type(e).__name__}: {e}');raise
 finally:r['seconds']=time.monotonic()-start;write_new_json(out/'result.json',r)
if __name__=='__main__':main()
