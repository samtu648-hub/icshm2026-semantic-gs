#!/usr/bin/env python3
"""Single all400 native Scaffold fit; resource estimates are not quality gates."""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256, write_new_json, verify_hashes
from scripts.run_quality_rgb import source_manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    a=p.parse_args(); run=a.run.resolve(); artifact=a.artifact_root.resolve()
    source=artifact/'r1/third_party/scaffold-gs'; cuda=artifact/'r1/toolchain/cuda-12.8'
    config=ROOT/'experiments/runs/RGB-120-ALL400/config.yaml'
    cfg=json.loads(config.read_text())
    with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (run/'started.json').exists():raise FileExistsError('Preserve previous run')
        frozen={x['path']:x['sha256'] for x in json.loads((ROOT/'experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json').read_text())['artifacts']}
        verify_hashes(artifact,frozen)
        manifest=source_manifest(source); write_new_json(run/'source_manifest.json',manifest)
        code=['scripts/run_quality_all400.py','scripts/quality_all400_worker.py','scripts/prepare_quality_all_data.py','src/quality_protocol.py']
        for n in code:
            dst=run/'code'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dst)
        env=os.environ.copy()
        env.update(CUDA_HOME=str(cuda),PYTHONPATH=os.pathsep.join(map(str,[source,source/'submodules/diff-gaussian-rasterization',source/'submodules/simple-knn',ROOT])),
            PATH=str(cuda/'bin')+os.pathsep+env.get('PATH',''),LD_LIBRARY_PATH=str(cuda/'lib64')+os.pathsep+env.get('LD_LIBRARY_PATH',''),
            TORCH_CUDA_ARCH_LIST='12.0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',WANDB_MODE='disabled',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TORCH_HOME=str(artifact/'r1/weights/torch'))
        cmd=[sys.executable,str(ROOT/'scripts/quality_all400_worker.py'),'--source',str(source),'--access',str(run/'access.json'),
            '--evidence',str(run/'worker_train.json'),'--seed','0','--','--source_path',str(run/'scene'),'--model_path',str(run/'model'),
            '--eval','--resolution','1','--data_device','cpu','--iterations',str(cfg['iterations']),
            '--test_iterations','30000','--save_iterations','7000','15000','30000',
            '--checkpoint_iterations','7000','15000','30000','--quiet','--skip_metric_eval']
        rec=dict(experiment_id=cfg['experiment_id'],status='running',seed=0,mode='full',config_sha256=sha256(config),
            split_sha256=sha256(run/'split.json'),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            command=cmd,code_sha256={n:sha256(ROOT/n) for n in code},quality_metrics=None,artifacts={})
        write_new_json(run/'started.json',rec); started=time.monotonic()
        try:
            with (run/'train.log').open('x') as log:subprocess.run(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            worker=json.loads((run/'worker_train.json').read_text())
            if worker['status']!='completed_native_execution' or len(worker['actual_train'])!=400 or worker['actual_evaluation']:raise ValueError('Actual training access mismatch')
            checkpoint=run/'model/point_cloud/iteration_30000/point_cloud.ply'
            if not checkpoint.is_file():raise ValueError('Missing endpoint')
            rec['artifacts']={str(p.relative_to(run)):sha256(p) for p in sorted((run/'model').rglob('*')) if p.is_file() and (p.suffix in ('.pt','.pth','.ply') or p.name=='cfg_args')}
            verify_hashes(artifact,frozen)
            if source_manifest(source)!=manifest:raise ValueError('Source changed')
            rec.update(status='completed_all400_rgb_pending_matched_semantics',worker=worker)
        except BaseException as e:rec.update(status='failed_preserved',error=f'{type(e).__name__}: {e}');raise
        finally:
            rec['seconds']=time.monotonic()-started;write_new_json(run/'result.json',rec)

if __name__=='__main__':main()
