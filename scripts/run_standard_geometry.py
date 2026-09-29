#!/usr/bin/env python3
"""Matched photo-placement arm; immutable run directory, source/config/data bindings."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,write_new_json
from scripts.run_quality_rgb import source_manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=30000);p.add_argument('--resume-state',type=Path)
    a=p.parse_args();a.output=a.output.resolve();a.config=a.config.resolve();c=json.loads(a.config.read_text())
    if a.iterations==30000 and not a.resume_state:
        for name in ['control-smoke-r1','candidate-smoke-r1']:
            q=json.loads((ROOT/'outputs/quality/rgb144'/name/'result.json').read_text())
            assert q['status']=='completed_smoke'
        events=[json.loads(s) for s in (ROOT/'outputs/quality/rgb144/candidate-smoke-r1/densification.jsonl').read_text().splitlines()]
        assert events[-1]['guided_total']>0,'Mechanism did not activate'
    a.output.mkdir(parents=True,exist_ok=False)
    source=ROOT/c['source'];scene=ROOT/c['data']['scene'];cuda=a.artifact_root/'r1/toolchain/cuda-12.8';python=a.artifact_root/'r1/envs/r1-cuda128/bin/python'
    assert sha256(scene/'sparse/0/points3D.ply')==c['data']['initial_ply_sha256']
    assert json.loads((ROOT/'outputs/quality/rgb144/preflight-r2/result.json').read_text())['status']=='passed'
    env=os.environ.copy();env.update(CUDA_HOME=str(cuda),PATH=str(cuda/'bin')+os.pathsep+env.get('PATH',''),
        LD_LIBRARY_PATH=str(cuda/'lib64')+os.pathsep+env.get('LD_LIBRARY_PATH',''),PYTHONPATH=os.pathsep.join(map(str,[source,source/'submodules/diff-gaussian-rasterization',source/'submodules/simple-knn',ROOT])),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TORCH_HOME=str(a.artifact_root/'r1/weights/torch'),WANDB_MODE='disabled',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    command=[str(python),str(ROOT/'scripts/worker_standard_geometry.py'),'--config',str(a.config),'--output',str(a.output)]
    if a.resume_state:command+=['--resume-state',str(a.resume_state.resolve())]
    command+=['--','--source',str(source),'--access',str(ROOT/'outputs/quality/all_data/RGB-120-r2/access.json'),
        '--evidence',str(a.output/'worker.json'),'--seed','0','--expected-init-policy','same RGB130 original 303-view DLT initialization; all400 photometric training','--',
        '--source_path',str(scene),'--model_path',str(a.output/'model'),'--resolution','1','--data_device','cpu','--iterations',str(a.iterations),
        '--densify_grad_threshold','0.0002','--densify_grad_abs_threshold','0.0004','--test_iterations','-1','--save_iterations',
        *([str(x) for x in (7000,15000,30000)] if a.iterations==30000 else [str(a.iterations)]),'--checkpoint_iterations',str(a.iterations),'--quiet']
    code=['scripts/run_standard_geometry.py','scripts/worker_standard_geometry.py','scripts/backbone_all400_worker.py','src/standard_geometry_split.py','src/photo_geometry.py']
    result={'experiment_id':c['experiment_id'],'status':'starting','command':command,'seed':0,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'config_sha256':sha256(a.config),'geometry_cache_sha256':sha256(ROOT/c['geometry']['cache']),'manifest_sha256':sha256(ROOT/c['dataset_manifest']),
        'split_sha256':sha256(ROOT/c['split_path']),'initialization_ply_sha256':sha256(scene/'sparse/0/points3D.ply'),'source':source_manifest(source),
        'code_sha256':{f:sha256(ROOT/f) for f in code},'registered_geometry':c['geometry'],'actual_iterations':a.iterations,'free_bytes_start':shutil.disk_usage(a.output).free}
    for f in code:
        dest=a.output/'code'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dest)
    shutil.copy2(a.config,a.output/'code/config.yaml');start=time.monotonic()
    try:
        with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);result['lock_wait_seconds']=time.monotonic()-start
            write_new_json(a.output/'started.json',result);training_start=time.monotonic()
            with (a.output/'train.log').open('x') as log,(a.output/'resources.jsonl').open('x') as monitor:
                process=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
                while process.poll() is None:
                    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
                    monitor.write(json.dumps({'seconds':time.monotonic()-training_start,'gpu':gpu.stdout.strip(),'free_bytes':shutil.disk_usage(a.output).free})+'\n');monitor.flush();time.sleep(5)
                if process.returncode:raise RuntimeError(f'Native worker exit {process.returncode}')
            w=json.loads((a.output/'worker.json').read_text());assert len(w['actual_train'])==400 and not w['actual_evaluation'];assert w['initialization']['common_point_count']==72292
            result.update(status='completed_all400' if a.iterations==30000 else 'completed_smoke',worker=w,training_wall_seconds_excluding_lock=time.monotonic()-training_start,
                artifacts={str(q.relative_to(a.output)):sha256(q) for q in (a.output/'model').rglob('*') if q.is_file()})
    except BaseException as exc:
        result.update(status='failed_preserved',error=repr(exc));raise
    finally:
        result.update(seconds=time.monotonic()-start,free_bytes_end=shutil.disk_usage(a.output).free);write_new_json(a.output/'result.json',result)


if __name__=='__main__':main()
