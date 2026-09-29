#!/usr/bin/env python3
"""RGB136 matched SEM370/371/372 training and inference, isolated outputs."""
import argparse, fcntl, json, os, shutil, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json
from scripts import backbone_comparison_stage as shared
EID = 'SEM-375-SCAFFOLD-NOAPPEARANCE-ALL300'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artifact-root', type=Path, required=True)
    p.add_argument('--stage', choices=['prepare', 'mask2former', 'inference'], required=True)
    a = p.parse_args(); a.artifact_root = a.artifact_root.resolve()
    cfgpath = ROOT/'experiments/runs'/EID/'config.yaml'
    cfg = json.loads(cfgpath.read_text()); a.batch = ROOT/cfg['output_path']; a.rgb_run = ROOT/cfg['rgb_run']
    a.method = 'scaffold-gs'
    a.batch.mkdir(parents=True, exist_ok=True)
    out = a.batch/a.stage; out.mkdir(exist_ok=False)
    start = time.monotonic()
    parent = json.loads((a.rgb_run/'result.json').read_text())
    export = json.loads((a.rgb_run/'export/result.json').read_text())
    old = json.loads((shared.SEM/'mask2former/result.json').read_text())
    code = {**old['code_sha256'], 'scripts/backbone_comparison_stage.py': sha256(ROOT/'scripts/backbone_comparison_stage.py'), 'scripts/run_sem375_matched.py':sha256(Path(__file__))}
    frozen = {x['path']:x['sha256'] for x in json.loads((ROOT/'experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json').read_text())['artifacts']}
    r = dict(status='starting',experiment_id=EID,rgb_experiment_id=parent['experiment_id'],stage=a.stage,git_commit=cfg['git_commit'],config_sha256=sha256(cfgpath),seed=cfg['seed'],dataset_manifest=cfg['dataset_manifest'],manifest_sha256=sha256(ROOT/cfg['dataset_manifest']),split_id=cfg['split_id'],split_sha256=sha256(ROOT/cfg['split_path']),code_sha256=code,parent_result_sha256=sha256(a.rgb_run/'result.json'),quality_scope='training-view fit only, original five-class Mask2Former',started_unix=time.time())
    def verify():
        verify_hashes(ROOT,code); verify_hashes(a.rgb_run,parent['artifacts'])
        verify_hashes(a.rgb_run/'export/images',export['image_hashes']); verify_hashes(a.artifact_root,frozen)
        assert sha256(cfgpath)==r['config_sha256']
    try:
        assert parent['experiment_id']=='RGB-136-SCAFFOLD-NOAPPEARANCE-ALL400' and parent['status']=='completed_all400'
        assert export['status']=='completed' and len(export['image_hashes'])==544
        assert r['split_sha256']==export['split_sha256']
        verify()
        if a.stage=='prepare':
            dest=a.batch/'export'; dest.mkdir(); (dest/'images').mkdir()
            for n in export['image_hashes']: shutil.copyfile(a.rgb_run/'export/images'/n,dest/'images'/n)
            shutil.copyfile(a.rgb_run/'export/result.json',dest/'result.json')
            verify_hashes(dest/'images',export['image_hashes'])
            snap=a.batch/'code';snap.mkdir()
            for n in code:
                target=snap/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,target)
            r.update(rgb_copies_exact=544,status='completed',free_gib=shutil.disk_usage(a.batch).free/2**30)
        else:
            with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
                print('Waiting for shared GPU lock',flush=True); t=time.monotonic();fcntl.flock(lock,fcntl.LOCK_EX);r['queue_seconds']=time.monotonic()-t
                apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
                r['gpu_processes_before']=apps
                if any(line.strip() and 'ToDesk' not in line for line in apps.splitlines()):raise RuntimeError('Other GPU compute process active outside shared lock')
                runtime,torch,np=shared.setup(a);torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
                write_new_json(out/'started.json',r)
                if a.stage=='mask2former':
                    shared.train_stage(a,out,r,torch,np)
                    std=json.loads((shared.BASE/'standard/mask2former/result.json').read_text())
                    assert r['initial_model_state_sha256']==std['initial_model_state_sha256']
                    r['initial_state_matches_sem371']=True
                else:
                    # Shared inference routes only exact 'scaffold-gs' to old SEM370.
                    # Explicitly select the current batch checkpoint for RGB136.
                    a.method='scaffold-noappearance'
                    shared.infer_stage(a,out,r,torch,np,runtime)
                r.update(status='completed',peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        verify();r.update(frozen18_exact=True,rgb_endpoint_unchanged=True,rgb544_hashes_unchanged=True)
    except BaseException as e:
        r.update(status='failed_preserved',error=f'{type(e).__name__}: {e}');raise
    finally:
        r['seconds']=time.monotonic()-start;write_new_json(out/'result.json',r)
if __name__=='__main__':main()
