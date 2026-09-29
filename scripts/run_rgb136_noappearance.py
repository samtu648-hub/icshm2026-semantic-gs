#!/usr/bin/env python3
"""Native Scaffold appearance_dim=0 all400 run; isolated attempts and shared GPU lock."""
import argparse,fcntl,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes,write_new_json
from scripts.run_quality_rgb import source_manifest,materialize_scene
EID='RGB-136-SCAFFOLD-NOAPPEARANCE-ALL400'
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--attempt',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 config=ROOT/'experiments/runs'/EID/'config.yaml';cfg=json.loads(config.read_text());parent=ROOT/cfg['parent_run'];out=ROOT/cfg['output_base']/a.attempt;out.mkdir(parents=True,exist_ok=False)
 source=a.artifact_root/'r1/third_party/scaffold-gs';cuda=a.artifact_root/'r1/toolchain/cuda-12.8';steps=32 if a.smoke else cfg['iterations']
 if not a.smoke:
  audit=ROOT/cfg['output_base']/'smoke-r3/uid_audit.json';assert json.loads(audit.read_text())['status']=='passed'
 materialize_scene(parent/'scene',out/'scene','scaffold');shutil.copy2(parent/'access.json',out/'access.json');shutil.copy2(parent/'split.json',out/'split.json')
 env=os.environ.copy();env.update(CUDA_HOME=str(cuda),PATH=str(cuda/'bin')+os.pathsep+env.get('PATH',''),LD_LIBRARY_PATH=str(cuda/'lib64')+os.pathsep+env.get('LD_LIBRARY_PATH',''),PYTHONPATH=os.pathsep.join(map(str,[source,source/'submodules/diff-gaussian-rasterization',source/'submodules/simple-knn',ROOT])),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TORCH_HOME=str(a.artifact_root/'r1/weights/torch'),WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
 saves=[str(steps)] if a.smoke else ['7000','15000','30000']
 cmd=[sys.executable,str(ROOT/'scripts/quality_all400_worker.py'),'--source',str(source),'--access',str(out/'access.json'),'--evidence',str(out/'worker_train.json'),'--seed',str(cfg['seed']),'--','--source_path',str(out/'scene'),'--model_path',str(out/'model'),'--eval','--resolution','1','--data_device','cpu','--iterations',str(steps),'--appearance_dim','0','--test_iterations',str(steps),'--save_iterations',*saves,'--checkpoint_iterations',*saves,'--quiet','--skip_metric_eval']
 code=['scripts/run_rgb136_noappearance.py','scripts/quality_all400_worker.py','scripts/run_quality_rgb.py','src/quality_protocol.py']
 for n in code:
  dst=out/'code'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dst)
 shutil.copy2(config,out/'code/config.json')
 r=dict(experiment_id=EID,status='queued',smoke=a.smoke,iterations=steps,seed=cfg['seed'],git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),command=cmd,config_sha256=sha256(config),manifest_sha256=sha256(ROOT/cfg['dataset_manifest']),split_sha256=sha256(out/'split.json'),code_sha256={n:sha256(ROOT/n) for n in code},started_unix=time.time(),artifacts={})
 write_new_json(out/'queued.json',r);start=time.monotonic()
 try:
  with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
   # Kernel lock waits without competing with other registered experiments.
   fcntl.flock(lock,fcntl.LOCK_EX)
   apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);r['gpu_processes_before']=apps
   if any(line.strip() and 'ToDesk' not in line for line in apps.splitlines()):raise RuntimeError('Uncoordinated GPU compute process; preserve attempt')
   r['queue_seconds']=time.monotonic()-start;native_start=time.monotonic()
   frozen={x['path']:x['sha256'] for x in json.loads((ROOT/'experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json').read_text())['artifacts']};verify_hashes(a.artifact_root,frozen)
   old=json.loads((parent/'result.json').read_text())['artifacts'];verify_hashes(parent,old)
   preparation=json.loads((parent/'scene/preparation_manifest.json').read_text());assert all(any((parent/'scene/images'/n).resolve().is_relative_to(base) for base in [(ROOT/'outputs').resolve(),a.artifact_root.resolve()]) and sha256(parent/'scene/images'/n)==h for n,h in preparation['image_hashes'].items()), 'Canonical shared image identity mismatch'
   manifest=source_manifest(source);assert manifest==json.loads((parent/'source_manifest.json').read_text()),'Upstream changed since RGB120'
   write_new_json(out/'source_manifest.json',manifest);r['status']='running';write_new_json(out/'started.json',r)
   with (out/'train.log').open('x') as log:subprocess.run(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   worker=json.loads((out/'worker_train.json').read_text());assert worker['status']=='completed_native_execution' and len(worker['actual_train'])==400 and not worker['actual_evaluation']
   from src.quality_scaffold import saved_namespace
   saved=saved_namespace(out/'model/cfg_args');previous=saved_namespace(parent/'model/cfg_args');assert saved['appearance_dim']==0 and saved['use_feat_bank'] is False
   assert {k:v for k,v in saved.items() if k not in ['appearance_dim','source_path','model_path']}=={k:v for k,v in previous.items() if k not in ['appearance_dim','source_path','model_path']}
   assert not list((out/'model').rglob('embedding_appearance.pt'))
   r['artifacts']={str(p.relative_to(out)):sha256(p) for p in sorted((out/'model').rglob('*')) if p.is_file() and (p.suffix in ['.pt','.pth','.ply'] or p.name=='cfg_args')}
   assert (out/f'model/point_cloud/iteration_{steps}/point_cloud.ply').exists()
   verify_hashes(parent,old);verify_hashes(a.artifact_root,frozen);verify_hashes(ROOT,r['code_sha256']);assert source_manifest(source)==manifest
   r.update(status='completed_smoke' if a.smoke else 'completed_all400',worker=worker,parent_and_frozen_exact=True,source_unchanged=True,native_seconds=time.monotonic()-native_start,free_bytes_end=shutil.disk_usage(out).free)
 except BaseException as e:r.update(status='failed_preserved',error=f'{type(e).__name__}: {e}');raise
 finally:
  r['seconds']=time.monotonic()-start;write_new_json(out/'result.json',r)
  with (ROOT/'experiments/runs'/EID/'report.md').open('a') as f:f.write(f'\n实际批次 {out.relative_to(ROOT)}：{r["status"]}，{r["seconds"]:.3f}秒，smoke={a.smoke}。质量验收另记。\n')
if __name__=='__main__':main()
