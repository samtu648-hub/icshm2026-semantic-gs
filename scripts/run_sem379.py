#!/usr/bin/env python3
"""Run only the registered SEM379 sampler intervention with SEM378's trainer."""
import argparse, fcntl, hashlib, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.quality_protocol import sha256, verify_hashes, write_new_json
import backbone_comparison_stage as shared
EID='SEM-379-CABLE-REGION-SAMPLING';BASE=ROOT/'outputs/quality/SEM-378/r1'
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--stage',choices=['mask2former','inference'],required=True);a=p.parse_args()
 cfgpath=ROOT/'experiments/runs'/EID/'config.yaml';cfg=read(cfgpath);a.batch=ROOT/cfg['output_path'];a.method='standard-3dgs';a.rgb_run=ROOT/cfg['rgb_run']
 bind=read(a.batch/'binding.json');export=read(a.batch/'export/result.json');prior=read(BASE/'mask2former/result.json')
 out=a.batch/a.stage;out.mkdir(exist_ok=False);start=time.monotonic()
 r=dict(schema_version=1,status='waiting_gpu',experiment_id=EID,git_commit=cfg['git_commit'],seed=cfg['seed'],config_sha256=sha256(cfgpath),binding_sha256=sha256(a.batch/'binding.json'),quality_scope='all300 training fit; unlabeled100 and trajectory144 qualitative only')
 def verify():
  assert sha256(cfgpath)==bind['config_sha256'];verify_hashes(ROOT,bind['code_sha256'])
  verify_hashes(a.batch/'export/images',export['image_hashes']);verify_hashes(a.rgb_run,bind['rgb140_artifacts']);verify_hashes(a.artifact_root,bind['frozen18'])
  verify_hashes(shared.SEM/'labels',bind['label_hashes']);assert sha256(BASE/'mask2former/model-6000.pt')==bind['baseline_model_sha256']
 try:
  verify();assert read(a.batch/'sampling-audit.json')['parent_6000_trace_exact']
  with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
   print('Waiting for shared GPU lock; no other processes will be interrupted',flush=True)
   while True:
    fcntl.flock(lock,fcntl.LOCK_EX)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
    if not any(l.strip() and 'ToDesk' not in l for l in apps.splitlines()):break
    fcntl.flock(lock,fcntl.LOCK_UN);print('Waiting for GPU tasks outside lock',flush=True);time.sleep(30)
   r['queue_seconds']=time.monotonic()-start;r['gpu_processes_before']=apps
   runtime,torch,np=shared.setup(a);torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
   r['runtime']={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'python':sys.version}
   write_new_json(out/'started.json',r)
   if a.stage=='mask2former':
    from scripts.run_quality_all300 import mask2former
    from src.quality_static_training import TrainingLabels
    import src.quality_mask2former as api
    import src.quality_mask2former_training as training
    from src.semantic_cable_sampling import CableRegionSampler
    split=read(ROOT/cfg['split_path']);recipe=read(ROOT/'experiments/runs/SEM-370-ALL300/config.yaml')
    labels=TrainingLabels(shared.SEM/'labels',split['semantic_train'],read(shared.SEM/'label_manifest.json'),[1320,989]);assert len(labels.allowed)==300
    r.update(trained_steps=0,access=dict(train=split['semantic_train'],independent_holdout=False),binding=dict(seed=cfg['seed'],split_sha256=bind['split_sha256'],label_manifest_sha256=bind['label_manifest_sha256'],export_result_sha256=sha256(a.batch/'export/result.json')))
    loader=api.load_official_model;parent_sample=training.sampled_crop
    def capture(*args,**kwargs):
     model,loaded=loader(*args,**kwargs)
     if 'initial_model_state_sha256' not in r:
      digest=hashlib.sha256()
      for k,v in sorted(model.state_dict().items()):
       digest.update(k.encode());digest.update(str(v.dtype).encode());digest.update(str(tuple(v.shape)).encode());digest.update(v.detach().cpu().contiguous().numpy().tobytes())
      r['initial_model_state_sha256']=digest.hexdigest();assert r['initial_model_state_sha256']==prior['initial_model_state_sha256']
      assert (out/'resolved_model.yaml').read_bytes()==(BASE/'mask2former/resolved_model.yaml').read_bytes()
     return model,loaded
    api.load_official_model=capture;training.sampled_crop=CableRegionSampler(parent_sample,cfg['seed'],cfg['mixture'])
    try:mask2former(a,out,r,labels,recipe)
    finally:api.load_official_model=loader;training.sampled_crop=parent_sample
    actual=[json.loads(l)['sample'] for l in (out/'training.jsonl').read_text().splitlines()]
    plan=[json.loads(l) for l in (a.batch/'sampling-plan.jsonl').read_text().splitlines()]
    assert actual==plan and len(actual)==6000
    r.update(sampling6000_matches_plan=True,resolved_recipe_matches_sem378=True,initial_state_matches_sem378=True,artifacts={p.name:sha256(p) for p in out.glob('*.pt')})
   else:shared.infer_stage(a,out,r,torch,np,runtime)
   r.update(status='completed',peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
  verify();r.update(rgb544_hashes_unchanged=True,rgb140_endpoint_unchanged=True,sem378_unchanged=True,frozen18_exact=True)
 except BaseException as e:r.update(status='failed_preserved',error=f'{type(e).__name__}: {e}');raise
 finally:r['seconds']=time.monotonic()-start;write_new_json(out/'result.json',r)
if __name__=='__main__':main()
