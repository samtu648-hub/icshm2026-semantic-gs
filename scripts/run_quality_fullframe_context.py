#!/usr/bin/env python3
"""SEM346 native whole-image context, saved-model forward only; no target GT."""
import argparse,fcntl,json,shutil,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes,write_new_json
CONFIG='experiments/runs/SEM-346-FULLFRAME-CONTEXT-DIAG/config.yaml'
CODE=['scripts/run_quality_fullframe_context.py','scripts/run_quality_external_camera.py','src/quality_mask2former.py','src/quality_mask2former_training.py']
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['artifact-root','output']:p.add_argument('--'+n,type=Path,required=True)
 for n in ['replay-from','probe-replay']:p.add_argument('--'+n,type=Path)
 p.add_argument('--fold',type=int,choices=range(3),required=True);p.add_argument('--mode',choices=['probe','full'],required=True);a=p.parse_args();a.output=a.output.resolve()
 if not a.output.is_relative_to(ROOT/'outputs/quality/cable_diagnostic'):p.error('Use separate cable_diagnostic output')
 c=read(ROOT/CONFIG)
 if a.mode=='probe' and a.fold!=0 or a.mode=='full' and not a.probe_replay:p.error('Probe fold0; full needs completed probe replay')
 with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
  r={k:c[k] for k in ['experiment_id','git_commit','seed','dataset_manifest','split_id']};r.update(status='started',mode=a.mode,fold=a.fold,config_sha256=sha256(ROOT/CONFIG),code_sha256={n:sha256(ROOT/n) for n in CODE},trained_steps=0,labels_decoded=False,real_photos_decoded=False,quality_metrics=None,artifacts={},views=[],replay_exact=False)
  try:
   verify_hashes(ROOT,c['input_hashes'])
   if shutil.disk_usage(ROOT).free/2**30<c['resource']['minimum_free_gib']+1:raise OSError('Disk reserve')
   write_new_json(a.output/'started.json',r)
   for n in [CONFIG,*CODE]:
    q=a.output/'code'/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,q)
   from src.quality_mask2former_training import Context
   ctx=Context(ROOT,a.artifact_root,f'oof-{a.fold}');r['runtime']=ctx.setup_runtime()
   import numpy as np
   import torch
   from src.quality_mask2former import load_official_model,semantic_probability
   from src.semantic_cable_binary import read_rgb
   from scripts.run_quality_external_camera import checked_model
   r['binding']=ctx.binding
   if a.mode=='full':
    proof=read(a.probe_replay/'result.json')
    if proof['status']!='passed_fullframe_context_stage' or not proof['replay_exact'] or proof['mode']!='probe' or proof['config_sha256']!=r['config_sha256'] or proof['code_sha256']!=r['code_sha256']:raise ValueError('Need matching native engineering acceptance')
    r['probe_replay_sha256']=sha256(a.probe_replay/'result.json')
   names=[c['probe_view']] if a.mode=='probe' else c['views'][f'oof-{a.fold}']
   if not set(names)<=set(ctx.access['prediction']):raise ValueError('Wrong OOF role')
   if a.replay_from:
    old=read(a.replay_from/'result.json')
    if old['status']!='passed_fullframe_context_stage' or any(old[k]!=r[k] for k in ['mode','fold','config_sha256','code_sha256','binding']):raise ValueError('Unmatched replay')
    verify_hashes(a.replay_from,old['artifacts']);r['replay_source_sha256']=sha256(a.replay_from/'result.json')
   run=ROOT/c['batch']/f'full-oof-{a.fold}';payload,parent=checked_model(run,'model.pt',ctx,'five');model,_=load_official_model(ctx.model_config(),ctx.runtime/'weights/model_final_6b4a3a.pkl');model.load_state_dict(payload['model'],strict=True);del payload;model.cuda();model.eval();torch.cuda.reset_peak_memory_stats()
   for name in names:
    ctx.guard();rgb=read_rgb(ctx.export_root/'images'/name)
    x=torch.from_numpy(rgb.transpose(2,0,1).copy()).float()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):scores=model([{'image':x,'height':989,'width':1320}])[0]['sem_seg']
    if scores.device.type!='cuda' or scores.shape!=(5,989,1320):raise RuntimeError('Wrong native grid or author CPU OOM fallback')
    prob=semantic_probability(scores.float().cpu().numpy());del scores
    rel=Path(name).stem+'.npz'
    if a.replay_from:
     with np.load(a.replay_from/rel,allow_pickle=False) as z:
      if not np.array_equal(prob,z['probabilities']):raise ValueError('Fresh wholeframe probability mismatch')
    else:np.savez_compressed(a.output/rel,probabilities=prob)
    r['views'].append(name);print(json.dumps({'mode':a.mode,'fold':a.fold,'view':name,'completed':len(r['views']),'total':len(names)}),flush=True)
   ctx.finish();verify_hashes(ROOT,c['input_hashes']);verify_hashes(ROOT,r['code_sha256']);verify_hashes(run,{'model.pt':parent['artifacts']['model.pt']})
   if sha256(ROOT/CONFIG)!=r['config_sha256']:raise ValueError('Config changed')
   r.update(status='passed_fullframe_context_stage',replay_exact=bool(a.replay_from),rgb_exports_verified=219,frozen_verified=18,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
  except BaseException as exc:r.update(status='failed_fullframe_context_stage',error=f'{type(exc).__name__}: {exc}');raise
  finally:
   r['artifacts']={p.name:sha256(p) for p in a.output.glob('*.npz')};r['seconds']=time.monotonic()-started;write_new_json(a.output/'result.json',r);print(json.dumps({'status':r['status'],'seconds':r['seconds']}),flush=True)
if __name__=='__main__':main()
