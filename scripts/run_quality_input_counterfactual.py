#!/usr/bin/env python3
"""Bounded frozen Mask2Former input/scale diagnostics. No labels or optimizer."""
import argparse,fcntl,hashlib,json,random,shutil,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);a=ap.parse_args();c=read(a.config);out=ROOT/c['output'];out.mkdir(exist_ok=False,parents=True);start=time.monotonic()
 r={k:c[k] for k in ['experiment_id','git_commit','seed','dataset_manifest','split_id']};r.update(status='running',config_sha256=sha(a.config),code_sha256=sha(__file__),inputs={},models={},rows=[],training=0,labels_decoded=False)
 def check(p,expected=None):
  h=sha(p)
  if expected is not None:assert h==expected,str(p)
  r['inputs'][str(p.relative_to(ROOT))]=h
  return h
 try:
  (out/'code').mkdir();shutil.copy2(__file__,out/'code'/Path(__file__).name);shutil.copy2(a.config,out/'config.json')
  runtime=ROOT/c['runtime_root'];sys.path[:0]=[str(runtime/n) for n in ['runtime','vendor/detectron2','vendor/Mask2Former','vendor/Mask2Former/mask2former/modeling/pixel_decoder/ops']]
  import torch
  from detectron2.modeling import build_model
  from src.quality_mask2former import config_for_model
  from src.quality_expert_scale import native_predict
  assert torch.cuda.is_available(),'GPU inaccessible'
  random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed']);torch.set_num_threads(4)
  torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  cfg=config_for_model(runtime/'vendor/Mask2Former',c['upstream_config'],False);cfg.defrost();cfg.DATASETS.TRAIN=('sem341_rendered_training',);cfg.SOLVER.MAX_ITER=6000;cfg.SOLVER.POLY_LR_POWER=.9;cfg.freeze()
  def load(run,names):
   rec=read(run/'result.json');check(run/'result.json');check(run/'model.pt',rec['artifacts']['model.pt'])
   assert set(names)<=set(rec['access']['prediction']) and not set(names)&set(rec['access']['train'])
   for path,h in rec['binding'].get('runtime_sha256',rec['binding'].get('upstream_runtime_sha256',{})).items():check(runtime/path,h)
   payload=torch.load(run/'model.pt',map_location='cpu',weights_only=True)
   assert payload['steps']==6000 and payload['binding']==rec['binding'] and payload['access']==rec['access']
   model=build_model(cfg);model.load_state_dict(payload['model'],strict=True);del payload;model.requires_grad_(False);model.cuda().eval()
   r['models'][str(run.relative_to(ROOT))]={'sha256':rec['artifacts']['model.pt'],'binding':rec['binding'],'access':rec['access']}
   return model,rec
  def forward(model,rec,run,root,name,size,arm,equality=False):
   p=root/name;check(p);rgb=np.array(Image.open(p).convert('RGB'));prob=native_predict(model,rgb,size)
   if equality:
    q=run/'probabilities'/(Path(name).stem+'.npz');check(q,rec['artifacts']['probabilities/'+Path(name).stem+'.npz'])
    with np.load(q,allow_pickle=False) as z:assert np.array_equal(prob,z['probabilities']),f'native probability mismatch {name}'
   d=out/arm;d.mkdir(exist_ok=True);Image.fromarray(prob.argmax(0).astype(np.uint8)).save(d/name)
   # Save only new photo probability; scale diagnostic needs hard error strata, not duplicate large caches.
   if arm=='photo':np.savez_compressed(d/(Path(name).stem+'.npz'),probabilities=prob)
   r['rows'].append({'name':name,'arm':arm,'size':size,'native_cached_exact':equality,'probability_sha256':hashlib.sha256(prob.tobytes()).hexdigest(),'prediction_sha256':sha(d/name)})
   print(arm,name,'done',flush=True)
  with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   if 'run' in c:
    run=ROOT/c['run'];model,rec=load(run,c['names'])
    for name in c['names']:
     forward(model,rec,run,ROOT/c['render_root'],name,768,'render',True)
     forward(model,rec,run,ROOT/c['real_root'],name,768,'photo')
    del model;torch.cuda.empty_cache()
   else:
    for fold,names in c['views'].items():
     for train,infer in [(768,864),(864,768)]:
      run=ROOT/c['run'+str(train)]/('full-'+fold);model,rec=load(run,names)
      forward(model,rec,run,ROOT/c['render_root'],names[0],train,fold+'-native'+str(train),True)
      for name in names:forward(model,rec,run,ROOT/c['render_root'],name,infer,'train'+str(train)+'-infer'+str(infer))
      del model;torch.cuda.empty_cache()
  for rel,h in r['inputs'].items():assert sha(ROOT/rel)==h,rel
  r.update(status='completed_frozen_input_diagnostic',peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,runtime={'torch':torch.__version__,'gpu':torch.cuda.get_device_name()})
 except BaseException as e:r.update(status='failed_preserved',error=repr(e));raise
 finally:
  r['seconds']=time.monotonic()-start;(out/'result.json').write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
