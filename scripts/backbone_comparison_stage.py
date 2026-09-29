#!/usr/bin/env python3
"""Native export, unchanged SEM370 M2F fit, and common training-view evaluation."""
import argparse,fcntl,json,os,random,sys,time,subprocess
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,write_new_json,verify_hashes
BASE=ROOT/'outputs/quality/backbone_comparison';PARENT=ROOT/'outputs/quality/all_data/RGB-120-r2';SEM=ROOT/'outputs/quality/all_data/SEM-370-r1'

def setup(a):
 runtime=ROOT/'outputs/quality/mask2former/SEM-340';source=a.artifact_root/'r1/third_party'/a.method
 sys.path[:0]=[str(p) for p in [source,source/'submodules/diff-gaussian-rasterization',source/'submodules/simple-knn',runtime/'runtime',runtime/'vendor/detectron2',runtime/'vendor/Mask2Former',runtime/'vendor/Mask2Former/mask2former/modeling/pixel_decoder/ops']]
 import numpy as np,torch
 torch.manual_seed(20260805);torch.cuda.manual_seed_all(20260805);random.seed(20260805);np.random.seed(20260805)
 torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 return runtime,torch,np

def render_stage(a,out,r,torch,np):
 from PIL import Image
 from src.quality_cameras import read_camera_requests
 from src.quality_native_gaussian import make_native_view
 from dataclasses import replace
 cameras=read_camera_requests(PARENT/'scene/sparse/0')
 trajectory=read_camera_requests(a.artifact_root/'results/frozen_scaffold_sem001/flythrough_v1/cameras')
 requests={**cameras,**{n:replace(c,model="PINHOLE",radial_k=0.) for n,c in trajectory.items()}}
 r['trajectory_source']='results/frozen_scaffold_sem001/flythrough_v1/cameras';r['trajectory_names']=sorted(trajectory)
 r['trajectory_metadata_sha256']={n:sha256(a.artifact_root/r['trajectory_source']/n) for n in ['images.txt','cameras.txt']}
 if a.method=='scaffold-gs':
  from src.quality_scaffold import ScaffoldQualityRenderer
  renderer=ScaffoldQualityRenderer(a.artifact_root/'r1/third_party/scaffold-gs',PARENT/'model',30000,train_camera_count=400)
  def render(c):return renderer.render_pinhole(c).rgb
 else:
  from gaussian_renderer import GaussianModel,render as native_render
  from arguments import PipelineParams
  model=GaussianModel(3);model.load_ply(str(a.rgb_run/'model/point_cloud/iteration_30000/point_cloud.ply'))
  parser=argparse.ArgumentParser();pp=PipelineParams(parser);pipe=pp.extract(parser.parse_args([]));bg=torch.zeros(3,device='cuda')
  def render(c):
   kwargs={'kernel_size':.1} if a.method=='mip-splatting' else {}
   return native_render(make_native_view(c),model,pipe,bg,**kwargs)['render'].clamp(0,1).permute(1,2,0).cpu().numpy()
 # Close the custom official-camera adapter against the author's native camera.
 if a.method!='scaffold-gs':
  import inspect
  from scene.dataset_readers import readColmapSceneInfo
  from utils.camera_utils import loadCam
  params=dict(path=str(PARENT/'scene'),images='images',depths='',eval=False,train_test_exp=False)
  info=readColmapSceneInfo(**{k:v for k,v in params.items() if k in inspect.signature(readColmapSceneInfo).parameters})
  selected={Path(c.image_name).stem:c for c in info.train_cameras}
  r['native_camera_closure']=[]
  with torch.inference_mode():
   for n in ['159.png','057.png','174.png']:
    kw=dict(args=SimpleNamespace(resolution=1,data_device='cpu',train_test_exp=False),id=0,cam_info=selected[Path(n).stem],resolution_scale=1,is_nerf_synthetic=False,is_test_dataset=False)
    view=loadCam(**{k:v for k,v in kw.items() if k in inspect.signature(loadCam).parameters})
    custom=make_native_view(cameras[n])
    matrix_error=float((view.full_proj_transform-custom.full_proj_transform).abs().max())
    assert matrix_error<1e-4, 'Official-to-native camera projection disagreement'
    kwargs={'kernel_size':.1} if a.method=='mip-splatting' else {}
    native=native_render(view,model,pipe,bg,**kwargs)['render'].clamp(0,1).permute(1,2,0).cpu().numpy()
    adapted=render(cameras[n]);delta=np.abs(native-adapted)
    r['native_camera_closure'].append(dict(name=n,projection_max_abs=matrix_error,rgb_mean_abs=float(delta.mean()),rgb_max_abs=float(delta.max())))
    assert float(delta.mean())<1e-4, 'Native/official camera render mismatch'
 (out/'images').mkdir();r['image_hashes']={};times=[]
 with torch.inference_mode():
  for i,(n,c) in enumerate(sorted(requests.items())):
   torch.cuda.synchronize();start=time.monotonic();rgb=render(c);torch.cuda.synchronize();times.append(time.monotonic()-start)
   assert rgb.shape==(989,1320,3) and np.isfinite(rgb).all()
   arr=np.rint(rgb*255).astype(np.uint8)
   if a.method=='scaffold-gs' and (SEM/'export/images'/n).exists():
    assert np.array_equal(arr,np.asarray(Image.open(SEM/'export/images'/n))), 'Scaffold SEM370 rendering mismatch'
   Image.fromarray(arr).save(out/'images'/n);r['image_hashes'][n]=sha256(out/'images'/n)
   if i%50==0:print(json.dumps(dict(rendered=i+1,total=len(requests))),flush=True)
  for n in ['159.png','057.png','174.png']:
   assert np.array_equal(np.rint(render(requests[n])*255).astype(np.uint8),np.asarray(Image.open(out/'images'/n)))
 r.update(render_seconds=times,render_median_seconds=float(np.median(times[1:])),rgb_repeat_exact=True,train_count=400,trajectory_count=len(trajectory))

def preflight(a,out,r,torch,np,runtime):
 from PIL import Image
 from src.quality_mask2former import config_for_model,load_official_model,author_optimizer,class_masks
 from src.quality_mask2former_training import sampled_crop,amp_update,poly_factor
 from src.semantic_cable_binary import read_rgb
 from detectron2.structures import Instances
 cfg=config_for_model(runtime/'vendor/Mask2Former','configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml',False)
 model,loaded=load_official_model(cfg,runtime/'weights/model_final_6b4a3a.pkl');model.cuda();model.train()
 opt=author_optimizer(model,cfg);rates=[g['lr'] for g in opt.param_groups];scaler=torch.amp.GradScaler('cuda',init_scale=1024)
 names=sorted(json.loads((PARENT/'split.json').read_text())['semantic_train'])
 # Same RNG sequence; decode lazily without altering sample draws.
 class Lazy(dict):
  def __init__(self,read):self.read=read
  def __missing__(self,key):self[key]=self.read(key);return self[key]
 images=Lazy(lambda n:read_rgb(a.batch/'export/images'/n));labels=Lazy(lambda n:np.asarray(Image.open(SEM/'labels'/n)))
 rng=random.Random(20260805);r['updates']=[]
 for step in range(1,9):
  x,y,trace=sampled_crop(rng,names,images,labels,768);ids,masks=class_masks(y)
  inst=Instances((768,768));inst.gt_classes=torch.from_numpy(ids);inst.gt_masks=torch.from_numpy(masks)
  for group,rate in zip(opt.param_groups,rates):group['lr']=rate*poly_factor(step-1)
  row=amp_update(model,opt,scaler,torch.from_numpy(x.transpose(2,0,1).copy()).float(),inst,max_attempts=12,clip=.01)
  r['updates'].append(dict(step=step,sample=trace,**row))
 r.update(trained_steps=8,pretrained_load=loaded,quality_gate=False)

def train_stage(a,out,r,torch,np):
 from scripts.run_quality_all300 import mask2former
 from src.quality_static_training import TrainingLabels
 split=json.loads((PARENT/'split.json').read_text());cfg=json.loads((ROOT/'experiments/runs/SEM-370-ALL300/config.yaml').read_text())
 labels=TrainingLabels(SEM/'labels',split['semantic_train'],json.loads((SEM/'label_manifest.json').read_text()),[1320,989])
 assert len(labels.allowed)==300
 export=json.loads((a.batch/'export/result.json').read_text());verify_hashes(a.batch/'export/images',export['image_hashes'])
 r.update(trained_steps=0,access=dict(train=split['semantic_train'],independent_holdout=False),binding=dict(seed=cfg['seed'],split_sha256=sha256(PARENT/'split.json'),label_manifest_sha256=sha256(SEM/'label_manifest.json'),export_result_sha256=sha256(a.batch/'export/result.json')))
 import hashlib
 import src.quality_mask2former as api
 original_loader=api.load_official_model
 def capture_initial(*args,**kwargs):
  model,loaded=original_loader(*args,**kwargs)
  if 'initial_model_state_sha256' not in r:
   digest=hashlib.sha256()
   for key,value in sorted(model.state_dict().items()):
    digest.update(key.encode());digest.update(str(value.dtype).encode());digest.update(str(tuple(value.shape)).encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
   r['initial_model_state_sha256']=digest.hexdigest()
  return model,loaded
 api.load_official_model=capture_initial
 try:mask2former(a,out,r,labels,cfg)
 finally:api.load_official_model=original_loader
 # The complete sampling trace must match the reused baseline.
 old=[json.loads(l)['sample'] for l in (SEM/'mask2former/training.jsonl').read_text().splitlines()]
 new=[json.loads(l)['sample'] for l in (out/'training.jsonl').read_text().splitlines()]
 assert old==new and len(new)==6000
 assert (out/'resolved_model.yaml').read_bytes()==(SEM/'mask2former/resolved_model.yaml').read_bytes()
 verify_hashes(a.batch/'export/images',export['image_hashes']);r.update(sampling_matches_sem370=True,resolved_recipe_matches_sem370=True,frozen_rgb_hashes_exact=True,artifacts={p.name:sha256(p) for p in out.glob('*.pt')})

def infer_stage(a,out,r,torch,np,runtime):
 from PIL import Image
 from src.quality_mask2former import config_for_model,load_official_model,native_predict
 from src.semantic_cable_binary import read_rgb
 from src.quality_semantic_audit import independent_confusion,metrics
 c=config_for_model(runtime/'vendor/Mask2Former','configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml',False)
 model,_=load_official_model(c,runtime/'weights/model_final_6b4a3a.pkl')
 ckpt=(SEM if a.method=='scaffold-gs' else a.batch)/'mask2former/model-6000.pt'
 fit_record=json.loads((ckpt.parent/'result.json').read_text());verify_hashes(ckpt.parent,{ckpt.name:fit_record['artifacts'][ckpt.name]})
 model.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True)['model']);model.cuda()
 split=json.loads((PARENT/'split.json').read_text());labeled=set(split['semantic_train'])
 export=json.loads((a.batch/'export/result.json').read_text());verify_hashes(a.batch/'export/images',export['image_hashes'])
 (out/'predictions').mkdir();total=np.zeros((5,5),np.int64);rows=[];times=[]
 names=sorted(export['image_hashes']);r.update(checkpoint_sha256=sha256(ckpt),pretrained_sha256=sha256(runtime/'weights/model_final_6b4a3a.pkl'))
 for i,n in enumerate(names):
  rgb=read_rgb(a.batch/'export/images'/n);torch.cuda.synchronize();start=time.monotonic();prob=native_predict(model,rgb);torch.cuda.synchronize();times.append(time.monotonic()-start)
  pred=prob.argmax(0).astype(np.uint8);Image.fromarray(pred).save(out/'predictions'/n)
  old=(SEM if a.method=='scaffold-gs' else a.batch)/'mask2former/probabilities'/(Path(n).stem+'.npz')
  if old.exists():
   with np.load(old) as z:assert np.array_equal(prob,z['probabilities']), 'Fresh semantic replay mismatch'
  if n in labeled:
   target=np.asarray(Image.open(SEM/'labels'/n));cm=independent_confusion(target,pred);total+=cm;rows.append(dict(name=n,**metrics(cm)))
  if i%25==0:print(json.dumps(dict(inferred=i+1,total=len(names))),flush=True)
 assert len(rows)==300
 r.update(metrics=metrics(total),per_view=rows,inference_seconds=times,inference_median_seconds=float(np.median(times[1:])),fresh_nine_probability_replay_exact=True,labeled_count=300,unlabeled_count=100,no_gt_trajectory_count=len(names)-400)
 verify_hashes(a.batch/'export/images',export['image_hashes'])

def rgb_eval(a,out,r,torch,np):
 from PIL import Image
 from scripts.evaluate_rgb import ssim,psnr,build_lpips_model,lpips_score
 lp,_,device=build_lpips_model('cuda');rows=[]
 split=json.loads((PARENT/'split.json').read_text())
 for i,n in enumerate(split['rgb_train']):
  ref=np.asarray(Image.open(PARENT/'scene/images'/n).convert('RGB'),dtype=np.float32)/255
  pred=np.asarray(Image.open(a.batch/'export/images'/n).convert('RGB'),dtype=np.float32)/255
  assert ref.shape==pred.shape==(989,1320,3)
  mse=float(np.mean((ref.astype(np.float64)-pred.astype(np.float64))**2));rows.append(dict(name=n,PSNR=psnr(ref,pred),SSIM=ssim(ref,pred),LPIPS=lpips_score(ref,pred,lp,torch,device)))
  if i%50==0:print(json.dumps(dict(scored=i+1,total=400)),flush=True)
 r.update(per_view=rows,metrics={k:float(np.mean([x[k] for x in rows])) for k in ['PSNR','SSIM','LPIPS']},protocol='mean over all400 PINHOLE uint8 training views, full1320x989; fixed project SSIM and LPIPS AlexNet v0.1')

def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--method',choices=['standard-3dgs','mip-splatting','scaffold-gs'],required=True);p.add_argument('--rgb-run',type=Path,required=True);p.add_argument('--batch',type=Path,required=True);p.add_argument('--stage',choices=['export','preflight','mask2former','inference','rgb-evaluation'],required=True);a=p.parse_args()
 a.batch=a.batch.resolve();a.rgb_run=a.rgb_run.resolve();a.batch.mkdir(exist_ok=True)
 out=a.batch/a.stage;out.mkdir(exist_ok=False);r=dict(status='starting',method=a.method,stage=a.stage,started_unix=time.time(),git_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),code_sha256={n:sha256(ROOT/n) for n in ['scripts/backbone_comparison_stage.py','scripts/run_quality_all300.py','src/quality_mask2former.py','src/quality_mask2former_training.py']});start=time.monotonic()
 ids={'scaffold-gs':('RGB-120-ALL400','SEM-370-ALL300'),'standard-3dgs':('RGB-130-STANDARD-ALL400','SEM-371-STANDARD-ALL300'),'mip-splatting':('RGB-131-MIP-ALL400','SEM-372-MIP-ALL300')}
 rgb_id,sem_id=ids[a.method]
 r.update(experiment_id=sem_id if a.stage in ['mask2former','preflight'] else 'QUAL-007-BACKBONE-COMPARISON',rgb_experiment_id=rgb_id,semantic_experiment_id=sem_id,rgb_seed=0,semantic_seed=20260805,dataset_manifest='data/manifests/contest_dataset_v1.json',split_id='released_all400_rgb_all300_sem_v1',split_sha256=sha256(PARENT/'split.json'),config_sha256={eid:sha256(ROOT/'experiments/runs'/eid/'config.yaml') for eid in [rgb_id,sem_id]},quality_scope='training-view fit only')
 parent=__import__('json').loads((a.rgb_run/'result.json').read_text())
 if a.stage in ['preflight','mask2former','inference']:
  r['sem370_reused_code_sha256']=json.loads((SEM/'mask2former/result.json').read_text())['code_sha256']
 try:
  assert parent['experiment_id']==rgb_id, 'Backbone experiment binding mismatch'
  if 'sem370_reused_code_sha256' in r:verify_hashes(ROOT,r['sem370_reused_code_sha256'])
  verify_hashes(a.rgb_run,parent['artifacts'])
  with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
   r['gpu_processes_before']=apps
   if any(line.strip() and 'ToDesk' not in line for line in apps.splitlines()):raise RuntimeError('Other GPU compute process active')
   runtime,torch,np=setup(a);torch.cuda.reset_peak_memory_stats()
   write_new_json(out/'started.json',r)
   if a.stage=='export':render_stage(a,out,r,torch,np)
   elif a.stage=='preflight':preflight(a,out,r,torch,np,runtime)
   elif a.stage=='mask2former':train_stage(a,out,r,torch,np)
   elif a.stage=='inference':infer_stage(a,out,r,torch,np,runtime)
   else:rgb_eval(a,out,r,torch,np)
   verify_hashes(a.rgb_run,parent['artifacts'])
   verify_hashes(ROOT,r['code_sha256'])
   if 'sem370_reused_code_sha256' in r:verify_hashes(ROOT,r['sem370_reused_code_sha256'])
   r.update(status='completed',peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
 except BaseException as e:r.update(status='failed_preserved',error=f'{type(e).__name__}: {e}');raise
 finally:
  r['seconds']=time.monotonic()-start
  try:
   import torch
   if torch.cuda.is_initialized():r.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
  except Exception:pass
  write_new_json(out/'result.json',r)
if __name__=='__main__':main()
