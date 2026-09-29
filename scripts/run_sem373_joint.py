#!/usr/bin/env python3
"""Isolated Standard 3DGS semantic-to-XYZ experiment; never patches upstream."""
import argparse, copy, fcntl, hashlib, json, math, os, random, subprocess, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT/'experiments/runs/SEM-373-STANDARD-JOINT-XYZ/config.yaml'


def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''): h.update(b)
 return h.hexdigest()


def write(p, v):
 Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def schedule(names, steps, seed):
 names=list(names);rng=random.Random(seed); out=[]
 if not names and steps:raise ValueError("Empty camera sampling universe")
 while len(out)<steps:
  epoch=sorted(names);rng.shuffle(epoch);out.extend(epoch)
 return out[:steps]


def tensor_hash(x):
 return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_hash(model, logits, semopt):
 return dict(model={n:tensor_hash(getattr(model,n)) for n in ['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity']},logits=tensor_hash(logits),semantic_optimizer={str(k):{n:tensor_hash(v) if hasattr(v,'detach') else v for n,v in s.items()} for k,s in semopt.state_dict()['state'].items()},rgb_optimizer={str(k):{n:tensor_hash(v) if hasattr(v,'detach') else v for n,v in s.items()} for k,s in model.optimizer.state_dict()['state'].items()},rgb_groups=[{k:v for k,v in x.items() if k!='params'} for x in model.optimizer.param_groups])


def setup(a,c):
 source=a.artifact_root/'r1/third_party/standard-3dgs'
 sys.path[:0]=[str(source),str(source/'submodules/diff-gaussian-rasterization'),str(source/'submodules/simple-knn')]
 import torch,numpy as np
 from PIL import Image
 from gaussian_renderer import GaussianModel,render
 from arguments import PipelineParams,OptimizationParams
 from src.quality_cameras import read_camera_requests
 from src.quality_native_gaussian import make_native_view
 torch.set_num_threads(4);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed']);np.random.seed(c['seed']);random.seed(c['seed'])
 torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 parser=argparse.ArgumentParser();pp=PipelineParams(parser);op=OptimizationParams(parser);args=parser.parse_args([])
 model=GaussianModel(3);model._exposure=torch.nn.Parameter(torch.eye(3,4,device='cuda')[None]);model.pretrained_exposures=None
 captured,iteration=torch.load(ROOT/c['rgb_parent']/'chkpnt30000.pth',map_location='cuda',weights_only=False)
 assert iteration==30000
 model.restore(captured,op.extract(args));model.update_learning_rate(30000)
 logits=torch.nn.Parameter(torch.zeros((len(model._xyz),5),device='cuda'));semopt=torch.optim.Adam([logits],lr=c['semantic_lr'])
 for n in ['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity']:getattr(model,n).requires_grad_(False)
 cams=read_camera_requests(ROOT/c['scene']/'sparse/0');views={n:make_native_view(cam) for n,cam in cams.items()}
 split=json.loads((ROOT/c['split_path']).read_text());assert len(split['rgb_train'])==400 and len(set(split['semantic_train']))==300
 assert set(split['semantic_train'])<=set(split['rgb_train'])==set(cams)
 class Lazy(dict):
  def __init__(self,fn):self.fn=fn
  def __missing__(self,k):self[k]=self.fn(k);return self[k]
 manifest=json.loads((ROOT/c['label_manifest']).read_text());records={r['image']:r for r in manifest['records']}
 assert set(records)==set(split['semantic_train'])
 def read_label(n):
  assert n in records
  path=ROOT/c['labels']/n;assert digest(path)==records[n]['label_png_sha256']
  return np.asarray(Image.open(path),dtype=np.uint8).copy()
 labels=Lazy(read_label)
 photos=Lazy(lambda n:np.asarray(Image.open(ROOT/c['scene']/'images'/n).convert('RGB'),dtype=np.uint8).copy())
 return SimpleNamespace(torch=torch,np=np,Image=Image,source=source,model=model,logits=logits,semopt=semopt,pipe=pp.extract(args),render=render,views=views,cams=cams,split=split,labels=labels,photos=photos,bg=torch.zeros(3,device='cuda'),c=c)


class GeometryView:
 def __init__(self,model,allow):self.model=model;self.allow=allow
 def __getattr__(self,n):
  if n in ['get_xyz','get_scaling','get_rotation','get_opacity']:
   v=getattr(self.model,n);return v if n in self.allow else v.detach()
  return getattr(self.model,n)


def semantic(s,view,allow=()):
 """Exact native alpha compositing of 5 probabilities via two 3-channel passes."""
 t=s.torch;p=s.logits.softmax(1);m=GeometryView(s.model,allow)
 first=s.render(view,m,s.pipe,t.tensor([1.,0.,0.],device='cuda'),override_color=p[:,:3].contiguous())['render']
 last=s.render(view,m,s.pipe,s.bg,override_color=t.cat([p[:,3:],t.zeros_like(p[:,:1])],1).contiguous())['render'][:2]
 return t.cat([first,last],0)


def nll(s,prob,name):
 assert name in s.split['semantic_train'], 'No fabricated labels on the unlabeled100'
 target=s.torch.from_numpy(s.labels[name].astype('int64')).cuda()
 assert target.shape==prob.shape[1:] and int(target.max())<5
 return s.torch.nn.functional.nll_loss(prob.clamp_min(1e-8).log()[None],target[None])


def rgb_loss(s,view,name):
 from utils.loss_utils import l1_loss,ssim
 rgb=s.render(view,s.model,s.pipe,s.bg)['render'];gt=s.torch.from_numpy(s.photos[name].transpose(2,0,1).copy()).cuda().float()/255
 return .8*l1_loss(rgb,gt)+.2*(1-ssim(rgb,gt))


def load_stage(s,path):
 q=s.torch.load(path,map_location='cuda',weights_only=False)
 # Existing Parameter identities remain bound to their optimizers.
 with s.torch.no_grad():
  for n,v in q['parameters'].items():getattr(s.model,n).copy_(v)
  s.logits.copy_(q['logits'])
 s.model.optimizer.load_state_dict(q['rgb_optimizer']);s.semopt.load_state_dict(q['semantic_optimizer'])
 return q


def save_stage(s,path,step):
 assert not path.exists()
 s.torch.save(dict(parameters={n:getattr(s.model,n).detach() for n in ['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity']},logits=s.logits.detach(),rgb_optimizer=s.model.optimizer.state_dict(),semantic_optimizer=s.semopt.state_dict(),step=step,config_sha256=digest(CONFIG)),path)


def finite(s):
 for p in [s.logits,*[g['params'][0] for g in s.model.optimizer.param_groups]]:
  if not s.torch.isfinite(p).all() or p.grad is not None and not s.torch.isfinite(p.grad).all():raise FloatingPointError('Non-finite parameter/gradient; diagnostic checkpoint preserved')


def preflight(s,out,r):
 t=s.torch;names=['159.png','057.png','174.png'];r['checks']=[]
 # The deployment PLY and true optimizer checkpoint must define identical RGB130.
 from gaussian_renderer import GaussianModel
 ply=GaussianModel(3);ply.load_ply(str(ROOT/s.c['rgb_parent']/'point_cloud/iteration_30000/point_cloud.ply'))
 r['parent_checkpoint_ply_parameter_exact']={n:bool(t.equal(getattr(ply,n),getattr(s.model,n))) for n in ['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity']}
 assert all(r['parent_checkpoint_ply_parameter_exact'].values())
 for name in names:
  with t.no_grad():
   rgb=s.render(s.views[name],s.model,s.pipe,s.bg)['render'];arr=s.np.rint(rgb.permute(1,2,0).cpu().numpy()*255).astype('uint8')
   assert s.np.array_equal(arr,s.np.asarray(s.Image.open(ROOT/s.c['parent_export']/name)))
   prob=semantic(s,s.views[name]);err=float((prob.sum(0)-1).abs().max());assert err<2e-5
  r['checks'].append(dict(name=name,rgb_parent_png_exact=True,probability_sum_max_error=err,shape=list(prob.shape)))
 # Nonconstant probe logits, discarded; avoids a zero-initialization geometry null test.
 with t.no_grad():s.logits.copy_(t.sin(t.arange(s.logits.numel(),device='cuda').reshape(s.logits.shape)*.17))
 params=[s.model._xyz,s.model._scaling,s.model._rotation,s.model._opacity]
 for p in params:p.requires_grad_(True)
 prob=semantic(s,s.views['159.png'],('get_xyz','get_scaling','get_rotation','get_opacity'));loss=nll(s,prob,'159.png')
 grads=t.autograd.grad(loss,[s.logits,*params]);r['native_semantic_gradient_norms']={n:float(g.norm()) for n,g in zip(['semantic','xyz','scaling','rotation','opacity'],grads)}
 assert all(t.isfinite(g).all() and g.norm()>0 for g in grads)
 # Directional derivative at one largest-absolute coordinate, multiple fixed eps.
 xyzgrad=grads[1];idx=int(xyzgrad.abs().argmax());analytic=float(xyzgrad.flatten()[idx]);base=float(s.model._xyz.flatten()[idx]);scale=float(s.model.spatial_lr_scale)
 r['finite_difference']=[]
 for eps in [scale*1e-5,scale*3e-6,scale*1e-6]:
  vals=[]
  for sign in [-1,1]:
   with t.no_grad():
    s.model._xyz.flatten()[idx]=base+sign*eps
    vals.append(float(nll(s,semantic(s,s.views['159.png']), '159.png')))
  with t.no_grad():s.model._xyz.flatten()[idx]=base
  r['finite_difference'].append(dict(epsilon=eps,analytic=analytic,numeric=(vals[1]-vals[0])/(2*eps)))
 # Control versus joint are exactly the same forward; only autograd routing differs.
 s.model.optimizer.zero_grad(set_to_none=True);s.semopt.zero_grad(set_to_none=True)
 p0=semantic(s,s.views['159.png']);nll(s,p0,'159.png').backward();assert s.model._xyz.grad is None
 g0=s.logits.grad.clone();s.semopt.zero_grad(set_to_none=True)
 p1=semantic(s,s.views['159.png'],('get_xyz',));nll(s,p1,'159.png').backward()
 r['routing_detail']=dict(forward_exact=bool(t.equal(p0,p1)),semantic_gradient_exact=bool(t.equal(g0,s.logits.grad)),semantic_gradient_max_abs=float((g0-s.logits.grad).abs().max()),semantic_gradient_scale=float(g0.abs().max()),joint_xyz_norm=float(s.model._xyz.grad.norm()))
 assert t.equal(p0,p1) and t.allclose(g0,s.logits.grad,rtol=1e-5,atol=1e-9) and s.model._xyz.grad.norm()>0
 assert all(p.grad is None for p in params[1:])
 r['forward_exact_semantic_gradient_close_control_geometry_blocked']=True
 finite(s)
 # Two diagnostic updates exercise Adam and reload, discarded before the fixed run.
 for p in params[1:]:p.requires_grad_(False)
 for n in ['_features_dc','_features_rest']:getattr(s.model,n).requires_grad_(True)
 r['smoke_steps']=[]
 for step,name in enumerate(['159.png','301.png'],1):
  s.model.optimizer.zero_grad(set_to_none=True);s.semopt.zero_grad(set_to_none=True)
  loss=rgb_loss(s,s.views[name],name)
  if name in s.split['semantic_train']:loss=loss+.1*nll(s,semantic(s,s.views[name],('get_xyz',)),name)
  loss.backward();finite(s);s.model.optimizer.step();s.semopt.step();finite(s)
  r['smoke_steps'].append(dict(step=step,name=name,semantic_supervised=name in s.split['semantic_train'],loss=float(loss)))
 save_stage(s,out/'diagnostic-only.pt',2);before=state_hash(s.model,s.logits,s.semopt);load_stage(s,out/'diagnostic-only.pt');assert before==state_hash(s.model,s.logits,s.semopt)
 r.update(reload_exact=True,diagnostic_updates_discarded=True,point_count=len(s.logits),spatial_lr_scale=scale,rgb_learning_rates={x['name']:x['lr'] for x in s.model.optimizer.param_groups})


def train(s,out,r,stage):
 t=s.torch;c=s.c;semantic_only=stage=='semantic'
 if not semantic_only:load_stage(s,ROOT/c['output']/'semantic-r1/endpoint.pt')
 for n in ['_xyz','_features_dc','_features_rest']:getattr(s.model,n).requires_grad_(not semantic_only)
 r['initial_state']=state_hash(s.model,s.logits,s.semopt)
 steps=c['semantic_steps'] if semantic_only else c['matched_steps']
 seq=schedule(s.split['semantic_train'] if semantic_only else s.split['rgb_train'],steps,c['seed']+(0 if semantic_only else 1))
 write(out/'sampling.json',seq);r['sampling_sha256']=digest(out/'sampling.json');r['label_visits']=0;r['successful_steps']=0
 frozen={n:tensor_hash(getattr(s.model,n)) for n in ['_scaling','_rotation','_opacity']+(['_xyz','_features_dc','_features_rest'] if semantic_only else [])}
 started=time.monotonic()
 with (out/'training.jsonl').open('x') as log:
  for step,name in enumerate(seq,1):
   s.model.optimizer.zero_grad(set_to_none=True);s.semopt.zero_grad(set_to_none=True)
   if semantic_only:sl=nll(s,semantic(s,s.views[name]),name);loss=sl;rl=None
   else:
    rl=rgb_loss(s,s.views[name],name);loss=rl;sl=None
    if name in s.split['semantic_train']:
     sl=nll(s,semantic(s,s.views[name],('get_xyz',) if stage=='joint' else ()),name);loss=loss+c['semantic_weight']*sl
   if sl is not None:r['label_visits']+=1
   loss.backward();finite(s)
   if not semantic_only:s.model.optimizer.step()
   s.semopt.step();finite(s);r['successful_steps']=step
   row=dict(step=step,name=name,rgb_loss=float(rl) if rl is not None else None,semantic_nll=float(sl) if sl is not None else None,seconds=time.monotonic()-started)
   log.write(json.dumps(row)+'\n')
   if step%100==0:log.flush();print(json.dumps(row),flush=True)
 r['training_seconds']=time.monotonic()-started
 assert all(tensor_hash(getattr(s.model,n))==v for n,v in frozen.items())
 r['frozen_parameters_exact']=list(frozen)
 if semantic_only:
  r['frozen_rgb_checks']={}
  with t.no_grad():
   for n in c['diagnostic_views']:
    rgb=s.render(s.views[n],s.model,s.pipe,s.bg)['render'];arr=s.np.rint(rgb.permute(1,2,0).cpu().numpy()*255).astype('uint8')
    assert s.np.array_equal(arr,s.np.asarray(s.Image.open(ROOT/c['parent_export']/n)))
    r['frozen_rgb_checks'][n]=True
 save_stage(s,out/'endpoint.pt',steps);r['endpoint_sha256']=digest(out/'endpoint.pt');r['final_state']=state_hash(s.model,s.logits,s.semopt)
 with t.no_grad():reference=semantic(s,s.views['159.png']).detach().clone()
 load_stage(s,out/'endpoint.pt')
 with t.no_grad():assert t.equal(reference,semantic(s,s.views['159.png']))
 r['endpoint_reload_exact']=True
 if stage=='joint':
  ctrl=json.loads((ROOT/c['output']/'control-r1/result.json').read_text());assert ctrl['initial_state']==json.loads(json.dumps(r['initial_state'])) and ctrl['sampling_sha256']==r['sampling_sha256'];r['matched_initial_state_and_sampling_exact']=True


def evaluate(s,out,r,stage,a):
 from dataclasses import replace
 from src.quality_cameras import read_camera_requests
 from src.quality_native_gaussian import make_native_view
 from scripts.evaluate_rgb import ssim,psnr,build_lpips_model,lpips_score
 from src.quality_semantic_audit import metrics
 t,np=s.torch,s.np;c=s.c
 parentxyz=s.model._xyz.detach().clone();frozen={n:tensor_hash(getattr(s.model,n)) for n in ['_scaling','_rotation','_opacity']}
 arm=stage.removeprefix('eval-');load_stage(s,ROOT/c['output']/({'baseline':'semantic','control':'control','joint':'joint'}[arm]+'-r1')/'endpoint.pt')
 assert all(tensor_hash(getattr(s.model,n))==v for n,v in frozen.items())
 displacement=(s.model._xyz-parentxyz).norm(dim=1);sizes=s.model.get_scaling.max(1).values
 r['geometry']=dict(point_count=len(displacement),frozen_shape_opacity_exact=True,xyz_displacement_quantiles=t.quantile(displacement,t.tensor([0,.5,.9,.99,1],device='cuda')).tolist(),displacement_over_max_axis_quantiles=t.quantile(displacement/sizes,t.tensor([.5,.9,.99,1],device='cuda')).tolist(),world_units='official COLMAP reconstruction units; not metric GT')
 traj=read_camera_requests(a.artifact_root/c['trajectory']);requests={**s.views,**{n:make_native_view(replace(cam,model='PINHOLE',radial_k=0.)) for n,cam in traj.items()}}
 r['trajectory_names']=sorted(traj);r['trajectory_hashes']={n:digest(a.artifact_root/c['trajectory']/n) for n in ['images.txt','cameras.txt']}
 for sub in ['rgb','semantic','depth']: (out/sub).mkdir()
 lp,_,device=build_lpips_model('cuda');rows=[];cm=np.zeros((5,5),np.int64);class_mae=np.zeros(5);class_count=np.zeros(5,np.int64);r['fixed_regions']=[]
 roi=json.loads((ROOT/'experiments/runs/QUAL-005-PHOTO-INPUT-DIAG/config.yaml').read_text())['rois'];structure=json.loads((ROOT/'experiments/runs/QUAL-004-CURRENT-BOTTLENECK/config.yaml').read_text())['crops']
 for i,(name,view) in enumerate(sorted(requests.items())):
  with t.no_grad():
   pack=s.render(view,s.model,s.pipe,s.bg);rgb=pack['render'];prob=semantic(s,view);assert t.isfinite(prob).all() and float((prob.sum(0)-1).abs().max())<2e-5
   arr=np.rint(rgb.permute(1,2,0).cpu().numpy()*255).astype(np.uint8);pred=prob.argmax(0).cpu().numpy().astype(np.uint8)
   if arm=='baseline' and name in s.views:assert np.array_equal(arr,np.asarray(s.Image.open(ROOT/c['parent_export']/name)))
   if name in c['diagnostic_views'] or name in traj:
    dep=pack['depth'][0].cpu().numpy();np.save(out/'depth'/(Path(name).stem+'.npy'),dep)
  s.Image.fromarray(arr).save(out/'rgb'/name);s.Image.fromarray(pred).save(out/'semantic'/name)
  if name in s.views:
   gt=s.photos[name].astype(np.float32)/255;rendered=arr.astype(np.float32)/255;err=np.abs(gt-rendered).mean(2)
   row=dict(name=name,PSNR=psnr(gt,rendered),SSIM=ssim(gt,rendered),LPIPS=lpips_score(gt,rendered,lp,t,device),MAE=float(err.mean()))
   if name in s.split['semantic_train']:
    target=s.labels[name];counts=np.bincount(target.ravel().astype(np.int64)*5+pred.ravel(),minlength=25).reshape(5,5);cm+=counts;row['semantic']=metrics(counts)
    for k in range(5):mask=target==k;class_mae[k]+=err[mask].sum(dtype=np.float64);class_count[k]+=mask.sum()
    if name in c['diagnostic_views']:
     for kind,box in [('shore_or_sky',roi.get(Path(name).stem,[0,0,1320,989])),('structure',structure.get(Path(name).stem,[0,0,1320,989]))]:
      x0,y0,x1,y1=box;yy=target[y0:y1,x0:x1];pp=pred[y0:y1,x0:x1];regcm=np.bincount(yy.ravel().astype(np.int64)*5+pp.ravel(),minlength=25).reshape(5,5)
      # Image-space GT boundaries diagnose geometry support, not 3D truth.
      boundary=np.zeros_like(yy,dtype=bool);boundary[1:]|=yy[1:]!=yy[:-1];boundary[:,1:]|=yy[:,1:]!=yy[:,:-1]
      from scipy.ndimage import binary_dilation
      boundary=binary_dilation(boundary,iterations=2);er=err[y0:y1,x0:x1]
      r['fixed_regions'].append(dict(name=name,kind=kind,box=box,rgb_mae=float(er.mean()),boundary_rgb_mae=float(er[boundary].mean()) if boundary.any() else None,semantic=metrics(regcm)))
   rows.append(row)
  if i%50==0:print(json.dumps(dict(evaluated=i+1,total=len(requests),arm=arm)),flush=True)
 assert len(rows)==400
 r.update(arm=arm,per_view=rows,rgb_metrics={k:float(np.mean([v[k] for v in rows])) for k in ['PSNR','SSIM','LPIPS','MAE']},semantic_metrics=metrics(cm),class_rgb_mae={n:float(class_mae[k]/class_count[k]) for k,n in enumerate(c['class_names'])},baseline_all400_rgb_exact=arm=='baseline',evaluation_role=c['evaluation'])
 # Fresh endpoint reload replay of one native RGB, semantic tensor and depth.
 with t.no_grad():reference=semantic(s,s.views['159.png']).clone();rgbref=s.render(s.views['159.png'],s.model,s.pipe,s.bg)['render'].clone()
 load_stage(s,ROOT/c['output']/({'baseline':'semantic','control':'control','joint':'joint'}[arm]+'-r1')/'endpoint.pt')
 with t.no_grad():assert t.equal(reference,semantic(s,s.views['159.png'])) and t.equal(rgbref,s.render(s.views['159.png'],s.model,s.pipe,s.bg)['render'])
 r['fresh_endpoint_replay_exact']=True


def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--stage',choices=['preflight','semantic','control','joint','eval-baseline','eval-control','eval-joint'],required=True);p.add_argument('--attempt',default='r1');a=p.parse_args();c=json.loads(CONFIG.read_text());out=ROOT/c['output']/(a.stage+'-'+a.attempt);out.mkdir(exist_ok=False)
 r=dict(experiment_id=c['experiment_id'],stage=a.stage,status='waiting_gpu_lock',config_sha256=digest(CONFIG),source_script_sha256=digest(__file__),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),seed=c['seed']);write(out/'started.json',r)
 start=time.monotonic();s=None
 try:
  with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
   print('Waiting shared GPU exclusive lock',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
   # Non-cooperating compute processes prevent accidental concurrent heavy work.
   while True:
    q=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
    other=[line for line in q.splitlines() if 'todesk' not in line.lower() and line.strip()]
    if not other:break
    print('Waiting external GPU compute: '+repr(other),flush=True);time.sleep(10)
   r['lock_wait_seconds']=time.monotonic()-start;r['status']='running';write(out/'started.json',r);start=time.monotonic()
   s=setup(a,c);s.torch.cuda.reset_peak_memory_stats()
   r['inputs']={str(f.relative_to(ROOT)):digest(f) for f in [CONFIG,ROOT/c['rgb_parent']/'chkpnt30000.pth',ROOT/c['rgb_parent']/'point_cloud/iteration_30000/point_cloud.ply',ROOT/c['split_path'],ROOT/c['label_manifest']]}
   r['source']={str(f.relative_to(s.source)):digest(f) for f in [s.source/'gaussian_renderer/__init__.py',s.source/'scene/gaussian_model.py',s.source/'arguments/__init__.py',s.source/'submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py']}
   r['source']['loaded_extension']=digest(Path(__import__('diff_gaussian_rasterization')._C.__file__))
   (out/'source.py').write_bytes(Path(__file__).read_bytes())
   r['runtime']=dict(torch=s.torch.__version__,cuda=s.torch.version.cuda,device=s.torch.cuda.get_device_name())
   if a.stage=='preflight':preflight(s,out,r)
   elif a.stage.startswith('eval-'):evaluate(s,out,r,a.stage,a)
   else:train(s,out,r,a.stage)
   for rel,h in r['inputs'].items():assert digest(ROOT/rel)==h
   r['status']='completed';s.torch.cuda.synchronize()
 except BaseException as e:
  r.update(status='failed_preserved',error=repr(e),traceback=traceback.format_exc())
  if s is not None:
   try:save_stage(s,out/'failure-state.pt',r.get('successful_steps',0))
   except BaseException as save_error:r['failure_save_error']=repr(save_error)
  raise
 finally:
  r['seconds']=time.monotonic()-start
  if s is not None:r.update(peak_allocated_mib=s.torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=s.torch.cuda.max_memory_reserved()/2**20)
  write(out/'result.json',r)

if __name__=='__main__':main()
