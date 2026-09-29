#!/usr/bin/env python3
"""Synchronized matched-budget two-arm Scaffold continuation. No original writes."""
import argparse,json,sys,time,shutil,traceback,random,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from src.quality_protocol import sha256

def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);a=p.parse_args();cp=ROOT/'experiments/runs/RGB-124-MATCHED-DENSIFICATION-PF/config.yaml';c=json.loads(cp.read_text());o=ROOT/c['output'];o.mkdir(parents=True,exist_ok=False)
 try:run(a,c,cp,o)
 except Exception as e:save(o/'failure.json',dict(status='failed_preserved',error=repr(e),traceback=traceback.format_exc()));raise

def run(a,c,cp,o):
 import torch
 from torch.utils.cpp_extension import load
 from src.scaffold_semantic_sidecar import configure_scaffold_import,pipeline,frozen_rgb_named_parameters,_raster_settings
 from src.quality_scaffold import saved_namespace
 from src.quality_cameras import read_camera_requests
 from src.quality_native_gaussian import make_native_view
 start=time.monotonic();torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);parent=ROOT/c['parent_run'];src=a.artifact_root/'r1/third_party/scaffold-gs';up=src/'submodules/diff-gaussian-rasterization';hashes=json.loads((parent/'result.json').read_text())['artifacts'];assert all(sha256(parent/k)==h for k,h in hashes.items())
 configure_scaffold_import(src)
 from scene.gaussian_model import GaussianModel
 from arguments import OptimizationParams
 from gaussian_renderer import prefilter_voxel,generate_neural_gaussians
 from utils.loss_utils import l1_loss,ssim
 import diff_gaussian_rasterization as dr
 native=dr._C;ext=o/'instrumented_source';shutil.copytree(ROOT/'outputs/quality/diagnostics/RGB-123-r1/instrumented_source',ext)
 f=ext/'cuda_rasterizer/backward.cu';t=f.read_text();old='atomicAdd(&dL_dmean2D[global_id].z, fabsf(dL_dG * dG_ddelx * ddelx_dx) + fabsf(dL_dG * dG_ddely * ddely_dy));';assert t.count(old)==1;t=t.replace(old,'atomicAdd(&dL_dmean2D[global_id].z, fabsf(dL_dG * dG_ddelx * ddelx_dx));\n            atomicAdd(&dL_dconic2D[global_id].z, fabsf(dL_dG * dG_ddely * ddely_dy));');f.write_text(t)
 f=ext/'rasterize_points.cu';t=f.read_text();needle='return std::make_tuple(dL_dmeans2D, dL_dcolors';assert t.count(needle)==1;t=t.replace(needle,'auto abs_xy = torch::stack({dL_dmeans2D.select(1,2), dL_dconic.select(1,1).select(1,0)}, 1);\n  dL_dmeans2D.select(1,2).zero_();\n  auto diagnostic = torch::cat({dL_dmeans2D, abs_xy}, 1);\n  return std::make_tuple(diagnostic, dL_dcolors');f.write_text(t);(o/'build').mkdir()
 lib=load(name='rgb124_absxy',sources=[str(ext/f) for f in ['cuda_rasterizer/rasterizer_impl.cu','cuda_rasterizer/forward.cu','cuda_rasterizer/backward.cu','rasterize_points.cu','ext.cpp']],extra_cuda_cflags=['-O3','-I'+str(up/'third_party/glm')],extra_cflags=['-O3'],build_directory=str(o/'build'),verbose=False)
 class Proxy:
  absolute=None
  def __getattr__(self,n):return getattr(lib,n)
  def rasterize_gaussians_backward(self,*args):
   v=list(lib.rasterize_gaussians_backward(*args));self.absolute=v[0][:,3:].detach().clone();v[0]=v[0][:,:3].contiguous();return tuple(v)
 proxy=Proxy();saved=saved_namespace(parent/'model/cfg_args');parser=argparse.ArgumentParser();op=OptimizationParams(parser);opt=op.extract(parser.parse_args([]));payload,step=torch.load(parent/c['checkpoint'],weights_only=False,map_location='cpu');assert step==7000
 def newmodel():
  args={k:saved[k] for k in ['feat_dim','n_offsets','voxel_size','update_depth','update_init_factor','update_hierachy_factor','use_feat_bank','appearance_dim','ratio','add_opacity_dist','add_cov_dist','add_color_dist'] if k in saved};m=GaussianModel(**args);m.set_appearance(400);m.restore(copy.deepcopy(payload),opt);m.train();return m
 models={arm:newmodel() for arm in c['arms']};pipe=pipeline();bg=torch.zeros(3,device='cuda');cams=read_camera_requests(parent/'scene/sparse/0');names=sorted(cams);rng=random.Random(0);order=[];stack=[]
 for _ in range(500):
  if not stack:stack=names.copy()
  order.append(stack.pop(rng.randrange(len(stack))))
 save(o/'binding.json',dict(experiment_id=c['experiment_id'],git_commit=c['git_commit'],seed=0,config_sha256=sha256(cp),script_sha256=sha256(Path(__file__)),checkpoint_sha256=sha256(parent/c['checkpoint']),parent_hashes=hashes,split_sha256=sha256(parent/'split.json'),manifest_sha256=sha256(ROOT/c['dataset_manifest']),camera_order=order,source_camera_uid='sorted full400 enumerate; inference uid0',instrumented_files={str(p.relative_to(ext)):sha256(p) for p in ext.rglob('*') if p.is_file()}))
 def equal_state(x,y):
  if torch.is_tensor(x):return torch.equal(x,y)
  if isinstance(x,dict):return x.keys()==y.keys() and all(equal_state(x[k],y[k]) for k in x)
  if isinstance(x,(list,tuple)):return len(x)==len(y) and all(equal_state(a,b) for a,b in zip(x,y))
  return x==y
 assert equal_state(models['signed'].capture(),models['absolute'].capture())
 def render(m,n,training,module=proxy):
  dr._C=module;v=make_native_view(cams[n]);v.uid=names.index(n) if training else 0
  if training:
   from utils.graphics_utils import getProjectionMatrix
   v.projection_matrix=getProjectionMatrix(znear=.01,zfar=100.,fovX=v.FoVx,fovY=v.FoVy).transpose(0,1).cuda();v.full_proj_transform=v.world_view_transform@v.projection_matrix
  visible=prefilter_voxel(v,m,pipe,bg);xyz,rgb,opacity,scale,rotation,_,active=generate_neural_gaussians(v,m,visible,is_training=True);means=torch.zeros_like(xyz,requires_grad=True);im,radii=dr.GaussianRasterizer(raster_settings=_raster_settings(v,bg))(means3D=xyz,means2D=means,shs=None,colors_precomp=rgb,opacities=opacity,scales=scale,rotations=rotation,cov3D_precomp=None);ids=(torch.nonzero(visible).flatten()[:,None]*m.n_offsets+torch.arange(m.n_offsets,device='cuda')).flatten()[active];return im,means,radii,scale,ids
 # Check every true model gradient before any optimizer step.
 m=models['signed'];im,means,_,_,_=render(m,'159.png',True,native);photo=torch.as_tensor(np.asarray(Image.open(parent/'scene/images/159.png')).copy(),device='cuda').permute(2,0,1).float()/255;loss=.8*l1_loss(im,photo)+.2*(1-ssim(im,photo));loss.backward();old={k:p.grad.detach().clone() for k,p in frozen_rgb_named_parameters(m) if p.grad is not None};base=im.detach();m.optimizer.zero_grad(set_to_none=True)
 ii,mm,_,_,_=render(m,'159.png',True);ll=.8*l1_loss(ii,photo)+.2*(1-ssim(ii,photo));ll.backward();assert torch.equal(base,ii);maxdiff=0.
 for k,p in frozen_rgb_named_parameters(m):
  if k in old:assert torch.allclose(old[k],p.grad,atol=2e-6,rtol=3e-4),k;maxdiff=max(maxdiff,float((old[k]-p.grad).abs().max()))
 m.optimizer.zero_grad(set_to_none=True)
 # Independent 8-pixel native oracle for both absolute components.
 im,means,_,_,_=render(m,'159.png',True,native);pix=[(80,760),(100,800),(200,920),(1100,790),(700,400),(550,650),(80,180),(900,820)];fixed=torch.zeros_like(im);fixed[:,[y for x,y in pix],[x for x,y in pix]]=1e-5;oracle=torch.zeros((len(means),2),device='cuda')
 for x,y in pix:
  one=torch.zeros_like(im);one[:,y,x]=1e-5;oracle+=torch.autograd.grad(im,means,one,retain_graph=True)[0][:,:2].abs()
 ii,mm,_,_,_=render(m,'159.png',True);torch.autograd.grad(ii,mm,fixed);assert torch.allclose(oracle,proxy.absolute,atol=2e-7,rtol=3e-4)
 save(o/'preflight.json',dict(initial_model_optimizer_exact=True,original_forward_exact=True,all_model_gradients_match=True,max_parameter_grad_diff=maxdiff,absolute_xy_native8pixel_oracle_error=float((oracle-proxy.absolute).abs().max()),optimizer_steps=0))
 del old,base,im,ii,loss,ll,oracle,photo;torch.cuda.empty_cache()
 acc={k:torch.zeros((len(m.get_anchor)*m.n_offsets,2),device='cuda') for k in models};counts={k:torch.zeros(len(m.get_anchor)*m.n_offsets,device='cuda') for k in models};growth=[];logs=[]
 def candidates(m,k,level,iteration):
  score=acc[k][:,0]/counts[k].clamp_min(1);qualified=(counts[k]>40)&((acc[k][:,1]/counts[k].clamp_min(1))>=opt.densify_grad_threshold*((m.update_hierachy_factor//2)**level));gen=torch.Generator(device='cuda');gen.manual_seed(iteration*10+level);qualified&=(torch.rand(len(qualified),generator=gen,device='cuda')>.5**(level+1));ix=torch.nonzero(qualified).flatten();size=m.voxel_size*(m.update_init_factor//m.update_hierachy_factor**level);xyz=(m.get_anchor[:,None]+m._offset*m.get_scaling[:,:3,None].transpose(1,2)).reshape(-1,3);coords=torch.round(xyz[ix]/size).int().cpu().numpy();existing=set(map(tuple,torch.round(m.get_anchor/size).int().cpu().numpy()));scor=score[ix].cpu().numpy();cells={}
  for i,(coord,s) in enumerate(zip(coords,scor)):
   cell=tuple(map(int,coord))
   if cell not in existing:cells.setdefault(cell,[]).append(i)
  ranked=sorted(cells,key=lambda cell:(-max(scor[cells[cell]]),cell));return ranked,cells,ix,size
 def add(m,data,chosen):
  ranked,cells,ix,size=data;coords=torch.tensor(chosen,dtype=torch.float32,device='cuda')*size;nn=len(chosen);feat=torch.stack([m._anchor_feat[(ix[torch.tensor(cells[cell],device='cuda')]//m.n_offsets)].max(0).values for cell in chosen]);tensors=dict(anchor=coords,offset=torch.zeros(nn,m.n_offsets,3,device='cuda'),anchor_feat=feat,scaling=torch.full((nn,6),float(np.log(size)),device='cuda'),rotation=torch.tensor([[1.,0,0,0]],device='cuda').repeat(nn,1),opacity=torch.full((nn,1),float(np.log(.1/.9)),device='cuda'));new=m.cat_tensors_to_optimizer(tensors)
  for name in new:setattr(m,'_'+name,new[name])
  m.max_radii2D=torch.cat([m.max_radii2D,torch.zeros(nn,device='cuda')])
 for it,n in enumerate(order,7001):
  photo=torch.as_tensor(np.asarray(Image.open(parent/'scene/images'/n)).copy(),device='cuda').permute(2,0,1).float()/255
  for k,m in models.items():
   m.update_learning_rate(it);im,means,radii,scale,ids=render(m,n,True);loss=.8*l1_loss(im,photo)+.2*(1-ssim(im,photo))+.01*scale.prod(1).mean();gg=torch.autograd.grad(loss,means,retain_graph=True)[0];absolute=proxy.absolute.clone();loss.backward();valid=radii>0;score=(gg[:,:2] if k=='signed' else absolute).norm(dim=1);acc[k][ids[valid],0]+=score[valid];acc[k][ids[valid],1]+=gg[valid,:2].norm(dim=1);counts[k][ids[valid]]+=1;logs.append(dict(iteration=it,arm=k,camera=n,loss=float(loss.detach())));del im,means,loss,gg,absolute
  if it%100==0:
   event=dict(iteration=it,levels=[])
   with torch.no_grad():
    for level,cap in enumerate(c['level_caps']):
     data={k:candidates(m,k,level,it) for k,m in models.items()};q=min(cap,*[len(x[0]) for x in data.values()]);entry=dict(level=level,quota=q,available={k:len(x[0]) for k,x in data.items()},selected={k:[list(z) for z in x[0][:q]] for k,x in data.items()})
     # Same-score deterministic selection invariant, plus synchronized quota.
     for k,x in data.items():assert x[0][:q]==sorted(x[0],key=lambda cell:(-max((acc[k][:,0]/counts[k].clamp_min(1))[x[2][torch.tensor(x[1][cell],device='cuda')]].cpu().numpy()),cell))[:q]
     if q:
      for k,m in models.items():add(m,data[k],data[k][0][:q]);acc[k]=torch.cat([acc[k],torch.zeros(q*m.n_offsets,2,device='cuda')]);counts[k]=torch.cat([counts[k],torch.zeros(q*m.n_offsets,device='cuda')])
     event['levels'].append(entry)
    assert len(models['signed'].get_anchor)==len(models['absolute'].get_anchor)
    for k in models:eligible=counts[k]>40;acc[k][eligible]=0;counts[k][eligible]=0
   growth.append(event);save(o/'growth.json',growth)
  for m in models.values():m.optimizer.step();m.optimizer.zero_grad(set_to_none=True)
  if it%50==0:save(o/'progress.json',dict(iteration=it,completed_steps_per_arm=it-7000,anchors={k:len(m.get_anchor) for k,m in models.items()}));print(it,flush=True)
 save(o/'training.json',dict(status='completed_500_each',logs=logs,growth=growth,anchors={k:len(m.get_anchor) for k,m in models.items()},seconds=time.monotonic()-start))
 for k,m in models.items():
  dest=o/k/'model';point=dest/'point_cloud/iteration_7500';point.mkdir(parents=True);shutil.copy2(parent/'model/cfg_args',dest/'cfg_args');torch.save((m.capture(),7500),dest/'chkpnt7500.pth');m.save_ply(str(point/'point_cloud.ply'));m.save_mlp_checkpoints(str(point));m.eval()
 # Evaluate all400 official reference cameras, uid0. No training labels used above.
 from scipy.ndimage import binary_dilation
 labelroot=ROOT/'outputs/quality/all_data/SEM-370-r1/labels';rows=[]
 for k in models:(o/k/'images').mkdir()
 with torch.no_grad():
  for n in names:
   photo=np.asarray(Image.open(parent/'scene/images'/n)).astype(float);images={}
   for k,m in models.items():
    im,*_=render(m,n,False);im=np.rint(im.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype('uint8');images[k]=im;Image.fromarray(im).save(o/k/'images'/n)
   masks={'frame':np.ones(photo.shape[:2],bool)}
   if (labelroot/n).exists():
    labels=np.asarray(Image.open(labelroot/n));masks.update(deck=labels==1,stay_cable=labels==2,structure=binary_dilation(np.isin(labels,[1,2,3,4]),iterations=4))
   oldcfg=json.loads((ROOT/'experiments/runs/RGB-122-LOCAL-OPACITY-PF/config.yaml').read_text())
   if n in oldcfg['shore_boxes']:
    mask=np.zeros(photo.shape[:2],bool)
    for x0,y0,x1,y1 in oldcfg['shore_boxes'][n]:mask[y0:y1,x0:x1]=True
    masks['shore']=mask
   metrics={key:{k:dict(mae=float(np.abs(im.astype(float)-photo)[mask].mean()),mse=float(((im.astype(float)-photo)[mask]**2).mean())) for k,im in images.items()} for key,mask in masks.items() if mask.any()};rows.append(dict(view=n,metrics=metrics))
 save(o/'evaluation.json',rows)
 assert all(sha256(parent/k)==h for k,h in hashes.items());save(o/'result.json',dict(status='completed_training_and400_evaluation',experiment_id=c['experiment_id'],steps_per_arm=500,source_parent_unchanged=True,seconds=time.monotonic()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
if __name__=='__main__':main()
