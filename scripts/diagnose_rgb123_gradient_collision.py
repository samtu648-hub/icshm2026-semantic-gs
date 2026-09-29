#!/usr/bin/env python3
"""Instrument a copied rasterizer; measure pixel cancellation without updating RGB."""
import argparse,json,sys,time,shutil,traceback,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from src.quality_protocol import sha256

def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--config',type=Path,default=ROOT/'experiments/runs/RGB-123-GRADIENT-COLLISION-DIAG/config.yaml');a=p.parse_args();cp=a.config;c=json.loads(cp.read_text());o=ROOT/c['output'];o.mkdir(parents=True,exist_ok=False)
 try:run(a,c,cp,o)
 except Exception as e:save(o/'failure.json',dict(status='failed_preserved',error=repr(e),traceback=traceback.format_exc()));raise

def run(a,c,cp,o):
 import torch
 from scipy.ndimage import binary_dilation
 from torch.utils.cpp_extension import load
 from src.quality_scaffold import ScaffoldQualityRenderer
 from src.quality_cameras import read_camera_requests
 from src.quality_native_gaussian import make_native_view
 from src.scaffold_semantic_sidecar import _raster_settings
 start=time.monotonic();torch.set_num_threads(4);torch.manual_seed(c['seed']);parent=ROOT/c['parent_run'];batch=ROOT/c['semantic_batch'];source=a.artifact_root/'r1/third_party/scaffold-gs';up=source/'submodules/diff-gaussian-rasterization';pr=json.loads((parent/'result.json').read_text());hashes={k:sha256(parent/k) for k in pr['artifacts']};assert hashes==pr['artifacts']
 # The only CUDA change writes sum(abs(pixel gx)+abs(pixel gy)) to previously unused z.
 ext=o/'instrumented_source';ext.mkdir();shutil.copytree(up/'cuda_rasterizer',ext/'cuda_rasterizer')
 for n in ['ext.cpp','rasterize_points.cu','rasterize_points.h','LICENSE.md']:shutil.copy2(up/n,ext/n)
 original={str(p.relative_to(ext)):sha256(up/p.relative_to(ext)) for p in ext.rglob('*') if p.is_file()}
 file=ext/'cuda_rasterizer/backward.cu';txt=file.read_text();needle='atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);';assert txt.count(needle)==1;txt=txt.replace(needle,needle+'\n            atomicAdd(&dL_dmean2D[global_id].z, fabsf(dL_dG * dG_ddelx * ddelx_dx) + fabsf(dL_dG * dG_ddely * ddely_dy));');file.write_text(txt)
 build=o/'build';build.mkdir();files=['cuda_rasterizer/rasterizer_impl.cu','cuda_rasterizer/forward.cu','cuda_rasterizer/backward.cu','rasterize_points.cu','ext.cpp']
 save(o/'binding.json',dict(experiment_id=c['experiment_id'],git_commit=c['git_commit'],seed=c['seed'],config_sha256=sha256(cp),script_sha256=sha256(Path(__file__)),parent_hashes=hashes,split_sha256=sha256(parent/'split.json'),manifest_sha256=sha256(ROOT/c['dataset_manifest']),upstream_url='https://github.com/city-super/Scaffold-GS',source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),extension_commit=subprocess.check_output(['git','-C',str(up),'rev-parse','HEAD'],text=True).strip(),license='upstream LICENSE.md retained; research use',original_extension_files=original,instrumented_backward_sha256=sha256(file)))
 diag=load(name='rgb123_abs_pixel_diagnostic',sources=[str(ext/n) for n in files],extra_cuda_cflags=['-O3','-I'+str(up/'third_party/glm')],extra_cflags=['-O3'],build_directory=str(build),verbose=False)
 r=ScaffoldQualityRenderer(source,parent/'model',c.get('iteration',30000),train_camera_count=400);sig=r.rgb_signature();cams=read_camera_requests(parent/'scene/sparse/0');import diff_gaussian_rasterization as dr
 from gaussian_renderer import prefilter_voxel,generate_neural_gaussians
 from utils.loss_utils import l1_loss,ssim
 native=dr._C;bg=torch.zeros(3,device='cuda');(o/'arrays').mkdir();rows=[];validation=[];input_hashes={}
 for n in c['views']:
  v=make_native_view(cams[n])
  with torch.no_grad():
   vis=prefilter_voxel(v,r.model,r.pipe,bg);xyz,rgb,opacity,sc,rot,_,active=generate_neural_gaussians(v,r.model,vis,is_training=True);ids=(torch.nonzero(vis).flatten()[:,None]*r.model.n_offsets+torch.arange(r.model.n_offsets,device='cuda')).flatten()[active];xyz,rgb,opacity,sc,rot=[x.detach() for x in [xyz,rgb,opacity,sc,rot]]
  def render(module):
   dr._C=module;means=torch.zeros_like(xyz,requires_grad=True);im,radii=dr.GaussianRasterizer(raster_settings=_raster_settings(v,bg))(means3D=xyz,means2D=means,shs=None,colors_precomp=rgb,opacities=opacity,scales=sc,rotations=rot,cov3D_precomp=None);return im,means,radii
  base,means,radii=render(native);photo_np=np.asarray(Image.open(parent/'scene/images'/n));photo=torch.as_tensor(photo_np.copy(),device='cuda').permute(2,0,1).float()/255;loss=.8*l1_loss(base,photo)+.2*(1-ssim(base,photo));image_grad=torch.autograd.grad(loss,base,retain_graph=True)[0].detach();native_grad=torch.autograd.grad(base,means,image_grad)[0].detach();base=base.detach()
  im,means,_=render(diag);assert torch.equal(im,base);grad=torch.autograd.grad(im,means,image_grad)[0].detach();maxerr=float((grad[:,:2]-native_grad[:,:2]).abs().max());assert torch.allclose(grad[:,:2],native_grad[:,:2],atol=2e-7,rtol=2e-4),maxerr
  baseline_png=np.rint(base.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype('uint8')
  if c.get('iteration',30000)==30000:assert np.array_equal(baseline_png,np.asarray(Image.open(batch/'export/images'/n)))
  else:Image.fromarray(baseline_png).save(o/('native-'+n))
  input_hashes[n]=dict(photo_sha256=sha256(parent/'scene/images'/n),label_sha256=sha256(batch/'labels'/n),export_sha256=sha256(batch/'export/images'/n))
  if n==c['views'][0]:
   # An independent native-backward oracle at 16 fixed pixels, including structure.
   pixels=[(x,y) for x,y in [(80,760),(100,800),(200,920),(1100,790),(700,400),(800,350),(550,650),(420,760),(80,180),(240,190),(490,210),(920,200),(300,880),(600,600),(680,250),(900,820)]]
   oracle=torch.zeros_like(native_grad);mask=torch.zeros_like(image_grad)
   for x,y in pixels:
    one=torch.zeros_like(image_grad);one[:,y,x]=image_grad[:,y,x];mask+=one;ii,mm,_=render(native);gg=torch.autograd.grad(ii,mm,one)[0].detach();oracle[:,:2]+=gg[:,:2];oracle[:,2]+=gg[:,:2].abs().sum(1)
   ii,mm,_=render(diag);actual=torch.autograd.grad(ii,mm,mask)[0].detach();assert torch.allclose(actual,oracle,atol=2e-8,rtol=2e-4)
   validation.append(dict(kind='native16pixel_sum_oracle',pixels=pixels,max_abs_error=float((actual-oracle).abs().max())))
  label=np.asarray(Image.open(batch/'labels'/n));protected=binary_dilation(np.isin(label,[1,2,3,4]),iterations=4);masks={'full':np.ones(label.shape,bool),'deck':label==1,'stay_cable':label==2}
  for group,boxes in c['roi_boxes'].get(n,{}).items():
   m=np.zeros(label.shape,bool)
   for x0,y0,x1,y1 in boxes:m[y0:y1,x0:x1]=True
   masks[group]=m&~protected
  for group,mask in masks.items():
   if not mask.any():continue
   if group=='full':g=grad
   else:
    ii,mm,_=render(diag);g=torch.autograd.grad(ii,mm,image_grad*torch.as_tensor(mask,device='cuda')[None])[0].detach()
   signed=g[:,:2].abs().sum(1).cpu().numpy();absolute=g[:,2].cpu().numpy();rad=radii.cpu().numpy();valid=absolute>1e-12
   assert np.all(signed<=absolute+2e-7)
   # Mass-weighted cancellation is the primary statistic. Not a training threshold.
   cancel=1-np.clip(signed/np.maximum(absolute,1e-30),0,1);ix=np.flatnonzero(valid);top=ix[np.argsort(-absolute[ix])[:10]]
   def summary(sel):
    return dict(count=int(sel.sum()),absolute_mass=float(absolute[sel].sum()),signed_l1_mass=float(signed[sel].sum()),mass_cancellation=float(1-signed[sel].sum()/absolute[sel].sum()) if absolute[sel].sum()>0 else None,median_cancellation=float(np.median(cancel[sel])) if sel.any() else None)
   row=dict(view=n,region=group,source_pixels=int(mask.sum()),photo_mae=float(np.abs(baseline_png.astype(float)-photo_np)[mask].mean()),**summary(valid),radius_strata={str(lo)+'-'+str(hi):summary(valid&(rad>=lo)&(rad<hi)) for lo,hi in [(0,8),(8,32),(32,128),(128,1000000)]},top_ids=ids[top].cpu().tolist())
   rows.append(row);np.savez_compressed(o/'arrays'/(Path(n).stem+'-'+group+'.npz'),stable_ids=ids.cpu().numpy(),signed_l1=signed,absolute_l1=absolute,cancellation=cancel,radii=rad,source_mask=mask)
  validation.append(dict(view=n,native_xy_max_abs_error=maxerr,forward_float_exact=True,export_png_exact=(True if c.get('iteration',30000)==30000 else None)));dr._C=native
  print(json.dumps(dict(view=n,regions=[x['region'] for x in rows if x['view']==n])),flush=True)
 assert r.rgb_signature()==sig
 assert all(sha256(parent/k)==h for k,h in hashes.items());assert all(sha256(up/k)==h for k,h in original.items())
 frozen=json.loads((ROOT/'experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json').read_text())['artifacts']
 for x in frozen:assert sha256(a.artifact_root/x['path'])==x['sha256']
 save(o/'result.json',dict(experiment_id=c['experiment_id'],status='completed_diagnostic_only',binding_sha256=sha256(o/'binding.json'),rows=rows,validation=validation,input_hashes=input_hashes,model_and_original_extension_unchanged=True,frozen18_exact=True,optimizer_steps=0,seconds=time.monotonic()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
if __name__=='__main__':main()
