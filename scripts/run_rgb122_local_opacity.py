#!/usr/bin/env python3
"""Fixed single-offset opacity ablation with all400 support and official trajectory."""
import argparse,json,sys,time,shutil,traceback
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from src.quality_protocol import sha256

def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);a=p.parse_args();cp=ROOT/'experiments/runs/RGB-122-LOCAL-OPACITY-PF/config.yaml';c=json.loads(cp.read_text());o=ROOT/c['output'];o.mkdir(parents=True,exist_ok=False)
 try:run(a,c,cp,o)
 except Exception as e:save(o/'failure.json',dict(status='failed_preserved',error=repr(e),traceback=traceback.format_exc()));raise

def run(a,c,cp,o):
 import torch
 from scipy.ndimage import binary_dilation
 from scipy.spatial.transform import Rotation
 from src.quality_scaffold import ScaffoldQualityRenderer
 from src.quality_cameras import read_camera_requests
 from src.quality_native_gaussian import make_native_view
 from src.camera_adapter import CameraFromWorld
 from src.scaffold_semantic_sidecar import _render_color_channels
 from scripts.create_scaffold_flythrough import catmull_rom,slerp,quaternion_to_rotation
 start=time.monotonic();torch.set_num_threads(4);torch.manual_seed(c['seed']);parent=ROOT/c['parent_run'];previous=ROOT/c['previous'];batch=ROOT/c['semantic_batch'];pr=json.loads((parent/'result.json').read_text());hashes={k:sha256(parent/k) for k in pr['artifacts']};assert hashes==pr['artifacts'];assert hashes['model/point_cloud/iteration_30000/point_cloud.ply']==c['ply_sha256']
 frozen=json.loads((ROOT/'experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json').read_text())['artifacts']
 for x in frozen:assert sha256(a.artifact_root/x['path'])==x['sha256']
 model=o/'model';(model/'point_cloud').mkdir(parents=True);shutil.copy2(parent/'model/cfg_args',model/'cfg_args');shutil.copytree(parent/'model/point_cloud/iteration_30000',model/'point_cloud/iteration_30000');copyhash={str(p.relative_to(model)):sha256(p) for p in model.rglob('*') if p.is_file()}
 save(o/'binding.json',dict(config_sha256=sha256(cp),script_sha256=sha256(Path(__file__)),git_commit=c['git_commit'],seed=c['seed'],manifest_sha256=sha256(ROOT/c['dataset_manifest']),split_sha256=sha256(parent/'split.json'),parent_hashes=hashes,copy_hashes=copyhash))
 r=ScaffoldQualityRenderer(a.artifact_root/'r1/third_party/scaffold-gs',model,30000,train_camera_count=400);sig=r.rgb_signature();cams=read_camera_requests(parent/'scene/sparse/0');assert len(cams)==400
 from gaussian_renderer import prefilter_voxel,generate_neural_gaussians
 bg=torch.zeros(3,device='cuda');(o/'images').mkdir();(o/'support').mkdir();rows=[];support=[];trajectory=[]
 def arrays(cam):
  v=make_native_view(cam);vis=prefilter_voxel(v,r.model,r.pipe,bg);xyz,rgb,op,sc,rot,_,active=generate_neural_gaussians(v,r.model,vis,is_training=True);ids=(torch.nonzero(vis).flatten()[:,None]*r.model.n_offsets+torch.arange(r.model.n_offsets,device='cuda')).flatten()[active];return v,(xyz,op,sc,rot,active),rgb,ids==c['stable_id']
 def image(g,rgb,v):return np.rint(_render_color_channels(g,rgb,v,bg).clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype('uint8')
 def evaluate(n,cam,novel=False):
  v,g,rgb,selected=arrays(cam);color=torch.zeros_like(rgb);color[selected]=1;mass=_render_color_channels(g,color,v,bg)[0];m=mass.cpu().numpy();nonzero=m>0;lp=batch/'labels'/n;protected=None
  if not novel and lp.exists():protected=binary_dilation(np.isin(np.asarray(Image.open(lp)),[1,2,3,4]),iterations=4)
  sr=dict(view=n,novel=novel,generated=int(selected.sum()),support_pixels=int(nonzero.sum()),max_Talpha=float(m.max()),sum_Talpha=float(m.sum()),protected_max_Talpha=float(m[protected].max()) if protected is not None else None)
  if nonzero.any():
   ys,xs=np.where(nonzero);sr['bbox']=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)];np.savez_compressed(o/'support'/(Path(n).stem+'.npz'),mass=m)
  support.append(sr)
  if not (novel or nonzero.any() or n in c['views']):return
  base=image(g,rgb,v);arms={'base':base}
  for name,factor in [('half',c['opacity_multiplier']),('zero',0.)]:
   opacity=g[1].clone();opacity[selected]*=factor;arms[name]=image((g[0],opacity,g[2],g[3],g[4]),rgb,v)
  for name,im in arms.items():Image.fromarray(im).save(o/'images'/(name+'-'+n))
  if n in c['views']:
   assert np.array_equal(base,np.asarray(Image.open(previous/'images'/('base-'+n))))
   old=previous/'single'/('44174-'+n)
   if old.exists():assert np.array_equal(arms['zero'],np.asarray(Image.open(old)))
  photo=None if novel else np.asarray(Image.open(parent/'scene/images'/n)).astype(float)
  metrics={}
  for name in ['half','zero']:
   diff=np.abs(arms[name].astype(float)-base);changed=diff.max(2)>0;masks={'frame':np.ones(nonzero.shape,bool),'original_support':nonzero,'changed':changed}
   if protected is not None:masks['protected']=protected
   if n in c['shore_boxes']:
    roi=np.zeros(nonzero.shape,bool)
    for x0,y0,x1,y1 in c['shore_boxes'][n]:roi[y0:y1,x0:x1]=True
    masks['shore']=roi
   metrics[name]={}
   for key,mask in masks.items():
    metrics[name][key]=dict(pixels=int(mask.sum()),changed_pixels=int(changed[mask].sum()),max_delta=float(diff[mask].max()) if mask.any() else 0.,rgb_delta_mae=float(diff[mask].mean()) if mask.any() else 0.)
    if photo is not None and mask.any():metrics[name][key].update(photo_mae_before=float(np.abs(base.astype(float)-photo)[mask].mean()),photo_mae_after=float(np.abs(arms[name].astype(float)-photo)[mask].mean()))
  rows.append(dict(view=n,novel=novel,metrics=metrics))
 with torch.inference_mode():
  # Replay original14 first, then enumerate every remaining official camera.
  order=c['views']+[n for n in sorted(cams) if n not in c['views']]
  for i,n in enumerate(order):
   evaluate(n,cams[n])
   if i%40==0:print(json.dumps(dict(stage='all400',completed=i+1,affected=sum(x['support_pixels']>0 for x in support))),flush=True)
  save(o/'all400_support.json',support)
  keys=[cams[n] for n in c['trajectory']['keyframes']];centers=[k.pose.center for k in keys];qs=[np.roll(Rotation.from_matrix(k.pose.rotation).as_quat(),1) for k in keys]
  for seg in range(2):
   for j in range(4):
    u=(j+.5)/4;center=catmull_rom(centers[max(0,seg-1)],centers[seg],centers[seg+1],centers[min(2,seg+2)],u);rot=quaternion_to_rotation(slerp(qs[seg],qs[seg+1],u*u*(3-2*u)));cam=replace(keys[seg],pose=CameraFromWorld(rot,-rot@center));n=f'novel-{seg}-{j}.png';trajectory.append(dict(view=n,u=u,segment=seg,rotation=rot.tolist(),translation=cam.pose.translation.tolist(),matrix=cam.matrix.tolist(),width=cam.width,height=cam.height));evaluate(n,cam,True)
  for n in ['159.png','160.png','161.png']:
   v,g,rgb,_=arrays(cams[n]);assert np.array_equal(image(g,rgb,v),np.asarray(Image.open(o/'images'/('base-'+n))))
 assert r.rgb_signature()==sig
 assert all(sha256(parent/k)==h for k,h in hashes.items());assert all(sha256(model/k)==h for k,h in copyhash.items())
 for x in frozen:assert sha256(a.artifact_root/x['path'])==x['sha256']
 save(o/'trajectory.json',trajectory);save(o/'result.json',dict(experiment_id=c['experiment_id'],status='completed_fixed_ablation_pending_visual_audit',binding_sha256=sha256(o/'binding.json'),all400_checked=400,baseline14_exact=True,zero_target3_exact=True,undo3_exact=True,old_and_copy_and18_frozen_exact=True,support=support,rows=rows,seconds=time.monotonic()-start))
 print(json.dumps(dict(stage='completed',seconds=time.monotonic()-start,rendered=len(rows))),flush=True)
if __name__=='__main__':main()
