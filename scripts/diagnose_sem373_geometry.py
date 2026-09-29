#!/usr/bin/env python3
"""Fixed9 post-fit geometry/SH counterfactual rendering, zero parameter updates."""
import argparse,fcntl,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_sem373_joint import CONFIG,setup,load_stage,digest,write

def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);a=p.parse_args();c=json.loads(CONFIG.read_text());base=ROOT/c['output'];out=base/'geometry-counterfactual-r1';out.mkdir(exist_ok=False);r=dict(status='running',config_sha256=digest(CONFIG),fixed_views=c['diagnostic_views'],rows=[],purpose='Separate direct XYZ effect from SH compensation on fixed9 only; not a new trained arm or deployment');start=time.monotonic()
 try:
  with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);s=setup(a,c);t,np=s.torch,s.np
   ctrl=t.load(base/'control-r1/endpoint.pt',map_location='cuda',weights_only=False);joint=t.load(base/'joint-r1/endpoint.pt',map_location='cuda',weights_only=False)
   rois=json.loads((ROOT/'experiments/runs/QUAL-005-PHOTO-INPUT-DIAG/config.yaml').read_text())['rois'];struct=json.loads((ROOT/'experiments/runs/QUAL-004-CURRENT-BOTTLENECK/config.yaml').read_text())['crops']
   control_radii={};r['projected_radius_deltas']=[]
   for arm,geometry,appearance in [('controlXYZ_jointSH',ctrl,joint),('jointXYZ_controlSH',joint,ctrl)]:
    (out/arm).mkdir()
    with t.no_grad():
     for n in ['_xyz','_scaling','_rotation','_opacity']:getattr(s.model,n).copy_(geometry['parameters'][n])
     for n in ['_features_dc','_features_rest']:getattr(s.model,n).copy_(appearance['parameters'][n])
     for name in c['diagnostic_views']:
      pack=s.render(s.views[name],s.model,s.pipe,s.bg);rgb=pack['render'];radii=pack['radii'].detach().cpu().numpy()
      if arm=='controlXYZ_jointSH':control_radii[name]=radii.copy()
      else:
       cr=control_radii[name];common=(cr>0)&(radii>0);delta=radii.astype(float)-cr.astype(float)
       r['projected_radius_deltas'].append(dict(name=name,control_visible=int((cr>0).sum()),joint_visible=int((radii>0).sum()),common_visible=int(common.sum()),increased_on_common=int((delta[common]>0).sum()),decreased_on_common=int((delta[common]<0).sum()),delta_mean_common_px=float(delta[common].mean()),delta_min_max_common_px=[float(delta[common].min()),float(delta[common].max())],note='Native integer 3-sigma screen radius; geometry diagnostic, not true cable diameter'))
      arr=np.rint(rgb.permute(1,2,0).cpu().numpy()*255).astype(np.uint8);s.Image.fromarray(arr).save(out/arm/name)
      gt=s.photos[name].astype(float);err=np.abs(arr.astype(float)-gt).mean(2)/255;target=s.labels[name]
      regions=[]
      for kind,box in [('full',[0,0,1320,989]),('shore_or_sky',rois.get(Path(name).stem,[0,0,1320,989])),('structure',struct.get(Path(name).stem,[0,0,1320,989]))]:
       x,y,x1,y1=box;regions.append(dict(kind=kind,box=box,MAE=float(err[y:y1,x:x1].mean())))
      r['rows'].append(dict(arm=arm,name=name,MAE=float(err.mean()),PSNR=float(-10*np.log10(np.square((arr.astype(float)-gt)/255).mean())),regions=regions,class_mae={n:float(err[target==k].mean()) if (target==k).any() else None for k,n in enumerate(c['class_names'])}))
   r.update(status='completed',optimizer_updates=0,source_sha256=digest(__file__),control_checkpoint_sha256=digest(base/'control-r1/endpoint.pt'),joint_checkpoint_sha256=digest(base/'joint-r1/endpoint.pt'))
 except BaseException as e:r.update(status='failed_preserved',error=repr(e),traceback=traceback.format_exc());raise
 finally:r['seconds']=time.monotonic()-start;write(out/'result.json',r)

if __name__=='__main__':main()
