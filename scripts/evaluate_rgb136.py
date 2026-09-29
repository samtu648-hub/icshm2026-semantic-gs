#!/usr/bin/env python3
"""Native export/evaluation and fresh closure of the UID-independent Scaffold endpoint."""
import argparse,fcntl,json,os,sys,time
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,write_new_json,verify_hashes
from src.quality_cameras import read_camera_requests
from src.quality_scaffold import ScaffoldQualityRenderer
EID='RGB-136-SCAFFOLD-NOAPPEARANCE-ALL400'
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['export','rgb-evaluation','fresh'],required=True);a=p.parse_args();o=a.run.resolve();out=o/a.stage;out.mkdir(exist_ok=False);c=json.loads((ROOT/'experiments/runs'/EID/'config.yaml').read_text());parent=ROOT/c['parent_run'];pr=json.loads((o/'result.json').read_text());assert pr['status']=='completed_all400';code=['scripts/evaluate_rgb136.py','scripts/backbone_comparison_stage.py','scripts/evaluate_rgb.py','src/quality_scaffold.py','src/scaffold_semantic_sidecar.py','src/quality_native_gaussian.py'];r=dict(code_sha256={n:sha256(ROOT/n) for n in code},experiment_id=EID,stage=a.stage,status='starting',git_commit=pr['git_commit'],seed=0,config_sha256=pr['config_sha256'],split_sha256=pr['split_sha256'],manifest_sha256=pr['manifest_sha256'],script_sha256=sha256(Path(__file__)),parent_result_sha256=sha256(o/'result.json'));start=time.monotonic()
 try:
  with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   import numpy as np,torch
   from PIL import Image
   torch.set_num_threads(4);torch.manual_seed(20260805);torch.cuda.manual_seed_all(20260805);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.reset_peak_memory_stats();os.environ['TORCH_HOME']=str(a.artifact_root/'r1/weights/torch');verify_hashes(o,pr['artifacts']);write_new_json(out/'started.json',r)
   if a.stage in ['export','fresh']:
    renderer=ScaffoldQualityRenderer(a.artifact_root/'r1/third_party/scaffold-gs',o/'model',30000,train_camera_count=400);assert renderer.model.appearance_dim==0 and renderer.model.embedding_appearance is None;sig=renderer.rgb_signature()
    cams=read_camera_requests(parent/'scene/sparse/0');trajectory_path=a.artifact_root/'results/frozen_scaffold_sem001/flythrough_v1/cameras';traj=read_camera_requests(trajectory_path);requests={**cams,**{n:replace(c,model='PINHOLE',radial_k=0.) for n,c in traj.items()}};assert len(cams)==400 and len(traj)==144
    if a.stage=='export':
     import inspect
     from types import SimpleNamespace
     from scene.dataset_readers import readColmapSceneInfo
     from utils.camera_utils import loadCam
     from src.quality_native_gaussian import make_native_view
     from src.scaffold_semantic_sidecar import render_frozen_rgb
     params=dict(path=str(o/'scene'),images='images',eval=False,lod=0)
     info=readColmapSceneInfo(**{k:v for k,v in params.items() if k in inspect.signature(readColmapSceneInfo).parameters})
     selected={Path(v.image_name).stem:v for v in info.train_cameras};r['native_camera_closure']=[]
     with torch.inference_mode():
      for n in ['159.png','057.png','174.png']:
       kw=dict(args=SimpleNamespace(resolution=1,data_device='cpu'),id=0,cam_info=selected[Path(n).stem],resolution_scale=1)
       v=loadCam(**{k:v for k,v in kw.items() if k in inspect.signature(loadCam).parameters});custom=make_native_view(cams[n]);delta=float((v.full_proj_transform-custom.full_proj_transform).abs().max());assert delta<1e-4
       native=render_frozen_rgb(v,renderer.model,renderer.pipe).clamp(0,1).permute(1,2,0).cpu().numpy();adapted=renderer.render_pinhole(cams[n]).rgb;error=float(np.abs(native-adapted).max());assert error<1e-4
       r['native_camera_closure'].append(dict(view=n,projection_max_abs=delta,rgb_max_abs=error))
    r.update(trajectory_names=sorted(traj),trajectory_metadata_sha256={n:sha256(trajectory_path/n) for n in ['images.txt','cameras.txt']},signature=sig,image_hashes={},render_seconds=[])
    if a.stage=='export':(out/'images').mkdir()
    with torch.inference_mode():
     for i,(n,cam) in enumerate(sorted(requests.items())):
      torch.cuda.synchronize();t=time.monotonic();rgb=renderer.render_pinhole(cam).rgb;torch.cuda.synchronize();r['render_seconds'].append(time.monotonic()-t);im=np.rint(rgb*255).astype('uint8');assert im.shape==(989,1320,3) and np.isfinite(rgb).all()
      if a.stage=='export':Image.fromarray(im).save(out/'images'/n);r['image_hashes'][n]=sha256(out/'images'/n)
      else:assert np.array_equal(im,np.asarray(Image.open(o/'export/images'/n))),n
      if i%50==0:print(f'{a.stage}: {i+1}/544',flush=True)
    assert sig==renderer.rgb_signature();r.update(native_png_exact=544 if a.stage=='fresh' else None,render_median_seconds=float(np.median(r['render_seconds'][1:])))
   else:
    from scripts.backbone_comparison_stage import rgb_eval
    from types import SimpleNamespace
    export=json.loads((o/'export/result.json').read_text());verify_hashes(o/'export/images',export['image_hashes']);rgb_eval(SimpleNamespace(batch=o),out,r,torch,np);verify_hashes(o/'export/images',export['image_hashes'])
   verify_hashes(ROOT,r['code_sha256']);verify_hashes(o,pr['artifacts']);frozen={x['path']:x['sha256'] for x in json.loads((ROOT/'experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json').read_text())['artifacts']};verify_hashes(a.artifact_root,frozen)
   r.update(status='completed',frozen18_exact=True,endpoint_unchanged=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
 except BaseException as e:r.update(status='failed_preserved',error=f'{type(e).__name__}: {e}');raise
 finally:r['seconds']=time.monotonic()-start;write_new_json(out/'result.json',r)
if __name__=='__main__':main()
