#!/usr/bin/env python3
"""Supplement: RGB120 native training-UID RGB fit, never change SEM370 inputs."""
import argparse,ast,fcntl,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes,write_new_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);out=a.output
 parent=ROOT/'outputs/quality/all_data/RGB-120-r2';source=a.artifact_root/'r1/third_party/scaffold-gs'
 sys.path[:0]=[str(x) for x in [source,source/'submodules/diff-gaussian-rasterization',source/'submodules/simple-knn']]
 import numpy as np,torch
 torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 from PIL import Image
 from src.quality_scaffold import ScaffoldQualityRenderer
 from src.quality_cameras import read_camera_requests
 from src.quality_native_gaussian import make_native_view
 from src.scaffold_semantic_sidecar import render_frozen_rgb
 from scripts.evaluate_rgb import psnr,ssim,build_lpips_model,lpips_score
 r=dict(experiment_id='QUAL-007-BACKBONE-COMPARISON',rgb_experiment_id='RGB-120-ALL400',status='starting',seed=0,role='supplemental native training-camera appearance; never semantic input',parent_sha256=sha256(parent/'result.json'),code_sha256=sha256(Path(__file__)),source_scene_sha256=sha256(source/'scene/__init__.py'),source_train_sha256=sha256(source/'train.py'),source_gaussian_model_sha256=sha256(source/'scene/gaussian_model.py'));start=time.monotonic()
 try:
  with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   import subprocess
   apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
   if any(line.strip() and 'ToDesk' not in line for line in apps.splitlines()):raise RuntimeError('Another GPU process active')
   saved=json.loads((parent/'result.json').read_text());verify_hashes(parent,saved['artifacts'])
   cameras=read_camera_requests(parent/'scene/sparse/0')
   # Pinned Scaffold training explicitly calls Scene(..., shuffle=False).
   # Its saved cameras.json IDs therefore equal the native training UIDs.
   names=[Path(x['img_name']).stem+'.png' for x in json.loads((parent/'model/cameras.json').read_text())]
   assert len(names)==400 and set(names)==set(cameras)
   source_record=json.loads((parent/'source_manifest.json').read_text())
   assert source_record['train.py']==sha256(source/'train.py')
   training=next(n for n in ast.parse((source/'train.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='training')
   scene_call=next(n for n in ast.walk(training) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='Scene')
   assert any(k.arg=='shuffle' and isinstance(k.value,ast.Constant) and k.value.value is False for k in scene_call.keywords)
   uids={n:i for i,n in enumerate(names)}
   r.update(uid_mapping=uids,uid_reconstruction='saved cameras.json order; pinned author training calls Scene(shuffle=False), verified by AST and original source hash',camera_json_sha256=sha256(parent/'model/cameras.json'))
   write_new_json(out/'started.json',r)
   renderer=ScaffoldQualityRenderer(source,parent/'model',30000,train_camera_count=400);model,_,device=build_lpips_model('cuda');rows=[]
   diagnostic=set(json.loads((ROOT/'experiments/runs/SEM-370-ALL300/config.yaml').read_text())['diagnostic_views']);(out/'images').mkdir()
   with torch.inference_mode():
    for i,n in enumerate(sorted(cameras)):
     view=make_native_view(cameras[n]);view.uid=uids[n]
     rgb=render_frozen_rgb(view,renderer.model,renderer.pipe).clamp(0,1).permute(1,2,0).cpu().numpy()
     pixels=np.rint(rgb*255).astype(np.uint8);pred=pixels.astype(np.float32)/255;ref=np.asarray(Image.open(parent/'scene/images'/n).convert('RGB'),dtype=np.float32)/255
     rows.append(dict(name=n,uid=uids[n],PSNR=psnr(ref,pred),SSIM=ssim(ref,pred),LPIPS=lpips_score(ref,pred,model,torch,device)))
     if n in diagnostic:Image.fromarray(pixels).save(out/'images'/n)
     if uids[n]==0:assert np.array_equal(pixels,np.asarray(Image.open(ROOT/'outputs/quality/backbone_comparison/scaffold/export/images'/n)))
     if i%50==0:print(json.dumps(dict(scored=i+1,total=400)),flush=True)
   verify_hashes(parent,saved['artifacts']);r.update(status='completed',per_view=rows,metrics={k:float(np.mean([x[k] for x in rows])) for k in ['PSNR','SSIM','LPIPS']},uid0_matches_sem370_deployment=True)
 except BaseException as e:r.update(status='failed_preserved',error=str(e));raise
 finally:r['seconds']=time.monotonic()-start;write_new_json(out/'result.json',r)
if __name__=='__main__':main()
