#!/usr/bin/env python3
"""Fresh native RGB reload after semantic fitting; official-camera PNG closure."""
import argparse,fcntl,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes,write_new_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);a=p.parse_args();start=time.monotonic()
 cfg=json.loads((ROOT/'experiments/runs/SEM-375-SCAFFOLD-NOAPPEARANCE-ALL300/config.yaml').read_text());batch=ROOT/cfg['output_path'];run=ROOT/cfg['rgb_run'];parent=json.loads((run/'result.json').read_text());exp=json.loads((run/'export/result.json').read_text())
 assert json.loads((batch/'mask2former/result.json').read_text())['status']=='completed'
 with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  import numpy as np,torch
  from PIL import Image
  from dataclasses import replace
  from src.quality_scaffold import ScaffoldQualityRenderer
  from src.quality_cameras import read_camera_requests
  torch.set_num_threads(4);torch.manual_seed(20260805);torch.cuda.manual_seed_all(20260805);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  verify_hashes(run,parent['artifacts']);renderer=ScaffoldQualityRenderer(a.artifact_root/'r1/third_party/scaffold-gs',run/'model',30000,train_camera_count=400);assert renderer.model.appearance_dim==0 and renderer.model.embedding_appearance is None
  cams=read_camera_requests(run/'scene/sparse/0');trajpath=a.artifact_root/'results/frozen_scaffold_sem001/flythrough_v1/cameras';traj=read_camera_requests(trajpath);cams.update({n:replace(c,model='PINHOLE',radial_k=0.) for n,c in traj.items()});assert set(cams)==set(exp['image_hashes']);verify_hashes(trajpath,exp['trajectory_metadata_sha256'])
  with torch.inference_mode():
   for n,c in sorted(cams.items()):
    actual=np.rint(renderer.render_pinhole(c).rgb*255).astype(np.uint8);assert np.array_equal(actual,np.asarray(Image.open(batch/'export/images'/n))),n
  verify_hashes(run,parent['artifacts']);verify_hashes(batch/'export/images',exp['image_hashes'])
  for path,hashes in json.loads((ROOT/'outputs/quality/scaffold_noappearance/old_model_identities.json').read_text()).items():verify_hashes(ROOT/path,hashes)
  frozen={x['path']:x['sha256'] for x in json.loads((ROOT/'experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json').read_text())['artifacts']};verify_hashes(a.artifact_root,frozen)
 write_new_json(batch/'fresh_rgb_audit.json',dict(status='completed',experiment_id=cfg['experiment_id'],fresh_native_png_exact=544,trajectory_metadata_exact=True,appearance_dim=0,old_three_rgb_models_unchanged=True,frozen18_exact=True,endpoint_unchanged=True,seconds=time.monotonic()-start,script_sha256=sha256(Path(__file__))))
if __name__=='__main__':main()
