"""Read-only author filter cache consistency probe, no saved model mutation."""
import argparse,fcntl,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);a=p.parse_args();out=ROOT/'outputs/quality/diagnostics/RGB-132-r1/mip-cache-probe.json';assert not out.exists()
 source=a.artifact_root/'r1/third_party/mip-splatting';sys.path[:0]=[str(source),str(source/'submodules/diff-gaussian-rasterization'),str(source/'submodules/simple-knn')]
 r=dict(status='started',experiment_id='RGB-132-THREE-BACKBONE-DIAG');start=time.monotonic()
 try:
  with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=process_name','--format=csv,noheader'],text=True)
   assert not any(x.strip() and 'ToDesk' not in x for x in apps.splitlines()),'Other GPU task active'
   import torch
   from gaussian_renderer import GaussianModel
   from src.quality_cameras import read_camera_requests
   from src.quality_native_gaussian import make_native_view
   torch.set_num_threads(4);torch.manual_seed(0)
   ply=ROOT/'outputs/quality/backbone_comparison/mip-full-r1/model/point_cloud/iteration_30000/point_cloud.ply';before=sha256(ply);m=GaussianModel(3);m.load_ply(str(ply));old=m.filter_3D.clone();cams=read_camera_requests(ROOT/'outputs/quality/all_data/RGB-120-r2/scene/sparse/0')
   m.compute_3D_filter([make_native_view(c) for c in cams.values()]);diff=(m.filter_3D-old).abs();relative=diff/old.clamp_min(1e-12)
   r.update(status='completed',count=len(old),changed=int((diff>0).sum()),max_abs=float(diff.max()),max_relative=float(relative.max()),relative_over_1pct=int((relative>.01).sum()),relative_over_10pct=int((relative>.1).sum()),relative_quantiles=torch.quantile(relative.flatten(),torch.tensor([.5,.9,.99,1.],device='cuda')).tolist(),parent_ply_sha256=before,source_train_sha256=sha256(source/'train.py'),source_model_sha256=sha256(source/'scene/gaussian_model.py'),code_sha256=sha256(Path(__file__)))
   assert sha256(ply)==before
 except Exception as e:r.update(status='failed_preserved',error=repr(e));raise
 finally:r['seconds']=time.monotonic()-start;out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
