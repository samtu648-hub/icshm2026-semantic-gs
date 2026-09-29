"""One fixed cache-only candidate against its unchanged Mip RGB131 baseline."""
import argparse,fcntl,json,subprocess,sys,time
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256
def save(p,r):p.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);a=p.parse_args();cp=ROOT/'experiments/runs/RGB-133-MIP-FILTER-REFRESH-PF/config.yaml';c=json.loads(cp.read_text());out=ROOT/c['output'];out.mkdir(parents=True,exist_ok=False);start=time.monotonic();r=dict(status='running',**c,config_sha256=sha256(cp),code_sha256=sha256(Path(__file__)))
 try:
  with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=process_name','--format=csv,noheader'],text=True)
   assert not any(x.strip() and 'ToDesk' not in x for x in apps.splitlines()),'Other GPU task active'
   source=a.artifact_root/'r1/third_party/mip-splatting';sys.path[:0]=[str(source),str(source/'submodules/diff-gaussian-rasterization'),str(source/'submodules/simple-knn')]
   import numpy as np,torch
   from PIL import Image
   from plyfile import PlyData
   from gaussian_renderer import GaussianModel,render
   from arguments import PipelineParams
   from src.quality_cameras import read_camera_requests
   from src.quality_native_gaussian import make_native_view
   from scripts.evaluate_rgb import psnr,ssim,build_lpips_model,lpips_score
   torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
   base=ROOT/'outputs/quality/backbone_comparison';parent=ROOT/'outputs/quality/all_data/RGB-120-r2';ply=base/'mip-full-r1/model/point_cloud/iteration_30000/point_cloud.ply';original_sha=sha256(ply);m=GaussianModel(3);m.load_ply(str(ply));pp=PipelineParams(argparse.ArgumentParser());pipe=pp.extract(argparse.Namespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False));bg=torch.zeros(3,device='cuda')
   cams=read_camera_requests(parent/'scene/sparse/0');trajectory=read_camera_requests(a.artifact_root/'results/frozen_scaffold_sem001/flythrough_v1/cameras');requests={**cams,**{n:replace(cam,model='PINHOLE',radial_k=0.) for n,cam in trajectory.items()}}
   def pixels(cam):return np.rint(render(make_native_view(cam),m,pipe,bg,kernel_size=.1)['render'].clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
   with torch.inference_mode():
    for n in c['views']:assert np.array_equal(pixels(cams[n]),np.asarray(Image.open(base/'mip/export/images'/n))),n
    original={k:v.detach().clone() for k,v in vars(m).items() if isinstance(v,torch.Tensor)}
    m.compute_3D_filter([make_native_view(cam) for cam in cams.values()])
    assert all(torch.equal(v,getattr(m,k)) for k,v in original.items() if k!='filter_3D')
    updated=m.filter_3D.clone();candidate=out/'model/point_cloud/iteration_30000/point_cloud.ply';m.save_ply(str(candidate))
    old= PlyData.read(ply)['vertex'].data;new=PlyData.read(candidate)['vertex'].data
    assert old.dtype.names==new.dtype.names and all(np.array_equal(old[k],new[k]) for k in old.dtype.names if k!='filter_3D')
    m.load_ply(str(candidate));assert torch.equal(updated,m.filter_3D)
    r.update(parent_ply_sha256=original_sha,candidate_ply_sha256=sha256(candidate),only_filter_changed=True,baseline_nine_replay_exact=True,source_train_sha256=sha256(source/'train.py'),source_model_sha256=sha256(source/'scene/gaussian_model.py'),split_sha256=sha256(parent/'split.json'))
    save(out/'started.json',r);(out/'images').mkdir();lp,_,device=build_lpips_model('cuda');rows=[];regions=[];times=[];hashes={};fresh={};oldmetrics={x['name']:x for x in json.loads((base/'mip/rgb-evaluation/result.json').read_text())['per_view']}
    for i,(n,cam) in enumerate(sorted(requests.items())):
     t=time.monotonic();im=pixels(cam);torch.cuda.synchronize();times.append(time.monotonic()-t);Image.fromarray(im).save(out/'images'/n);hashes[n]=sha256(out/'images'/n)
     if n in c['views']:fresh[n]=im.copy()
     orig=np.asarray(Image.open(base/'mip/export/images'/n));delta=np.abs(im.astype(float)-orig.astype(float));row=dict(name=n,changed_pixels=int((delta.max(2)>0).sum()),rgb_delta_mae_255=float(delta.mean()),max_rgb_delta_255=float(delta.max()))
     if n in cams:
      ref=np.asarray(Image.open(parent/'scene/images'/n),dtype=np.float32)/255;pred=im.astype(np.float32)/255
      row.update(PSNR=psnr(ref,pred),SSIM=ssim(ref,pred),LPIPS=lpips_score(ref,pred,lp,torch,device));row['metric_deltas']={k:row[k]-oldmetrics[n][k] for k in ['PSNR','SSIM','LPIPS']}
      gtpath=ROOT/'outputs/quality/all_data/SEM-370-r1/labels'/n; masks={}
      if gtpath.exists():
       gt=np.asarray(Image.open(gtpath));masks={cl:gt==k for k,cl in enumerate(['background','deck','stay_cable','tower','foundation'])}
      for key,reg in c['regions'].items():
       if reg['view']!=n:continue
       mask=np.zeros(im.shape[:2],bool)
       for x0,y0,x1,y1 in reg['boxes']:mask[y0:y1,x0:x1]=True
       masks[key]=mask
      photo=np.asarray(Image.open(parent/'scene/images'/n),dtype=float)
      for key,mask in masks.items():
       if mask.any():regions.append(dict(view=n,region=key,pixels=int(mask.sum()),before_mae=float(np.abs(orig.astype(float)-photo)[mask].mean()),after_mae=float(np.abs(im.astype(float)-photo)[mask].mean()),changed_pixels=int((delta.max(2)[mask]>0).sum())))
     rows.append(row)
     if i%50==0:print(json.dumps(dict(rendered=i+1,total=len(requests))),flush=True)
    m.load_ply(str(candidate))
    for n,im in fresh.items():assert np.array_equal(im,pixels(cams[n]))
    assert sha256(ply)==original_sha
    r.update(status='completed',rows=rows,regions=regions,metrics={k:float(np.mean([x[k] for x in rows if k in x])) for k in ['PSNR','SSIM','LPIPS']},image_hashes=hashes,candidate_nine_reload_exact=True,render_median_ms=float(np.median(times[1:]))*1000,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
 except BaseException as e:r.update(status='failed_preserved',error=repr(e));raise
 finally:r['seconds']=time.monotonic()-start;save(out/'result.json',r)
if __name__=='__main__':main()
