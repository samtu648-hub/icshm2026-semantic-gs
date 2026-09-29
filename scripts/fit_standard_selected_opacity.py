"""RGB135 fixed-budget, all400 refit of only the registered opacity logits."""
import argparse,fcntl,json,random,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);a=p.parse_args();cp=ROOT/'experiments/runs/RGB-135-STANDARD-LOCAL-REPAIR/config.yaml';c=json.loads(cp.read_text());out=ROOT/c['fit_output'];out.mkdir(parents=True,exist_ok=False);(out/'source.py').write_text(Path(__file__).read_text());save(out/'config.json',c);r=dict(status='waiting_gpu',experiment_id=c['experiment_id'],config_sha256=sha256(cp),source_sha256=sha256(Path(__file__)));start=time.monotonic();save(out/'progress.json',r)
 try:
  with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
   print('Waiting shared GPU lock',flush=True);fcntl.flock(lock,fcntl.LOCK_EX);drain=time.monotonic()
   while True:
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=process_name','--format=csv,noheader'],text=True)
    if not any(x.strip() and 'ToDesk' not in x for x in apps.splitlines()):break
    if time.monotonic()-drain>60:raise RuntimeError('Prior CUDA context did not exit')
    time.sleep(1)
   begin=time.monotonic();source=a.artifact_root/'r1/third_party/standard-3dgs';sys.path[:0]=[str(source),str(source/'submodules/diff-gaussian-rasterization'),str(source/'submodules/simple-knn')]
   import numpy as np,torch
   from PIL import Image
   from plyfile import PlyData
   from gaussian_renderer import GaussianModel,render
   from utils.loss_utils import ssim
   from src.quality_cameras import read_camera_requests
   from src.quality_native_gaussian import make_native_view
   try:
    from fused_ssim import fused_ssim
    similarity=lambda a,b:fused_ssim(a[None],b[None])
    r['ssim_implementation']='fused_ssim'
   except ImportError:similarity=ssim;r['ssim_implementation']='native utils.loss_utils.ssim'
   torch.set_num_threads(4);torch.manual_seed(c['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
   parent=ROOT/'outputs/quality/all_data/RGB-120-r2';base=ROOT/'outputs/quality/backbone_comparison';ply=base/'standard-full-r1/model/point_cloud/iteration_30000/point_cloud.ply';h=sha256(ply);m=GaussianModel(3);m.load_ply(str(ply))
   for value in vars(m).values():
    if isinstance(value,torch.Tensor):value.requires_grad_(False)
   original=m._opacity.detach().clone();idx=torch.tensor(c['gaussian_ids'],device='cuda');alpha=torch.nn.Parameter(original[idx].clone());optimizer=torch.optim.Adam([alpha],lr=c['opacity_lr'],eps=1e-15)
   cams=read_camera_requests(parent/'scene/sparse/0');names=sorted(cams);assert len(names)==400;views={n:make_native_view(cam) for n,cam in cams.items()};bg=torch.zeros(3,device='cuda');pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False,antialiasing=False)
   def photo(n):return torch.tensor(np.asarray(Image.open(parent/'scene/images'/n)).copy(),device='cuda').permute(2,0,1).float()/255
   def rgb(n):return render(views[n],m,pipe,bg)['render']
   def loss(n):
    pred=rgb(n);target=photo(n);return .8*(pred-target).abs().mean()+.2*(1-similarity(pred,target))
   with torch.inference_mode():
    for n in c['views']:assert np.array_equal((rgb(n)*255).round().byte().permute(1,2,0).cpu().numpy(),np.asarray(Image.open(base/'standard/export/images'/n))),n
    before=float(np.mean([float(loss(n)) for n in names]))
   # Routing preflight: only15 registered trainable scalars enter native RGB loss.
   m._opacity=original.index_copy(0,idx,alpha);probe=loss('159.png');g=torch.autograd.grad(probe,alpha)[0];assert torch.isfinite(g).all() and g.abs().max()>0
   r.update(baseline_nine_exact=True,trainable_scalar_count=alpha.numel(),preflight_gradient_max=float(g.abs().max()),parent_ply_sha256=h,before_photometric_objective=before,split_sha256=sha256(parent/'split.json'),source_renderer_sha256=sha256(source/'gaussian_renderer/__init__.py'))
   rng=random.Random(c['seed']);schedule=[]
   for epoch in range(c['epochs']):order=names.copy();rng.shuffle(order);schedule.extend(order)
   assert len(schedule)==c['iterations'] and all(schedule.count(n)==10 for n in names);save(out/'sampling.json',schedule);r['sampling_sha256']=sha256(out/'sampling.json');save(out/'started.json',r);train_start=time.monotonic();trace=[]
   for step,n in enumerate(schedule,1):
    optimizer.zero_grad(set_to_none=True);m._opacity=original.index_copy(0,idx,alpha);objective=loss(n);assert torch.isfinite(objective);objective.backward();assert torch.isfinite(alpha.grad).all();optimizer.step()
    if step%200==0:
     item=dict(step=step,loss=float(objective.detach()),training_seconds=time.monotonic()-train_start);trace.append(item);save(out/'progress.json',item);print(json.dumps(item),flush=True)
   training_seconds=time.monotonic()-train_start
   with torch.inference_mode():
    m._opacity=torch.nn.Parameter(original.index_copy(0,idx,alpha.detach()),requires_grad=False);after=float(np.mean([float(loss(n)) for n in names]));candidate=out/'point_cloud.ply';m.save_ply(str(candidate));torch.save(dict(ids=c['gaussian_ids'],logits=alpha.detach().cpu(),optimizer=optimizer.state_dict(),steps=c['iterations']),out/'optimizer.pt')
    old=PlyData.read(ply)['vertex'].data;new=PlyData.read(candidate)['vertex'].data;assert old.dtype.names==new.dtype.names;assert all(np.array_equal(old[k],new[k]) for k in old.dtype.names if k!='opacity');changed=np.flatnonzero(old['opacity']!=new['opacity']);assert set(changed).issubset(set(c['gaussian_ids']));assert sha256(ply)==h
    finalalpha=alpha.detach().clone();m.load_ply(str(candidate));assert torch.equal(m._opacity[idx],finalalpha)
   r.update(status='completed',steps=len(schedule),training_seconds=training_seconds,gpu_seconds=time.monotonic()-begin,after_photometric_objective=after,trace=trace,opacity_before=torch.sigmoid(original[idx]).flatten().tolist(),opacity_after=torch.sigmoid(finalalpha).flatten().tolist(),ids=c['gaussian_ids'],only_selected_opacity_changed=True,candidate_ply_sha256=sha256(candidate),parent_unchanged=True,reload_exact=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
 except BaseException as e:r.update(status='failed_preserved',error=repr(e));raise
 finally:r['wall_seconds']=time.monotonic()-start;save(out/'result.json',r)
if __name__=='__main__':main()
