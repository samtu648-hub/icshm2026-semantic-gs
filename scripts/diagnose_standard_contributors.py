"""RGB134: fixed ROI derivative ranking, followed by all-view counterfactuals."""
import argparse,fcntl,json,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);a=p.parse_args()
 cp=ROOT/'experiments/runs/RGB-134-STANDARD-CONTRIBUTION-DIAG/config.yaml';c=json.loads(cp.read_text());out=ROOT/c['output'];out.mkdir(parents=True,exist_ok=False)
 r=dict(status='waiting_gpu',**c,code_sha256=sha256(Path(__file__)),config_sha256=sha256(cp));save(out/'progress.json',r);start=time.monotonic()
 try:
  with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
   print('Waiting shared GPU lock',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
   # A previous lock holder can release before its CUDA context exits.
   drain=time.monotonic()
   while True:
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=process_name','--format=csv,noheader'],text=True)
    if not any(x.strip() and 'ToDesk' not in x for x in apps.splitlines()):break
    if time.monotonic()-drain>60:raise RuntimeError('GPU context remains occupied after lock acquisition: '+apps)
    time.sleep(1)
   begin=time.monotonic();source=a.artifact_root/'r1/third_party/standard-3dgs';sys.path[:0]=[str(source),str(source/'submodules/diff-gaussian-rasterization'),str(source/'submodules/simple-knn')]
   import numpy as np,torch
   from PIL import Image
   from gaussian_renderer import GaussianModel,render
   from src.quality_cameras import read_camera_requests
   from src.quality_native_gaussian import make_native_view
   torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
   base=ROOT/'outputs/quality/backbone_comparison';parent=ROOT/'outputs/quality/all_data/RGB-120-r2';ply=base/'standard-full-r1/model/point_cloud/iteration_30000/point_cloud.ply';h=sha256(ply);m=GaussianModel(3);m.load_ply(str(ply));pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False,antialiasing=False);bg=torch.zeros(3,device='cuda');cams=read_camera_requests(parent/'scene/sparse/0')
   for value in vars(m).values():
    if isinstance(value,torch.Tensor):value.requires_grad_(False)
   m._opacity.requires_grad_(True);original=m._opacity.detach().clone();count=len(original)
   def rgb(n):return render(make_native_view(cams[n]),m,pipe,bg)['render']
   def quant(t):return (t*255).round().to(torch.uint8).permute(1,2,0)
   def mask_region(reg):
    mask=torch.zeros((989,1320),device='cuda',dtype=torch.bool)
    for x0,y0,x1,y1 in reg['boxes']:mask[y0:y1,x0:x1]=True
    return mask
   region_masks={k:mask_region(v) for k,v in c['regions'].items()};rank=[];chosen=set()
   for key,reg in c['regions'].items():
    n=reg['view'];target=torch.tensor(np.asarray(Image.open(parent/'scene/images'/n)).copy(),device='cuda').permute(2,0,1).float()/255
    pred=rgb(n);expected=np.asarray(Image.open(base/'standard/export/images'/n));assert np.array_equal(quant(pred.detach()).cpu().numpy(),expected),n
    loss=(pred-target).abs()[:,region_masks[key]].mean();g=torch.autograd.grad(loss,m._opacity)[0].flatten();idx=torch.topk(g,c['top_per_region']).indices;ids=[int(i) for i in idx if g[i]>0];chosen.update(ids)
    rank.append(dict(region=key,view=n,loss=float(loss),ids=ids,derivatives=[float(g[i]) for i in ids]));del pred,loss,g,target
   ids=sorted(chosen);assert ids;geom=[]
   for i in ids:
    scales=m.get_scaling[i].detach();geom.append(dict(id=i,xyz=m.get_xyz[i].tolist(),scales=scales.tolist(),anisotropy=float(scales.max()/scales.min()),opacity=float(m.get_opacity[i])))
   selection=dict(ids=ids,ranking=rank,geometry=geom,parent_ply_sha256=h,count=count,scope='frozen before all400 counterfactuals');save(out/'selection.json',selection);print('Frozen candidates',ids,flush=True)
   r.update(status='running',selection_sha256=sha256(out/'selection.json'),parent_ply_sha256=h,split_sha256=sha256(parent/'split.json'),source_renderer_sha256=sha256(source/'gaussian_renderer/__init__.py'));save(out/'progress.json',r)
   (out/'images').mkdir();rows=[]
   with torch.inference_mode():
    for vi,n in enumerate(sorted(cams)):
     m._opacity.copy_(original);old=quant(rgb(n));oldfloat=old.float();photo=torch.tensor(np.asarray(Image.open(parent/'scene/images'/n)).copy(),device='cuda').float();base_error=(oldfloat-photo).abs().mean(2)
     if n in c['views']:assert np.array_equal(old.cpu().numpy(),np.asarray(Image.open(base/'standard/export/images'/n))),n
     masks={k:v for k,v in region_masks.items() if c['regions'][k]['view']==n};gtpath=ROOT/'outputs/quality/all_data/SEM-370-r1/labels'/n
     if gtpath.exists():
      gt=torch.tensor(np.asarray(Image.open(gtpath)).copy(),device='cuda');masks.update({cl:gt==j for j,cl in enumerate(['background','deck','stay_cable','tower','foundation'])})
     for i in ids:
      m._opacity[i]=-100.;new=quant(rgb(n));m._opacity[i]=original[i];change=(new!=old).any(2);delta=(new.float()-photo).abs().mean(2)-base_error
      row=dict(view=n,id=i,changed_pixels=int(change.sum()),mae_delta_255=float(delta.mean()),changed_mae_delta_255=float(delta[change].mean()) if change.any() else 0.,regions={k:dict(pixels=int(mask.sum()),changed=int((mask&change).sum()),mae_delta_255=float(delta[mask].mean())) for k,mask in masks.items() if mask.any()});rows.append(row)
      if n in c['views']:Image.fromarray(new.cpu().numpy()).save(out/'images'/f'{i}-{n}')
     if vi%25==0:save(out/'progress.json',dict(status='running',views=vi+1,total=400,candidates=len(ids),gpu_seconds=time.monotonic()-begin));print('Views',vi+1,'/400',flush=True)
    m._opacity.copy_(original);assert torch.equal(m._opacity,original);assert sha256(ply)==h
   summary=[]
   for i in ids:
    rr=[x for x in rows if x['id']==i];keys=sorted(set(k for x in rr for k in x['regions']));summary.append(dict(id=i,support_views=sum(x['changed_pixels']>0 for x in rr),better_views=sum(x['mae_delta_255']<0 for x in rr),worse_views=sum(x['mae_delta_255']>0 for x in rr),mean_mae_delta_255=float(np.mean([x['mae_delta_255'] for x in rr])),changed_pixels=sum(x['changed_pixels'] for x in rr),regions={k:dict(changed=sum(x['regions'][k]['changed'] for x in rr if k in x['regions']),mean_mae_delta_255=float(np.mean([x['regions'][k]['mae_delta_255'] for x in rr if k in x['regions']]))) for k in keys}))
   r.update(status='completed',rows=rows,summary=summary,parent_unchanged=True,baseline_nine_exact=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,gpu_seconds=time.monotonic()-begin)
 except BaseException as e:r.update(status='failed_preserved',error=repr(e));raise
 finally:r['wall_seconds']=time.monotonic()-start;save(out/'result.json',r)
if __name__=='__main__':main()
