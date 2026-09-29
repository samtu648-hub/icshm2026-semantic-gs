#!/usr/bin/env python3
"""Two frozen semantic/RGB crossover arms; unchanged native Mask2Former inference."""
import argparse,fcntl,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes,write_new_json
from scripts.backbone_comparison_stage import setup
EID='SEM-376-CROSS-RGB-INFERENCE'
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--arm',choices=['standard_sem375','scaffold_sem371'],required=True);p.add_argument('--fresh',action='store_true');a=p.parse_args();cfgpath=ROOT/'experiments/runs'/EID/'config.yaml';cfg=json.loads(cfgpath.read_text());arm=cfg['arms'][a.arm];rgb=ROOT/arm['rgb'];sem=ROOT/arm['semantic'];batch=ROOT/cfg['output_path']/a.arm;batch.mkdir(parents=True,exist_ok=True);out=batch/('fresh' if a.fresh else 'inference');out.mkdir(exist_ok=False)
 exp=json.loads((rgb/'export/result.json').read_text());fit=json.loads((sem/'mask2former/result.json').read_text());split=json.loads((ROOT/cfg['split_path']).read_text());fixed=json.loads((ROOT/'experiments/runs/SEM-370-ALL300/config.yaml').read_text())['diagnostic_views'];ckpt=sem/'mask2former/model-6000.pt';code={n:sha256(ROOT/n) for n in ['scripts/run_sem376_cross.py','scripts/backbone_comparison_stage.py','src/quality_mask2former.py','src/quality_mask2former_training.py','src/quality_semantic_audit.py','src/semantic_cable_binary.py']};start=time.monotonic()
 r=dict(status='starting',experiment_id=EID,arm=a.arm,stage='fresh' if a.fresh else 'inference',git_commit=cfg['git_commit'],seed=cfg['seed'],config_sha256=sha256(cfgpath),manifest_sha256=sha256(ROOT/cfg['dataset_manifest']),dataset_manifest=cfg['dataset_manifest'],split_id=cfg['split_id'],split_sha256=sha256(ROOT/cfg['split_path']),rgb_experiment_id=arm['rgb_id'],semantic_experiment_id=arm['semantic_id'],rgb_export_sha256=sha256(rgb/'export/result.json'),checkpoint_sha256=sha256(ckpt),training_steps=0,code_sha256=code,started_unix=time.time())
 def verify():
  verify_hashes(rgb/'export/images',exp['image_hashes']);verify_hashes(sem/'mask2former',{'model-6000.pt':fit['artifacts']['model-6000.pt']});verify_hashes(ROOT,code);assert sha256(cfgpath)==r['config_sha256']
 try:
  verify();assert len(exp['image_hashes'])==544 and len(split['semantic_train'])==300
  with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
   print('Waiting for shared GPU lock',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
   apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);r['gpu_processes_before']=apps
   if any(s.strip() and 'ToDesk' not in s for s in apps.splitlines()):raise RuntimeError('Other GPU compute process active')
   a.method='standard-3dgs';runtime,torch,np=setup(a);torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
   from PIL import Image
   from src.quality_mask2former import config_for_model,load_official_model,native_predict
   from src.semantic_cable_binary import read_rgb
   from src.quality_semantic_audit import independent_confusion,metrics
   c=config_for_model(runtime/'vendor/Mask2Former','configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml',False);model,_=load_official_model(c,runtime/'weights/model_final_6b4a3a.pkl');model.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True)['model']);model.cuda()
   write_new_json(out/'started.json',r);names=fixed if a.fresh else sorted(exp['image_hashes']);total=np.zeros((5,5),np.int64);rows=[];times=[];labelled=set(split['semantic_train'])
   if not a.fresh:(out/'predictions').mkdir();(out/'probabilities').mkdir()
   for i,n in enumerate(names):
    x=read_rgb(rgb/'export/images'/n);torch.cuda.synchronize();t=time.monotonic();prob=native_predict(model,x);torch.cuda.synchronize();times.append(time.monotonic()-t)
    assert prob.shape==(5,989,1320) and np.isfinite(prob).all();pred=prob.argmax(0).astype(np.uint8)
    if a.fresh:
     with np.load(batch/'inference/probabilities'/(Path(n).stem+'.npz')) as z:assert np.array_equal(prob,z['probabilities']),n
     assert np.array_equal(pred,np.asarray(Image.open(batch/'inference/predictions'/n))),n
    else:
     Image.fromarray(pred).save(out/'predictions'/n)
     if n in fixed:np.savez_compressed(out/'probabilities'/(Path(n).stem+'.npz'),probabilities=prob)
     if n in labelled:
      gt=np.asarray(Image.open(ROOT/'outputs/quality/all_data/SEM-370-r1/labels'/n));cm=independent_confusion(gt,pred);total+=cm;rows.append(dict(name=n,**metrics(cm)))
    if i%50==0:print(json.dumps(dict(arm=a.arm,fresh=a.fresh,views=i+1,total=len(names))),flush=True)
   if a.fresh:r.update(fresh_probability_exact=len(names),fresh_png_exact=len(names))
   else:
    assert len(rows)==300;r.update(metrics=metrics(total),per_view=rows,labeled_count=300,unlabeled_count=100,trajectory_count=144,prediction_hashes={n:sha256(out/'predictions'/n) for n in names})
   r.update(inference_seconds=times,inference_median_seconds=float(np.median(times[1:])),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
  verify();r.update(status='completed',rgb544_unchanged=True,checkpoint_unchanged=True)
 except BaseException as e:r.update(status='failed_preserved',error=f'{type(e).__name__}: {e}');raise
 finally:r['seconds']=time.monotonic()-start;write_new_json(out/'result.json',r)
if __name__=='__main__':main()
