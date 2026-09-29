#!/usr/bin/env python3
"""All400 fit and fixed144 evaluation for the registered RGB142/143 arms."""
import argparse
import fcntl
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--experiment',required=True)
    a=p.parse_args();out=a.run/'evaluation';out.mkdir(exist_ok=False)
    start=time.monotonic();result={'experiment_id':a.experiment,'status':'started','seed':0,
        'quality_scope':'all400 training-view fit; fixed144 has no GT',
        'code_sha256':sha256(Path(__file__))}
    try:
        training=json.loads((a.run/'result.json').read_text())
        assert training['status']=='completed_all400'
        verify_hashes(a.run,training['artifacts'])
        result.update(git_commit=training['git_commit'],config_sha256=training['config_sha256'],
            run_result_sha256=sha256(a.run/'result.json'),split_sha256=training['split_sha256'],manifest_sha256=training['manifest_sha256'])
        # Leave the requested post-training inference + RGB140 verification
        # slot to the independent semantic task. Only scheduling state is read.
        queue=ROOT/'outputs/quality/SEM-379/r1/completion-queue.json'
        waiting=time.monotonic()
        while queue.exists():
            try:
                state=json.loads(queue.read_text())
            except json.JSONDecodeError:
                time.sleep(1);continue
            completed={x['name'] for x in state.get('stages',[])}
            if 'verify_sem379_rgb' in completed or state['status'] in ['failed_preserved','artifacts_ready_pending_visual_review']:
                break
            (out/'coordination_wait.json').write_text(json.dumps({'status':'waiting_independent_inference_and_rgb_verification',
                'queue_status':state['status'],'seconds':time.monotonic()-waiting})+'\n')
            time.sleep(10)
        result['coordination_wait_seconds']=time.monotonic()-waiting
        with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            source=ROOT/'outputs/quality/rgb140/source-r1'
            sys.path[:0]=[str(source),str(source/'submodules/diff-gaussian-rasterization'),str(source/'submodules/simple-knn')]
            import torch
            import numpy as np
            from PIL import Image
            from scipy.ndimage import binary_dilation,binary_erosion
            from types import SimpleNamespace
            from gaussian_renderer import GaussianModel,render
            from src.quality_cameras import read_camera_requests
            from src.quality_native_gaussian import make_native_view
            from scripts.evaluate_rgb import psnr,ssim,build_lpips_model,lpips_score
            torch.set_num_threads(4);torch.manual_seed(0)
            torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            parent=ROOT/'outputs/quality/all_data/RGB-120-r2';cams=read_camera_requests(parent/'scene/sparse/0')
            trajroot=a.artifact_root/'results/frozen_scaffold_sem001/flythrough_v1/cameras'
            traj=read_camera_requests(trajroot);expected=json.loads((ROOT/'outputs/quality/backbone_comparison/standard/export/result.json').read_text())
            result['trajectory_metadata_sha256']={name:sha256(trajroot/name) for name in ['images.txt','cameras.txt']}
            assert result['trajectory_metadata_sha256']==expected['trajectory_metadata_sha256']
            assert sorted(cams)==sorted(json.loads((parent/'split.json').read_text())['rgb_train']) and len(traj)==144
            requests={**cams,**{n:replace(c,model='PINHOLE',radial_k=0.) for n,c in traj.items()}}
            ply=a.run/'model/point_cloud/iteration_30000/point_cloud.ply';model=GaussianModel(3);model.load_ply(str(ply))
            pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False,antialiasing=False);bg=torch.zeros(3,device='cuda')
            def rgb(m,c):return render(make_native_view(c),m,pipe,bg)['render']
            def quant(t):return np.rint(t.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            fixed=json.loads((ROOT/'experiments/runs/RGB-132-THREE-BACKBONE-DIAG/config.yaml').read_text())
            historical=GaussianModel(3)
            historical.load_ply(str(ROOT/'outputs/quality/rgb140/full-r2/model/point_cloud/iteration_30000/point_cloud.ply'))
            reference_check={}
            with torch.inference_mode():
                for name in fixed['views']:
                    fresh_reference=quant(rgb(historical,cams[name]))
                    saved_reference=np.asarray(Image.open(ROOT/'outputs/quality/rgb140/full-r2/evaluation/images'/name))
                    diff=np.abs(fresh_reference.astype(int)-saved_reference.astype(int))
                    reference_check[name]={'max_byte_difference':int(diff.max()),'changed_pixels':int((diff>0).any(2).sum())}
            result['historical_rgb140_fixed9_replay']=reference_check
            del historical
            torch.cuda.empty_cache()
            lp,_,device=build_lpips_model('cuda');rows=[];regions=[];hashes={};times=[];classes=['background','deck','stay_cable','tower','foundation'];classvalues={k:[] for k in classes}
            (out/'images').mkdir();torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                for index,(name,cam) in enumerate(sorted(requests.items())):
                    torch.cuda.synchronize();tick=time.monotonic();image=quant(rgb(model,cam));torch.cuda.synchronize();times.append(time.monotonic()-tick)
                    Image.fromarray(image).save(out/'images'/name);hashes[name]=sha256(out/'images'/name)
                    if name in cams:
                        photo=np.asarray(Image.open(parent/'scene/images'/name));ref=photo.astype(np.float32)/255;pred=image.astype(np.float32)/255
                        row={'name':name,'PSNR':psnr(ref,pred),'SSIM':ssim(ref,pred),'LPIPS':lpips_score(ref,pred,lp,torch,device)}
                        err=np.abs(image.astype(float)-photo).mean(2)
                        label=ROOT/'outputs/quality/all_data/SEM-370-r1/labels'/name
                        if label.exists():
                            gt=np.asarray(Image.open(label));row['class_rgb_mae_255']={k:float(err[gt==j].mean()) for j,k in enumerate(classes) if (gt==j).any()}
                            for k,v in row['class_rgb_mae_255'].items():classvalues[k].append(v)
                            if name in fixed['views']:
                                structure=gt!=0
                                masks={'deck':gt==1,'stay_cable':gt==2,
                                    'structure-label-boundary':binary_dilation(structure,iterations=2)&~binary_erosion(structure,iterations=2)}
                                for key,mask in masks.items():
                                    if mask.any():regions.append({'region':Path(name).stem+'-'+key,'view':name,
                                        'pixels':int(mask.sum()),'mae_255':float(err[mask].mean())})
                        rows.append(row)
                        for key,region in fixed['regions'].items():
                            if region['view']!=name:continue
                            mask=np.zeros(image.shape[:2],bool)
                            for x0,y0,x1,y1 in region['boxes']:mask[y0:y1,x0:x1]=True
                            regions.append({'region':key,'view':name,'pixels':int(mask.sum()),'mae_255':float(err[mask].mean())})
                    if index%50==0:print('evaluation',a.experiment,index+1,flush=True)
                fresh=GaussianModel(3);fresh.load_ply(str(ply))
                for name in fixed['views']:
                    assert np.array_equal(quant(rgb(fresh,cams[name])),np.asarray(Image.open(out/'images'/name)))
                # Same nine cameras, warm-up then three repeats, CUDA events only.
                for name in fixed['views']:rgb(model,cams[name])
                gpu_ms=[]
                for repeat in range(3):
                    for name in fixed['views']:
                        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                        begin.record();rgb(model,cams[name]);end.record();end.synchronize();gpu_ms.append(begin.elapsed_time(end))
            result.update(status='completed',metrics={k:float(np.mean([x[k] for x in rows])) for k in ['PSNR','SSIM','LPIPS']},
                per_view=rows,regions=regions,class_rgb_mae_255={k:float(np.mean(v)) for k,v in classvalues.items()},
                image_hashes=hashes,train_count=len(rows),trajectory_count=len(traj),fresh_fixed9_exact=True,
                render_median_ms=float(np.median(times[1:])*1000),fixed9_cuda_render_median_ms=float(np.median(gpu_ms)),
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                final_gaussian_count=len(model.get_xyz),model_mib=ply.stat().st_size/2**20)
    except BaseException as exc:result.update(status='failed_preserved',error=repr(exc));raise
    finally:
        result['seconds']=time.monotonic()-start;(out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
