#!/usr/bin/env python3
"""RGB141: frozen RGB140 contribution, residual and marginal-support audit."""
import argparse
import fcntl
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(exist_ok=False,parents=True)
    result={'status':'started','experiment_id':'RGB-141-STANDARD-SUPPORT-DIAG'}
    started=time.monotonic()
    try:
        with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            source=ROOT/'outputs/quality/rgb140/source-r1'
            sys.path[:0]=[str(source),str(source/'submodules/diff-gaussian-rasterization'),str(source/'submodules/simple-knn')]
            import numpy as np
            import torch
            from PIL import Image
            from scipy.ndimage import binary_dilation,binary_erosion
            from types import SimpleNamespace
            from gaussian_renderer import render,GaussianModel
            from utils.loss_utils import l1_loss,ssim
            from src.quality_cameras import read_camera_requests
            from src.quality_native_gaussian import make_native_view
            from src.standard_support_densification import support_weight
            torch.set_num_threads(4);torch.manual_seed(0)
            scene=ROOT/'outputs/quality/all_data/RGB-120-r2/scene'
            ply=ROOT/'outputs/quality/rgb140/full-r2/model/point_cloud/iteration_30000/point_cloud.ply'
            model=GaussianModel(3);model.load_ply(str(ply))
            for v in vars(model).values():
                if isinstance(v,torch.Tensor): v.requires_grad_(False)
            model._opacity.requires_grad_(True)
            old=model._opacity.detach().clone();pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False,antialiasing=False)
            bg=torch.zeros(3,device='cuda');cams=read_camera_requests(scene/'sparse/0')
            def pred(n):return render(make_native_view(cams[n]),model,pipe,bg)['render']
            def target(n):return torch.tensor(np.asarray(Image.open(scene/'images'/n)).copy(),device='cuda').permute(2,0,1).float()/255
            fixed=json.loads((ROOT/'experiments/runs/RGB-132-THREE-BACKBONE-DIAG/config.yaml').read_text())
            regions={}
            for key in ['159-shore','159-far-occlusion','159-water-ghost']:
                mask=np.zeros((989,1320),bool)
                for x0,y0,x1,y1 in fixed['regions'][key]['boxes']:mask[y0:y1,x0:x1]=True
                regions[key]=mask
            labels=np.asarray(Image.open(ROOT/'outputs/quality/all_data/SEM-370-r1/labels/159.png'))
            regions['159-stay_cable']=labels==2
            structure=labels!=0
            regions['159-structure-boundary']=binary_dilation(structure,iterations=2)&~binary_erosion(structure,iterations=2)
            regions={k:torch.tensor(v,device='cuda') for k,v in regions.items()}
            selection=[];ids=set()
            for key,mask in regions.items():
                im=pred('159.png');loss=(im-target('159.png')).abs()[:,mask].mean()
                g=torch.autograd.grad(loss,model._opacity)[0].flatten()
                chosen=torch.argsort(g.abs(),descending=True,stable=True)[:2].tolist();ids.update(chosen)
                selection.append({'region':key,'mae_255':float(loss)*255,'ids':chosen,'opacity_derivatives':[float(g[i]) for i in chosen]})
            ids=sorted(ids);(a.output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
            e=torch.zeros(len(old),device='cuda');r=torch.zeros_like(e);rows=[];positive=torch.zeros_like(e);negative=torch.zeros_like(e)
            before=sha256(ply)
            for vi,n in enumerate(sorted(cams)):
                im=pred(n);gt=target(n);loss=.8*l1_loss(im,gt)+.2*(1-ssim(im,gt))
                g=torch.autograd.grad(loss,model._opacity)[0].flatten();mag=g.abs();tau=mag[mag>0].median();den=mag+tau
                e.add_(-g/den);r.add_(mag/den);positive.add_((g<0).float());negative.add_((g>0).float())
                base=im.detach();baseerr=(base-gt).abs().mean()
                for i in ids:
                    with torch.no_grad():
                        model._opacity[i]=-100;abl=pred(n);model._opacity[i]=old[i]
                    delta=abl-base
                    rows.append({'view':n,'id':i,'visible_changed_pixels':int((delta.abs().max(0).values>1e-6).sum()),
                        'opacity_gradient':float(g[i]),'normalized_evidence':float(-g[i]/den[i]),
                        'removal_mae_delta_255':float((abl-gt).abs().mean()-baseerr)*255})
                if vi%50==0:print('diagnostic views',vi+1,flush=True)
            weights=support_weight(e,r)
            result.update(status='completed_readonly',selection=selection,rows=rows,
                summary=[{'id':i,'weight':float(weights[i]),'exposure':float(r[i]),'signed_evidence':float(e[i]),
                    'helpful_gradient_views':int(positive[i]),'harmful_gradient_views':int(negative[i]),
                    'contribution_views':sum(x['visible_changed_pixels']>0 for x in rows if x['id']==i),
                    'removal_better_views':sum(x['removal_mae_delta_255']<0 for x in rows if x['id']==i),
                    'removal_worse_views':sum(x['removal_mae_delta_255']>0 for x in rows if x['id']==i)} for i in ids],
                parent_ply_sha256=before,parent_unchanged=sha256(ply)==before and torch.equal(model._opacity,old),
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                scope='all400 frozen training-fit marginal evidence; labels used for diagnostic masks only; no geometry ground truth')
            np.savez_compressed(a.output/'support_arrays.npz',signed=e.cpu().numpy(),exposure=r.cpu().numpy(),weight=weights.cpu().numpy())
    except BaseException as exc:
        result.update(status='failed_preserved',error=repr(exc));raise
    finally:
        result.update(seconds=time.monotonic()-started,code_sha256=sha256(Path(__file__)))
        (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
