#!/usr/bin/env python3
"""Check hook isolation and native gradient evidence on frozen RGB140."""
import argparse
import fcntl
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False,parents=True)
    with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        source=ROOT/'outputs/quality/rgb140/source-r1'
        sys.path[:0]=[str(source),str(source/'submodules/diff-gaussian-rasterization'),str(source/'submodules/simple-knn')]
        import torch
        import numpy as np
        from PIL import Image
        from types import SimpleNamespace
        from gaussian_renderer import GaussianModel,render
        from src.quality_cameras import read_camera_requests
        from src.quality_native_gaussian import make_native_view
        from src.standard_support_densification import install,support_weight
        from utils.loss_utils import l1_loss,ssim
        torch.set_num_threads(4)
        model=GaussianModel(3);model.load_ply(str(ROOT/'outputs/quality/rgb140/full-r2/model/point_cloud/iteration_30000/point_cloud.ply'))
        n=len(model.get_xyz);model.xyz_gradient_accum=torch.zeros((n,1),device='cuda');model.xyz_gradient_accum_abs=torch.zeros_like(model.xyz_gradient_accum);model.denom=torch.zeros_like(model.xyz_gradient_accum)
        camera=make_native_view(read_camera_requests(ROOT/'outputs/quality/all_data/RGB-120-r2/scene/sparse/0')['159.png'])
        camera.image_name='159'
        pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False,antialiasing=False);bg=torch.zeros(3,device='cuda')
        package=render(camera,model,pipe,bg)
        target=torch.tensor(np.asarray(Image.open(ROOT/'outputs/quality/all_data/RGB-120-r2/scene/images/159.png')).copy(),device='cuda').permute(2,0,1).float()/255
        loss=.8*l1_loss(package['render'],target)+.2*(1-ssim(package['render'],target));loss.backward()
        parameters={k:v for k,v in vars(model).items() if isinstance(v,torch.Tensor) and v.grad is not None}
        gradients={k:v.grad.clone() for k,v in parameters.items()}
        pixel=package['render'].detach().clone();screen=package['viewspace_points'].grad.clone()
        config=json.loads((ROOT/'experiments/runs/RGB-143-STANDARD-SUPPORT-DENSIFY/config.yaml').read_text())['support']
        observe=install(GaussianModel,config,a.output);observe(1,camera,loss,model)
        with torch.no_grad():
            model.add_densification_stats(package['viewspace_points'],package['visibility_filter'])
            again=render(camera,model,pipe,bg)['render']
        result={'status':'passed','forward_max_abs_diff':float((pixel-again).abs().max()),
                'parameter_gradient_max_abs_diff':{k:float((v.grad-gradients[k]).abs().max()) for k,v in parameters.items()},
                'screen_grad_unchanged':torch.equal(screen,package['viewspace_points'].grad),
                'absolute_finite':bool(torch.isfinite(screen[:,2:]).all()),'absolute_nonzero':int((screen[:,2:].norm(dim=1)>0).sum()),
                'weight_min':float(support_weight(model._mv_e,model._mv_r).min()),'weight_max':float(support_weight(model._mv_e,model._mv_r).max()),
                'neutral_unobserved':float(support_weight(torch.tensor(0.),torch.tensor(0.))),
                'historical_native_absolute_oracle':'outputs/quality/rgb140/preflight-r3/result.json'}
        assert result['forward_max_abs_diff']==0 and not any(result['parameter_gradient_max_abs_diff'].values()) and result['screen_grad_unchanged']
        assert result['absolute_finite'] and result['absolute_nonzero']>0 and .5<=result['weight_min']<=result['weight_max']<=1.5
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
