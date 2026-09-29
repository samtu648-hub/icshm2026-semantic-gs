#!/usr/bin/env python3
"""GPU checks for native forward/gradients and narrowly scoped split placement."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        source=ROOT/'outputs/quality/rgb140/source-r1';sys.path[:0]=[str(source),str(source/'submodules/diff-gaussian-rasterization'),str(source/'submodules/simple-knn')]
        import torch
        import numpy as np
        from PIL import Image
        from types import SimpleNamespace
        from gaussian_renderer import GaussianModel,render
        from arguments import OptimizationParams
        from src.quality_cameras import read_camera_requests
        from src.quality_native_gaussian import make_native_view
        from src.standard_geometry_split import install,associate
        from src.photo_geometry import project
        from utils.loss_utils import l1_loss,ssim
        torch.set_num_threads(4);torch.manual_seed(0)
        config=json.loads((ROOT/'experiments/runs/RGB-146-STANDARD-GEOMETRY-SPLIT/config.yaml').read_text())['geometry']
        original_split=GaussianModel.densify_and_split;original_postfix=GaussianModel.densification_postfix
        def reset():GaussianModel.densify_and_split=original_split;GaussianModel.densification_postfix=original_postfix
        # Frozen real forward and gradients; hook does not touch forward/stats.
        model=GaussianModel(3);model.load_ply(str(ROOT/'outputs/quality/rgb140/full-r2/model/point_cloud/iteration_30000/point_cloud.ply'))
        n=len(model.get_xyz)
        model.xyz_gradient_accum=torch.zeros((n,1),device='cuda');model.xyz_gradient_accum_abs=torch.zeros_like(model.xyz_gradient_accum);model.denom=torch.zeros_like(model.xyz_gradient_accum)
        camera=make_native_view(read_camera_requests(ROOT/'outputs/quality/all_data/RGB-120-r2/scene/sparse/0')['159.png']);camera.image_name='159'
        pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False,antialiasing=False);bg=torch.zeros(3,device='cuda')
        gt=torch.tensor(np.asarray(Image.open(ROOT/'outputs/quality/all_data/RGB-120-r2/scene/images/159.png')).copy(),device='cuda').permute(2,0,1).float()/255
        def forward():
            pkg=render(camera,model,pipe,bg);loss=.8*l1_loss(pkg['render'],gt)+.2*(1-ssim(pkg['render'],gt));loss.backward()
            return pkg,loss
        pkg,loss=forward();pixel=pkg['render'].detach().clone();screen=pkg['viewspace_points'].grad.clone()
        params=[model._xyz,model._features_dc,model._features_rest,model._opacity,model._scaling,model._rotation]
        gradients=[v.grad.clone() for v in params]
        model.add_densification_stats(pkg['viewspace_points'],pkg['visibility_filter'])
        stats=[v.clone() for v in (model.xyz_gradient_accum,model.xyz_gradient_accum_abs,model.denom)]
        observe,state=install(GaussianModel,config,[],{},a.output)
        observe(1,camera,loss,model)
        diffs=[float((v.grad-g).abs().max()) for v,g in zip(params,gradients)]
        assert not any(diffs) and torch.equal(screen,pkg['viewspace_points'].grad)
        for v in (model.xyz_gradient_accum,model.xyz_gradient_accum_abs,model.denom):v.zero_()
        model.add_densification_stats(pkg['viewspace_points'],pkg['visibility_filter'])
        with torch.no_grad():again=render(camera,model,pipe,bg)['render']
        assert torch.equal(pixel,again)
        for s,v in zip(stats,(model.xyz_gradient_accum,model.xyz_gradient_accum_abs,model.denom)):assert torch.equal(s,v)
        assert torch.isfinite(screen).all() and (screen[:,2:].abs()>0).any()
        del model,pkg,again,params,gradients,stats;torch.cuda.empty_cache();reset()
        parser=argparse.ArgumentParser();op=OptimizationParams(parser);opt=op.extract(parser.parse_args([]))
        fields=['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity']
        def toy():
            m=GaussianModel(3);nn=torch.nn.Parameter
            m._xyz=nn(torch.tensor([[i*5.,0.,10.] for i in range(8)],device='cuda'))
            m._features_dc=nn(torch.arange(24,device='cuda',dtype=torch.float).reshape(8,1,3)/24)
            m._features_rest=nn(torch.zeros(8,15,3,device='cuda'));m._scaling=nn(torch.zeros(8,3,device='cuda'))
            m._rotation=nn(torch.tensor([[1.,0,0,0]]*8,device='cuda'));m._opacity=nn(torch.zeros(8,1,device='cuda'))
            m._exposure=nn(torch.eye(3,4,device='cuda')[None]);m.pretrained_exposures=None;m.spatial_lr_scale=1.;m.training_setup(opt)
            m.max_radii2D=torch.zeros(8,device='cuda');m.tmp_radii=torch.zeros(8,device='cuda')
            # Exercise initialized optimizer moments, not only empty Adam state.
            sum(getattr(m,k).sum() for k in fields).backward();m.optimizer.step();m.optimizer.zero_grad(set_to_none=True)
            return m
        cameras={str(i):SimpleNamespace(matrix=np.array([[1000.,0,660],[0,1000.,494.5],[0,0,1.]]),pose=SimpleNamespace(rotation=np.eye(3),translation=np.array([-x,0.,0.]))) for i,x in enumerate([-1.,0.,1.,2.])}
        proposals=[]
        for i in range(8):
            xyz=np.array([i*5.+.1,0.,10.]);proposals.append({'id':i,'xyz':xyz.tolist(),'confidence':.9,'observations':[{'camera':n,'xy':project(c,xyz)[0].tolist()} for n,c in cameras.items()]})
        outputs={};rngs={};optimizer={}
        for arm in ['native','zero','control','candidate']:
            reset();folder=a.output/arm;folder.mkdir();m=toy()
            if arm!='native':
                obs,st=install(GaussianModel,config|{'arm':'control' if arm in ['zero','control'] else 'candidate'},[] if arm=='zero' else proposals,cameras,folder);st['iteration']=14900
            torch.manual_seed(123);m.densify_and_split(torch.ones(8,1,device='cuda'),.0004,1.)
            outputs[arm]={k:getattr(m,k).detach().clone() for k in fields};rngs[arm]=torch.cuda.get_rng_state().clone();optimizer[arm]=m.optimizer.state_dict()
        for arm in ['zero','control']:
            for k in fields:assert torch.equal(outputs['native'][k],outputs[arm][k]),(arm,k)
        for arm in rngs:assert torch.equal(rngs['native'],rngs[arm])
        for k in fields[1:]:assert torch.equal(outputs['control'][k],outputs['candidate'][k]),k
        changed=(outputs['control']['_xyz']!=outputs['candidate']['_xyz']).any(1).nonzero().flatten().tolist()
        assert len(changed)==2 and max(changed)<8,changed
        for k,value in optimizer['control']['state'].items():
            for name,t in value.items():assert torch.equal(t,optimizer['candidate']['state'][k][name])
        chosen,_=associate(np.array([[.1,0.,10.]]*8),np.ones((8,3)),np.tile(np.eye(3),(8,1,1)),proposals,cameras,set(range(8)))
        assert not chosen
        result={'status':'passed','native_forward_exact':True,'parameter_gradient_max_abs_diff':diffs,'gradient_check':'exact preservation across hook; stats replay identical native screen gradient, no second stochastic CUDA reduction',
                'absolute_stats_match':True,'native_zero_and_control_topology_exact':True,'candidate_only_two_first_block_child_xyz_changed':changed,
                'all_other_attributes_optimizer_states_and_cuda_rng_exact':True,'used_proposals_not_reused':True}
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));reset()


if __name__=='__main__':main()
