#!/usr/bin/env python3
"""Native RGB140 process with local placement hook and complete resume checkpoints."""
import argparse
import json
import os
from pathlib import Path
import random
import runpy
import sys
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-state',type=Path);p.add_argument('worker_args',nargs=argparse.REMAINDER);a=p.parse_args()
    c=json.loads(a.config.read_text());source=ROOT/c['source'];sys.path[:0]=[str(source),str(source/'submodules/diff-gaussian-rasterization'),str(source/'submodules/simple-knn'),str(ROOT)]
    import numpy as np
    import torch
    from scene import GaussianModel
    from src.standard_geometry_split import install
    from src.quality_cameras import read_camera_requests
    from src.quality_protocol import sha256
    cache=ROOT/c['geometry']['cache']; proposals=json.loads(cache.read_text())['proposals']
    observe,state=install(GaussianModel,c['geometry'],proposals,read_camera_requests(ROOT/c['data']['scene']/'sparse/0'),a.output)
    binding={'config_sha256':sha256(a.config),'cache_sha256':sha256(cache),
             'hook_sha256':sha256(ROOT/'src/standard_geometry_split.py'),'worker_sha256':sha256(Path(__file__))}

    def resume(model,opt,scene):
        if not a.resume_state:return 0,scene.getTrainCameras().copy(),list(range(len(scene.getTrainCameras()))),0.,0.
        q=torch.load(a.resume_state,weights_only=False)
        if q['binding']!=binding:raise ValueError('Resume scientific binding mismatch')
        model._exposure.data.copy_(q['exposure']);model.restore(q['model'],opt)
        model.exposure_optimizer.load_state_dict(q['exposure_optimizer'])
        names={cam.image_name:cam for cam in scene.getTrainCameras()}
        stack=[names[n] for n in q['viewpoint_stack']]
        random.setstate(q['python_rng']);np.random.set_state(q['numpy_rng']);torch.set_rng_state(q['torch_rng']);torch.cuda.set_rng_state_all(q['cuda_rng'])
        state.update(q['geometry_state']);state['used']=set(state['used']);state['pending']=None
        return q['iteration'],stack,q['viewpoint_indices'],q['ema_loss'],q['ema_depth']

    def checkpoint(iteration,model,stack,indices,ema_loss,ema_depth,total):
        if iteration%1000 and iteration!=total:return
        assert state['pending'] is None
        q={'binding':binding,'iteration':iteration,'model':model.capture(),'exposure':model._exposure.detach(),
           'exposure_optimizer':model.exposure_optimizer.state_dict(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
           'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'viewpoint_stack':[cam.image_name for cam in stack],
           'viewpoint_indices':indices,'ema_loss':ema_loss,'ema_depth':ema_depth,'geometry_state':state|{'used':sorted(state['used'])}}
        # Retain periodic states; never discard interruption evidence. Atomic rename prevents partial resume.
        folder=a.output/'recovery';folder.mkdir(exist_ok=True)
        dest=folder/f'state-{iteration:05d}.pth';tmp=dest.with_suffix('.tmp');torch.save(q,tmp);os.replace(tmp,dest)
        (folder/'latest.json').write_text(json.dumps({'path':str(dest),'iteration':iteration,'binding':binding})+'\n')

    original=runpy.run_path
    def patched(path,*args,**kwargs):
        if Path(path).resolve()!=(source/'train.py').resolve():return original(path,*args,**kwargs)
        text=Path(path).read_text()
        replacements={
            '        loss.backward()\n':'        loss.backward()\n        geometry_observe(iteration, viewpoint_cam, loss, gaussians)\n',
            '    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")':
                '    first_iter, viewpoint_stack, viewpoint_indices, ema_loss_for_log, ema_Ll1depth_for_log = geometry_resume(gaussians, opt, scene)\n    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")',
            '            if (iteration in checkpoint_iterations):':
                '            geometry_checkpoint(iteration, gaussians, viewpoint_stack, viewpoint_indices, ema_loss_for_log, ema_Ll1depth_for_log, opt.iterations)\n\n            if (iteration in checkpoint_iterations):'}
        for needle,replacement in replacements.items():
            if text.count(needle)!=1:raise RuntimeError('Native hook location changed')
            text=text.replace(needle,replacement)
        (a.output/'executed_train.py').write_text(text)
        scope=dict(__name__='__main__',__file__=str(path),geometry_observe=observe,geometry_resume=resume,geometry_checkpoint=checkpoint)
        exec(compile(text,str(path),'exec'),scope);return scope
    runpy.run_path=patched
    args=a.worker_args[1:] if a.worker_args[:1]==['--'] else a.worker_args
    sys.argv=[str(ROOT/'scripts/backbone_all400_worker.py'),*args];original(sys.argv[0],run_name='__main__')


if __name__=='__main__':main()
