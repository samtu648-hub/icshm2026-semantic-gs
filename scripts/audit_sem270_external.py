#!/usr/bin/env python3
"""Exercise trained foundation classifiers on official native and small cameras."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_sem270_foundation_binary_oof import initialize, array_hash
from src.semantic_foundation_binary import fuse_foundation
from src.semantic2d import read_json, write_json, sha256
from src.submission_sem161 import sliding


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('artifact-root','data-root','bundle','run-dir','reference-dir','output-dir'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'experiments/runs/SEM-270-FOUNDATION-BINARY-OOF/config.yaml')
    args=p.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error();hf_logging.disable_progress_bar()
    cfg=read_json(args.config);rows=[];start=time.perf_counter()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    for size,camera in [('native',args.data_root/'camera_parameters/cameras.txt'),('small',args.reference_dir/'small_cameras.txt')]:
        rendered=out/size
        subprocess.run([sys.executable,str(ROOT/'scripts/render_submission_sem161.py'),'--bundle',str(args.bundle),
            '--cameras',str(camera),'--images',str(args.data_root/'camera_parameters/images.txt'),
            '--image-name','140.png','--output-dir',str(rendered)],env=env,cwd=ROOT,check=True)
        rgb,base=[np.asarray(Image.open(rendered/stage/'140.png')) for stage in ('rgb','semantic')]
        ref=args.reference_dir/('reload1' if size=='native' else 'small')
        if any(not np.array_equal(a,np.asarray(Image.open(ref/stage/'140.png'))) for a,stage in ((rgb,'rgb'),(base,'semantic'))): raise ValueError('Frozen replay differs')
        for arm in cfg['arm_names']:
            path=args.run_dir/f'fold0_{arm}'/'model.pth';payload=torch.load(path,map_location='cpu',weights_only=True)
            meta=payload['metadata'];record=read_json(args.run_dir/f'fold0_{arm}'/'training_result.json')
            if meta['config_sha256']!=sha256(args.config) or meta['fold']!=0 or meta['arm']!=arm or sha256(path)!=record['sha256']: raise ValueError('Wrong trained branch')
            model=initialize(args.artifact_root/cfg['parents']['model_config'],arm,cfg['seed']).cuda().eval().requires_grad_(False)
            model.load_state_dict(payload['model_state_dict'],strict=True);before=array_hash(rgb)
            torch.cuda.reset_peak_memory_stats();classes=5 if arm=='five_control' else 2
            prob=sliding(model,rgb,crop=960,stride=480,classes=classes)[4 if classes==5 else 1]
            zero=fuse_foundation(base,prob,strength=0);new=fuse_foundation(base,prob)
            target=rendered/f'{arm}_140.png';Image.fromarray(new).save(target)
            checks={'rgb_frozen_replay_exact':True,'baseline_frozen_replay_exact':True,
                'disabled_exact':bool(np.array_equal(zero,base)),
                'protected_masks_exact':all(np.array_equal(base==k,new==k) for k in (1,2,3)),
                'rgb_input_unchanged':array_hash(rgb)==before}
            if not all(checks.values()): raise ValueError('External trained branch gate failed')
            rows.append({'size':size,'arm':arm,'shape':list(base.shape),'camera':'140.png',
                'pose':'COLMAP_CAMERA_FROM_WORLD','camera_model':'SIMPLE_RADIAL','checks':checks,
                'changed_pixels':int((new!=base).sum()),'added_foundation':int(((base==0)&(new==4)).sum()),
                'removed_foundation':int(((base==4)&(new==0)).sum()),'checkpoint_sha256':sha256(path),
                'prediction_sha256':sha256(target),'rgb_sha256':sha256(rendered/'rgb/140.png'),
                'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/2**20})
            del model,payload;torch.cuda.empty_cache()
    result={'experiment_id':cfg['experiment_id'],'status':'passed','scope':'Trained fold0 two-arm engineering, one official camera at two sizes; no hidden/final quality claim',
        'target_labels_read':[],'reference_rgb_inputs':[],'checks':rows,'seconds':time.perf_counter()-start,'code_sha256':sha256(Path(__file__))}
    write_json(out/'result.json',result)
    print({'status':'passed','probes':len(rows),'seconds':result['seconds']},flush=True)


if __name__=='__main__': main()
