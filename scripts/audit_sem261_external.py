#!/usr/bin/env python3
"""Replay official cameras and exercise the TRAINED refinement head without GT."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from run_sem261_point_refinement import load_parent, extract, infer_heads
from src.semantic_refinement import point_head, fuse_residual, mutable_lane
from src.semantic2d import read_json, write_json, sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('artifact-root','data-root','bundle','run-dir','reference-dir','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'experiments/runs/SEM-261-POINT-REFINEMENT-OOF/config.yaml')
    parser.add_argument('--checkpoint-name',default='fold0_fine_candidate.pth')
    args=parser.parse_args()
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    cfg=read_json(args.config)
    checkpoint=args.run_dir/args.checkpoint_name
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if payload['metadata']['config_sha256']!=sha256(args.config):
        raise ValueError('Head protocol changed')
    checkpoint_digest=sha256(checkpoint)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    rows=[]
    start=time.perf_counter()
    for size,camera_path in [('native',args.data_root/'camera_parameters/cameras.txt'),
                            ('small',args.reference_dir/'small_cameras.txt')]:
        rendered=out/size
        subprocess.run([sys.executable,str(ROOT/'scripts/render_submission_sem161.py'),
            '--bundle',str(args.bundle),'--cameras',str(camera_path),
            '--images',str(args.data_root/'camera_parameters/images.txt'),
            '--image-name','140.png','--output-dir',str(rendered),'--write-fallbacks'],
            env=env,cwd=ROOT,check=True)
        rgb,base,static=[np.asarray(Image.open(rendered/stage/'140.png')) for stage in ('rgb','semantic','SEM-001')]
        reference=args.reference_dir/('reload1' if size=='native' else 'small')
        if any(not np.array_equal(value,np.asarray(Image.open(reference/stage/'140.png')))
               for value,stage in ((rgb,'rgb'),(base,'semantic'))):
            raise ValueError('Frozen external replay mismatch')
        model,parent=load_parent(args.artifact_root,cfg,0,'deck')
        head=point_head().cuda().eval()
        head.load_state_dict(payload['state_dict'],strict=True)
        torch.cuda.reset_peak_memory_stats()
        rgb_hash=sha256(rendered/'rgb/140.png')
        fine,p=extract(model,rgb)
        zero=fuse_residual(base,static,np.zeros(base.shape,np.float32))
        new=infer_heads({'fine_candidate':head},fine,rgb,p,base,static,cfg)['fine_candidate']
        path=rendered/'refined_140.png'
        Image.fromarray(new).save(path)
        protected=~mutable_lane(base,static)
        record={'size':size,'shape':list(base.shape),'pose':'COLMAP_CAMERA_FROM_WORLD',
                'camera_model':'SIMPLE_RADIAL','rgb_frozen_replay_exact':True,
                'baseline_semantic_frozen_replay_exact':True,'zero_head_semantic_exact':bool(np.array_equal(zero,base)),
                'protected_exact':bool(np.array_equal(new[protected],base[protected])),
                'rgb_after_head_exact':sha256(rendered/'rgb/140.png')==rgb_hash,
                'changed_pixels':int((new!=base).sum()),'prediction_sha256':sha256(path),
                'rgb_sha256':rgb_hash,'trained_fold0_head_sha256':checkpoint_digest,
                'parent_checkpoint_sha256':parent['sha256'],
                'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
        if not all(record[k] for k in ('zero_head_semantic_exact','protected_exact','rgb_after_head_exact')):
            raise ValueError('External head gate failed')
        rows.append(record)
        del model,head,fine
        torch.cuda.empty_cache()
    result={'experiment_id':cfg['experiment_id'],'status':'completed_trained_branch_external_engineering',
            'scope':'fold0 engineering only; no final model, GT or hidden-set quality claim',
            'target_labels_read':[],'reference_rgb_inputs':[], 'seconds':time.perf_counter()-start,
            'checks':rows,'code_sha256':sha256(Path(__file__)),'checkpoint_sha256':checkpoint_digest}
    write_json(out/'result.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
