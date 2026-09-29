#!/usr/bin/env python3
"""Add a sealed SEM314 reviewer to a fresh official-camera SEM161 render."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
import numpy as np
from PIL import Image
import torch
from run_sem314_final_review import CONFIG,context,check_source_snapshot
from src.semantic2d import sha256,write_json
from src.semantic_foundation_review_final import EXP,load_final_head
from src.semantic_foundation_review import review_foundation
from src.submission_sem161 import sliding
from audit_sem260_error_budget import png


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("artifact-root","model-dir","train-dir","bundle","render-python","cameras","images","output-dir"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--image-name",default="140.png"); p.add_argument("--config",type=Path,default=CONFIG)
    args=p.parse_args(); args.output_dir=args.output_dir.resolve(); args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=False
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar(); started=time.perf_counter()
    try:
        cfg,_,_,_,_,_,_,_=context(args); check_source_snapshot(args.train_dir)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
        render=args.output_dir/"base"
        subprocess.run([str(args.render_python),str(ROOT/"scripts/render_submission_sem161.py"),"--bundle",str(args.bundle),
                        "--cameras",str(args.cameras),"--images",str(args.images),"--image-name",args.image_name,
                        "--output-dir",str(render)],cwd=ROOT,env=env,check=True)
        image=png(render/"rgb"/args.image_name); before=image.copy(); b=png(render/"semantic"/args.image_name)
        model,record=load_final_head(args.model_dir,args.train_dir,cfg,sha256(args.config))
        prob=sliding(model,image,crop=896,stride=448,classes=5)[4]; new=review_foundation(b,prob)
        if not np.array_equal(image,before) or not np.array_equal(review_foundation(b,prob,enabled=False),b):
            raise ValueError("External final RGB/bypass changed")
        if any(not np.array_equal(new==i,b==i) for i in (1,2,3)):
            raise ValueError("External protected class changed")
        path=args.output_dir/"semantic.png"; Image.fromarray(new).save(path); np.save(args.output_dir/"probability.npy",prob)
        write_json(args.output_dir/"result.json",{"experiment_id":EXP,"status":"external_review_passed",
                   "camera":args.image_name,"shape":list(b.shape),"pose":"COLMAP_CAMERA_FROM_WORLD",
                   "checkpoint_sha256":record["checkpoint_sha256"],"config_sha256":sha256(args.config),
                   "probability_sha256":sha256(args.output_dir/"probability.npy"),"prediction_sha256":sha256(path),
                   "rgb_sha256":sha256(render/"rgb"/args.image_name),"base_sha256":sha256(render/"semantic"/args.image_name),
                   "camera_file_sha256":sha256(args.cameras),"rgb_exact":True,"protected_exact":True,"disabled_exact":True,
                   "changed_pixels":int((new!=b).sum()),"target_labels_decoded":0,"seconds":time.perf_counter()-started})
        print({"status":"external_review_passed","shape":list(b.shape)},flush=True)
    except Exception as exc:
        write_json(args.output_dir/"failure.json",{"error_type":type(exc).__name__,"error":str(exc)})
        raise


if __name__=="__main__":main()
