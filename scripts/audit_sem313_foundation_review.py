#!/usr/bin/env python3
"""Independent review reconstruction, actual parent heads, official camera probes."""
from __future__ import annotations
import argparse
from pathlib import Path
from types import SimpleNamespace
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
import numpy as np
from PIL import Image
import torch
from audit_sem311_frozen_dinov2 import audit as parent_audit, load_head
from run_sem313_foundation_review import CONFIG, checked, context, verify_build
from src.semantic2d import sha256, read_json, write_json
from src.semantic_foundation_review import review_foundation
from src.semantic_frozen_dinov2 import ARMS
from src.submission_sem161 import sliding
from audit_sem260_error_budget import png
from run_sem250_colmap_track_deck_diag import verify_freeze


def audit(args):
    cfg,parent,run,names,base,rgb,parent_pred,freeze=context(args)
    built,pred=verify_build(args); count=0
    for name in names:
        b=png(checked(*base[name]))
        for arm in ARMS:
            r=pred[arm][name]; inherited=parent_pred[arm][name]
            parent_png=png(checked(inherited["path"],inherited["sha256"]))
            expected=b.copy(); expected[(b==4)&(parent_png==0)]=0
            if not np.array_equal(expected,png(checked(r["path"],r["sha256"]))):
                raise ValueError("Independent review from inherited PNG differs")
            if r["fold"]!=inherited["fold"] or r["parent_probability_sha256"]!=inherited["probability_sha256"]:
                raise ValueError("Inherited provenance differs")
            count+=1
    # Existing independent audit verifies all parent probabilities/sources,
    # replays six heads, and freshly renders official COLMAP140 at both sizes.
    pargs=SimpleNamespace(**vars(args)); pargs.config=ROOT/cfg["parent_config"]; pargs.run_dir=args.parent_dir
    parent_out=args.output_dir/"parent_replay"; parent_out.mkdir()
    evidence=parent_audit(pargs,parent_out); write_json(parent_out/"result.json",evidence)
    probes=[]
    for size in ("native","small"):
        render=parent_out/size; image=png(render/"rgb/140.png"); b=png(render/"semantic/140.png")
        for arm in ARMS:
            model,record=load_head(pargs,parent,run,0,arm)
            before=image.copy(); probability=sliding(model,image,crop=896,stride=448,classes=5)[4]
            new=review_foundation(b,probability)
            expected=b.copy(); expected[(b==4)&(probability<.5)]=0
            if not np.array_equal(new,expected) or not np.array_equal(before,image):
                raise ValueError("Actual reviewer probe failed")
            if not np.array_equal(b,review_foundation(b,probability,enabled=False)):
                raise ValueError("Actual reviewer disabled equivalence failed")
            path=args.output_dir/f"{size}_{arm}_140.png"; Image.fromarray(new).save(path)
            probes.append({"size":size,"arm":arm,"camera":"140.png","shape":list(b.shape),
                           "pose":"COLMAP_CAMERA_FROM_WORLD","checkpoint_sha256":record["sha256"],
                           "rgb_exact":True,"protected_exact":True,"disabled_exact":True,"review_rule_exact":True,
                           "changed_foundation_pixels":int((new!=b).sum()),"prediction_sha256":sha256(path)})
            del model; torch.cuda.empty_cache()
    if verify_freeze(ROOT/cfg["freeze_manifest"],args.artifact_root)!=freeze:
        raise ValueError("Frozen production changed")
    return {"experiment_id":cfg["experiment_id"],"status":"passed_independent_review_engineering",
            "build_result_sha256":sha256(args.run_dir/"result.json"),"config_sha256":sha256(args.config),
            "predictions_checked":count,"native_replays":evidence["independent_native_replays"],
            "parent_audit_sha256":sha256(parent_out/"result.json"),"review_external_probes":probes,
            "target_labels_decoded":0,"frozen_artifacts_verified":freeze["count"],
            "audit_code_sha256":sha256(Path(__file__))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("artifact-root","parent-dir","run-dir","output-dir","model-dir","bundle","data-root","reference-dir","render-python"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--config",type=Path,default=CONFIG)
    args=p.parse_args(); args.output_dir=args.output_dir.resolve(); args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=False
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
    started=time.perf_counter()
    try:
        result=audit(args); result["seconds"]=time.perf_counter()-started
        write_json(args.output_dir/"result.json",result)
        print({"status":result["status"],"checked":result["predictions_checked"],"probes":len(result["review_external_probes"]),"seconds":result["seconds"]},flush=True)
    except Exception as exc:
        write_json(args.output_dir/"failure.json",{"error_type":type(exc).__name__,"error":str(exc),"seconds":time.perf_counter()-started})
        raise


if __name__=="__main__":
    main()
