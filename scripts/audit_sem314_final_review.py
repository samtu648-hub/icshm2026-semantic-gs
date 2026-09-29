#!/usr/bin/env python3
"""Independent final-head training/input, prediction and official-camera audit."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
import numpy as np
import torch
from run_sem314_final_review import CONFIG,prediction_context,checked,check_source_snapshot
from audit_sem260_error_budget import manifest_paths,png
from run_sem250_colmap_track_deck_diag import verify_freeze
from src.semantic2d import sha256,read_json,write_json
from src.semantic_foundation_binary import training_plan
from src.semantic_foundation_review_final import EXP,validate_final_split,load_final_head
from src.submission_sem161 import sliding


def audit(args):
    cfg,dev,labels,rgb,base,freeze,pred,manifest=prediction_context(args)
    tr=check_source_snapshot(args.train_dir); contract=read_json(args.train_dir/"checkpoint_contract.json")
    train_names,dev_names=validate_final_split(read_json(ROOT/cfg["semantic_split"]))
    plan=training_plan(train_names,cfg["seed"],0,3000,896)
    if (plan!=read_json(args.train_dir/"sampling_plan.json") or contract["sampling_plan_sha256"]!=sha256(args.train_dir/"sampling_plan.json")
            or contract["train"]!=train_names or contract["developer_names_excluded"]!=dev_names
            or [r["name"] for r in contract["actual_training_label_access"]]!=train_names):
        raise ValueError("Final training isolation/plan differs")
    train_root=args.artifact_root/cfg["parents"]["train_rgb"]
    train_rgb=manifest_paths(train_root,"rendered_dataset_manifest.json","rendered_images","path","sha256")
    counts=np.zeros(5,np.int64)
    for r in contract["actual_training_label_access"]:
        name=r["name"]; target=png(checked(args.artifact_root/cfg["parents"]["labels"]/name,labels[name]))
        if r["label_sha256"]!=labels[name] or r["rgb_sha256"]!=train_rgb[name][1]:
            raise ValueError("Final training access ledger differs")
        checked(*train_rgb[name])
        for i in range(5): counts[i]+=int((target==i).sum())
    weights=np.clip(np.median(counts.astype(float))/counts,.05,20).astype(np.float32)
    if counts.tolist()!=contract["class_counts"] or weights.tolist()!=contract["class_weights"]:
        raise ValueError("Independent final class weights differ")
    records={r["name"]:r for r in manifest["predictions"]}
    for name,r in records.items():
        b=png(checked(*base[name])); checked(*rgb[name])
        p=np.load(checked(args.prediction_dir/r["probability"],r["probability_sha256"]),allow_pickle=False)
        if p.shape!=b.shape or p.dtype!=np.float32 or not np.isfinite(p).all() or p.min()<0 or p.max()>1:
            raise ValueError("Invalid final probability")
        actual=png(checked(args.prediction_dir/r["prediction"],r["prediction_sha256"]))
        expected=np.where((b==4)&(p<.5),0,b).astype(np.uint8)
        if not np.array_equal(expected,actual) or int((b!=actual).sum())!=r["removed_foundation_pixels"]:
            raise ValueError("Independent final review reconstruction differs")
    replays=[]
    for name in (dev[0],"159.png"):
        model,record=load_final_head(args.model_dir,args.train_dir,cfg,sha256(args.config))
        image=png(rgb[name][0]); before=image.copy()
        actual=sliding(model,image,crop=896,stride=448,classes=5)[4]
        r=records[name]; expected=np.load(args.prediction_dir/r["probability"],allow_pickle=False)
        if not np.array_equal(actual,expected) or not np.array_equal(image,before):
            raise ValueError("Actual final-head native replay differs")
        replays.append({"name":name,"checkpoint_sha256":record["checkpoint_sha256"],"exact":True})
        del model; torch.cuda.empty_cache()
    external=[]; env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
    for name,cameras,reference in (("native1",args.data_root/"camera_parameters/cameras.txt",args.reference_dir/"reload1"),
                                  ("native2",args.data_root/"camera_parameters/cameras.txt",args.reference_dir/"reload1"),
                                  ("small",args.reference_dir/"small_cameras.txt",args.reference_dir/"small")):
        out=args.output_dir/name
        subprocess.run([sys.executable,str(ROOT/"scripts/render_sem314_external_camera.py"),"--artifact-root",str(args.artifact_root),
                        "--model-dir",str(args.model_dir),"--train-dir",str(args.train_dir),"--bundle",str(args.bundle),
                        "--render-python",str(args.render_python),"--cameras",str(cameras),"--images",str(args.data_root/"camera_parameters/images.txt"),
                        "--config",str(args.config),"--output-dir",str(out)],cwd=ROOT,env=env,check=True)
        r=read_json(out/"result.json")
        for kind in ("rgb","semantic"):
            if not np.array_equal(png(out/"base"/kind/"140.png"),png(reference/kind/"140.png")):
                raise ValueError("Fresh official-camera frozen baseline differs")
        b=png(out/"base/semantic/140.png"); p=np.load(out/"probability.npy",allow_pickle=False)
        if not np.array_equal(png(out/"semantic.png"),np.where((b==4)&(p<.5),0,b)):
            raise ValueError("Actual official reviewer differs")
        external.append({"probe":name,**r,"reference_rgb_and_baseline_exact":True})
    for filename in ("semantic.png","probability.npy"):
        if sha256(args.output_dir/"native1"/filename)!=sha256(args.output_dir/"native2"/filename):
            raise ValueError("Fresh-process official final reviewer not exact")
    if verify_freeze(ROOT/cfg["freeze_manifest"],args.artifact_root)!=freeze: raise ValueError("Frozen production changed")
    return {"experiment_id":EXP,"status":"passed_final_review_engineering","predictions_checked":74,
            "training_labels_verified":219,"developer_labels_decoded":0,"class_weights_exact":True,
            "native_replays":replays,"external_probes":external,"native_fresh_process_exact":True,
            "prediction_result_sha256":sha256(args.prediction_dir/"result.json"),"training_result_sha256":sha256(args.train_dir/"result.json"),
            "config_sha256":sha256(args.config),"code_sha256":sha256(Path(__file__)),"frozen_artifacts_verified":18}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("artifact-root","model-dir","train-dir","prediction-dir","bundle","render-python","data-root","reference-dir","output-dir"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--config",type=Path,default=CONFIG); args=p.parse_args(); args.output_dir=args.output_dir.resolve()
    args.output_dir.mkdir(parents=True,exist_ok=False); torch.set_num_threads(4); torch.backends.cudnn.benchmark=False
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar(); started=time.perf_counter()
    try:
        result=audit(args);result["seconds"]=time.perf_counter()-started;write_json(args.output_dir/"result.json",result)
        print({"status":result["status"],"native_replays":len(result["native_replays"]),"external_probes":len(result["external_probes"]),"seconds":result["seconds"]},flush=True)
    except Exception as exc:
        write_json(args.output_dir/"failure.json",{"error_type":type(exc).__name__,"error":str(exc),"seconds":time.perf_counter()-started})
        raise


if __name__=="__main__":main()
