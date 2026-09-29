#!/usr/bin/env python3
"""Fresh SEM326 renderer/matcher plus unchanged SEM328 veto at official raw pixels."""
from copy import copy
from dataclasses import replace
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
import numpy as np
from PIL import Image
from src.semantic2d import read_json,write_json,sha256,label_array
from src.semantic_second_source_veto import second_source,review_second_source
from src.semantic_source_access import read_pose_only
from src.geometry_prior.camera_io import read_cameras_txt
import render_sem326_external as legacy
from run_sem329_final import CONFIG,EXP,setup
from run_sem320_dense_preflight import checked
from run_sem250_colmap_track_deck_diag import verify_freeze


def external_label_array(path):
    # External output dimensions come from incoming camera metadata, unlike native training labels.
    with Image.open(path) as image:
        array=np.asarray(image).copy()
    if array.ndim!=2 or not np.issubdtype(array.dtype,np.integer) or array.min()<0 or array.max()>4:
        raise ValueError("External semantic output must be a single-channel0-4 ID map")
    return array


def probe(args):
    d=setup(args);cfg=d["cfg"];c=d["c"]
    old=copy(args);old.config=ROOT/cfg["parent_config"];old.prepare_dir=ROOT/cfg["matching_root"]/"prepare"
    old.output_dir=args.output_dir/"dense_stage";old.output_dir.mkdir(parents=True,exist_ok=False)
    original=legacy.probe(old)
    write_json(old.output_dir/"result.json",original)
    good=[r["source"] for r in original["pairs"] if r["checks"]["coverage"] and r["checks"]["epipolar_retention"]]
    secondary=second_source({"accessible_sources":good,"selected_source":original["selected_source"]})
    base_path=old.output_dir/"base/semantic"/args.image_name;dense_path=old.output_dir/"semantic.png"
    rgb_path=old.output_dir/"base/rgb"/args.image_name
    baseline=external_label_array(base_path);dense=external_label_array(dense_path)
    seal={"config_sha256":sha256(args.config),"wrapper_sha256":sha256(Path(__file__)),
          "veto_core_sha256":sha256(ROOT/cfg["veto_core"]),"original_result_sha256":sha256(old.output_dir/"result.json"),
          "second_source":secondary,"second_source_label_sha256":sha256(c["source_paths"][secondary]) if secondary else None,
          "rgb_sha256":sha256(rgb_path),"baseline_sha256":sha256(base_path),"dense_sha256":sha256(dense_path),
          "camera_sha256":sha256(args.cameras),"pose_sha256":sha256(args.images)}
    write_json(args.output_dir/"access_seal.json",seal)
    kwargs={"target_is_distorted":True}
    if secondary:
        selected=next(r for r in original["pairs"] if r["source"]==secondary);maps=[]
        for record in selected["maps"]:
            with np.load(checked(old.output_dir/record["path"],record["sha256"]),allow_pickle=False) as data:
                maps.append((data["warp"].astype(np.float64),data["overlap"].astype(np.float64)))
        a=read_pose_only(args.images)[args.image_name]
        ca=replace(read_cameras_txt(args.cameras)[a.camera_id],camera_id=-1);a=replace(a,camera_id=-1)
        b=c["poses"][secondary];cb=c["cameras"][b.camera_id]
        kwargs.update(source_labels=label_array(c["source_paths"][secondary]),maps=maps,target_pose=a,source_pose=b,
                      target_camera=ca,source_camera=cb,filters=c["filters"])
    candidate,stats=review_second_source(baseline,dense,**kwargs)
    independently,other=review_second_source(baseline,dense,**kwargs,independent=True)
    if not np.array_equal(candidate,independently) or stats!=other:raise ValueError("Independent official veto differs")
    bypass,_=review_second_source(baseline,dense,enabled=False)
    if not np.array_equal(bypass,dense):raise ValueError("Disabled official veto differs")
    if any(not np.array_equal(candidate==k,baseline==k) for k in (1,2,3)):raise ValueError("Protected output differs")
    path=args.output_dir/"semantic.png";Image.fromarray(candidate).save(path)
    for path0,key in ((rgb_path,"rgb_sha256"),(base_path,"baseline_sha256"),(dense_path,"dense_sha256")):checked(path0,seal[key])
    if verify_freeze(ROOT/c["cfg"]["freeze_manifest"],args.artifact_root)!=c["frozen"]:raise ValueError("Freeze differs")
    return {"experiment_id":EXP,"status":"passed_actual_external_second_source","config_sha256":sha256(args.config),
            "shape":list(candidate.shape),"camera":args.image_name,"selected_source":original["selected_source"],
            "second_source":secondary,"native":stats,"prediction_sha256":sha256(path),
            "rgb_sha256":seal["rgb_sha256"],"baseline_sha256":seal["baseline_sha256"],"dense_sha256":seal["dense_sha256"],
            "pinhole_rgb_sha256":original["pinhole_rgb_sha256"],"original_result_sha256":seal["original_result_sha256"],
            "access_seal_sha256":sha256(args.output_dir/"access_seal.json"),"independent_native_exact":True,
            "protected_masks_exact":True,"rgb_exact":True,"disabled_exact":True,"target_label_decodes":0,
            "fresh_renderer_and_matcher":True,"peak_cuda_allocated_mib":original["peak_cuda_allocated_mib"]}


def run_all(args):
    cfg=read_json(args.config);parent=read_json(ROOT/cfg["parent_config"])
    previous_path=checked(ROOT/cfg["parent_external"],cfg["parent_external_sha256"])
    previous=read_json(previous_path);previous_rows={r["probe"]:r for r in previous["probes"]}
    probe_cfg=parent["external_probes"]
    a=read_pose_only(args.data_root/"camera_parameters/images.txt")[probe_cfg["reference_image"]]
    novel=replace(a,name=probe_cfg["novel_target"],tvec=a.tvec+np.asarray(probe_cfg["novel_translation_w2c_delta"]))
    pose_file=args.output_dir/"novel_images.txt"
    values=[str(novel.image_id),*map(str,novel.qvec),*map(str,novel.tvec),str(novel.camera_id),novel.name]
    pose_file.write_text(" ".join(values)+"\n\n")
    rows=[]
    for name,cameras,images,target in (
        ("native1",args.data_root/"camera_parameters/cameras.txt",args.data_root/"camera_parameters/images.txt","140.png"),
        ("native2",args.data_root/"camera_parameters/cameras.txt",args.data_root/"camera_parameters/images.txt","140.png"),
        ("small",args.reference_dir/"small_cameras.txt",args.data_root/"camera_parameters/images.txt","140.png"),
        ("novel",args.data_root/"camera_parameters/cameras.txt",pose_file,novel.name)):
        out=args.output_dir/name
        command=[sys.executable,str(Path(__file__).resolve()),"probe","--cameras",str(cameras),"--images",str(images),
                 "--image-name",target,"--output-dir",str(out)]
        for key in ("config","artifact_root","data_root","roma_dir","dinov3_dir","weights","bundle","render_python"):
            command += ["--"+key.replace("_","-"),str(getattr(args,key))]
        subprocess.run(command,cwd=ROOT,check=True)
        row=read_json(out/"result.json");old=previous_rows[name]
        checked(previous_path.parent/name/"result.json",old["result_sha256"])
        for newkey,oldkey in (("rgb_sha256","rgb_sha256"),("baseline_sha256","baseline_sha256"),
                              ("dense_sha256","prediction_sha256"),("pinhole_rgb_sha256","pinhole_rgb_sha256")):
            if row[newkey]!=old[oldkey]:raise ValueError("Frozen external RGB/baseline/dense reference differs")
        rows.append({"probe":name,"result_sha256":sha256(out/"result.json"),**row})
    for key in ("prediction_sha256","rgb_sha256","baseline_sha256","dense_sha256","pinhole_rgb_sha256","native"):
        if rows[0][key]!=rows[1][key]:raise ValueError("Current-rule native replay differs")
    return {"experiment_id":EXP,"status":"passed_external_second_source_final_audit","config_sha256":sha256(args.config),
            "wrapper_sha256":sha256(Path(__file__)),"parent_external_sha256":sha256(previous_path),"probes":rows,
            "native_replay_exact":True,"frozen_and_dense_references_exact":True,"target_label_decodes":0,
            "fresh_external_probes":4}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=("all","probe"))
    p.add_argument("--config",type=Path,default=CONFIG)
    for k in ("artifact-root","data-root","roma-dir","dinov3-dir","weights","bundle","render-python","output-dir"):
        p.add_argument("--"+k,type=Path,required=True)
    for k in ("cameras","images","reference-dir"):p.add_argument("--"+k,type=Path)
    p.add_argument("--image-name",default="140.png");args=p.parse_args()
    for k,v in vars(args).items():
        if isinstance(v,Path):setattr(args,k,legacy.argument_path(k,v))
    args.partition="developer";args.output_dir.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    try:
        result=run_all(args) if args.mode=="all" else probe(args)
        result["seconds"]=time.perf_counter()-started
        if result["seconds"]>read_json(args.config)["resource_budget"]["maximum_seconds_per_stage"]:raise RuntimeError("Stage time budget")
        write_json(args.output_dir/"result.json",result);print({k:result[k] for k in ("status","seconds")},flush=True)
    except Exception as exc:
        write_json(args.output_dir/"failure.json",{"experiment_id":EXP,"error_type":type(exc).__name__,"error":str(exc)})
        raise
