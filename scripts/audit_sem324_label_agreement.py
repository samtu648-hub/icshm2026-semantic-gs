#!/usr/bin/env python3
"""Read-only class agreement and baseline-error coverage on sealed SEM323 maps."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
import cv2
import numpy as np
from src.semantic2d import read_json,write_json,sha256,label_array
from src.semantic_source_access import read_pose_only
from src.geometry_prior.camera_io import read_cameras_txt
from src.semantic_dense_geometry import normalized_grid,cycle_support,fundamental_from_w2c,symmetric_epipolar_pixels
from src.semantic_dense_label_audit import sample_labels,count_evidence,aggregate
from run_sem250_colmap_track_deck_diag import verify_freeze
from run_sem320_dense_preflight import checked

EXP="SEM-324-DENSE-LABEL-AGREEMENT"
CONFIG=ROOT/"experiments/runs"/EXP/"config.yaml"


def geometry_support(pair,maps,poses,cameras,filters,expected):
    (ab,oa),(ba,ob)=maps
    a,b=poses[pair["target"]],poses[pair["source"]]
    ca,cb=cameras[a.camera_id],cameras[b.camera_id]
    if ca.model!="SIMPLE_RADIAL" or cb.model!="SIMPLE_RADIAL": raise ValueError("Wrong camera model")
    def matrix(camera):
        f,cx,cy,k=camera.params
        return np.array([[f,0,cx],[0,f,cy],[0,0,1.]]),np.array([k,0.,0.,0.,0.])
    ka,_=matrix(ca); kb,db=matrix(cb)
    pa=(normalized_grid(*oa.shape)+1)*[ca.width/2,ca.height/2]
    raw=(ab+1)*[cb.width/2,cb.height/2]
    pb=cv2.undistortPoints(raw.reshape(-1,1,2),kb,db,P=kb).reshape(raw.shape)
    support=cycle_support(ab,oa,ba,ob,(ca.width,ca.height),filters["overlap_each_direction_minimum"],filters["cycle_maximum_native_pixels"])
    f=fundamental_from_w2c(a.rotation_world_to_camera,a.tvec,b.rotation_world_to_camera,b.tvec,ka,kb)
    keep=support["cycle_kept"]&(symmetric_epipolar_pixels(pa,pb,f)<=filters["epipolar_maximum_native_pixels"])
    tiles=[[float(t.mean()) for t in np.array_split(band,4,axis=1)] for band in np.array_split(keep,4)]
    if int(keep.sum())!=expected["reliable_points"] or tiles!=expected["coverage_4x4"]: raise ValueError("Sealed geometry support differs")
    return pa,pb,keep


def run(args):
    cfg=read_json(args.config)
    if cfg["experiment_id"]!=EXP: raise ValueError("Wrong diagnostic")
    for key in ("prepare","runtime","geometry","parent_config","label_manifest"):
        checked(ROOT/cfg[key],cfg[key+"_sha256"])
    geo=read_json(ROOT/cfg["geometry"]); audit=read_json(ROOT/cfg["geometry_audit"])
    if not geo["passed"] or not audit["passed"] or audit["status"]!="passed_independent_access_audit" or audit["primary_result_sha256"]!=cfg["geometry_sha256"]:
        raise ValueError("Parent geometry not independently qualified")
    plan=read_json(ROOT/cfg["prepare"]); runtime=read_json(ROOT/cfg["runtime"])
    if geo["prepare_sha256"]!=cfg["prepare_sha256"] or geo["runtime_sha256"]!=cfg["runtime_sha256"]:
        raise ValueError("Parent linkage differs")
    for name,h in plan["source_sha256"].items():
        checked(ROOT/name,h); checked((ROOT/cfg["prepare"]).parent/"source_snapshot"/name,h)
    parent=read_json(ROOT/cfg["parent_config"])
    original=read_json(checked(ROOT/parent["parent_config"],parent["parent_config_sha256"]))
    filters=read_json(checked(ROOT/parent["geometry_config"],parent["geometry_config_sha256"]))["filters"]
    frozen=verify_freeze(ROOT/original["freeze_manifest"],args.artifact_root)
    if frozen!=geo["freeze"]: raise ValueError("Frozen baseline changed")
    split=read_json(ROOT/original["semantic_split"])
    folds={f["fold"]:f for f in read_json(ROOT/original["oof_split"])["folds"]}
    targets=geo["summary"]["targets"]
    if len(targets)!=105 or len({t["target"] for t in targets})!=105: raise ValueError("Target scope changed")
    label_records={r["image"]:r for r in read_json(ROOT/cfg["label_manifest"])["records"]}
    baseline_root=args.artifact_root/cfg["baseline"]
    bm=checked(baseline_root/"prediction_manifest.json",cfg["baseline_manifest_sha256"])
    baseline_records={r["name"]:r for r in read_json(bm)["predictions"]}
    allowed=set(t["target"] for t in targets)
    for target in targets:
        fold=folds[target["fold"]]
        if target["target"] not in fold["oof"]: raise ValueError("Target is not OOF")
        source=target["selected_source"]
        if source:
            if source not in fold["train"] or source in fold["oof"] or source in fold["guard"]: raise ValueError("Invalid source label role")
            allowed.add(source)
    forbidden=set(split["semantic_validation_labeled"])|set(split["buffer_excluded_labeled"])
    if allowed&forbidden or not allowed<=set(split["semantic_train_labeled"]): raise ValueError("Forbidden label name")
    label_paths={n:checked(args.artifact_root/cfg["labels"]/n,label_records[n]["label_png_sha256"]) for n in sorted(allowed)}
    base_paths={t["target"]:checked(baseline_root/baseline_records[t["target"]]["prediction"],baseline_records[t["target"]]["prediction_sha256"]) for t in targets}
    camera_dir=args.data_root/"camera_parameters"
    poses=read_pose_only(checked(camera_dir/"images.txt",plan["camera_images_sha256"]))
    cameras=read_cameras_txt(checked(camera_dir/"cameras.txt",plan["camera_intrinsics_sha256"]))
    # This access seal is written BEFORE any label PNG is decoded.
    sources={str(args.config.relative_to(ROOT)):sha256(args.config),str(Path(__file__).relative_to(ROOT)):sha256(Path(__file__))}
    for module in tuple(sys.modules.values()):
        location=getattr(module,"__file__",None)
        if location:
            p=Path(location).resolve()
            if p.is_relative_to(ROOT) and p.suffix==".py" and not p.is_relative_to(ROOT/"outputs"):
                sources[str(p.relative_to(ROOT))]=sha256(p)
    for name in sources:
        dest=args.output_dir/"source_snapshot"/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/name,dest)
    access={"experiment_id":EXP,"config_sha256":sha256(args.config),"geometry_audit_sha256":sha256(ROOT/cfg["geometry_audit"]),
        "target_labels":[t["target"] for t in targets],"source_label_roles":[{"fold":t["fold"],"target":t["target"],"source":t["selected_source"]} for t in targets if t["selected_source"]],
        "label_sha256":{n:sha256(p) for n,p in label_paths.items()},"baseline_sha256":{n:sha256(p) for n,p in base_paths.items()},"source_sha256":sources}
    write_json(args.output_dir/"access_seal.json",access)
    records={p["name"]:p for p in runtime["probes"]}
    pairs={(p["fold"],p["target"],p["source"]):(i,p) for i,p in enumerate(plan["pairs"])}
    expected={(r["fold"],r["target"],r["source"]):r for r in geo["pairs"]}
    label_cache={}
    def label(name):
        if name not in label_cache: label_cache[name]=label_array(label_paths[name])
        return label_cache[name]
    rows=[]; independent_rows=[]
    import torch
    torch.set_num_threads(4)
    for target in targets:
        name,source=target["target"],target["selected_source"]
        ca=cameras[poses[name].camera_id]
        pa=(normalized_grid(512,512)+1)*[ca.width/2,ca.height/2]
        pb=pa; support=np.zeros((512,512),bool)
        if source:
            key=(target["fold"],name,source); i,pair=pairs[key]; maps=[]
            for direction in ("ab","ba"):
                rec=records[f"pair{i}_{direction}"]
                with np.load(checked((ROOT/cfg["runtime"]).parent/rec["path"],rec["sha256"]),allow_pickle=False) as data:
                    maps.append((data["warp"].astype(np.float64),data["overlap"].astype(np.float64)))
            pa,pb,support=geometry_support(pair,maps,poses,cameras,filters,expected[key])
        target_png=label(name); base_png=label_array(base_paths[name]); source_png=label(source) if source else np.zeros_like(target_png)
        counts=[]
        for independent in (False,True):
            gt,valid_a=sample_labels(target_png,pa,independent)
            base,_=sample_labels(base_png,pa,independent)
            other,valid_b=sample_labels(source_png,pb,independent)
            if not valid_a.all(): raise ValueError("Target grid outside frame")
            usable=support&valid_b
            counts.append(count_evidence(gt,base,other,usable,independent))
        if counts[0]!=counts[1]: raise ValueError("Independent class sampling/counting differs: "+name)
        row={"fold":target["fold"],"target":name,"source":source,"selected_rank":target["selected_rank"],
             "reliable_geometric_points":int(support.sum()),"outside_label_grid_points":int((support&~valid_b).sum()),**counts[0]}
        rows.append(row); independent_rows.append({"fold":target["fold"],**counts[1]})
    if set(label_cache)!=allowed: raise ValueError("Label access set differs")
    for n,p in label_paths.items(): checked(p,access["label_sha256"][n])
    for n,p in base_paths.items(): checked(p,access["baseline_sha256"][n])
    if verify_freeze(ROOT/original["freeze_manifest"],args.artifact_root)!=frozen: raise ValueError("Frozen model changed")
    summary=aggregate(rows)
    if summary!=aggregate(independent_rows): raise ValueError("Independent aggregate differs")
    return {"experiment_id":EXP,"status":"completed_read_only_label_diagnostic","summary":summary,
        "folds":[{"fold":f,**aggregate([r for r in rows if r["fold"]==f])} for f in sorted(folds)],"targets":rows,
        "access_seal_sha256":sha256(args.output_dir/"access_seal.json"),"config_sha256":sha256(args.config),
        "independent_sampling_counting_exact":True,"frozen_artifacts_verified":frozen["count"],
        "unique_labels_decoded":len(label_cache),"target_label_observations":len(targets),"source_label_observations":sum(bool(t["selected_source"]) for t in targets),
        "baseline_pngs_decoded":len(base_paths),"rgb_decoded":0,"new_model_inference":0,"semantic_prediction_exports":0,
        "claim_boundary":"512-grid conditional agreement and hypothetical direct-replacement counts only; not native full-frame mIoU or production candidate. Same-fold source-label isolation, reused calibration targets."}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("artifact-root","data-root","output-dir"): p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--config",type=Path,default=CONFIG); args=p.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path): setattr(args,key,value.resolve())
    args.output_dir.mkdir(parents=True,exist_ok=False); start=time.perf_counter()
    try:
        cfg=read_json(args.config)
        if shutil.disk_usage(args.output_dir).free/2**30<cfg["resource_budget"]["minimum_free_disk_gib"]: raise RuntimeError("Disk floor")
        result=run(args); result["seconds"]=time.perf_counter()-start
        if result["seconds"]>cfg["resource_budget"]["maximum_seconds"]: raise RuntimeError("Diagnostic budget exceeded")
        write_json(args.output_dir/"result.json",result)
        print({k:result[k] for k in ("status","seconds","unique_labels_decoded")})
    except Exception as exc:
        write_json(args.output_dir/"failure.json",{"experiment_id":EXP,"error_type":type(exc).__name__,"error":str(exc)})
        raise


if __name__=="__main__": main()
