#!/usr/bin/env python3
"""Seal one matched auxiliary-review pair; independently gated quality stages."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
import numpy as np
from PIL import Image
from audit_sem260_error_budget import manifest_paths, png
from run_sem250_colmap_track_deck_diag import verify_freeze
from src.semantic2d import read_json, write_json, sha256, confusion_metrics
from src.semantic_protected_fusion import partition_training_names
from src.semantic_foundation_review import review_foundation
from src.semantic_frozen_dinov2 import ARMS, quality_gates

CONFIG = ROOT/"experiments/runs/SEM-313-DINOV2-FOUNDATION-REVIEW-OOF/config.yaml"


def checked(path, digest):
    path = Path(path)
    if sha256(path)!=digest:
        raise ValueError(f"Sealed input changed: {path.name}")
    return path


def context(args):
    cfg = read_json(args.config)
    if cfg["threshold"] != 0.5:
        raise ValueError("Fixed rule threshold cannot change")
    parent = read_json(checked(ROOT/cfg["parent_config"],cfg["parent_config_sha256"]))
    run = read_json(checked(args.parent_dir/"result.json",cfg["parent_result_sha256"]))
    if run["status"]!="completed_calibration_gate_failed" or run["config_sha256"]!=sha256(ROOT/cfg["parent_config"]):
        raise ValueError("Requires the frozen failed parent")
    for key in ("semantic_split","oof_split"):
        checked(ROOT/parent[key],parent[key+"_sha256"])
    names = read_json(ROOT/parent["semantic_split"])["semantic_train_labeled"]
    baseline = manifest_paths(args.artifact_root/parent["parents"]["sem160"],"prediction_manifest.json","predictions","prediction","prediction_sha256")
    rgb = manifest_paths(args.artifact_root/parent["parents"]["rgb"],"rendered_dataset_manifest.json","rendered_images","path","sha256")
    seal = read_json(checked(args.parent_dir/"prediction_seal.json",run["prediction_seal_sha256"]))
    pred = read_json(checked(args.parent_dir/"prediction_manifest.json",seal["prediction_manifest_sha256"]))
    freeze = verify_freeze(ROOT/cfg["freeze_manifest"],args.artifact_root)
    if freeze != run["freeze_after"] or set(baseline)!=set(names) or set(rgb)!=set(names):
        raise ValueError("Frozen parent inputs differ")
    return cfg,parent,run,names,baseline,rgb,pred,freeze


def check_review_contract(base, result):
    if result.shape!=base.shape or not np.isin(result,(0,1,2,3,4)).all():
        raise ValueError("Invalid review prediction")
    changed = result!=base
    if np.any(changed&((base!=4)|(result!=0))):
        raise ValueError("Review exceeded foundation-to-background authority")


def build(args):
    cfg,parent,run,names,base,rgb,pred,freeze = context(args)
    folds = {n:f["fold"] for f in read_json(ROOT/parent["oof_split"])["folds"] for n in f["oof"]}
    sources = [Path(__file__), ROOT/"scripts/audit_sem313_foundation_review.py",
               ROOT/"src/semantic_foundation_review.py", ROOT/"src/semantic_frozen_dinov2.py",
               ROOT/"src/semantic2d.py", ROOT/"src/semantic_protected_fusion.py", args.config]
    snapshot = args.output_dir/"source_snapshot"; snapshot.mkdir()
    hashes = {}
    for path in sources:
        shutil.copyfile(path,snapshot/path.name); hashes[str(path.relative_to(ROOT))]=sha256(path)
    outputs = {a:{} for a in ARMS}
    for a in ARMS:
        if set(pred[a])!=set(names):
            raise ValueError("Unexpected inherited OOF coverage")
        (args.output_dir/a).mkdir()
    for name in names:
        b = png(checked(*base[name])); checked(*rgb[name])
        for arm in ARMS:
            r = pred[arm][name]
            if r["fold"] != folds[name]:
                raise ValueError("Wrong inherited fold")
            p = np.load(checked(r["probability_path"],r["probability_sha256"]),allow_pickle=False)
            checked(r["path"],r["sha256"])
            new = review_foundation(b,p); check_review_contract(b,new)
            if not np.array_equal(b,review_foundation(b,p,enabled=False)):
                raise ValueError("Disabled branch differs")
            path = args.output_dir/arm/name; Image.fromarray(new).save(path)
            outputs[arm][name] = {"path":str(path),"sha256":sha256(path),"fold":folds[name],
                                  "parent_probability_sha256":r["probability_sha256"],"baseline_sha256":base[name][1]}
    write_json(args.output_dir/"prediction_manifest.json",outputs)
    after = verify_freeze(ROOT/cfg["freeze_manifest"],args.artifact_root)
    if freeze != after:
        raise ValueError("Frozen artifacts changed")
    for name in names:
        checked(*rgb[name]); checked(*base[name])
    return {"experiment_id":cfg["experiment_id"],"status":"predictions_sealed_no_quality",
            "repository_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            "seed":cfg["seed"],"config_sha256":sha256(args.config),"source_sha256":hashes,
            "parent_result_sha256":cfg["parent_result_sha256"],"dataset_manifest_sha256":sha256(ROOT/cfg["dataset_manifest"]),
            "prediction_manifest_sha256":sha256(args.output_dir/"prediction_manifest.json"),
            "folds":folds,"prediction_count":sum(len(v) for v in outputs.values()),
            "freeze_before":freeze,"freeze_after":after,"rgb_files_verified_unchanged":len(names),
            "labels_decoded":0,"training_steps":0,"rule":cfg["rule"]}


def verify_build(args):
    build_run = read_json(args.run_dir/"result.json")
    if build_run["config_sha256"]!=sha256(args.config) or build_run["status"]!="predictions_sealed_no_quality":
        raise ValueError("Review build identity changed")
    for path,digest in build_run["source_sha256"].items():
        checked(ROOT/path,digest); checked(args.run_dir/"source_snapshot"/Path(path).name,digest)
    pred = read_json(checked(args.run_dir/"prediction_manifest.json",build_run["prediction_manifest_sha256"]))
    return build_run,pred


def score(args):
    cfg,parent,run,names,base,rgb,parent_pred,freeze = context(args)
    built,pred = verify_build(args)
    audit = read_json(args.audit_dir/"result.json")
    if (audit["status"]!="passed_independent_review_engineering" or audit["build_result_sha256"]!=sha256(args.run_dir/"result.json")
            or audit["predictions_checked"]!=438 or len(audit["native_replays"])!=6 or len(audit["review_external_probes"])!=4):
        raise ValueError("Independent actual-head engineering gate required")
    if args.mode=="check":
        cal = read_json(args.calibration_dir/"result.json")
        if (cal["scope"]!="calibration" or not cal["passed"] or cal["config_sha256"]!=sha256(args.config)
                or cal["build_result_sha256"]!=sha256(args.run_dir/"result.json")):
            raise ValueError("Check cannot open without matched calibration pass")
    scope = "calibration" if args.mode=="calibration" else "check"
    targets = partition_training_names(names,expected_count=219)[scope]
    totals = {a:np.zeros((5,5),np.int64) for a in ("baseline",*ARMS)}
    fold_cms = {str(i):{a:np.zeros((5,5),np.int64) for a in totals} for i in range(3)}
    rows = []; access = []; changes={a:{"corrected":0,"harmed":0,"neutral":0,"removed_foundation":0} for a in ARMS}
    for name in targets:
        label = args.artifact_root/parent["parents"]["labels"]/name
        if scope=="calibration":
            old_access={r["name"]:r["label_sha256"] for r in run["calibration"]["target_access"]}
            checked(label,old_access[name])
        gt = png(label); b = png(checked(*base[name])); fid=str(built["folds"][name])
        if gt.shape!=b.shape or not np.isin(gt,(0,1,2,3,4)).all():
            raise ValueError("Invalid quality target")
        row={"name":name,"fold":fid}
        for arm in totals:
            if arm=="baseline":
                new=b
            else:
                r=pred[arm][name]; new=png(checked(r["path"],r["sha256"]))
                check_review_contract(b,new)
                changed=b!=new; good=changed&(new==gt); bad=changed&(b==gt)
                for k,v in (("corrected",good),("harmed",bad),("neutral",changed&~good&~bad),("removed_foundation",changed)):
                    changes[arm][k]+=int(v.sum())
            cm=np.bincount((gt.astype(np.int64)*5+new).reshape(-1),minlength=25).reshape(5,5)
            totals[arm]+=cm; fold_cms[fid][arm]+=cm; row[arm]=confusion_metrics(cm)
        rows.append(row); access.append({"name":name,"label_sha256":sha256(label)})
    metrics={a:confusion_metrics(cm) for a,cm in totals.items()}
    folds={fid:{a:confusion_metrics(cm) for a,cm in byarm.items()} for fid,byarm in fold_cms.items()}
    gates=quality_gates(metrics,folds,cfg["gates"]); gates["protected_exact"]=True
    after=verify_freeze(ROOT/cfg["freeze_manifest"],args.artifact_root)
    if after!=freeze:
        raise ValueError("Frozen artifacts changed")
    return {"experiment_id":cfg["experiment_id"],"status":"completed_"+scope,
            "scope":scope,"passed":all(gates.values()),"checks":gates,"metrics":metrics,"fold_metrics":folds,
            "per_view":rows,"changes":changes,"target_access":access,"views":len(targets),
            "build_result_sha256":sha256(args.run_dir/"result.json"),"engineering_audit_sha256":sha256(args.audit_dir/"result.json"),
            "config_sha256":sha256(args.config),"seed":cfg["seed"],"freeze_after":after,
            "developer_labels_decoded":0,"training_steps":0,"final_fit":False,"production_changed":False,
            "interpretation":"Reused development scope, not blind;108 motivated the direction. No post-score adjustment."}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("build","calibration","check"),required=True)
    for k in ("artifact-root","parent-dir","output-dir"):
        p.add_argument("--"+k,type=Path,required=True)
    for k in ("run-dir","audit-dir","calibration-dir"):
        p.add_argument("--"+k,type=Path)
    p.add_argument("--config",type=Path,default=CONFIG)
    args=p.parse_args(); args.output_dir=args.output_dir.resolve(); args.output_dir.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    try:
        result=build(args) if args.mode=="build" else score(args)
        result["seconds"]=time.perf_counter()-started; write_json(args.output_dir/"result.json",result)
        print({k:result[k] for k in ("status","seconds","passed","checks") if k in result},flush=True)
    except Exception as exc:
        write_json(args.output_dir/"failure.json",{"error_type":type(exc).__name__,"error":str(exc),"seconds":time.perf_counter()-started})
        raise


if __name__=="__main__":
    main()
