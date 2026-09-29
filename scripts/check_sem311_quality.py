#!/usr/bin/env python3
"""Independently recompute ONLY SEM-311's already-opened quality partitions."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
import numpy as np
from audit_sem260_error_budget import manifest_paths, png
from src.semantic2d import read_json, write_json, sha256
from src.semantic_frozen_dinov2 import ARMS, quality_gates
from src.semantic_protected_fusion import partition_training_names


def check_metrics(cm,record):
    if cm.tolist()!=record["confusion_matrix_rows_gt_columns_prediction"]:
        raise ValueError("Independent confusion matrix mismatch")
    ious=[]
    for i,name in enumerate(("background","deck","stay_cable","tower","foundation")):
        tp=int(cm[i,i]); fp=int(cm[:,i].sum()-tp); fn=int(cm[i,:].sum()-tp)
        item=record["per_class"][name]
        if (item["tp"],item["fp"],item["fn"])!=(tp,fp,fn):
            raise ValueError("Class count mismatch")
        for key,denominator in (("iou",tp+fp+fn),("precision",tp+fp),("recall",tp+fn)):
            expected=tp/denominator if denominator else None
            if expected is None:
                if item[key] is not None:
                    raise ValueError("Absent-class metric mismatch")
            elif item[key] is None or abs(expected-item[key])>1e-12:
                raise ValueError("Independent metric mismatch")
        if tp+fp+fn:
            ious.append(tp/(tp+fp+fn))
    if abs(sum(ious)/len(ious)-record["mIoU_with_background"])>1e-12:
        raise ValueError("Independent mIoU mismatch")


def verify(args):
    cfg=read_json(args.config); run=read_json(args.run_dir/"result.json")
    if run["mode"]!="full" or run["config_sha256"]!=sha256(args.config):
        raise ValueError("Requires completed registered full run")
    seal=args.run_dir/"prediction_seal.json"
    if sha256(seal)!=run["prediction_seal_sha256"]:
        raise ValueError("Seal changed")
    manifest=args.run_dir/"prediction_manifest.json"
    if sha256(manifest)!=read_json(seal)["prediction_manifest_sha256"]:
        raise ValueError("Prediction manifest changed")
    predictions=read_json(manifest)
    names=read_json(ROOT/cfg["semantic_split"])["semantic_train_labeled"]
    partition=partition_training_names(names,expected_count=219)
    baseline=manifest_paths(args.artifact_root/cfg["parents"]["sem160"],"prediction_manifest.json","predictions","prediction","prediction_sha256")
    calibration_passed=run["calibration"]["passed"]
    if run["check_quality_scored_this_run"]!=calibration_passed or run["developer_quality_scored_this_run"]:
        raise ValueError("Quality access gate mismatch")
    if not calibration_passed and ("check" in run or (args.run_dir/"check_sealed.json").exists()):
        raise ValueError("Check opened after calibration failure")
    scopes=("calibration","check") if calibration_passed else ("calibration",)
    rows=[]
    for scope in scopes:
        summary=run[scope]
        if read_json(args.run_dir/f"{scope}_sealed.json")!=summary:
            raise ValueError("Scored result changed")
        target_names=partition[scope]
        if [r["name"] for r in summary["target_access"]]!=target_names:
            raise ValueError("Unexpected quality label access")
        per_view={r["name"]:r for r in summary["per_view"]}
        totals={k:np.zeros((5,5),np.int64) for k in ("baseline",*ARMS)}
        folds={str(i):{k:np.zeros((5,5),np.int64) for k in totals} for i in range(3)}
        changes={a:{"corrected":0,"harmed":0,"added_foundation":0,"removed_foundation":0} for a in ARMS}
        access=[]
        for access_record in summary["target_access"]:
            name=access_record["name"]; path=args.artifact_root/cfg["parents"]["labels"]/name
            if sha256(path)!=access_record["label_sha256"] or sha256(baseline[name][0])!=baseline[name][1]:
                raise ValueError("Quality input changed")
            target=png(path); base=png(baseline[name][0]); fid=str(predictions[ARMS[0]][name]["fold"])
            for key in totals:
                if key=="baseline":
                    predicted=base
                else:
                    record=predictions[key][name]
                    if str(record["fold"])!=fid or sha256(Path(record["path"]))!=record["sha256"]:
                        raise ValueError("Prediction changed")
                    predicted=png(Path(record["path"]))
                    if any(not np.array_equal(base==i,predicted==i) for i in (1,2,3)):
                        raise ValueError("Protected class changed")
                    for label,mask in (("corrected",(base!=predicted)&(predicted==target)),("harmed",(base!=predicted)&(base==target)),
                                       ("added_foundation",(base==0)&(predicted==4)),("removed_foundation",(base==4)&(predicted==0))):
                        changes[key][label]+=int(mask.sum())
                # Independent construction, separate from the runner's local_confusion.
                cm=np.zeros((5,5),np.int64)
                np.add.at(cm,(target.reshape(-1),predicted.reshape(-1)),1)
                check_metrics(cm,per_view[name][key])
                totals[key]+=cm; folds[fid][key]+=cm
            access.append(access_record)
        for key,cm in totals.items():
            check_metrics(cm,summary["metrics"][key])
        for fid,by_arm in folds.items():
            for key,cm in by_arm.items():
                check_metrics(cm,summary["fold_metrics"][fid][key])
        if changes!=summary["changes"]:
            raise ValueError("Correction/harm count mismatch")
        gates=quality_gates(summary["metrics"],summary["fold_metrics"],cfg["gates"])
        gates["protected_exact"]=True
        if gates!=summary["checks"] or all(gates.values())!=summary["passed"]:
            raise ValueError("Prospective decision differs")
        rows.append({"scope":scope,"views":len(target_names),"metrics_exact":True,
                     "fold_and_view_metrics_exact":True,"changes_exact":True,"gates_exact":True,"target_access":access})
    return {"experiment_id":cfg["experiment_id"],"status":"passed_independent_quality_recomputation",
            "run_result_sha256":sha256(args.run_dir/"result.json"),"config_sha256":sha256(args.config),
            "code_sha256":sha256(Path(__file__)),"scopes":rows,"new_quality_partitions_opened":[],"developer_labels_decoded":0}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("artifact-root","run-dir","output-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--config",type=Path,default=ROOT/"experiments/runs/SEM-311-DINOV2-FROZEN-FEATURE-OOF/config.yaml")
    args=parser.parse_args(); out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    try:
        result=verify(args); result["seconds"]=time.perf_counter()-started
        write_json(out/"result.json",result)
        print({"status":result["status"],"scopes":[(r["scope"],r["views"]) for r in result["scopes"]],"seconds":result["seconds"]})
    except Exception as exc:
        write_json(out/"failure.json",{"error_type":type(exc).__name__,"error":str(exc),"seconds":time.perf_counter()-started})
        raise


if __name__=="__main__":
    main()
