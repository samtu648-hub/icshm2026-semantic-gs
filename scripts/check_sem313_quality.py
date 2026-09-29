#!/usr/bin/env python3
"""Independent per-pixel recomputation of an ALREADY completed SEM313 score."""
import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
import numpy as np
from audit_sem260_error_budget import png
from check_sem311_quality import check_metrics
from run_sem313_foundation_review import CONFIG, checked, context, verify_build
from src.semantic2d import read_json, write_json, sha256
from src.semantic_protected_fusion import partition_training_names
from src.semantic_frozen_dinov2 import ARMS, quality_gates


def verify(args):
    cfg,parent,run,names,base,rgb,inherited,freeze=context(args)
    built,pred=verify_build(args); result=read_json(args.quality_dir/"result.json")
    if (result["config_sha256"]!=sha256(args.config)
            or result["build_result_sha256"]!=sha256(args.run_dir/"result.json")):
        raise ValueError("Quality provenance mismatch")
    scope=result["scope"]
    if scope not in ("calibration","check"):
        raise ValueError("Invalid already-opened scope")
    expected=partition_training_names(names,expected_count=219)[scope]
    if expected!=[r["name"] for r in result["target_access"]]:
        raise ValueError("Scope mismatch")
    rows={r["name"]:r for r in result["per_view"]}
    cms={a:np.zeros((5,5),np.int64) for a in ("baseline",*ARMS)}
    folds={str(i):{a:np.zeros((5,5),np.int64) for a in cms} for i in range(3)}
    changes={a:{"corrected":0,"harmed":0,"neutral":0,"removed_foundation":0} for a in ARMS}
    for access in result["target_access"]:
        name=access["name"]; fid=str(built["folds"][name])
        gt=png(checked(args.artifact_root/parent["parents"]["labels"]/name,access["label_sha256"]))
        b=png(checked(*base[name]))
        for arm in cms:
            if arm=="baseline":
                new=b
            else:
                r=pred[arm][name]; new=png(checked(r["path"],r["sha256"]))
                old=inherited[arm][name]; parent_png=png(checked(old["path"],old["sha256"]))
                expected_image=np.where((b==4)&(parent_png==0),0,b).astype(np.uint8)
                if not np.array_equal(new,expected_image):
                    raise ValueError("Review output differs from parent-PNG reconstruction")
                changed=b!=new
                transition=np.zeros((5,5,5),np.int64)
                np.add.at(transition,(gt[changed],b[changed],new[changed]),1)
                if transition.sum()!=transition[:,4,0].sum():
                    raise ValueError("Unauthorized transition")
                counts=transition[:,4,0]
                for key,value in (("corrected",counts[0]),("harmed",counts[4]),
                                  ("neutral",counts[1:4].sum()),("removed_foundation",counts.sum())):
                    changes[arm][key]+=int(value)
            cm=np.zeros((5,5),np.int64)
            np.add.at(cm,(gt.reshape(-1),new.reshape(-1)),1)
            check_metrics(cm,rows[name][arm]); cms[arm]+=cm; folds[fid][arm]+=cm
    for arm,cm in cms.items():
        check_metrics(cm,result["metrics"][arm])
    for fid,byarm in folds.items():
        for arm,cm in byarm.items():
            check_metrics(cm,result["fold_metrics"][fid][arm])
    if changes!=result["changes"]:
        raise ValueError("Independent transition counts differ")
    gates=quality_gates(result["metrics"],result["fold_metrics"],cfg["gates"])
    gates["protected_exact"]=True
    if gates!=result["checks"] or all(gates.values())!=result["passed"]:
        raise ValueError("Quality decision differs")
    return {"experiment_id":cfg["experiment_id"],"status":"passed_independent_quality_recomputation",
            "scope":scope,"views":len(expected),"quality_passed":result["passed"],
            "quality_result_sha256":sha256(args.quality_dir/"result.json"),"config_sha256":sha256(args.config),
            "code_sha256":sha256(Path(__file__)),"metrics_views_folds_exact":True,"transitions_exact":True,
            "new_quality_partitions_opened":[],"developer_labels_decoded":0,"target_access":result["target_access"]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("artifact-root","parent-dir","run-dir","quality-dir","output-dir"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--config",type=Path,default=CONFIG)
    args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=False); started=time.perf_counter()
    try:
        result=verify(args); result["seconds"]=time.perf_counter()-started
        write_json(args.output_dir/"result.json",result)
        print({k:result[k] for k in ("status","scope","views","quality_passed","seconds")})
    except Exception as exc:
        write_json(args.output_dir/"failure.json",{"error_type":type(exc).__name__,"error":str(exc)})
        raise


if __name__=="__main__":
    main()
