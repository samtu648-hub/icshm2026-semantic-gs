#!/usr/bin/env python3
"""Independently recompute already-scored74 and render registered diagnostic panels."""
import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
import numpy as np
from PIL import Image,ImageDraw
from audit_sem260_error_budget import png
from check_sem311_quality import check_metrics
from run_sem314_final_review import CONFIG,checked,prediction_context,quality_summaries
from src.semantic2d import read_json,write_json,sha256
from src.semantic_foundation_review_final import EXP,final_gates
from src.semantic_render_domain import semantic_rgb


def verify(args):
    cfg,dev,labels,rgb,base,freeze,pred,manifest=prediction_context(args)
    result=read_json(args.quality_dir/"result.json")
    if (result["config_sha256"]!=sha256(args.config) or result["prediction_result_sha256"]!=sha256(args.prediction_dir/"result.json")
            or [r["name"] for r in result["target_access"]]!=dev):
        raise ValueError("Already-scored final quality scope differs")
    per_view={r["name"]:r for r in result["per_view"]}; total={a:np.zeros((5,5),np.int64) for a in ("baseline","candidate")}
    transitions=np.zeros((5,5,5),np.int64); panels=[]; (args.output_dir/"panels").mkdir()
    for rec in manifest["predictions"]:
        name=rec["name"];gt=png(checked(args.artifact_root/cfg["parents"]["labels"]/name,labels[name]))
        b=png(checked(*base[name]));new=png(checked(args.prediction_dir/rec["prediction"],rec["prediction_sha256"]))
        changed=b!=new;np.add.at(transitions,(gt[changed],b[changed],new[changed]),1)
        for arm,image in (("baseline",b),("candidate",new)):
            cm=np.zeros((5,5),np.int64);np.add.at(cm,(gt.reshape(-1),image.reshape(-1)),1)
            check_metrics(cm,per_view[name][arm]);total[arm]+=cm
        if name in cfg["visual_samples"]:
            image=png(checked(*rgb[name])); error=np.zeros_like(image)
            error[changed&(new==gt)]=(40,220,80);error[changed&(b==gt)]=(255,50,50)
            error[changed&(new!=gt)&(b!=gt)]=(240,170,40)
            columns=[image,semantic_rgb(gt),semantic_rgb(b),semantic_rgb(new),error]
            titles=["Frozen RGB","GT (reused developer)","SEM161 baseline","SEM314 final review","Green corrected / red harmed / amber neutral"]
            ys,xs=np.where((gt==4)|(b==4))
            if len(xs):
                x0,x1=max(0,int(xs.min())-16),min(b.shape[1],int(xs.max())+17)
                y0,y1=max(0,int(ys.min())-16),min(b.shape[0],int(ys.max())+17)
            else:
                x0,x1,y0,y1=0,b.shape[1],0,b.shape[0]
            w,h=x1-x0,y1-y0; panel=Image.new("RGB",(5*w,h+44),"white");draw=ImageDraw.Draw(panel)
            for i,(column,title) in enumerate(zip(columns,titles)):
                panel.paste(Image.fromarray(column[y0:y1,x0:x1]),(i*w,44))
                draw.text((i*w+3,3),name+" "+title,fill="black")
            path=args.output_dir/"panels"/name;panel.save(path)
            panels.append({"name":name,"path":str(path),"sha256":sha256(path),"crop_xyxy":[x0,y0,x1,y1]})
    for arm,cm in total.items():check_metrics(cm,result["metrics"][arm])
    if transitions.sum()!=transitions[:,4,0].sum(): raise ValueError("Review authority differs")
    c=transitions[:,4,0]
    changes={"corrected":int(c[0]),"harmed":int(c[4]),"neutral":int(c[1:4].sum()),"removed_foundation":int(c.sum())}
    if changes!=result["changes"]:raise ValueError("Independent transition counts differ")
    metrics,groups,subsets=quality_summaries(result["per_view"],manifest["groups"])
    if metrics!=result["metrics"] or groups!=result["group_metrics"] or subsets!=result["subsets"]:
        raise ValueError("Quality subgroup aggregation differs")
    gates=final_gates(metrics,groups,subsets,cfg["developer_gates"]);gates["protected_exact"]=True
    if gates!=result["checks"] or all(gates.values())!=result["passed"]:raise ValueError("Independent final decision differs")
    return {"experiment_id":EXP,"status":"passed_independent_final_quality_recomputation","views":74,"quality_passed":result["passed"],
            "quality_result_sha256":sha256(args.quality_dir/"result.json"),"config_sha256":sha256(args.config),
            "code_sha256":sha256(Path(__file__)),"per_view_group_aggregate_metrics_exact":True,"transitions_exact":True,
            "target_access":result["target_access"],"new_partitions_opened":[],"panels":panels}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("artifact-root","train-dir","prediction-dir","quality-dir","output-dir"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--config",type=Path,default=CONFIG);args=p.parse_args();args.output_dir=args.output_dir.resolve()
    args.output_dir.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    try:
        result=verify(args);result["seconds"]=time.perf_counter()-started;write_json(args.output_dir/"result.json",result)
        print({"status":result["status"],"quality_passed":result["quality_passed"],"panels":len(result["panels"]),"seconds":result["seconds"]})
    except Exception as exc:
        write_json(args.output_dir/"failure.json",{"error_type":type(exc).__name__,"error":str(exc)});raise


if __name__=="__main__":main()
