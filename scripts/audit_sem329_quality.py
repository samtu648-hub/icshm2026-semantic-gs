#!/usr/bin/env python3
"""Independent three-arm final arithmetic and the six fixed native-pixel panels."""
import argparse
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
import numpy as np
from PIL import Image,ImageDraw
from src.semantic2d import read_json,write_json,sha256,label_array
from src.semantic_render_domain import semantic_rgb
from run_sem329_final import CONFIG,EXP,setup,verify_prediction,require_engineering
from run_sem320_dense_preflight import checked
from check_sem311_quality import check_metrics


def independent_miou(cm):
    values=[]
    for i in range(5):
        union=int(cm[i,:].sum()+cm[:,i].sum()-cm[i,i])
        if union:values.append(int(cm[i,i])/union)
    return sum(values)/len(values)


def run(args):
    d=setup(args);c=d["c"];prediction=verify_prediction(args,d);require_engineering(args)
    q=read_json(args.quality_dir/"result.json")
    if q["config_sha256"]!=sha256(args.config) or q["prediction_result_sha256"]!=sha256(args.prediction_dir/"result.json"):
        raise ValueError("Final quality provenance differs")
    rows={r["target"]:r for r in q["targets"]};old={r["target"]:r for r in d["targets"]}
    matrices={};panels=[];restored_total=np.zeros(5,np.int64);remaining_total=np.zeros(5,np.int64)
    for rec in prediction["predictions"]:
        n=rec["target"]
        gtpath=checked(args.artifact_root/c["parent"]["labels"]/n,c["labels"][n]["label_png_sha256"])
        gt=label_array(gtpath);base=label_array(c["baseline_paths"][n])
        dense=label_array(d["old"].prediction_dir/old[n]["path"]);candidate=label_array(args.prediction_dir/rec["path"])
        matrices[n]={}
        for a,im in (("baseline",base),("dense",dense),("candidate",candidate)):
            cm=np.zeros((5,5),np.int64);np.add.at(cm,(gt.ravel(),im.ravel()),1)
            if not np.array_equal(cm,rows[n][a]):raise ValueError("Per-view confusion differs")
            matrices[n][a]=cm
            if abs(independent_miou(cm)-q["per_view"][n][a])>1e-12:raise ValueError("Per-view mIoU differs")
        restore=candidate!=dense;remain=candidate!=base
        restored=np.array([int(np.sum(restore&(gt==k))) for k in range(5)])
        remaining=np.array([int(np.sum(remain&(gt==k))) for k in range(5)])
        if restored.tolist()!=rows[n]["restored_by_gt_class"] or remaining.tolist()!=rows[n]["remaining_deletions_by_gt_class"]:
            raise ValueError("Independent per-class changes differ")
        restored_total+=restored;remaining_total+=remaining
        if n in d["cfg"]["visual_samples"]:
            rgbpath=next(args.artifact_root/p for p in c["rgb"] if Path(p).name==n)
            with Image.open(rgbpath) as im:rgb=np.asarray(im).copy()
            effect=np.zeros_like(rgb)
            effect[restore&(gt==4)]=(35,210,80);effect[restore&(gt==0)]=(245,45,45)
            effect[restore&(gt!=0)&(gt!=4)]=(240,170,40)
            ys,xs=np.where((gt==4)|(base==4))
            x0,x1=(max(0,int(xs.min())-16),min(base.shape[1],int(xs.max())+17)) if xs.size else (0,base.shape[1])
            y0,y1=(max(0,int(ys.min())-16),min(base.shape[0],int(ys.max())+17)) if ys.size else (0,base.shape[0])
            width,height=x1-x0,y1-y0;cell=max(320,width+16);header=50
            panel=Image.new("RGB",(cell*3,2*(height+header)),"white");draw=ImageDraw.Draw(panel)
            pictures=[rgb,semantic_rgb(gt),semantic_rgb(base),semantic_rgb(dense),semantic_rgb(candidate),effect]
            titles=[n+" frozen RGB","GT (reused developer)","SEM161 baseline","SEM326 dense","SEM329 second-source veto",
                    "Restored: green true foundation\nred background / amber other GT"]
            for i,(picture,title) in enumerate(zip(pictures,titles)):
                dx,dy=(i%3)*cell,(i//3)*(height+header)
                tile=Image.fromarray(picture[y0:y1,x0:x1]);left=dx+(cell-width)//2
                panel.paste(tile,(left,dy+header));draw.multiline_text((dx+6,dy+5),title,fill="black",spacing=4)
                if panel.crop((left,dy+header,left+width,dy+header+height)).tobytes()!=tile.tobytes():raise ValueError("Display pixels differ")
            p=args.output_dir/"panels"/n;p.parent.mkdir(parents=True,exist_ok=True);panel.save(p)
            panels.append({"name":n,"path":str(p.relative_to(args.output_dir)),"sha256":sha256(p),"crop_xyxy":[x0,y0,x1,y1],"native_pixels_exact":True})
    if set(matrices)!=set(c["dev"]) or len(matrices)!=74:raise ValueError("Audit scope differs")
    def check_aggregate(names,record):
        for a in ("baseline","dense","candidate"):
            cm=sum((matrices[n][a] for n in names),np.zeros((5,5),np.int64));check_metrics(cm,record[a])
        for reference in ("baseline","dense"):
            r=record[reference];cc=record["candidate"]
            expected={"miou":100*(cc["mIoU_with_background"]-r["mIoU_with_background"]),
                      **{k:100*(cc["per_class"]["foundation"][k]-r["per_class"]["foundation"][k]) for k in ("iou","precision","recall")}}
            if expected!=record["delta_vs_"+reference+"_pp"]:raise ValueError("Paired arithmetic differs")
    check_aggregate(c["dev"],q["summary"])
    for name,names in c["groups"].items():check_aggregate(names,q["group_metrics"][name])
    if restored_total.tolist()!=q["summary"]["restored_by_gt_class"] or remaining_total.tolist()!=q["summary"]["remaining_deletions_by_gt_class"]:
        raise ValueError("Global restored/remaining counts differ")
    worst=sorted(matrices,key=lambda n:(independent_miou(matrices[n]["baseline"]),n))[:8]
    challenge=[n for n in matrices if 140<=int(Path(n).stem)<=146]
    for key,names in (("baseline_worst10",worst),("challenge_140_146",challenge)):
        row=q["subsets"][key]
        if names!=row["names"]:raise ValueError("Fixed subset differs")
        for a in ("baseline","dense","candidate"):
            expected=sum(independent_miou(matrices[n][a]) for n in names)/len(names)
            if abs(expected-row[a+"_mean_miou"])>1e-12:raise ValueError("Subset mean differs")
    # Historical cutoffs are descriptive; independently check both them and benefit signs.
    s=q["summary"];g=c["cfg"]["developer_gates"];b=s["baseline"];cc=s["candidate"]
    bf,cf=b["per_class"]["foundation"],cc["per_class"]["foundation"]
    gates={"miou_gain":cc["mIoU_with_background"]>=b["mIoU_with_background"]+g["miou_gain"],
           "foundation_iou_gain":cf["iou"]>=bf["iou"]+g["foundation_iou_gain"],
           "recall_retention":cf["recall"]>=bf["recall"]-g["maximum_recall_drop"],"recall_floor":cf["recall"]>=g["recall_floor"]}
    for name,row in q["group_metrics"].items():
        a,z=row["baseline"],row["candidate"]
        gates[name+"_miou_safety"]=z["mIoU_with_background"]>=a["mIoU_with_background"]-g["maximum_group_miou_drop"]
        ai,zi=a["per_class"]["foundation"]["iou"],z["per_class"]["foundation"]["iou"]
        if ai is not None or zi is not None:gates[name+"_foundation_safety"]=ai is not None and zi is not None and zi>=ai-g["maximum_group_foundation_iou_drop"]
    for name,row in q["subsets"].items():gates[name+"_miou_safety"]=row["candidate_mean_miou"]>=row["baseline_mean_miou"]-g["maximum_difficult_miou_drop"]
    if gates!=q["historical_sem326_gates_on_candidate"]:raise ValueError("Historical context differs")
    for name in ("baseline","dense"):
        delta=s["delta_vs_"+name+"_pp"]
        benefit={"miou_positive":delta["miou"]>0,"foundation_iou_positive":delta["iou"]>0,
                 "foundation_recall_delta_pp":delta["recall"],"foundation_precision_delta_pp":delta["precision"]}
        if benefit!=q["benefits"][name]:raise ValueError("Benefit conclusion differs")
    return {"experiment_id":EXP,"status":"passed_independent_three_arm_quality_audit","views":74,
            "config_sha256":sha256(args.config),"quality_result_sha256":sha256(args.quality_dir/"result.json"),
            "auditor_sha256":sha256(Path(__file__)),"all_view_group_subset_metrics_exact":True,
            "change_accounting_exact":True,"benefit_and_historical_gate_arithmetic_exact":True,"panels":panels,
            "new_partitions_opened":[],"target_label_roles_decoded":74}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path,default=CONFIG)
    for k in ("artifact-root","data-root","prediction-dir","audit-dir","external-dir","quality-dir","output-dir"):
        p.add_argument("--"+k,type=Path,required=True)
    args=p.parse_args()
    for k,v in vars(args).items():
        if isinstance(v,Path):setattr(args,k,v.resolve())
    args.partition="developer";args.output_dir.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    try:
        r=run(args);r["seconds"]=time.perf_counter()-started;write_json(args.output_dir/"result.json",r)
        print({k:r[k] for k in ("status","views","seconds")})
    except Exception as exc:
        write_json(args.output_dir/"failure.json",{"experiment_id":EXP,"error_type":type(exc).__name__,"error":str(exc)})
        raise
