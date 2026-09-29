#!/usr/bin/env python3
"""Independently count the already-scored74 and save fixed SEM326 diagnostic panels."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
import numpy as np
from PIL import Image, ImageDraw
from src.semantic2d import read_json,write_json,sha256,label_array
from src.semantic_render_domain import semantic_rgb
from run_sem326_review import CONFIG,EXP,context,verify_prediction_seal
from check_sem311_quality import check_metrics
from run_sem320_dense_preflight import checked


def run(args):
    c=context(args); pred=verify_prediction_seal(args,c)
    quality=read_json(args.quality_dir/"result.json")
    if quality["prediction_result_sha256"]!=sha256(args.prediction_dir/"result.json") or quality["config_sha256"]!=sha256(args.config):
        raise ValueError("Quality provenance differs")
    if {r["target"] for r in quality["targets"]}!=set(c["dev"]):raise ValueError("Target scope differs")
    rows={r["target"]:r for r in quality["targets"]}; matrices={}; panels=[]
    changes={"changed":0,"corrected":0,"harmed":0,"still_wrong":0}
    label_access=[]
    for rec in pred["predictions"]:
        n=rec["target"]
        path=checked(args.artifact_root/c["parent"]["labels"]/n,c["labels"][n]["label_png_sha256"])
        gt=label_array(path); base=label_array(c["baseline_paths"][n]); candidate=label_array(args.prediction_dir/rec["path"])
        label_access.append({"name":n,"sha256":sha256(path)})
        matrices[n]={}
        for arm,image in (("baseline",base),("candidate",candidate)):
            cm=np.zeros((5,5),np.int64);np.add.at(cm,(gt.ravel(),image.ravel()),1)
            if not np.array_equal(cm,rows[n][arm]):raise ValueError("Independent per-view confusion differs")
            matrices[n][arm]=cm
        changed=base!=candidate; counts=np.bincount(gt[changed],minlength=5)
        values={"changed":int(counts.sum()),"corrected":int(counts[0]),"harmed":int(counts[4]),"still_wrong":int(counts[1:4].sum())}
        for k,v in values.items():
            if v!=rows[n][k]:raise ValueError("Transition accounting differs")
            changes[k]+=v
        if n in c["cfg"]["visual_samples"]:
            rgb_path=next(args.artifact_root/p for p in c["rgb"] if Path(p).name==n)
            with Image.open(rgb_path) as im:rgb=np.asarray(im).copy()
            error=np.zeros_like(rgb);error[changed&(gt==0)]=(35,210,80);error[changed&(gt==4)]=(245,45,45)
            error[changed&(gt!=0)&(gt!=4)]=(240,170,40)
            ys,xs=np.where((gt==4)|(base==4))
            x0,x1=(max(0,int(xs.min())-16),min(base.shape[1],int(xs.max())+17)) if xs.size else (0,base.shape[1])
            y0,y1=(max(0,int(ys.min())-16),min(base.shape[0],int(ys.max())+17)) if ys.size else (0,base.shape[0])
            w,h=x1-x0,y1-y0; canvas=Image.new("RGB",(3*w,2*(h+32)),"white");draw=ImageDraw.Draw(canvas)
            pictures=[rgb,semantic_rgb(gt),semantic_rgb(base),semantic_rgb(candidate),error,np.full_like(rgb,245)]
            titles=[n+" frozen RGB","GT (reused developer)","SEM161 baseline","SEM326 dense review","green fixed / red harmed / amber wrong","Only4-to0; no display cleanup"]
            for i,(im,title) in enumerate(zip(pictures,titles)):
                dx,dy=(i%3)*w,(i//3)*(h+32)
                canvas.paste(Image.fromarray(im[y0:y1,x0:x1]),(dx,dy+32));draw.text((dx+3,dy+5),title,fill="black")
            out=args.output_dir/"panels"/n;out.parent.mkdir(parents=True,exist_ok=True);canvas.save(out)
            panels.append({"name":n,"path":str(out.relative_to(args.output_dir)),"sha256":sha256(out),"crop_xyxy":[x0,y0,x1,y1]})
    for arm in ("baseline","candidate"):
        check_metrics(sum((m[arm] for m in matrices.values()),np.zeros((5,5),np.int64)),quality["metrics"][arm])
    for group,names in c["groups"].items():
        for arm in ("baseline","candidate"):
            check_metrics(sum((matrices[n][arm] for n in names),np.zeros((5,5),np.int64)),quality["group_metrics"][group][arm])
    if changes!={k:quality["summary"][k] for k in changes}:raise ValueError("Aggregate changes differ")
    # Independent mean-IoU arithmetic and fixed subset membership, not scorer helpers.
    def miou(cm):
        d=cm.diagonal();u=cm.sum(0)+cm.sum(1)-d
        return float(np.mean(d[u>0]/u[u>0]))
    by_name={n:{a:miou(cm) for a,cm in m.items()} for n,m in matrices.items()}
    worst=sorted(by_name,key=lambda n:(by_name[n]["baseline"],n))[:8]
    challenge=[n for n in c["dev"] if 140<=int(Path(n).stem)<=146]
    for key,names in (("baseline_worst10",worst),("challenge_140_146",challenge)):
        row=quality["subsets"][key]
        if names!=row["names"]:raise ValueError("Fixed subset differs")
        for arm in ("baseline","candidate"):
            if not np.isclose(np.mean([by_name[n][arm] for n in names]),row[arm+"_mean_miou"],rtol=0,atol=1e-12):raise ValueError("Subset arithmetic differs")
    g=c["cfg"]["developer_gates"];b,cc=quality["metrics"]["baseline"],quality["metrics"]["candidate"]
    bf,cf=b["per_class"]["foundation"],cc["per_class"]["foundation"]
    checks={"miou_gain":cc["mIoU_with_background"]-b["mIoU_with_background"]>=g["miou_gain"],
            "foundation_iou_gain":cf["iou"]-bf["iou"]>=g["foundation_iou_gain"],
            "recall_retention":cf["recall"]-bf["recall"]>=-g["maximum_recall_drop"],"recall_floor":cf["recall"]>=g["recall_floor"]}
    for name,row in quality["group_metrics"].items():
        x,y=row["baseline"],row["candidate"]
        checks[name+"_miou_safety"]=y["mIoU_with_background"]-x["mIoU_with_background"]>=-g["maximum_group_miou_drop"]
        xi,yi=x["per_class"]["foundation"]["iou"],y["per_class"]["foundation"]["iou"]
        if xi is not None or yi is not None:checks[name+"_foundation_safety"]=xi is not None and yi is not None and yi-xi>=-g["maximum_group_foundation_iou_drop"]
    for name,row in quality["subsets"].items():checks[name+"_miou_safety"]=row["candidate_mean_miou"]-row["baseline_mean_miou"]>=-g["maximum_difficult_miou_drop"]
    if checks!=quality["gates"] or all(checks.values())!=quality["passed"]:raise ValueError("Independent final decision differs")
    return {"experiment_id":EXP,"status":"passed_independent_final_quality_audit","quality_passed":quality["passed"],
            "quality_result_sha256":sha256(args.quality_dir/"result.json"),"config_sha256":sha256(args.config),
            "auditor_sha256":sha256(Path(__file__)),"views":74,"labels":label_access,"new_partitions_opened":[],
            "per_view_group_subset_metrics_exact":True,"decision_exact":True,"panels":panels}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("artifact-root","data-root","matching-root","prediction-dir","quality-dir","output-dir"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--config",type=Path,default=CONFIG);args=p.parse_args()
    for k,v in vars(args).items():
        if isinstance(v,Path):setattr(args,k,v.resolve())
    args.partition="developer";args.prepare_dir=args.matching_root/"prepare"
    args.output_dir.mkdir(parents=True,exist_ok=False)
    try:
        result=run(args);write_json(args.output_dir/"result.json",result);print({k:result[k] for k in ("status","quality_passed","views")})
    except Exception as exc:
        write_json(args.output_dir/"failure.json",{"experiment_id":EXP,"error_type":type(exc).__name__,"error":str(exc)});raise
