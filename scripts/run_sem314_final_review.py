#!/usr/bin/env python3
"""One final head, sealed developer predictions, then gated developer scoring."""
from __future__ import annotations
import argparse
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from audit_sem260_error_budget import manifest_paths,png
from run_sem250_colmap_track_deck_diag import verify_freeze
from run_sem311_frozen_dinov2 import verify_sources,environment
from src.semantic2d import read_json,write_json,sha256,confusion_metrics
from src.semantic_cable_binary import normalize_rgb
from src.semantic_foundation_binary import training_plan
from src.semantic_foundation_review import review_foundation
from src.semantic_foundation_review_final import (EXP,validate_final_split,segment_groups,
    final_gates,require_engineering,load_final_head)
from src.semantic_frozen_dinov2 import load_segmenter,state_hash,ARMS
from src.submission_sem161 import sliding

CONFIG=ROOT/"experiments/runs"/EXP/"config.yaml"
SOURCE_FILES=["scripts/run_sem314_final_review.py","scripts/audit_sem314_final_review.py",
              "scripts/check_sem314_quality.py","scripts/render_sem314_external_camera.py",
              "src/semantic_foundation_review_final.py","src/semantic_foundation_review.py",
              "src/semantic_frozen_dinov2.py","src/semantic_foundation_binary.py",
              "src/semantic_cable_binary.py","src/semantic2d.py","src/submission_sem161.py",
              "src/semantic_protected_fusion.py","scripts/run_sem311_frozen_dinov2.py",
              "scripts/run_sem250_colmap_track_deck_diag.py","scripts/audit_sem260_error_budget.py"]


def checked(path,digest):
    path=Path(path)
    if sha256(path)!=digest:
        raise ValueError(f"Immutable input changed: {path.name}")
    return path


def context(args,*,developer=False):
    cfg=read_json(args.config)
    for key in ("dataset_manifest","semantic_split","label_manifest","trajectory_segments","parent_config",
                "qualified_check","qualified_check_audit"):
        checked(ROOT/cfg[key],cfg[key+"_sha256"])
    qualified=read_json(ROOT/cfg["qualified_check"])
    audit=read_json(ROOT/cfg["qualified_check_audit"])
    if not qualified["passed"] or not all(qualified["checks"].values()) or not audit["quality_passed"]:
        raise ValueError("Qualified OOF parent required")
    if audit["quality_result_sha256"]!=cfg["qualified_check_sha256"]:
        raise ValueError("OOF audit provenance differs")
    if cfg["experiment_id"]!=EXP or cfg["inference"]!={"crop":896,"stride":448,"threshold":.5,"only_transition":"4_to_0"}:
        raise ValueError("Frozen final recipe differs")
    parent=read_json(ROOT/cfg["parent_config"])
    for k in ("iterations","crop","decoder_width","learning_rate","weight_decay"):
        if cfg["training"][k]!=parent["training"][k]:
            raise ValueError("Final training recipe differs from OOF")
    split=read_json(ROOT/cfg["semantic_split"]); train,dev=validate_final_split(split)
    labels={r["image"]:r["label_png_sha256"] for r in read_json(ROOT/cfg["label_manifest"])["records"]}
    tr=args.artifact_root/cfg["parents"]["train_rgb"]
    checked(tr/"rendered_dataset_manifest.json",cfg["parents"]["train_rgb_manifest_sha256"])
    rgb=manifest_paths(tr,"rendered_dataset_manifest.json","rendered_images","path","sha256")
    if set(rgb)!=set(train):
        raise ValueError("Unexpected training input coverage")
    freeze=verify_freeze(ROOT/cfg["freeze_manifest"],args.artifact_root)
    if developer:
        vr=args.artifact_root/cfg["parents"]["developer_rgb"]
        br=args.artifact_root/cfg["parents"]["baseline"]
        checked(vr/"rendered_validation_manifest.json",cfg["parents"]["developer_rgb_manifest_sha256"])
        checked(br/"prediction_manifest.json",cfg["parents"]["baseline_manifest_sha256"])
        checked(br/"result.json",cfg["parents"]["baseline_result_sha256"])
        rgb=manifest_paths(vr,"rendered_validation_manifest.json","rendered_images","path","sha256")
        base=manifest_paths(br,"prediction_manifest.json","predictions","prediction","prediction_sha256")
        if set(rgb)!=set(dev) or set(base)!=set(dev):
            raise ValueError("Unexpected developer coverage")
    else:
        base={}
    return cfg,parent,train,dev,labels,rgb,base,freeze


def budget(args,cfg,started):
    if shutil.disk_usage(args.output_dir).free/2**30<cfg["minimum_free_disk_gib"]:
        raise RuntimeError("Registered disk floor reached")
    if time.perf_counter()-started>cfg["maximum_gpu_hours"]*3600:
        raise RuntimeError("Registered run budget reached")


def check_source_snapshot(train_dir):
    r=read_json(train_dir/"result.json")
    for name,digest in r["source_sha256"].items():
        checked(ROOT/name,digest); checked(train_dir/"source_snapshot"/Path(name).name,digest)
    return r


def train(args,started):
    cfg,parent,names,dev,label_hashes,rgb,base,freeze=context(args)
    budget(args,cfg,started); source=verify_sources(args.model_dir,parent)
    out=args.output_dir; snapshot=out/"source_snapshot"; snapshot.mkdir()
    paths=[ROOT/p for p in SOURCE_FILES]+[args.config]
    source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in paths}
    for path in paths: shutil.copyfile(path,snapshot/path.name)
    plan=training_plan(names,cfg["seed"],0,cfg["training"]["iterations"],cfg["training"]["crop"])
    write_json(out/"sampling_plan.json",plan)
    counts=np.zeros(5,np.int64); cache={}; access=[]
    for name in names:
        image=png(checked(*rgb[name]))
        target_path=checked(args.artifact_root/cfg["parents"]["labels"]/name,label_hashes[name]); target=png(target_path)
        if image.shape!=(989,1320,3) or target.shape!=(989,1320) or not np.isin(target,(0,1,2,3,4)).all():
            raise ValueError("Native training grid or labels changed")
        counts+=np.bincount(target.reshape(-1),minlength=5); cache[name]=(image,target)
        access.append({"name":name,"label_sha256":label_hashes[name],"rgb_sha256":rgb[name][1]})
    if np.any(counts==0): raise ValueError("Missing training class")
    weights=np.clip(np.median(counts.astype(float))/counts,.05,20).astype(np.float32)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])
    model=load_segmenter(args.model_dir,ARMS[1],width=cfg["training"]["decoder_width"],seed=cfg["seed"]).cuda().train()
    encoder_hash=state_hash(model.backbone); initial_head={"projection":state_hash(model.projection),"decoder":state_hash(model.decoder)}
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=cfg["training"]["learning_rate"],weight_decay=cfg["training"]["weight_decay"])
    cw=torch.from_numpy(weights).cuda(); torch.cuda.reset_peak_memory_stats(); losses=[]; begin=time.perf_counter()
    crop=cfg["training"]["crop"]
    for step,row in enumerate(plan,1):
        image,target=cache[row["name"]]; y,x=row["top"],row["left"]
        image,target=image[y:y+crop,x:x+crop],target[y:y+crop,x:x+crop]
        if row["flip"]: image,target=image[:,::-1],target[:,::-1]
        pixels=normalize_rgb(np.ascontiguousarray(image)).unsqueeze(0).cuda()
        targets=torch.from_numpy(target.copy().astype(np.int64)).unsqueeze(0).cuda()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            logits=F.interpolate(model(pixel_values=pixels).logits,(crop,crop),mode="bilinear",align_corners=False)
            loss=F.cross_entropy(logits.float(),targets,weight=cw)
        if not torch.isfinite(loss): raise FloatingPointError("Nonfinite final loss")
        loss.backward()
        if model.backbone.training or any(p.grad is not None for p in model.backbone.parameters()):
            raise ValueError("Final backbone was unfrozen")
        opt.step(); losses.append(float(loss.detach()))
        if step==1 or step%250==0:
            budget(args,cfg,started)
            print({"stage":"final_training","step":step,"total":len(plan),"loss":losses[-1],"seconds":time.perf_counter()-begin},flush=True)
    torch.cuda.synchronize(); elapsed=time.perf_counter()-begin; peak=torch.cuda.max_memory_allocated()/2**20
    if state_hash(model.backbone)!=encoder_hash: raise ValueError("Frozen encoder changed")
    model.eval()
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16): expected=model(pixel_values=pixels).logits.cpu()
    contract={"experiment_id":EXP,"config_sha256":sha256(args.config),"seed":cfg["seed"],"iterations":len(plan),
              "train":names,"developer_names_excluded":dev,"actual_training_label_access":access,
              "class_counts":counts.tolist(),"class_weights":weights.tolist(),"sampling_plan_sha256":sha256(out/"sampling_plan.json"),
              "backbone_state_sha256":encoder_hash,"initial_head_sha256":initial_head,
              "pretrained_source":source,"training_parameters":sum(p.numel() for p in params),
              "frozen_parameters":sum(p.numel() for p in model.backbone.parameters()),"final_only":True}
    torch.save({"metadata":contract,"head_state_dict":model.head_state()},out/"head.pth")
    write_json(out/"checkpoint_contract.json",contract)
    del model,opt,params,logits,loss,cache; torch.cuda.empty_cache()
    fresh=load_segmenter(args.model_dir,ARMS[1],width=cfg["training"]["decoder_width"],seed=cfg["seed"]).cuda().eval()
    payload=torch.load(out/"head.pth",map_location="cpu",weights_only=True); fresh.load_head(payload["head_state_dict"])
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16): actual=fresh(pixel_values=pixels).logits.cpu()
    if not torch.equal(actual,expected) or state_hash(fresh.backbone)!=encoder_hash: raise ValueError("Final head fresh reload differs")
    del fresh; torch.cuda.empty_cache()
    after=verify_freeze(ROOT/cfg["freeze_manifest"],args.artifact_root)
    if freeze!=after: raise ValueError("Production changed")
    for name in names: checked(*rgb[name])
    return {"experiment_id":EXP,"status":"completed_final_head_sealed","config_sha256":sha256(args.config),
            "source_sha256":source_hashes,"repository_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            "seed":cfg["seed"],"checkpoint_sha256":sha256(out/"head.pth"),"contract_sha256":sha256(out/"checkpoint_contract.json"),
            "training_views":219,"training_steps":len(plan),"training_seconds":elapsed,"peak_training_cuda_allocated_mib":peak,
            "loss":{"first":losses[0],"last":losses[-1],"last100":float(np.mean(losses[-100:]))},
            "fresh_reload_exact":True,"frozen_encoder_exact":True,"freeze_before":freeze,"freeze_after":after,
            "training_rgb_verified":219,"developer_labels_decoded":0,"developer_predictions":0,"environment":environment()}


def predict(args,started):
    cfg,parent,names,dev,labels,rgb,base,freeze=context(args,developer=True)
    check_source_snapshot(args.train_dir); verify_sources(args.model_dir,parent)
    model,record=load_final_head(args.model_dir,args.train_dir,cfg,sha256(args.config))
    groups=segment_groups(read_json(ROOT/cfg["trajectory_segments"]),dev)
    out=args.output_dir; (out/"predictions").mkdir(); (out/"probabilities").mkdir(); records=[]
    for name in dev:
        image=png(checked(*rgb[name])); before=image.copy(); b=png(checked(*base[name]))
        p=sliding(model,image,crop=896,stride=448,classes=5)[4]
        new=review_foundation(b,p)
        if not np.array_equal(before,image) or not np.array_equal(b,review_foundation(b,p,enabled=False)):
            raise ValueError("Prediction RGB/bypass differs")
        if any(not np.array_equal(b==k,new==k) for k in (1,2,3)): raise ValueError("Protected prediction changed")
        pp=out/"probabilities"/(Path(name).stem+".npy"); path=out/"predictions"/name
        np.save(pp,p.astype(np.float32)); Image.fromarray(new).save(path)
        records.append({"name":name,"probability":str(pp.relative_to(out)),"probability_sha256":sha256(pp),
                        "prediction":str(path.relative_to(out)),"prediction_sha256":sha256(path),
                        "rgb_sha256":rgb[name][1],"baseline_sha256":base[name][1],"removed_foundation_pixels":int((b!=new).sum())})
        if len(records)%18==0: print({"stage":"sealed_inference","views":len(records),"labels_decoded":0},flush=True)
        budget(args,cfg,started)
    write_json(out/"prediction_manifest.json",{"predictions":records,"groups":groups,"developer_labels_decoded":0})
    after=verify_freeze(ROOT/cfg["freeze_manifest"],args.artifact_root)
    if freeze!=after: raise ValueError("Frozen production changed")
    for name in dev: checked(*rgb[name]); checked(*base[name])
    checked(args.train_dir/"head.pth",record["checkpoint_sha256"])
    return {"experiment_id":EXP,"status":"all74_predictions_sealed_before_labels","views":74,
            "config_sha256":sha256(args.config),"training_result_sha256":sha256(args.train_dir/"result.json"),
            "checkpoint_sha256":record["checkpoint_sha256"],"prediction_manifest_sha256":sha256(out/"prediction_manifest.json"),
            "freeze_before":freeze,"freeze_after":after,"rgb_unchanged":74,"developer_labels_decoded":0}


def prediction_context(args):
    cfg,parent,names,dev,labels,rgb,base,freeze=context(args,developer=True)
    tr=check_source_snapshot(args.train_dir); r=read_json(args.prediction_dir/"result.json")
    if (r["status"]!="all74_predictions_sealed_before_labels" or r["config_sha256"]!=sha256(args.config)
            or r["training_result_sha256"]!=sha256(args.train_dir/"result.json")):
        raise ValueError("Prediction seal identity changed")
    checked(args.train_dir/"head.pth",tr["checkpoint_sha256"])
    m=read_json(checked(args.prediction_dir/"prediction_manifest.json",r["prediction_manifest_sha256"]))
    if [x["name"] for x in m["predictions"]]!=dev or m["developer_labels_decoded"]!=0:
        raise ValueError("Unexpected prediction coverage/access")
    if m["groups"]!=segment_groups(read_json(ROOT/cfg["trajectory_segments"]),dev):
        raise ValueError("Frozen spatial groups changed")
    return cfg,dev,labels,rgb,base,freeze,r,m


def quality_summaries(views,groups):
    metrics={a:confusion_metrics(sum((np.array(r[a]["confusion_matrix_rows_gt_columns_prediction"]) for r in views),np.zeros((5,5),np.int64))) for a in ("baseline","candidate")}
    by_name={r["name"]:r for r in views}
    grouped={g:{a:confusion_metrics(sum((np.array(by_name[n][a]["confusion_matrix_rows_gt_columns_prediction"]) for n in ns),np.zeros((5,5),np.int64))) for a in metrics} for g,ns in groups.items()}
    worst=sorted(views,key=lambda r:(r["baseline"]["mIoU_with_background"],r["name"]))[:max(1,int(np.ceil(.1*len(views))))]
    subsets={"baseline_worst10":[r["name"] for r in worst],"challenge_140_146":[r["name"] for r in views if 140<=int(Path(r["name"]).stem)<=146]}
    if len(subsets["challenge_140_146"])!=7: raise ValueError("Missing challenge views")
    sub={k:{"names":ns,**{a+"_mean_miou":float(np.mean([by_name[n][a]["mIoU_with_background"] for n in ns])) for a in metrics}} for k,ns in subsets.items()}
    return metrics,grouped,sub


def score(args,started):
    cfg,dev,labels,rgb,base,freeze,pred,manifest=prediction_context(args)
    audit=read_json(args.audit_dir/"result.json")
    require_engineering(audit,sha256(args.prediction_dir/"result.json"),sha256(args.train_dir/"result.json"))
    views=[]; access=[]; changes={"corrected":0,"harmed":0,"neutral":0,"removed_foundation":0}
    for rec in manifest["predictions"]:
        name=rec["name"]
        target_path=checked(args.artifact_root/cfg["parents"]["labels"]/name,labels[name]); gt=png(target_path)
        b=png(checked(*base[name])); new=png(checked(args.prediction_dir/rec["prediction"],rec["prediction_sha256"]))
        changed=b!=new
        if np.any(changed&((b!=4)|(new!=0))): raise ValueError("Unauthorized quality transition")
        row={"name":name}
        for arm,image in (("baseline",b),("candidate",new)):
            cm=np.bincount((gt.astype(np.int64)*5+image).reshape(-1),minlength=25).reshape(5,5)
            row[arm]=confusion_metrics(cm)
        good=changed&(new==gt); bad=changed&(b==gt)
        for k,v in (("corrected",good),("harmed",bad),("neutral",changed&~good&~bad),("removed_foundation",changed)):
            changes[k]+=int(v.sum())
        views.append(row); access.append({"name":name,"label_sha256":labels[name]})
    metrics,groups,subsets=quality_summaries(views,manifest["groups"])
    prior=read_json(args.artifact_root/cfg["parents"]["baseline"]/"result.json")["metrics"]
    if metrics["baseline"]["confusion_matrix_rows_gt_columns_prediction"]!=prior["confusion_matrix_rows_gt_columns_prediction"]:
        raise ValueError("Baseline recomputation differs from frozen SEM161")
    gates=final_gates(metrics,groups,subsets,cfg["developer_gates"]); gates["protected_exact"]=True
    after=verify_freeze(ROOT/cfg["freeze_manifest"],args.artifact_root)
    if freeze!=after: raise ValueError("Frozen production changed")
    return {"experiment_id":EXP,"status":"completed_developer_gate_passed" if all(gates.values()) else "completed_developer_gate_failed",
            "passed":all(gates.values()),"checks":gates,"metrics":metrics,"group_metrics":groups,"subsets":subsets,
            "per_view":views,"changes":changes,"target_access":access,"views":len(dev),"developer_only":True,
            "config_sha256":sha256(args.config),"training_result_sha256":sha256(args.train_dir/"result.json"),
            "prediction_result_sha256":sha256(args.prediction_dir/"result.json"),"engineering_audit_sha256":sha256(args.audit_dir/"result.json"),
            "freeze_after":after,"production_overwritten":False,"claim_boundary":"Historically reused74 developer audit, one seed, not blind or official hidden evaluation"}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--mode",choices=("train","predict","score"),required=True)
    for k in ("artifact-root","output-dir"): p.add_argument("--"+k,type=Path,required=True)
    for k in ("model-dir","train-dir","prediction-dir","audit-dir"): p.add_argument("--"+k,type=Path)
    p.add_argument("--config",type=Path,default=CONFIG); args=p.parse_args()
    args.output_dir=args.output_dir.resolve(); args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=False
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar(); started=time.perf_counter()
    try:
        result={"train":train,"predict":predict,"score":score}[args.mode](args,started)
        result["seconds"]=time.perf_counter()-started; write_json(args.output_dir/"result.json",result)
        print({k:result[k] for k in ("status","seconds","passed","checks") if k in result},flush=True)
    except Exception as exc:
        write_json(args.output_dir/"failure.json",{"error_type":type(exc).__name__,"error":str(exc),"seconds":time.perf_counter()-started})
        raise


if __name__=="__main__": main()
