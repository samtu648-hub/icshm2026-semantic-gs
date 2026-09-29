"""Pure final-review split/quality contracts; learned model import is lazy."""
from __future__ import annotations
import numpy as np

EXP = "SEM-314-DINOV2-FOUNDATION-REVIEW-FINAL"


def validate_final_split(split):
    train=split["semantic_train_labeled"]; dev=split["semantic_validation_labeled"]
    guard=split["buffer_excluded_labeled"]
    if len(train)!=219 or len(dev)!=74 or len(guard)!=7:
        raise ValueError("Final split count changed")
    groups=[set(train),set(dev),set(guard),set(split["excluded_unlabeled"])]
    if any(len(g)!=len(v) for g,v in zip(groups,(train,dev,guard,split["excluded_unlabeled"]))):
        raise ValueError("Duplicate split name")
    if any(groups[i]&groups[j] for i in range(4) for j in range(i)):
        raise ValueError("Training/developer/guard/unlabeled overlap")
    return sorted(train),sorted(dev)


def segment_groups(segments,names):
    targets=set(names)
    groups={s["segment_id"]:sorted(targets&set(s["images"])) for s in segments["segments"]}
    groups={k:v for k,v in groups.items() if v}
    combined=[n for v in groups.values() for n in v]
    if len(combined)!=len(names) or set(combined)!=targets:
        raise ValueError("Segment groups do not partition developer views")
    return groups


def final_gates(metrics,groups,subsets,cfg):
    b=metrics["baseline"]; c=metrics["candidate"]
    bf=b["per_class"]["foundation"]; cf=c["per_class"]["foundation"]
    checks={"miou_gain":c["mIoU_with_background"]>=b["mIoU_with_background"]+cfg["miou_gain"],
            "foundation_iou_gain":cf["iou"]>=bf["iou"]+cfg["foundation_iou_gain"],
            "recall_retention":cf["recall"]>=bf["recall"]-cfg["maximum_recall_drop"],
            "recall_floor":cf["recall"]>=cfg["recall_floor"]}
    for name,row in groups.items():
        bb=row["baseline"]; cc=row["candidate"]
        bi=bb["per_class"]["foundation"]["iou"]; ci=cc["per_class"]["foundation"]["iou"]
        checks[name+"_miou_safety"]=cc["mIoU_with_background"]>=bb["mIoU_with_background"]-cfg["maximum_group_miou_drop"]
        if bi is not None or ci is not None:
            checks[name+"_foundation_safety"]=(bi is not None and ci is not None
                and ci>=bi-cfg["maximum_group_foundation_iou_drop"])
    for name,row in subsets.items():
        checks[name+"_miou_safety"]=row["candidate_mean_miou"]>=row["baseline_mean_miou"]-cfg["maximum_difficult_miou_drop"]
    return checks


def require_engineering(audit, prediction_hash, training_hash):
    if (audit.get("status")!="passed_final_review_engineering"
            or audit.get("prediction_result_sha256")!=prediction_hash
            or audit.get("training_result_sha256")!=training_hash
            or audit.get("predictions_checked")!=74 or len(audit.get("native_replays",[]))!=2
            or len(audit.get("external_probes",[]))!=3):
        raise ValueError("Final-head engineering gate required before developer labels")


def load_final_head(model_dir,train_dir,cfg,config_hash):
    import torch
    from src.semantic2d import read_json,sha256
    from src.semantic_frozen_dinov2 import load_segmenter,state_hash,ARMS
    record=read_json(train_dir/"result.json")
    if record["status"]!="completed_final_head_sealed" or record["config_sha256"]!=config_hash:
        raise ValueError("Final training identity differs")
    path=train_dir/"head.pth"
    if sha256(path)!=record["checkpoint_sha256"] or sha256(train_dir/"checkpoint_contract.json")!=record["contract_sha256"]:
        raise ValueError("Final head/contract changed")
    contract=read_json(train_dir/"checkpoint_contract.json")
    payload=torch.load(path,map_location="cpu",weights_only=True)
    if payload["metadata"]!=contract or contract["experiment_id"]!=EXP or contract["config_sha256"]!=config_hash:
        raise ValueError("Final checkpoint metadata differs")
    model=load_segmenter(model_dir,ARMS[1],width=cfg["training"]["decoder_width"],seed=cfg["seed"]).cuda().eval()
    model.load_head(payload["head_state_dict"]); model.requires_grad_(False)
    if state_hash(model.backbone)!=contract["backbone_state_sha256"]:
        raise ValueError("Frozen final backbone differs")
    return model,record
