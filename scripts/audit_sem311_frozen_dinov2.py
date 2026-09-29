#!/usr/bin/env python3
"""Independent-process SEM-311 artifact replay and optional official-camera probes.

No target label is decoded. Audit results are engineering-only; quality is recorded separately.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from PIL import Image
import torch

from audit_sem260_error_budget import manifest_paths, png
from run_sem250_colmap_track_deck_diag import verify_freeze
from run_sem311_frozen_dinov2 import initialize, verify_sources, arrays_equal
from src.semantic2d import read_json, sha256, write_json
from src.semantic_foundation_binary import training_plan, fuse_foundation, validate_fold
from src.semantic_frozen_dinov2 import ARMS, state_hash
from src.submission_sem161 import sliding


def load_head(args, cfg, run, fold, arm):
    record = run["folds"][fold]["arms"][arm]["training"]
    path = args.run_dir / f"fold{fold}_{arm}/head.pth"
    if sha256(path) != record["sha256"]:
        raise ValueError("Trained checkpoint changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    if (metadata != record["metadata"] or metadata["fold"] != fold
            or metadata["arm"] != arm or metadata["config_sha256"] != sha256(args.config)):
        raise ValueError("Checkpoint metadata mismatch")
    model = initialize(args, cfg, arm, False).cuda().eval()
    model.load_head(payload["head_state_dict"])
    model.requires_grad_(False)
    if state_hash(model.backbone) != metadata["backbone_state_sha256"]:
        raise ValueError("Immutable encoder fingerprint mismatch")
    return model, record


def audit(args, out):
    cfg, run = read_json(args.config), read_json(args.run_dir / "result.json")
    if run["mode"] not in ("smoke", "full"):
        raise ValueError("No trained run to audit")
    if run["config_sha256"] != sha256(args.config):
        raise ValueError("Config changed")
    for name, digest in run["source_sha256"].items():
        if sha256(ROOT/name) != digest or sha256(args.run_dir/"source_snapshot"/Path(name).name) != digest:
            raise ValueError("Run source fingerprint changed")
    seal = args.run_dir / "prediction_seal.json"
    if sha256(seal) != run["prediction_seal_sha256"]:
        raise ValueError("Prediction seal changed")
    sealed = read_json(seal)
    if (sealed["config_sha256"] != run["config_sha256"]
            or sealed["source_sha256"] != run["source_sha256"]
            or sealed["folds"] != run["folds"]
            or sha256(args.run_dir/"prediction_manifest.json") != sealed["prediction_manifest_sha256"]):
        raise ValueError("Seal contents mismatch")
    art = args.artifact_root
    freeze = verify_freeze(ROOT/"experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json", art)
    if freeze != run["preflight"]["freeze"] or freeze != run["freeze_after"]:
        raise ValueError("Frozen production source changed")
    for name, digest in run["preflight"]["control_source"].items():
        if sha256(art/cfg["parents"]["model_config"]/name) != digest:
            raise ValueError("Control model source changed")
    if run["mode"] != "prepare":
        verify_sources(args.model_dir, cfg)
    names = read_json(ROOT/cfg["semantic_split"])["semantic_train_labeled"]
    split = read_json(ROOT/cfg["oof_split"])
    centers = {r["name"]:r["center"] for r in read_json(ROOT/cfg["camera_graph"])["nodes"]}
    for key in ("semantic_split", "oof_split"):
        if sha256(ROOT/cfg[key]) != cfg[key+"_sha256"]:
            raise ValueError("Split changed")
    if sha256(ROOT/cfg["camera_graph"]) != split["camera_graph_sha256"]:
        raise ValueError("Camera graph changed")
    rgb = manifest_paths(art/cfg["parents"]["rgb"],"rendered_dataset_manifest.json","rendered_images","path","sha256")
    base = manifest_paths(art/cfg["parents"]["sem160"],"prediction_manifest.json","predictions","prediction","prediction_sha256")
    for mapping in (rgb, base):
        if set(mapping) != set(names) or any(sha256(p)!=h for p,h in mapping.values()):
            raise ValueError("Frozen dataset inputs changed")
    pred = read_json(args.run_dir/"prediction_manifest.json")
    small = run["mode"] != "full"
    selected = split["folds"][:1] if small else split["folds"]
    expected = {n for f in selected for n in (sorted(f["oof"])[:1] if small else f["oof"])}
    if set(pred) != set(ARMS) or any(set(pred[a])!=expected for a in ARMS):
        raise ValueError("Unexpected prediction set")
    checked, replays = [], []
    for fold in selected:
        fid = fold["fold"]
        validate_fold(fold,names,centers,split["guard_policy"]["radius"])
        plan = training_plan(sorted(fold["train"]),cfg["seed"],fid,2 if small else cfg["training"]["iterations"],cfg["training"]["crop"])
        plan_path = args.run_dir/f"fold{fid}_sampling_plan.json"
        if plan != read_json(plan_path):
            raise ValueError("Registered train sampling changed")
        metas = [run["folds"][fid]["arms"][a]["training"]["metadata"] for a in ARMS]
        for key in ("common_decoder_initial_sha256","actual_label_access","class_weights","class_counts","sampling_plan_sha256"):
            if metas[0][key] != metas[1][key]:
                raise ValueError("Paired training differs")
        train_names = sorted({r["name"] for r in plan}) if small else sorted(fold["train"])
        for meta in metas:
            if (meta["train"] != train_names or meta["outer_train"] != fold["train"]
                    or meta["oof"] != fold["oof"] or meta["guard"] != fold["guard"]
                    or meta["sampling_plan_sha256"] != sha256(plan_path)
                    or meta["seed"] != cfg["seed"] or meta["iterations"] != len(plan)
                    or [r["name"] for r in meta["actual_label_access"]] != train_names):
                raise ValueError("Training isolation metadata mismatch")
            for r in meta["actual_label_access"]:
                if sha256(art/cfg["parents"]["labels"]/r["name"]) != r["label_sha256"] or rgb[r["name"]][1] != r["rgb_sha256"]:
                    raise ValueError("Training source changed")
        for arm in ARMS:
            held = sorted(set(fold["oof"]) & expected)
            for name in held:
                r = pred[arm][name]
                if (r["fold"] != fid or sha256(Path(r["path"])) != r["sha256"]
                        or sha256(Path(r["probability_path"])) != r["probability_sha256"]):
                    raise ValueError("Prediction provenance changed")
                p = np.load(r["probability_path"],allow_pickle=False)
                b, new = png(base[name][0]), png(Path(r["path"]))
                if p.shape != b.shape or p.dtype != np.float32 or not np.isfinite(p).all() or p.min()<0 or p.max()>1:
                    raise ValueError("Invalid probability tensor")
                arrays_equal(b,new,np.zeros(1),np.zeros(1))
                if not np.array_equal(new,fuse_foundation(b,p)) or not np.array_equal(b,fuse_foundation(b,p,strength=0)):
                    raise ValueError("Saved probability/fusion replay mismatch")
                checked.append({"arm":arm,"fold":fid,"name":name})
            model, record = load_head(args,cfg,run,fid,arm)
            name = held[0]; before = png(rgb[name][0]); image = before.copy()
            torch.cuda.reset_peak_memory_stats()
            actual = sliding(model,image,crop=cfg["inference"]["crop"],stride=cfg["inference"]["stride"],classes=5)[4]
            if not np.array_equal(actual,np.load(pred[arm][name]["probability_path"],allow_pickle=False)) or not np.array_equal(image,before):
                raise ValueError("Independent native inference not exact")
            replays.append({"fold":fid,"arm":arm,"name":name,"exact":True,
                            "peak_cuda_allocated_mib":torch.cuda.max_memory_allocated()/2**20})
            del model; torch.cuda.empty_cache()
    external = []
    if args.bundle:
        env = dict(os.environ,PYTHONDONTWRITEBYTECODE="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
        for size, cameras in (("native",args.data_root/"camera_parameters/cameras.txt"),("small",args.reference_dir/"small_cameras.txt")):
            render = out/size
            subprocess.run([str(args.render_python),str(ROOT/"scripts/render_submission_sem161.py"),"--bundle",str(args.bundle),
                "--cameras",str(cameras),"--images",str(args.data_root/"camera_parameters/images.txt"),
                "--image-name","140.png","--output-dir",str(render)],cwd=ROOT,env=env,check=True)
            reference = args.reference_dir/("reload1" if size=="native" else "small")
            image,b = png(render/"rgb/140.png"),png(render/"semantic/140.png")
            if not np.array_equal(image,png(reference/"rgb/140.png")) or not np.array_equal(b,png(reference/"semantic/140.png")):
                raise ValueError("Frozen external camera replay changed")
            for arm in ARMS:
                model,record = load_head(args,cfg,run,0,arm)
                before = image.copy(); torch.cuda.reset_peak_memory_stats()
                p = sliding(model,image,crop=cfg["inference"]["crop"],stride=cfg["inference"]["stride"],classes=5)[4]
                new = fuse_foundation(b,p); arrays_equal(b,new,image,before)
                if not np.array_equal(b,fuse_foundation(b,p,strength=0)):
                    raise ValueError("External disabled branch differs")
                path = render/f"{arm}_140.png"; Image.fromarray(new).save(path)
                external.append({"size":size,"shape":list(b.shape),"arm":arm,"camera":"140.png",
                    "pose":"COLMAP_CAMERA_FROM_WORLD","camera_model":"SIMPLE_RADIAL",
                    "frozen_rgb_and_semantic_replay_exact":True,"protected_exact":True,"rgb_input_unchanged":True,
                    "disabled_exact":True,"checkpoint_sha256":record["sha256"],"prediction_sha256":sha256(path),
                    "rgb_sha256":sha256(render/"rgb/140.png"),"camera_file_sha256":sha256(cameras),
                    "peak_cuda_allocated_mib":torch.cuda.max_memory_allocated()/2**20})
                del model; torch.cuda.empty_cache()
    after = verify_freeze(ROOT/"experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json",art)
    if freeze != after:
        raise ValueError("Frozen artifact changed during audit")
    return {"experiment_id":cfg["experiment_id"],"status":"passed_engineering_only",
        "run_mode":run["mode"],"quality_claim":False,"target_labels_decoded":0,
        "target_real_rgb_used":False,"run_result_sha256":sha256(args.run_dir/"result.json"),
        "frozen_artifacts_verified":freeze["count"],"predictions_checked":checked,
        "independent_native_replays":replays,"external_probes":external,
        "audit_code_sha256":sha256(Path(__file__)),"renderer_code_sha256":sha256(ROOT/"scripts/render_submission_sem161.py")}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("artifact-root","model-dir","run-dir","output-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ("bundle","data-root","reference-dir","render-python"):
        parser.add_argument("--"+name,type=Path)
    parser.add_argument("--config",type=Path,default=ROOT/"experiments/runs/SEM-311-DINOV2-FROZEN-FEATURE-OOF/config.yaml")
    args=parser.parse_args()
    optional=(args.bundle,args.data_root,args.reference_dir,args.render_python)
    if any(optional) and not all(optional):
        parser.error("External probes require bundle, data-root, reference-dir and render-python together")
    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=False
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
    started=time.perf_counter()
    try:
        result=audit(args,out); result["seconds"]=time.perf_counter()-started
        write_json(out/"result.json",result)
        print({"status":result["status"],"native_replays":len(result["independent_native_replays"]),
               "external_probes":len(result["external_probes"]),"seconds":result["seconds"]},flush=True)
    except Exception as exc:
        write_json(out/"failure.json",{"error_type":type(exc).__name__,"error":str(exc),"seconds":time.perf_counter()-started})
        raise


if __name__=="__main__":
    main()
