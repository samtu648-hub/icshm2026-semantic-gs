#!/usr/bin/env python3
"""Prepare official weights, smoke, train and audit the SEM-311 paired experiment.

All paths come from the CLI/config. Every invocation writes a new output folder.
Only official public pretrained weights are accepted; no random fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from audit_sem260_error_budget import manifest_paths, png
from run_sem250_colmap_track_deck_diag import verify_freeze
from src.semantic2d import read_json, write_json, sha256, confusion_metrics
from src.semantic_cable_binary import normalize_rgb
from src.semantic_foundation_binary import training_plan, validate_fold, fuse_foundation
from src.semantic_frozen_dinov2 import ARMS, FrozenFeatureSegmenter, load_segmenter, state_hash, quality_gates
from src.semantic_protected_fusion import partition_training_names, local_confusion
from src.semantic_render_domain import semantic_rgb
from src.submission_sem161 import sliding

EXP = "SEM-311-DINOV2-FROZEN-FEATURE-OOF"
CONFIG = ROOT / "experiments/runs" / EXP / "config.yaml"
SOURCE_FILES = [Path(__file__), ROOT / "src/semantic_frozen_dinov2.py",
                ROOT / "src/semantic_foundation_binary.py", ROOT / "src/semantic_cable_binary.py",
                ROOT / "src/submission_sem161.py", ROOT / "src/semantic_protected_fusion.py",
                ROOT / "src/semantic2d.py", ROOT / "src/semantic_render_domain.py",
                ROOT / "scripts/audit_sem260_error_budget.py",
                ROOT / "scripts/run_sem250_colmap_track_deck_diag.py"]


def log(**values):
    print(json.dumps(values), flush=True)


def arrays_equal(base, new, rgb, before):
    if not np.array_equal(rgb, before):
        raise ValueError("RGB array mutated")
    if any(not np.array_equal(base == k, new == k) for k in (1, 2, 3)):
        raise ValueError("Protected class mask changed")


def environment():
    import transformers
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def verify_sources(model_dir, cfg):
    manifest = read_json(model_dir / "official_source_manifest.json")
    if (manifest["model_id"] != cfg["dinov2"]["model_id"]
            or manifest["revision"] != cfg["dinov2"]["revision"]):
        raise ValueError("Wrong official model source")
    if set(manifest["files"]) != set(cfg["dinov2"]["required_files"]):
        raise ValueError("Incomplete official source manifest")
    for name, record in manifest["files"].items():
        if sha256(model_dir / name) != record["sha256"]:
            raise ValueError(f"Model source file changed: {name}")
    return manifest


def prepare_sources(model_dir, cfg):
    """Use only the official public Hub repository, without authentication.

    Error text/token values are deliberately not recorded. No mirror fallback.
    Existing local files may be verified against official git/LFS fingerprints.
    """
    from huggingface_hub import HfApi, hf_hub_download
    expected = cfg["dinov2"]
    info = HfApi(token=False).model_info(expected["model_id"], revision=expected["revision"], files_metadata=True)
    if info.gated:
        raise ValueError("SEM-311 requires an ungated official model")
    if info.sha != expected["revision"]:
        raise ValueError("Official revision changed")
    metadata = {f.rfilename: f for f in info.siblings}
    records = {}
    for name in expected["required_files"]:
        path = model_dir / name
        if not path.is_file():
            try:
                hf_hub_download(expected["model_id"], name, revision=expected["revision"], local_dir=model_dir, token=False)
            except Exception as exc:
                return {"status": "blocked_official_source_download", "model_id": info.id,
                        "revision": info.sha, "gated": info.gated,
                        "token_used": False, "error_type": type(exc).__name__,
                        "http_status": getattr(getattr(exc, "response", None), "status_code", None),
                        "missing_file": name, "files_verified": records,
                        "quality_training_started": False, "quality_labels_read": 0}
        record = metadata[name]
        digest = sha256(path)
        if name == "model.safetensors" and digest != expected["weight_sha256"]:
            raise ValueError("Registered DINOv2 weight checksum mismatch")
        if record.lfs:
            if digest != record.lfs.sha256:
                raise ValueError(f"Official LFS checksum mismatch: {name}")
        else:
            blob = hashlib.sha1()
            blob.update(f"blob {path.stat().st_size}\0".encode())
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    blob.update(chunk)
            if blob.hexdigest() != record.blob_id:
                raise ValueError(f"Official git blob mismatch: {name}")
        records[name] = {"sha256": digest, "bytes": path.stat().st_size,
                         "official_git_blob": record.blob_id,
                         "official_lfs_sha256": record.lfs.sha256 if record.lfs else None}
    manifest = {"model_id": info.id, "revision": info.sha, "files": records,
                "source": expected["source"], "license": expected["license"]}
    write_json(model_dir / "official_source_manifest.json", manifest)
    return {"status": "official_sources_verified", "source": manifest,
            "quality_training_started": False, "quality_labels_read": 0}


def context(args, cfg, out):
    art = args.artifact_root.resolve()
    freeze = verify_freeze(ROOT / "experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json", art)
    for key in ("semantic_split", "oof_split"):
        if sha256(ROOT / cfg[key]) != cfg[key + "_sha256"]:
            raise ValueError("Split fingerprint changed")
    names = read_json(ROOT / cfg["semantic_split"])["semantic_train_labeled"]
    split = read_json(ROOT / cfg["oof_split"])
    graph = ROOT / cfg["camera_graph"]
    if sha256(graph) != split["camera_graph_sha256"]:
        raise ValueError("Camera graph changed")
    centers = {r["name"]: r["center"] for r in read_json(graph)["nodes"]}
    held = [n for fold in split["folds"] for n in fold["oof"]]
    if len(held) != len(set(held)) or set(held) != set(names):
        raise ValueError("OOF coverage changed")
    for fold in split["folds"]:
        validate_fold(fold, names, centers, split["guard_policy"]["radius"])
    rgb = manifest_paths(art / cfg["parents"]["rgb"], "rendered_dataset_manifest.json",
                         "rendered_images", "path", "sha256")
    baseline = manifest_paths(art / cfg["parents"]["sem160"], "prediction_manifest.json",
                              "predictions", "prediction", "prediction_sha256")
    for mapping in (rgb, baseline):
        if set(mapping) != set(names) or any(sha256(p) != h for p, h in mapping.values()):
            raise ValueError("Frozen RGB/baseline image manifest changed")
    model_dir = art / cfg["parents"]["model_config"]
    control_source = {p.name: sha256(p) for p in model_dir.iterdir()
                      if p.suffix in (".json", ".bin", ".safetensors")}
    if not any(n.endswith((".bin", ".safetensors")) for n in control_source):
        raise ValueError("Control pretrained weights unavailable")
    evidence = {"freeze": freeze, "semantic_views": len(names), "folds": len(split["folds"]),
                "rgb_files_verified": len(rgb), "baseline_files_verified": len(baseline),
                "control_source": control_source, "environment": environment(),
                "dataset_manifest_sha256": sha256(ROOT / cfg["dataset_manifest"]),
                "free_disk_gib": shutil.disk_usage(out).free / 2**30}
    write_json(out / "preflight.json", evidence)
    return art, split, names, rgb, baseline, evidence


def initialize(args, cfg, arm, architecture_only=False):
    kwargs = {"width": cfg["training"]["decoder_width"], "seed": cfg["seed"]}
    if architecture_only:
        raise ValueError("SEM-311 forbids random candidate initialization")
    path = (args.artifact_root / cfg["parents"]["model_config"] if arm == ARMS[0] else args.model_dir)
    return load_segmenter(path, arm, **kwargs)


def resource_gate(out, cfg, started):
    if shutil.disk_usage(out).free < cfg["minimum_free_disk_gib"] * 2**30:
        raise RuntimeError("Registered disk floor reached")
    if time.perf_counter() - started > cfg["maximum_gpu_hours"] * 3600:
        raise RuntimeError("Registered wall-clock budget reached")


def fit(args, cfg, out, fold, arm, plan, rgb, labels, started):
    folder = out / f"fold{fold['fold']}_{arm}"
    folder.mkdir()
    small = args.mode != "full"
    names = sorted({r["name"] for r in plan}) if small else sorted(fold["train"])
    counts, cache, access = np.zeros(5, np.int64), {}, []
    for name in names:
        if name not in fold["train"] or name in fold["oof"] or name in fold["guard"]:
            raise ValueError("Forbidden training label")
        image, target = png(rgb[name][0]), png(labels / name)
        if image.shape != (989, 1320, 3) or target.shape != (989, 1320) or target.max() > 4:
            raise ValueError("Native RGB/label grid or class IDs changed")
        counts += np.bincount(target.ravel(), minlength=5)
        cache[name] = (image, target)
        access.append({"name": name, "label_sha256": sha256(labels / name), "rgb_sha256": rgb[name][1]})
    if np.any(counts == 0):
        raise ValueError("Training sample misses an official class")
    weights = np.clip(np.median(counts.astype(float)) / counts, .05, 20).astype(np.float32)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    model = initialize(args, cfg, arm, False).cuda().train()
    backbone_hash = state_hash(model.backbone)
    common_hash = state_hash(model.decoder)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    class_weights = torch.from_numpy(weights).cuda()
    torch.cuda.reset_peak_memory_stats()
    begin, losses = time.perf_counter(), []
    crop = cfg["training"]["crop"]
    for step, row in enumerate(plan, 1):
        image, target = cache[row["name"]]
        y, x = row["top"], row["left"]
        image, target = image[y:y+crop, x:x+crop], target[y:y+crop, x:x+crop]
        if row["flip"]:
            image, target = image[:, ::-1], target[:, ::-1]
        pixels = normalize_rgb(np.ascontiguousarray(image)).unsqueeze(0).cuda()
        targets = torch.from_numpy(target.copy().astype(np.int64)).unsqueeze(0).cuda()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = F.interpolate(model(pixel_values=pixels).logits, (crop, crop), mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits.float(), targets, weight=class_weights)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        if any(p.grad is not None for p in model.backbone.parameters()) or model.backbone.training:
            raise ValueError("Backbone unfreezing detected")
        opt.step(); losses.append(float(loss.detach()))
        if step == 1 or step % 250 == 0 or step == len(plan):
            resource_gate(out, cfg, started)
            log(stage="training", mode=args.mode, fold=fold["fold"], arm=arm, step=step,
                total=len(plan), loss=losses[-1], seconds=time.perf_counter()-begin)
    torch.cuda.synchronize()
    seconds, peak = time.perf_counter()-begin, torch.cuda.max_memory_allocated()/2**20
    if state_hash(model.backbone) != backbone_hash:
        raise ValueError("Frozen encoder parameters/buffers changed")
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = model(pixel_values=pixels).logits.cpu()
    meta = {"experiment_id": EXP, "arm": arm, "fold": fold["fold"], "mode": args.mode,
            "seed": cfg["seed"], "iterations": len(plan), "config_sha256": cfg["_sha256"],
            "backbone_state_sha256": backbone_hash, "common_decoder_initial_sha256": common_hash,
            "train": names, "outer_train": fold["train"], "oof": fold["oof"], "guard": fold["guard"],
            "actual_label_access": access, "class_counts": counts.tolist(), "class_weights": weights.tolist(),
            "sampling_plan_sha256": sha256(out / f"fold{fold['fold']}_sampling_plan.json"),
            "trainable_parameters": sum(p.numel() for p in params),
            "frozen_parameters": sum(p.numel() for p in model.backbone.parameters()),
            "candidate_pretrained": True}
    path = folder / "head.pth"
    torch.save({"metadata": meta, "head_state_dict": model.head_state()}, path)
    del model, opt, params, logits, loss, cache
    torch.cuda.empty_cache()
    fresh = initialize(args, cfg, arm).cuda().eval()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    fresh.load_head(payload["head_state_dict"])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        actual = fresh(pixel_values=pixels).logits.cpu()
    if not torch.equal(expected, actual) or state_hash(fresh.backbone) != backbone_hash:
        raise ValueError("Fresh checkpoint/encoder reload mismatch")
    result = {"metadata": meta, "sha256": sha256(path), "reload_exact": True,
              "backbone_unchanged": True, "seconds": seconds, "peak_cuda_allocated_mib": peak,
              "loss": {"first": losses[0], "last": losses[-1], "last100": float(np.mean(losses[-100:]))}}
    write_json(folder / "training_result.json", result)
    return fresh, result


def evaluate(names, predictions, baseline, labels, cfg):
    keys = ("baseline", *ARMS)
    matrices = {k: np.zeros((5,5), np.int64) for k in keys}
    folds = {str(i): {k: np.zeros((5,5), np.int64) for k in keys} for i in range(3)}
    rows, access = [], []
    changed = {a: {"corrected":0, "harmed":0, "added_foundation":0, "removed_foundation":0} for a in ARMS}
    for name in names:
        target, base = png(labels/name), png(baseline[name][0])
        fold = str(predictions[ARMS[0]][name]["fold"])
        cm = local_confusion(target, base)
        matrices["baseline"] += cm; folds[fold]["baseline"] += cm
        row = {"name":name, "fold":fold, "baseline":confusion_metrics(cm)}
        for arm in ARMS:
            record = predictions[arm][name]
            if sha256(Path(record["path"])) != record["sha256"]:
                raise ValueError("Sealed prediction changed")
            new = png(Path(record["path"]))
            arrays_equal(base, new, np.zeros(1), np.zeros(1))
            cm = local_confusion(target, new)
            matrices[arm] += cm; folds[fold][arm] += cm
            row[arm] = confusion_metrics(cm)
            for key, mask in (("corrected", (base!=new)&(new==target)),
                              ("harmed", (base!=new)&(base==target)),
                              ("added_foundation", (base==0)&(new==4)),
                              ("removed_foundation", (base==4)&(new==0))):
                changed[arm][key] += int(mask.sum())
        rows.append(row); access.append({"name":name,"label_sha256":sha256(labels/name)})
    metrics = {k:confusion_metrics(v) for k,v in matrices.items()}
    fold_metrics = {f:{k:confusion_metrics(v) for k,v in row.items()} for f,row in folds.items()}
    gates = quality_gates(metrics, fold_metrics, cfg["gates"])
    gates["protected_exact"] = True
    return {"metrics":metrics,"fold_metrics":fold_metrics,"checks":gates,"passed":all(gates.values()),
            "per_view":rows,"target_access":access,"changes":changed}


def panels(names, rgb, baseline, labels, predictions, out):
    out.mkdir()
    records = []
    for name in names:
        image, target, base = png(rgb[name][0]), png(labels/name), png(baseline[name][0])
        control, candidate = [png(Path(predictions[a][name]["path"])) for a in ARMS]
        change = np.zeros_like(image)
        change[(candidate!=base)&(candidate==target)] = (40,220,80)
        change[(candidate!=base)&(base==target)] = (255,50,50)
        columns = [image, semantic_rgb(target), semantic_rgb(base), semantic_rgb(control), semantic_rgb(candidate), change]
        titles = ["Rendered RGB","GT","SEM160","Frozen MiT control","Frozen DINOv2","Corrected green / harmed red"]
        ys,xs = np.where((target==4)|(base==4)|(candidate==4))
        if not len(xs):
            continue
        box = (max(0,int(xs.min())-16), max(0,int(ys.min())-16),
               min(image.shape[1],int(xs.max())+17), min(image.shape[0],int(ys.max())+17))
        x0,y0,x1,y1 = box; w,h = x1-x0,y1-y0
        panel = Image.new("RGB",(w*6,h+24),"white"); draw = ImageDraw.Draw(panel)
        for i,(column,title) in enumerate(zip(columns,titles)):
            panel.paste(Image.fromarray(column[y0:y1,x0:x1]),(i*w,24))
            draw.text((i*w+3,4),title,fill="black")
        path = out/name; panel.save(path)
        records.append({"name":name,"sha256":sha256(path),"crop_xyxy":box})
    write_json(out/"manifest.json",records)
    return records


def run(args, cfg, out, started):
    torch.set_num_threads(4)
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
    if cfg["experiment_id"] != EXP or tuple(cfg["arm_names"]) != ARMS:
        raise ValueError("Wrong experiment config")
    sources = {str(p.relative_to(ROOT)):sha256(p) for p in SOURCE_FILES}
    snapshot = out/"source_snapshot"; snapshot.mkdir()
    for path in (*SOURCE_FILES,args.config):
        shutil.copyfile(path,snapshot/path.name)
    art, split, names, rgb, baseline, evidence = context(args,cfg,out)
    resource_gate(out,cfg,started)
    result = {"experiment_id":EXP,"mode":args.mode,"seed":cfg["seed"],
              "config_sha256":cfg["_sha256"],"source_sha256":sources,"preflight":evidence,
              "repository_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
              "check_quality_scored_this_run":False,"developer_quality_scored_this_run":False}
    if args.mode == "prepare":
        args.model_dir.mkdir(parents=True,exist_ok=True)
        result.update(prepare_sources(args.model_dir,cfg))
        return result
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Visible bf16 CUDA is required")
    torch.backends.cudnn.benchmark = False
    if args.mode != "prepare":
        result["dinov2_source"] = verify_sources(args.model_dir,cfg)
    labels = art/cfg["parents"]["labels"]
    predictions = {a:{} for a in ARMS}; foldrows=[]
    small = args.mode != "full"
    for fold in split["folds"][:1] if small else split["folds"]:
        plan = training_plan(sorted(fold["train"]),cfg["seed"],fold["fold"],
                             2 if small else cfg["training"]["iterations"],cfg["training"]["crop"])
        write_json(out/f"fold{fold['fold']}_sampling_plan.json",plan)
        row={"fold":fold["fold"],"arms":{}}
        for arm in ARMS:
            model,training = fit(args,cfg,out,fold,arm,plan,rgb,labels,started)
            begin=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
            targets=sorted(fold["oof"])[:1] if small else sorted(fold["oof"])
            directory=out/arm; directory.mkdir(exist_ok=True)
            for index,name in enumerate(targets):
                resource_gate(out,cfg,started)
                image,base=png(rgb[name][0]),png(baseline[name][0]); before=image.copy()
                probability=sliding(model,image,crop=cfg["inference"]["crop"],stride=cfg["inference"]["stride"],classes=5)[4]
                new=fuse_foundation(base,probability)
                arrays_equal(base,new,image,before)
                if not np.array_equal(fuse_foundation(base,probability,strength=0),base):
                    raise ValueError("Disabled equivalence failed")
                path,pp=directory/name,directory/(Path(name).stem+".npy")
                Image.fromarray(new).save(path); np.save(pp,probability.astype(np.float32))
                predictions[arm][name]={"path":str(path),"sha256":sha256(path),"probability_path":str(pp),
                                        "probability_sha256":sha256(pp),"fold":fold["fold"]}
                if index%20==0 or index+1==len(targets):
                    log(stage="inference",arm=arm,fold=fold["fold"],views=index+1,total=len(targets))
            row["arms"][arm]={"training":training,"inference_seconds":time.perf_counter()-begin,
                              "inference_peak_cuda_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
                              "protected_exact":True,"rgb_exact":True,"disabled_exact_views":len(targets)}
            del model; torch.cuda.empty_cache()
        a,b=[row["arms"][arm]["training"]["metadata"] for arm in ARMS]
        for field in ("common_decoder_initial_sha256","sampling_plan_sha256","actual_label_access","class_weights"):
            if a[field]!=b[field]:
                raise ValueError(f"Paired training differs: {field}")
        foldrows.append(row); write_json(out/f"fold{fold['fold']}_result.json",row)
    write_json(out/"prediction_manifest.json",predictions)
    write_json(out/"prediction_seal.json",{"config_sha256":cfg["_sha256"],"source_sha256":sources,
        "prediction_manifest_sha256":sha256(out/"prediction_manifest.json"),"folds":foldrows})
    if sha256(args.config)!=cfg["_sha256"] or any(sha256(ROOT/p)!=h for p,h in sources.items()):
        raise ValueError("Code/config changed during execution")
    after=verify_freeze(ROOT/"experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json",art)
    if after!=evidence["freeze"]:
        raise ValueError("Frozen production artifacts changed")
    for name,digest in evidence["control_source"].items():
        if sha256(art/cfg["parents"]["model_config"]/name)!=digest:
            raise ValueError("Control pretrained source changed")
    verify_sources(args.model_dir,cfg)
    result.update(folds=foldrows,freeze_after=after,prediction_seal_sha256=sha256(out/"prediction_seal.json"),
                  status="completed_pretrained_smoke")
    if args.mode=="full":
        if any(set(predictions[a])!=set(names) for a in ARMS):
            raise ValueError("Incomplete paired OOF outputs")
        partition=partition_training_names(names,expected_count=219)
        calibration=evaluate(partition["calibration"],predictions,baseline,labels,cfg)
        write_json(out/"calibration_sealed.json",calibration)
        result.update(calibration=calibration,status="completed_calibration_gate_failed")
        if calibration["passed"]:
            check=evaluate(partition["check"],predictions,baseline,labels,cfg)
            write_json(out/"check_sealed.json",check)
            result.update(check=check,check_quality_scored_this_run=True,
                          status="quality_passed_pending_independent_audit" if check["passed"] else "completed_check_gate_failed")
        if not set(cfg["visual_samples"]).issubset(partition["calibration"]):
            raise ValueError("Visual view outside calibration")
        result["panels"]=panels(cfg["visual_samples"],rgb,baseline,labels,predictions,out/"panels")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("prepare","smoke","full"),required=True)
    for name in ("artifact-root","model-dir","output-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--config",type=Path,default=CONFIG)
    args=parser.parse_args()
    args.artifact_root=args.artifact_root.resolve(); args.model_dir=args.model_dir.resolve()
    cfg=read_json(args.config); cfg["_sha256"]=sha256(args.config)
    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    try:
        result=run(args,cfg,out,started)
        result["seconds"]=time.perf_counter()-started
        write_json(out/"result.json",result)
        log(stage="completed",status=result["status"],seconds=result["seconds"])
        return 2 if result["status"].startswith("blocked_") else 0
    except Exception as exc:
        write_json(out/"failure.json",{"experiment_id":EXP,"mode":args.mode,
            "error_type":type(exc).__name__,"error":str(exc),"seconds":time.perf_counter()-started,
            "status":"incomplete_not_quality_result"})
        raise


if __name__=="__main__":
    raise SystemExit(main())
