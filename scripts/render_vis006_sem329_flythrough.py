#!/usr/bin/env python3
"""Batch the unchanged SEM329 external-camera recipe; no target GT interface."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import numpy as np
from PIL import Image
from src.semantic2d import read_json, write_json, sha256, label_array
from src.semantic_source_access import rank_sources, read_pose_only
from src.geometry_prior.camera_io import read_cameras_txt
from src.semantic_source_geometry import geometry_row, compare_rows
from src.semantic_dense_final import review_official_output
from src.semantic_second_source_veto import review_second_source, second_source
from scripts.run_sem326_matching import load_plan
from scripts.run_sem320_dense_preflight import checked, offline_official_load, array_digest, git_sources
from scripts.run_sem250_colmap_track_deck_diag import verify_freeze

CONFIG = ROOT / "experiments/runs/VIS-006-SEM329-FLYTHROUGH/config.yaml"


def png(path):
    with Image.open(path) as im:
        return np.asarray(im).copy()


def child(command, output, env):
    with output.open("w") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def run(args):
    vis = read_json(args.config)
    freeze_path = ROOT / "experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json"
    freeze = verify_freeze(freeze_path, args.artifact_root)
    for key in ("dataset_manifest", "semantic_split"):
        checked(ROOT / vis[key], vis[key + "_sha256"])
    trajectory = args.artifact_root / vis["trajectory"]
    for name, digest in vis["trajectory_hashes"].items():
        checked(trajectory / name, digest)
    poses = read_pose_only(trajectory / "images.txt")
    names = list(poses)
    if names != [f"frame{i:04d}.png" for i in range(vis["frames"])]:
        raise ValueError("Trajectory sequence differs")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    snapshot = args.output_dir / "source_snapshot"
    snapshot.mkdir()
    source_paths = [Path(__file__), args.config, ROOT / "scripts/render_sem326_base.py",
                    ROOT / "scripts/run_sem329_external.py", ROOT / "scripts/render_sem326_external.py",
                    ROOT / "src/semantic_dense_final.py", ROOT / "src/semantic_second_source_veto.py"]
    hashes = {}
    for source in source_paths:
        relative = source.relative_to(ROOT)
        dest = snapshot / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        hashes[str(relative)] = sha256(source)
    write_json(args.output_dir / "input_seal.json", {"config_sha256": sha256(args.config), "sources": hashes,
               "trajectory": vis["trajectory_hashes"], "freeze": freeze, "source_bank": "train219", "target_gt_decodes": 0})
    common = []
    for key in ("artifact_root", "data_root", "roma_dir", "dinov3_dir", "weights", "bundle", "render_python"):
        common += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    for name in vis["probe_frames"]:
        out = args.output_dir / "single_probes" / Path(name).stem
        out.parent.mkdir(exist_ok=True)
        command = [str(args.render_python), str(ROOT / "scripts/run_sem329_external.py"), "probe", *common,
                   "--cameras", str(trajectory / "cameras.txt"), "--images", str(trajectory / "images.txt"),
                   "--image-name", name, "--output-dir", str(out)]
        child(command, args.output_dir / (Path(name).stem + "-probe.log"), env)
        print(json.dumps({"stage": "fresh_original_probe", "frame": name}), flush=True)
    base = args.output_dir / "base"
    render_env = dict(env)
    render_env.pop("PYTHONPATH", None)
    child([str(args.render_python), str(ROOT / "scripts/render_sem326_base.py"), "--bundle", str(args.bundle),
           "--cameras", str(trajectory / "cameras.txt"), "--images", str(trajectory / "images.txt"),
           "--output-dir", str(base)], args.output_dir / "base-render.log", render_env)
    base_manifest = read_json(base / "manifest.json")
    if [r["pose"] for r in base_manifest["outputs"]] != names:
        raise ValueError("Batch renderer frame order differs")
    legacy_args = argparse.Namespace(**vars(args))
    legacy_args.config = ROOT / "experiments/runs/SEM-326-DENSE-FOUNDATION-REVIEW-FINAL/config.yaml"
    legacy_args.prepare_dir = ROOT / "outputs/sem326/matching-20260908-r2/prepare"
    cfg, original, geo, frozen, plan = load_plan(legacy_args)
    if frozen != freeze:
        raise ValueError("Frozen state differs")
    bank = {r["name"]: r for r in plan["reference_bank"]}
    split = read_json(ROOT / vis["semantic_split"])
    if len(bank) != 219 or set(bank) != set(split["semantic_train_labeled"]):
        raise ValueError("Source roles differ")
    for kind, path in (("roma", args.roma_dir), ("dinov3", args.dinov3_dir)):
        if git_sources(path, original["sources"][kind + "_commit"]) != plan["upstream_sha256"][kind]:
            raise ValueError("Upstream changed")
    checked(args.weights, plan["weight_sha256"])
    source_poses = read_pose_only(args.data_root / "camera_parameters/images.txt")
    source_cameras = read_cameras_txt(args.data_root / "camera_parameters/cameras.txt")
    incoming_cameras = read_cameras_txt(trajectory / "cameras.txt")
    import torch
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    if torch.cuda.mem_get_info()[0] / 2**20 < original["runtime"]["minimum_free_gpu_mib"]:
        raise RuntimeError("GPU headroom unavailable")
    torch.cuda.reset_peak_memory_stats()
    sys.path.insert(0, str(args.roma_dir / "src"))
    from romav2 import RoMaV2
    from romav2.local_correlation import local_corr
    if local_corr is not None:
        raise ValueError("Fixed native matcher backend changed")
    records = []
    # Evaluate the three representative frames first, then every remaining pose in order.
    order = vis["probe_frames"] + [n for n in names if n not in vis["probe_frames"]]
    source_cache, label_cache = {}, {}
    with offline_official_load(args, original):
        model = RoMaV2(RoMaV2.Cfg(setting="fast", compile=False)).eval().requires_grad_(False)
        if (model.H_lr, model.W_lr, model.H_hr, model.W_hr, model.bidirectional) != (512,512,None,None,False):
            raise ValueError("Matcher settings differ")
        for name in order:
            if time.perf_counter() - args.started > vis["budget"]["maximum_seconds"]:
                raise RuntimeError("Visualization runtime budget exceeded")
            target, baseline, rgb = [png(base / k / name) for k in ("pinhole_rgb", "semantic", "rgb")]
            a = replace(poses[name], camera_id=-1)
            ca = replace(incoming_cameras[poses[name].camera_id], camera_id=-1)
            allposes, cameras = {**source_poses, name: a}, {**source_cameras, -1: ca}
            ranking = rank_sources(name, bank, allposes, 60.)
            shortlist = [r["source"] for r in ranking if r["eligible"]][:3]
            out = args.output_dir / "frames" / Path(name).stem
            out.mkdir(parents=True)
            good, maps_by_source, pairs = [], {}, []
            for rank, source_name in enumerate(shortlist, 1):
                rec = bank[source_name]
                if source_name not in source_cache:
                    source_cache[source_name] = png(checked(args.data_root / rec["source_path"], rec["source_sha256"]))
                source = source_cache[source_name]
                before = array_digest(source)
                maps, map_records = [], []
                for direction, x, y in (("ab", target, source), ("ba", source, target)):
                    with torch.inference_mode():
                        result = model.match(x, y)
                    warp = result["warp_AB"].float().cpu().numpy()[0]
                    overlap = result["overlap_AB"].float().cpu().numpy()[0, ..., 0]
                    if (warp.shape != (512,512,2) or overlap.shape != (512,512) or
                            not np.isfinite(warp).all() or not np.isfinite(overlap).all() or overlap.min()<0 or overlap.max()>1):
                        raise ValueError("Invalid maps")
                    path = out / f"rank{rank}_{direction}.npz"
                    np.savez_compressed(path, warp=warp, overlap=overlap)
                    maps.append((warp.astype(np.float64), overlap.astype(np.float64)))
                    map_records.append({"path": path.name, "sha256": sha256(path)})
                if before != array_digest(source):
                    raise ValueError("Source RGB changed")
                pair = {"fold":"external", "target":name, "source":source_name, "rank":rank}
                row = geometry_row(pair, maps, allposes, cameras, geo)
                compare_rows(row, geometry_row(pair, maps, allposes, cameras, geo, independent=True))
                pairs.append({**row, "maps": map_records})
                maps_by_source[source_name] = maps
                if row["checks"]["coverage"] and row["checks"]["epipolar_retention"]:
                    good.append(source_name)
            first = good[0] if good else None
            second = second_source({"accessible_sources":good, "selected_source":first})
            def source_args(source_name):
                rec = bank[source_name]
                if source_name not in label_cache:
                    label_cache[source_name] = label_array(checked(args.artifact_root / rec["label_path"], rec["label_sha256"]))
                b = source_poses[source_name]
                return label_cache[source_name], maps_by_source[source_name], b, source_cameras[b.camera_id]
            dense = baseline.copy()
            if first:
                labels, maps, b, cb = source_args(first)
                dense, stats = review_official_output(baseline, labels, maps, a,b,ca,cb,geo["filters"])
                check, stats2 = review_official_output(baseline, labels, maps, a,b,ca,cb,geo["filters"],independent=True)
                if not np.array_equal(dense, check) or stats != stats2:
                    raise ValueError("Independent dense output differs")
            kwargs = {"target_is_distorted":True}
            if second:
                labels, maps, b, cb = source_args(second)
                kwargs.update(source_labels=labels,maps=maps,target_pose=a,source_pose=b,target_camera=ca,source_camera=cb,filters=geo["filters"])
            current, stats = review_second_source(baseline, dense, **kwargs)
            independent, stats2 = review_second_source(baseline, dense, independent=True, **kwargs)
            bypass, _ = review_second_source(baseline, dense, enabled=False)
            if not np.array_equal(current,independent) or stats != stats2 or not np.array_equal(bypass,dense):
                raise ValueError("Independent second-source or bypass differs")
            for result in (dense, current):
                if any(not np.array_equal(result==k,baseline==k) for k in (1,2,3)):
                    raise ValueError("Protected classes differ")
            outputs = {}
            for key, array in (("sem326",dense),("sem329",current)):
                p = out / (key + ".png"); Image.fromarray(array).save(p)
                outputs[key] = {"path": str(p.relative_to(args.output_dir)),"sha256":sha256(p)}
            exact = None
            if name in vis["probe_frames"]:
                reference = args.output_dir / "single_probes" / Path(name).stem
                checks = [(rgb,reference/"dense_stage/base/rgb"/name), (baseline,reference/"dense_stage/base/semantic"/name),
                          (target,reference/"dense_stage/base/pinhole_rgb"/name),(dense,reference/"dense_stage/semantic.png"),(current,reference/"semantic.png")]
                if not all(np.array_equal(x,png(p)) for x,p in checks):
                    raise ValueError("Batch and fresh original single-frame recipe differ: " + name)
                old = read_json(reference/"result.json")
                if (first,second,stats)!=(old["selected_source"],old["second_source"],old["native"]):
                    raise ValueError("Source selection or counts differ")
                exact = True
            row = {"name":name,"selected_source":first,"second_source":second,"shortlist":shortlist,
                   "native":stats,"pairs":pairs,"outputs":outputs,"fresh_original_exact":exact,
                   "rgb_sha256":sha256(base/"rgb"/name),"sem161_sha256":sha256(base/"semantic"/name),
                   "protected_exact":True,"independent_geometry_and_native_exact":True,"target_gt_decodes":0}
            write_json(out/"result.json",row); records.append(row)
            print(json.dumps({"stage":"review","done":len(records),"frame":name,"veto_pixels":stats["foundation_veto_pixels"]}),flush=True)
    peak = torch.cuda.max_memory_allocated()/2**20
    if peak > original["runtime"]["maximum_peak_allocated_mib"]:
        raise RuntimeError("Matcher memory budget exceeded")
    if verify_freeze(freeze_path,args.artifact_root)!=freeze:
        raise ValueError("Frozen artifacts changed")
    for source,digest in hashes.items(): checked(ROOT/source,digest)
    for rec in base_manifest["outputs"]:
        for output in rec["outputs"].values(): checked(base/output["path"],output["sha256"])
    return {"experiment_id":vis["experiment_id"],"status":"completed_verified_flythrough_render",
            "config_sha256":sha256(args.config),"source_sha256":hashes,"trajectory_sha256":vis["trajectory_hashes"],
            "base_manifest_sha256":sha256(base/"manifest.json"),"frames":sorted(records,key=lambda r:r["name"]),
            "frame_count":len(records),"fps":vis["fps"],"frozen":freeze,"target_gt_decodes":0,"training_steps":0,
            "source_labels_decoded":sorted(label_cache),"three_original_probe_frames_exact":True,"rgb_exact":True,
            "protected_exact":True,"matcher_peak_allocated_mib":peak,"base_render_peak_allocated_mib":base_manifest["peak_cuda_allocated_mib"],
            "seconds":time.perf_counter()-args.started,"claim_boundary":vis["claim_boundary"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=CONFIG)
    for key in ("artifact-root","data-root","roma-dir","dinov3-dir","weights","bundle","render-python","output-dir"):
        p.add_argument("--"+key,type=Path,required=True)
    args=p.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path): setattr(args,key,value.absolute() if key=="render_python" else value.resolve())
    args.output_dir.mkdir(parents=True,exist_ok=False)
    args.started=time.perf_counter()
    try:
        if shutil.disk_usage(args.output_dir).free/2**30<read_json(args.config)["budget"]["minimum_free_disk_gib"]:
            raise RuntimeError("Disk floor")
        result=run(args); write_json(args.output_dir/"result.json",result)
        print(json.dumps({"status":result["status"],"seconds":result["seconds"]}),flush=True)
    except Exception as exc:
        write_json(args.output_dir/"failure.json",{"experiment_id":"VIS-006-SEM329-FLYTHROUGH","error":str(exc),"type":type(exc).__name__,"seconds":time.perf_counter()-args.started})
        raise


if __name__=="__main__": main()
