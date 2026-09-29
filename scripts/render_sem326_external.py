#!/usr/bin/env python3
"""Fresh official-camera SEM161 renders with fixed source-only dense review."""
import argparse
from dataclasses import replace
import os
from pathlib import Path
import random
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from PIL import Image
from src.semantic2d import read_json, write_json, sha256, label_array
from src.semantic_source_access import rank_sources, read_pose_only
from src.geometry_prior.camera_io import read_cameras_txt
from src.semantic_source_geometry import geometry_row, compare_rows
from src.semantic_dense_final import review_official_output
from run_sem326_matching import CONFIG, EXP, load_plan
from run_sem320_dense_preflight import checked, offline_official_load, array_digest, git_sources
from run_sem250_colmap_track_deck_diag import verify_freeze


def png(path):
    with Image.open(path) as image: return np.asarray(image).copy()


def argument_path(name, value):
    # Resolving a venv executable symlink selects the system interpreter and loses its packages.
    return value.absolute() if name == "render_python" else value.resolve()


def probe(args):
    cfg, original, geo, frozen, plan = load_plan(args)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("PYTHONPATH", None)  # renderer uses its unchanged installed Pillow, not RoMa's scoped overlay
    render = args.output_dir / "base"
    subprocess.run([str(args.render_python), str(ROOT / "scripts/render_sem326_base.py"), "--bundle", str(args.bundle),
                    "--cameras", str(args.cameras), "--images", str(args.images), "--image-name", args.image_name,
                    "--output-dir", str(render)], cwd=ROOT, env=env, check=True)
    target = png(render / "pinhole_rgb" / args.image_name)
    baseline = png(render / "semantic" / args.image_name)
    official_rgb = png(render / "rgb" / args.image_name)
    target_hash, raw_hash, base_hash = array_digest(target), array_digest(official_rgb), array_digest(baseline)
    source_poses = read_pose_only(args.data_root / "camera_parameters/images.txt")
    source_cameras = read_cameras_txt(args.data_root / "camera_parameters/cameras.txt")
    incoming = read_pose_only(args.images)[args.image_name]
    target_camera = replace(read_cameras_txt(args.cameras)[incoming.camera_id], camera_id=-1)
    a = replace(incoming, camera_id=-1)
    poses = {**source_poses, args.image_name: a}; cameras = {**source_cameras, -1: target_camera}
    bank = {r["name"]: r for r in plan["reference_bank"]}
    if args.image_name in bank: raise ValueError("Registered external target must not be a training source")
    ranking = rank_sources(args.image_name, bank, poses, 60.)
    shortlist = [r["source"] for r in ranking if r["eligible"]][:3]
    seal = {"config_sha256": sha256(args.config), "reference_plan_sha256": sha256(args.prepare_dir / "result.json"),
            "target_pose": {"qvec": a.qvec.tolist(), "tvec": a.tvec.tolist()}, "ranking": ranking, "sources": shortlist,
            "target_pinhole_rgb_sha256": sha256(render / "pinhole_rgb" / args.image_name),
            "wrapper_sha256": sha256(Path(__file__)), "renderer_sha256": sha256(ROOT / "scripts/render_sem326_base.py"),
            "core_sha256": sha256(ROOT / "src/semantic_dense_review.py"), "adapter_sha256": sha256(ROOT / "src/semantic_dense_final.py")}
    write_json(args.output_dir / "access_seal.json", seal)
    for kind, path in (("roma", args.roma_dir), ("dinov3", args.dinov3_dir)):
        if git_sources(path, original["sources"][kind + "_commit"]) != plan["upstream_sha256"][kind]: raise ValueError("Upstream changed")
    checked(args.weights, plan["weight_sha256"])
    import torch
    torch.set_num_threads(4); torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    if torch.cuda.mem_get_info()[0] / 2**20 < original["runtime"]["minimum_free_gpu_mib"]: raise RuntimeError("GPU headroom")
    torch.cuda.reset_peak_memory_stats()
    sys.path.insert(0, str(args.roma_dir / "src"))
    from romav2 import RoMaV2
    from romav2.local_correlation import local_corr
    if local_corr is not None: raise ValueError("Fixed matcher backend changed")
    results, saved_maps, selected = [], {}, None
    with offline_official_load(args, original):
        model = RoMaV2(RoMaV2.Cfg(setting="fast", compile=False)).eval().requires_grad_(False)
        if (model.H_lr, model.W_lr, model.H_hr, model.W_hr, model.bidirectional) != (512,512,None,None,False): raise ValueError("Matcher setting differs")
        for rank, name in enumerate(shortlist, 1):
            rec = bank[name]
            source = png(checked(args.data_root / rec["source_path"], rec["source_sha256"]))
            before = array_digest(source); maps, records = [], []
            for direction, x, y in (("ab", target, source), ("ba", source, target)):
                with torch.inference_mode(): output = model.match(x,y)
                warp = output["warp_AB"].float().cpu().numpy()[0]
                overlap = output["overlap_AB"].float().cpu().numpy()[0,...,0]
                if not np.isfinite(warp).all() or not np.isfinite(overlap).all() or overlap.min()<0 or overlap.max()>1: raise ValueError("Invalid maps")
                path = args.output_dir / f"rank{rank}_{direction}.npz"
                np.savez_compressed(path, warp=warp, overlap=overlap)
                records.append({"path": path.name, "sha256": sha256(path), "warp_sha256": array_digest(warp), "overlap_sha256": array_digest(overlap)})
                maps.append((warp.astype(np.float64), overlap.astype(np.float64)))
            if before != array_digest(source): raise ValueError("Source RGB mutated")
            pair = {"fold": "external", "target": args.image_name, "source": name, "rank": rank}
            row = geometry_row(pair, maps, poses, cameras, geo)
            compare_rows(row, geometry_row(pair, maps, poses, cameras, geo, independent=True))
            results.append({**row, "maps": records}); saved_maps[name] = maps
            if selected is None and row["checks"]["coverage"] and row["checks"]["epipolar_retention"]: selected = name
        peak = torch.cuda.max_memory_allocated() / 2**20
        if peak > original["runtime"]["maximum_peak_allocated_mib"]: raise RuntimeError("GPU budget")
    candidate = baseline.copy(); stats = {"changed_pixels": 0}
    if selected:
        rec = bank[selected]; labels = label_array(checked(args.artifact_root / rec["label_path"], rec["label_sha256"]))
        b = source_poses[selected]; cb = source_cameras[b.camera_id]
        candidate, stats = review_official_output(baseline, labels, saved_maps[selected], a,b,target_camera,cb,geo["filters"])
        independently, other_stats = review_official_output(baseline, labels, saved_maps[selected], a,b,target_camera,cb,geo["filters"], independent=True)
        if not np.array_equal(candidate, independently) or stats != other_stats: raise ValueError("Independent official review differs")
        bypass,_ = review_official_output(baseline, labels, saved_maps[selected], a,b,target_camera,cb,geo["filters"],enabled=False)
        if not np.array_equal(baseline,bypass): raise ValueError("Disabled review differs")
    if any(not np.array_equal(candidate==k,baseline==k) for k in (1,2,3)): raise ValueError("Protected masks differ")
    changed = candidate != baseline
    if np.any(changed & ((baseline!=4) | (candidate!=0))): raise ValueError("Illegal transition")
    if (array_digest(target),array_digest(official_rgb),array_digest(baseline)) != (target_hash,raw_hash,base_hash): raise ValueError("Base output changed")
    path = args.output_dir / "semantic.png"; Image.fromarray(candidate).save(path)
    if verify_freeze(ROOT / cfg["freeze_manifest"],args.artifact_root) != frozen: raise ValueError("Freeze changed")
    return {"experiment_id":EXP,"status":"passed_external_dense_review","camera":args.image_name,"shape":list(candidate.shape),
            "selected_source":selected,"native":stats,"pairs":results,"config_sha256":sha256(args.config),
            "access_seal_sha256":sha256(args.output_dir/"access_seal.json"),"prediction_sha256":sha256(path),
            "rgb_sha256":sha256(render/"rgb"/args.image_name),"baseline_sha256":sha256(render/"semantic"/args.image_name),
            "pinhole_rgb_sha256":sha256(render/"pinhole_rgb"/args.image_name),"peak_cuda_allocated_mib":peak,
            "rgb_exact":True,"protected_exact":True,"independent_geometry_and_native_exact":True,"disabled_exact":True,
            "target_label_decodes":0,"source_label_decodes":int(selected is not None),"official_pose_frame":"COLMAP_CAMERA_FROM_WORLD"}


def run_all(args):
    cfg = read_json(args.config)
    source_pose = read_pose_only(args.data_root/"camera_parameters/images.txt")[cfg["external_probes"]["reference_image"]]
    novel = replace(source_pose, name=cfg["external_probes"]["novel_target"],
                    tvec=source_pose.tvec+np.array(cfg["external_probes"]["novel_translation_w2c_delta"]))
    novel_file = args.output_dir/"novel_images.txt"
    values = [str(novel.image_id), *map(str,novel.qvec), *map(str,novel.tvec), str(novel.camera_id), novel.name]
    novel_file.write_text(" ".join(values)+"\n\n")
    rows=[]
    for name,cameras,images,image_name in (
        ("native1",args.data_root/"camera_parameters/cameras.txt",args.data_root/"camera_parameters/images.txt","140.png"),
        ("native2",args.data_root/"camera_parameters/cameras.txt",args.data_root/"camera_parameters/images.txt","140.png"),
        ("small",args.reference_dir/"small_cameras.txt",args.data_root/"camera_parameters/images.txt","140.png"),
        ("novel",args.data_root/"camera_parameters/cameras.txt",novel_file,novel.name)):
        out=args.output_dir/name
        command=[sys.executable,str(Path(__file__).resolve()),"probe","--image-name",image_name,"--cameras",str(cameras),"--images",str(images),"--output-dir",str(out)]
        for key in ("config","artifact_root","data_root","roma_dir","dinov3_dir","weights","prepare_dir","bundle","render_python"):
            command += ["--"+key.replace("_","-"),str(getattr(args,key))]
        subprocess.run(command,cwd=ROOT,check=True)
        result=read_json(out/"result.json")
        if name != "novel":
            reference=args.reference_dir/("small" if name=="small" else "reload1")
            for kind in ("rgb","semantic"):
                if not np.array_equal(png(out/"base"/kind/image_name),png(reference/kind/image_name)): raise ValueError("Frozen official output differs")
        rows.append({"probe":name,"result_sha256":sha256(out/"result.json"),**result})
    for key in ("prediction_sha256","rgb_sha256","baseline_sha256","pinhole_rgb_sha256"):
        if rows[0][key]!=rows[1][key]: raise ValueError("External new-process replay differs")
    for x,y in zip(rows[0]["pairs"],rows[1]["pairs"],strict=True):
        for u,v in zip(x["maps"],y["maps"],strict=True):
            if u["warp_sha256"]!=v["warp_sha256"] or u["overlap_sha256"]!=v["overlap_sha256"]: raise ValueError("External map replay differs")
    return {"experiment_id":EXP,"status":"passed_external_dense_review_audit","config_sha256":sha256(args.config),
            "probes":rows,"native_replay_exact":True,"frozen_known_output_comparisons":3,"target_label_decodes":0,
            "code_sha256":sha256(Path(__file__)),"renderer_sha256":sha256(ROOT/"scripts/render_sem326_base.py")}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=("all","probe"))
    for key in ("artifact-root","data-root","roma-dir","dinov3-dir","weights","prepare-dir","bundle","render-python","output-dir"):
        p.add_argument("--"+key,type=Path,required=True)
    for key in ("cameras","images","reference-dir"):p.add_argument("--"+key,type=Path)
    p.add_argument("--image-name",default="140.png");p.add_argument("--config",type=Path,default=CONFIG)
    args=p.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,argument_path(key,value))
    args.output_dir.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    try:
        result=probe(args) if args.mode=="probe" else run_all(args)
        result["seconds"]=time.perf_counter()-start
        if result["seconds"]>read_json(args.config)["resource_budget"]["maximum_seconds_per_stage"]:raise RuntimeError("Stage time budget")
        write_json(args.output_dir/"result.json",result);print({"status":result["status"],"seconds":result["seconds"]},flush=True)
    except Exception as exc:
        write_json(args.output_dir/"failure.json",{"experiment_id":EXP,"error_type":type(exc).__name__,"error":str(exc)});raise
