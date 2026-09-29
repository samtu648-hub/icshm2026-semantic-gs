#!/usr/bin/env python3
"""SEM326 full219 reference bank and fixed74 developer matching, no target GT."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import random
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from PIL import Image
from run_sem320_dense_preflight import checked, git_sources, offline_official_load, array_digest
from run_sem250_colmap_track_deck_diag import verify_freeze
from src.semantic2d import read_json, write_json, sha256
from src.semantic_protected_fusion import partition_training_names
from src.geometry_prior.camera_io import read_cameras_txt
from src.semantic_source_access import read_pose_only
from src.semantic_source_check import select_plan, summarize_access
from src.semantic_source_geometry import geometry_row, compare_rows

EXP = "SEM-326-DENSE-FOUNDATION-REVIEW-FINAL"
CONFIG = ROOT / "experiments/runs" / EXP / "config.yaml"


def context(args):
    cfg = read_json(args.config)
    if cfg["experiment_id"] != EXP: raise ValueError("Wrong experiment")
    quality = read_json(checked(ROOT / cfg["qualified_check"], cfg["qualified_check_sha256"]))
    if not quality["passed"] or quality["config_sha256"] != cfg["oof_config_sha256"]:
        raise ValueError("Calibration must qualify before check matching")
    checked(ROOT / cfg["oof_config"], cfg["oof_config_sha256"])
    audit = read_json(checked(ROOT / cfg["qualified_check_audit"], cfg["qualified_check_audit_sha256"]))
    if not audit["quality_passed"] or audit["score_sha256"] != cfg["qualified_check_sha256"]:
        raise ValueError("OOF independent quality audit missing")
    parent = read_json(checked(ROOT / cfg["parent_config"], cfg["parent_config_sha256"]))
    geo = read_json(checked(ROOT / cfg["geometry_config"], cfg["geometry_config_sha256"]))
    for key in ("dataset_manifest", "semantic_split", "oof_split"):
        checked(ROOT / parent[key], parent[key + "_sha256"])
    frozen = verify_freeze(ROOT / parent["freeze_manifest"], args.artifact_root)
    return cfg, parent, geo, frozen


def split_scope(parent):
    from src.semantic_foundation_review_final import validate_final_split
    split = read_json(ROOT / parent["semantic_split"])
    train, dev = validate_final_split(split)
    folds = [{"fold": "final", "train": train, "oof": dev, "guard": split["buffer_excluded_labeled"]}]
    return split, folds, {"check": dev, "calibration": train}


def independent_ranking(plans, folds, poses):
    """Inverse homogeneous transforms; does not call rank_sources/select_plan."""
    transforms = {}
    for name, pose in poses.items():
        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = pose.rotation_world_to_camera, pose.tvec
        transforms[name] = np.linalg.inv(w2c)
    for plan in plans:
        fold = next(f for f in folds if f["fold"] == plan["fold"])
        rows = []
        a = transforms[plan["target"]]
        for source in fold["train"]:
            b = transforms[source]
            axis_a, axis_b = a[:3, 2], b[:3, 2]
            cosine = np.dot(axis_a, axis_b) / (np.linalg.norm(axis_a) * np.linalg.norm(axis_b))
            angle = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
            distance = float(np.sqrt(np.sum((a[:3, 3] - b[:3, 3]) ** 2)))
            rows.append({"source": source, "center_distance": distance, "angle_degrees": angle, "eligible": angle <= 60.})
        rows.sort(key=lambda r: (r["center_distance"], r["angle_degrees"], r["source"]))
        for row, old in zip(rows, plan["ranking"], strict=True):
            if row["source"] != old["source"] or row["eligible"] != old["eligible"]:
                raise ValueError("Independent pose ranking differs")
            for key in ("angle_degrees", "center_distance"):
                if not np.isclose(row[key], old[key], rtol=0, atol=1e-6): raise ValueError("Pose metric differs")
        if [r["source"] for r in rows if r["eligible"]][:3] != plan["sources"]:
            raise ValueError("Independent top3 differs")


def prepare(args):
    cfg, parent, _, frozen = context(args)
    split, folds, partition = split_scope(parent)
    camera_dir = args.data_root / "camera_parameters"
    poses = read_pose_only(camera_dir / "images.txt")
    plans = select_plan(folds, partition["check"], poses,
                        cfg["retrieval"]["candidates_per_target"], cfg["retrieval"]["maximum_optical_axis_angle_degrees"])
    if len(plans) != 74 or {p["target"] for p in plans} != set(partition["check"]):
        raise ValueError("Expanded target selection changed")
    independent_ranking(plans, folds, poses)
    forbidden = set(split["semantic_validation_labeled"]) | set(split["buffer_excluded_labeled"])
    render_root = args.artifact_root / cfg["parents"]["developer_rgb"]
    manifest = checked(render_root / "rendered_validation_manifest.json", cfg["parents"]["developer_rgb_manifest_sha256"])
    rendered = {r["name"]: r for r in read_json(manifest)["rendered_images"]}
    pairs = []
    for plan in plans:
        fold = next(f for f in folds if f["fold"] == plan["fold"])
        record = rendered[plan["target"]]
        target = checked(render_root / record["path"], record["sha256"])
        for rank, source_name in enumerate(plan["sources"], 1):
            if (source_name in forbidden or plan["target"] not in partition["check"] or source_name not in fold["train"]
                    or source_name in fold["oof"] or source_name in fold["guard"]):
                raise ValueError("Forbidden source/target")
            source = args.data_root / "images" / source_name
            pairs.append({"fold": plan["fold"], "target": plan["target"], "source": source_name, "rank": rank,
                "target_path": str(target.relative_to(args.artifact_root)), "target_sha256": sha256(target),
                "source_path": str(source.relative_to(args.data_root)), "source_sha256": sha256(source),
                "source_historical_partition": "all219_train"})
    label_manifest = read_json(checked(ROOT / cfg["label_manifest"], cfg["label_manifest_sha256"]))
    label_hashes = {r["image"]: r["label_png_sha256"] for r in label_manifest["records"]}
    bank = []
    for name in partition["calibration"]:
        image = args.data_root / "images" / name
        label = checked(args.artifact_root / cfg["parents"]["labels"] / name, label_hashes[name])
        bank.append({"name": name, "source_path": str(image.relative_to(args.data_root)), "source_sha256": sha256(image),
                     "label_path": str(label.relative_to(args.artifact_root)), "label_sha256": label_hashes[name]})
    upstream = {"roma": git_sources(args.roma_dir, parent["sources"]["roma_commit"]),
                "dinov3": git_sources(args.dinov3_dir, parent["sources"]["dinov3_commit"])}
    checked(args.weights, parent["sources"]["weight_sha256"])
    if args.weights.stat().st_size != parent["sources"]["weight_bytes"]: raise ValueError("Weight size differs")
    files = {Path(__file__).resolve(), args.config, ROOT / cfg["parent_config"], ROOT / cfg["geometry_config"]}
    # Imported project modules capture transitive project dependencies, not packages.
    for module in tuple(sys.modules.values()):
        location = getattr(module, "__file__", None)
        if location:
            path = Path(location).resolve()
            if path.is_relative_to(ROOT) and path.suffix == ".py" and not path.is_relative_to(ROOT / "outputs"):
                files.add(path)
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}
    for name in sources:
        destination = args.output_dir / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    return {"experiment_id": EXP, "status": "pose_plan_sealed", "started_unix": time.time(),
        "config_sha256": sha256(args.config), "source_sha256": sources, "plans": plans, "pairs": pairs,
        "independent_pose_ranking_exact": True, "upstream_sha256": upstream, "reference_bank": bank,
        "weight_sha256": parent["sources"]["weight_sha256"], "freeze": frozen,
        "camera_images_sha256": sha256(camera_dir / "images.txt"),
        "camera_intrinsics_sha256": sha256(camera_dir / "cameras.txt"),
        "labels_decoded": 0, "rgb_decoded": 0, "observations_parsed": 0}


def load_plan(args):
    cfg, parent, geo, frozen = context(args)
    plan = read_json(args.prepare_dir / "result.json")
    if plan["config_sha256"] != sha256(args.config) or plan["freeze"] != frozen: raise ValueError("Plan changed")
    for name, digest in plan["source_sha256"].items():
        checked(ROOT / name, digest); checked(args.prepare_dir / "source_snapshot" / name, digest)
    checked(args.data_root / "camera_parameters/images.txt", plan["camera_images_sha256"])
    checked(args.data_root / "camera_parameters/cameras.txt", plan["camera_intrinsics_sha256"])
    for pair in plan["pairs"]:
        checked(args.artifact_root / pair["target_path"], pair["target_sha256"])
        checked(args.data_root / pair["source_path"], pair["source_sha256"])
    return cfg, parent, geo, frozen, plan


def runtime(args):
    cfg, parent, _, frozen, plan = load_plan(args)
    for kind, path in (("roma", args.roma_dir), ("dinov3", args.dinov3_dir)):
        if git_sources(path, parent["sources"][kind + "_commit"]) != plan["upstream_sha256"][kind]:
            raise ValueError("Upstream changed")
    checked(args.weights, plan["weight_sha256"])
    import torch
    sys.path.insert(0, str(args.roma_dir / "src"))
    from romav2 import RoMaV2
    from romav2.local_correlation import local_corr
    if local_corr is not None or not torch.cuda.is_available(): raise RuntimeError("Fixed CUDA backend unavailable")
    torch.set_num_threads(4); torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    free, total = torch.cuda.mem_get_info()
    if free / 2**20 < parent["runtime"]["minimum_free_gpu_mib"]: raise RuntimeError("GPU headroom below limit")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter(); records, access = [], []
    with offline_official_load(args, parent):
        model = RoMaV2(RoMaV2.Cfg(setting="fast", compile=False)).eval().requires_grad_(False)
        if (model.H_lr, model.W_lr, model.H_hr, model.W_hr, model.bidirectional) != (512, 512, None, None, False):
            raise ValueError("Fixed matcher settings changed")
        loading_seconds = time.perf_counter() - started
        pairs = plan["pairs"][:1] if args.mode == "replay" else plan["pairs"]
        for i, pair in enumerate(pairs):
            target_path, source_path = args.artifact_root / pair["target_path"], args.data_root / pair["source_path"]
            with Image.open(target_path) as im: a = np.asarray(im.convert("RGB"))
            with Image.open(source_path) as im: b = np.asarray(im.convert("RGB"))
            hashes = array_digest(a), array_digest(b)
            access.append({"target": pair["target"], "source": pair["source"], "target_kind": "frozen_rendered",
                "source_kind": "same_fold_train_real", "source_historical_partition": pair["source_historical_partition"],
                "target_array_sha256": hashes[0], "source_array_sha256": hashes[1]})
            directions = [("ab", a, b)] if args.mode == "replay" else [("ab", a, b), ("ba", b, a)]
            for direction, first, second in directions:
                torch.cuda.synchronize(); tick = time.perf_counter()
                with torch.inference_mode(): prediction = model.match(first, second)
                torch.cuda.synchronize(); seconds = time.perf_counter() - tick
                warp = prediction["warp_AB"].float().cpu().numpy()[0]
                overlap = prediction["overlap_AB"].float().cpu().numpy()[0, ..., 0]
                if (warp.shape != (512, 512, 2) or overlap.shape != (512, 512) or not np.isfinite(warp).all()
                        or not np.isfinite(overlap).all() or overlap.min() < 0 or overlap.max() > 1):
                    raise ValueError("Invalid correspondence output")
                name = f"pair{i}_{direction}"
                destination = args.output_dir / (name + ".npz")
                np.savez_compressed(destination, warp=warp, overlap=overlap)
                records.append({"name": name, "path": destination.name, "sha256": sha256(destination),
                    "warp_array_sha256": array_digest(warp), "overlap_array_sha256": array_digest(overlap), "seconds": seconds})
                if torch.cuda.max_memory_allocated() / 2**20 > parent["runtime"]["maximum_peak_allocated_mib"]:
                    raise RuntimeError("GPU allocation exceeded")
                if time.time() - plan["started_unix"] > cfg["resource_budget"]["maximum_total_seconds"]:
                    raise RuntimeError("Total experiment time budget exceeded")
                print({"pair": i, "target": pair["target"], "source": pair["source"], "direction": direction, "seconds": round(seconds, 3)}, flush=True)
            if (array_digest(a), array_digest(b)) != hashes: raise ValueError("RGB arrays changed")
            checked(target_path, pair["target_sha256"]); checked(source_path, pair["source_sha256"])
    replay_exact = None
    if args.mode == "replay":
        old = read_json(args.run_dir / "result.json")
        if old["prepare_sha256"] != sha256(args.prepare_dir / "result.json"): raise ValueError("Replay parent changed")
        reference = old["probes"][0]
        checked(args.run_dir / reference["path"], reference["sha256"])
        for key in ("warp_array_sha256", "overlap_array_sha256"):
            if reference[key] != records[0][key]: raise ValueError("Fresh process replay differs")
        replay_exact = True
    if verify_freeze(ROOT / parent["freeze_manifest"], args.artifact_root) != frozen: raise ValueError("Production changed")
    return {"experiment_id": EXP, "status": "passed_new_process_replay" if args.mode == "replay" else "matching_maps_sealed",
        "config_sha256": sha256(args.config), "prepare_sha256": sha256(args.prepare_dir / "result.json"),
        "probes": records, "fresh_process_exact": replay_exact, "input_access": access,
        "freeze": frozen, "rgb_exact": True, "labels_decoded": 0, "real_target_slot_rgb_decoded": 0,
        "developer_rgb_decoded": 0, "observations_parsed": 0, "training_steps": 0, "semantic_outputs": 0,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20, "gpu_name": torch.cuda.get_device_name(),
        "free_gpu_before_mib": free / 2**20, "total_gpu_mib": total / 2**20, "loading_seconds": loading_seconds,
        "versions": {n: importlib.metadata.version(n) for n in ("torch", "torchvision", "numpy", "pillow", "einops")}}


def diagnose(args):
    cfg, parent, geo, frozen, plan = load_plan(args)
    run, replay = (read_json(directory / "result.json") for directory in (args.run_dir, args.replay_dir))
    for r in (run, replay):
        if r["prepare_sha256"] != sha256(args.prepare_dir / "result.json") or r["freeze"] != frozen:
            raise ValueError("Runtime provenance differs")
    if not replay["fresh_process_exact"]: raise ValueError("Replay missing")
    records = {r["name"]: r for r in run["probes"]}
    expected_names = {f"pair{i}_{direction}" for i in range(len(plan["pairs"])) for direction in ("ab", "ba")}
    if set(records) != expected_names or len(run["probes"]) != len(expected_names): raise ValueError("Incomplete pair maps")
    poses = read_pose_only(args.data_root / "camera_parameters/images.txt")
    cameras = read_cameras_txt(args.data_root / "camera_parameters/cameras.txt")
    _, folds, partition = split_scope(parent)
    reranked = select_plan(folds, partition["check"], poses)
    if reranked != plan["plans"]: raise ValueError("Source scope/ranking changed")
    independent_ranking(plan["plans"], folds, poses)
    reference = read_json(args.result_dir / "result.json") if args.mode == "audit" else None
    if reference is not None:
        if reference["prepare_sha256"] != sha256(args.prepare_dir / "result.json") or reference["runtime_sha256"] != sha256(args.run_dir / "result.json"):
            raise ValueError("Geometry parent differs")
    results = []
    for i, pair in enumerate(plan["pairs"]):
        maps = []
        for direction in ("ab", "ba"):
            record = records[f"pair{i}_{direction}"]
            with np.load(checked(args.run_dir / record["path"], record["sha256"]), allow_pickle=False) as data:
                maps.append((data["warp"].astype(np.float64), data["overlap"].astype(np.float64)))
        row = geometry_row(pair, maps, poses, cameras, geo, independent=reference is not None)
        if reference is not None: compare_rows(reference["pairs"][i], row)
        results.append(row)
    summary = summarize_access(plan["plans"], results)
    if reference is not None and summary != reference["summary"]: raise ValueError("Independent source-access decision differs")
    if verify_freeze(ROOT / parent["freeze_manifest"], args.artifact_root) != frozen: raise ValueError("Frozen production changed")
    status = "passed_independent_access_audit" if reference is not None else ("completed_access_screen_passed" if summary["passed"] else "completed_access_screen_failed")
    return {"experiment_id": EXP, "status": status, "passed": summary["passed"], "summary": summary, "pairs": results,
        "config_sha256": sha256(args.config), "prepare_sha256": sha256(args.prepare_dir / "result.json"),
        "runtime_sha256": sha256(args.run_dir / "result.json"), "replay_sha256": sha256(args.replay_dir / "result.json"),
        "primary_result_sha256": sha256(args.result_dir / "result.json") if reference is not None else None,
        "independent_pose_ranking_exact": True, "independent_geometry": reference is not None,
        "freeze": frozen, "rgb_decoded": 0, "labels_decoded": 0, "new_model_inference": 0,
        "observations_parsed": 0, "semantic_outputs": 0,
        "claim_boundary": "Fixed74 developer targets with219 train references; pair access is descriptive, not semantic quality or blind evaluation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run", "replay", "geometry", "audit"), required=True)
    for key in ("artifact-root", "data-root", "output-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("roma-dir", "dinov3-dir", "weights", "prepare-dir", "run-dir", "replay-dir", "result-dir"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path): setattr(args, key, value.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        cfg = read_json(args.config)
        if shutil.disk_usage(args.output_dir).free / 2**30 < cfg["resource_budget"]["minimum_free_disk_gib"]:
            raise RuntimeError("Free disk below registered floor")
        if args.mode == "prepare": result = prepare(args)
        elif args.mode in ("run", "replay"): result = runtime(args)
        else:
            if args.mode == "audit":
                import torch
                torch.set_num_threads(4)
            result = diagnose(args)
        result["seconds"] = time.perf_counter() - started
        write_json(args.output_dir / "result.json", result)
        print({k: result[k] for k in ("status", "seconds")}, flush=True)
    except Exception as exc:
        write_json(args.output_dir / "failure.json", {"experiment_id": EXP, "mode": args.mode,
            "error_type": type(exc).__name__, "error": str(exc), "seconds": time.perf_counter() - started})
        raise


if __name__ == "__main__": main()
