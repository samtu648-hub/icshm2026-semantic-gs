#!/usr/bin/env python3
"""Seal inputs, test the fixed official RoMa model offline, then replay independently."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from PIL import Image
from run_sem250_colmap_track_deck_diag import verify_freeze
from src.geometry_prior.camera_io import read_images_txt
from src.semantic2d import read_json, write_json, sha256
from src.semantic_dense_preflight import choose_pairs, require_pair_scope, pixel_center
from src.semantic_protected_fusion import partition_training_names

EXP = "SEM-320-ROMAV2-DENSE-PREFLIGHT"
CONFIG = ROOT / "experiments/runs" / EXP / "config.yaml"
SOURCES = ["scripts/run_sem320_dense_preflight.py", "src/semantic_dense_preflight.py",
           "src/geometry_prior/camera_io.py", "src/semantic2d.py",
           "src/semantic_protected_fusion.py", "scripts/run_sem250_colmap_track_deck_diag.py"]


def checked(path, digest):
    path = Path(path)
    if sha256(path) != digest:
        raise ValueError("Immutable input changed: " + path.name)
    return path


def git_sources(path, commit):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
    if git("rev-parse", "HEAD") != commit or git("status", "--porcelain"):
        raise ValueError("Upstream checkout changed: " + path.name)
    return {name: sha256(path / name) for name in git("ls-files").splitlines()}


def context(args):
    cfg = read_json(args.config)
    for k in ("dataset_manifest", "semantic_split", "oof_split"):
        checked(ROOT / cfg[k], cfg[k + "_sha256"])
    if cfg["experiment_id"] != EXP or cfg["runtime"]["setting"] != "fast":
        raise ValueError("Wrong preflight recipe")
    return cfg, verify_freeze(ROOT / cfg["freeze_manifest"], args.artifact_root)


def prepare(args):
    cfg, freeze = context(args)
    split = read_json(ROOT / cfg["semantic_split"])
    folds = read_json(ROOT / cfg["oof_split"])["folds"]
    calibration = partition_training_names(split["semantic_train_labeled"], expected_count=219)["calibration"]
    camera_file = args.data_root / "camera_parameters/images.txt"
    cameras = read_images_txt(camera_file)
    metadata = {c.name: {"center": c.center.tolist(), "tracks": sorted(c.point3d_ids)} for c in cameras.values()}
    pairs = choose_pairs(folds, calibration, metadata)
    require_pair_scope(pairs, folds, calibration, split["semantic_validation_labeled"], split["buffer_excluded_labeled"])
    render_root = args.artifact_root / cfg["rendered_rgb"]
    manifest = checked(render_root / "rendered_dataset_manifest.json", cfg["rendered_manifest_sha256"])
    rendered = {r["name"]: r for r in read_json(manifest)["rendered_images"]}
    for p in pairs:
        record = rendered[p["target"]]
        target = checked(render_root / record["path"], record["sha256"])
        source = args.data_root / "images" / p["source"]
        p.update(target_path=str(target.relative_to(args.artifact_root)), target_sha256=sha256(target),
                 source_path=str(source.relative_to(args.data_root)), source_sha256=sha256(source))
    sources = {"roma": git_sources(args.roma_dir, cfg["sources"]["roma_commit"]),
               "dinov3": git_sources(args.dinov3_dir, cfg["sources"]["dinov3_commit"])}
    checked(args.weights, cfg["sources"]["weight_sha256"])
    if args.weights.stat().st_size != cfg["sources"]["weight_bytes"]:
        raise ValueError("Official weight size differs")
    snapshot = args.output_dir / "source_snapshot"
    snapshot.mkdir()
    source_hashes = {name: sha256(ROOT / name) for name in SOURCES}
    source_hashes[str(args.config.relative_to(ROOT))] = sha256(args.config)
    for name in source_hashes:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    return {"experiment_id": EXP, "status": "source_and_pair_plan_sealed", "pairs": pairs,
            "config_sha256": sha256(args.config), "source_sha256": source_hashes,
            "upstream_sha256": sources, "weight_sha256": cfg["sources"]["weight_sha256"],
            "camera_images_sha256": sha256(camera_file),
            "camera_intrinsics_sha256": sha256(args.data_root / "camera_parameters/cameras.txt"),
            "freeze_before": freeze, "labels_decoded": 0, "real_target_rgb_decoded": 0,
            "developer_views_decoded": 0, "semantic_outputs_created": 0}


def load_plan(args):
    cfg, freeze = context(args)
    plan = read_json(args.prepare_dir / "result.json")
    if plan["config_sha256"] != sha256(args.config) or plan["freeze_before"] != freeze:
        raise ValueError("Preparation contract changed")
    for name, digest in plan["source_sha256"].items():
        checked(ROOT / name, digest)
        checked(args.prepare_dir / "source_snapshot" / name, digest)
    for kind, path in (("roma", args.roma_dir), ("dinov3", args.dinov3_dir)):
        if git_sources(path, cfg["sources"][kind + "_commit"]) != plan["upstream_sha256"][kind]:
            raise ValueError("Source tree changed")
    checked(args.weights, plan["weight_sha256"])
    checked(args.data_root / "camera_parameters/images.txt", plan["camera_images_sha256"])
    checked(args.data_root / "camera_parameters/cameras.txt", plan["camera_intrinsics_sha256"])
    return cfg, freeze, plan


@contextmanager
def offline_official_load(args, cfg):
    """Only reroute official retrieval to identical local content; no model patch."""
    import torch
    original_hub = torch.hub.load

    def state(url, *pos, **kwargs):
        if url != cfg["sources"]["weight_url"]:
            raise ValueError("Unexpected checkpoint request")
        return torch.load(args.weights, map_location="cpu", weights_only=True)

    def architecture(repo_or_dir, model, *pos, **kwargs):
        expected = "facebookresearch/dinov3:" + cfg["sources"]["dinov3_commit"]
        if repo_or_dir != expected or model != "dinov3_vitl16" or kwargs.get("pretrained") is not False or kwargs.get("weights") is not None:
            raise ValueError("Unexpected backbone request")
        return original_hub(str(args.dinov3_dir), model, source="local", pretrained=False)

    def denied(*args, **kwargs):
        raise RuntimeError("Network forbidden during model execution")

    with patch.object(torch.hub, "load", architecture), patch.object(torch.hub, "load_state_dict_from_url", state), \
            patch.object(socket.socket, "connect", denied), patch.object(socket, "create_connection", denied):
        yield


def array_digest(a):
    import hashlib
    return hashlib.sha256(a.tobytes()).hexdigest()


def run(args):
    cfg, freeze, plan = load_plan(args)
    import torch
    sys.path.insert(0, str(args.roma_dir / "src"))
    from romav2 import RoMaV2
    from romav2.local_correlation import local_corr
    if local_corr is not None:
        raise ValueError("Fixed native correlation backend was replaced")
    if not torch.cuda.is_available():
        raise RuntimeError("Actual CUDA GPU required")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    free, total = torch.cuda.mem_get_info()
    if free / 2**20 < cfg["runtime"]["minimum_free_gpu_mib"]:
        raise RuntimeError("Registered GPU headroom unavailable")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with offline_official_load(args, cfg):
        model = RoMaV2(RoMaV2.Cfg(setting="fast", compile=False)).eval()
        model.requires_grad_(False)
        if (model.H_lr, model.W_lr, model.H_hr, model.W_hr, model.bidirectional) != (512, 512, None, None, False):
            raise ValueError("Official fixed setting differs")
        loading_seconds = time.perf_counter() - start
        records = []

        def match(a, b, name):
            torch.cuda.synchronize(); before = time.perf_counter()
            result = model.match(a, b)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - before
            warp = result["warp_AB"].float().cpu().numpy()[0]
            overlap = result["overlap_AB"].float().cpu().numpy()[0, ..., 0]
            if (warp.shape != (512, 512, 2) or overlap.shape != (512, 512) or
                    not np.isfinite(warp).all() or not np.isfinite(overlap).all() or
                    overlap.min() < 0 or overlap.max() > 1):
                raise ValueError("Invalid dense correspondence schema")
            destination = args.output_dir / (name + ".npz")
            np.savez_compressed(destination, warp=warp, overlap=overlap)
            record = {"name": name, "path": destination.name, "sha256": sha256(destination),
                      "warp_array_sha256": array_digest(warp), "overlap_array_sha256": array_digest(overlap),
                      "seconds": seconds, "shape": list(warp.shape), "finite": True,
                      "in_bounds_fraction": float((np.abs(warp).max(-1) <= 1 - 1 / 512).mean()),
                      "overlap_mean": float(overlap.mean()),
                      "overlap_at_least_half_fraction": float((overlap >= .5).mean())}
            if torch.cuda.max_memory_allocated() / 2**20 > cfg["runtime"]["maximum_peak_allocated_mib"]:
                raise RuntimeError("Registered allocation limit exceeded")
            if time.perf_counter() - start > cfg["runtime"]["maximum_seconds"]:
                raise RuntimeError("Registered runtime budget exceeded")
            records.append(record)
            print({"probe": name, "seconds": round(seconds, 3), "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2)}, flush=True)
            return warp, overlap

        pairs = plan["pairs"][:1] if args.mode == "replay" else plan["pairs"]
        input_access = []
        for i, pair in enumerate(pairs):
            a_path = checked(args.artifact_root / pair["target_path"], pair["target_sha256"])
            b_path = checked(args.data_root / pair["source_path"], pair["source_sha256"])
            a = np.asarray(Image.open(a_path).convert("RGB"))
            b = np.asarray(Image.open(b_path).convert("RGB"))
            a_hash, b_hash = array_digest(a), array_digest(b)
            input_access += [{"kind": "rendered_target", "name": pair["target"], "sha256": pair["target_sha256"]},
                             {"kind": "real_same_fold_training_source", "name": pair["source"], "sha256": pair["source_sha256"]}]
            match(a, b, f"pair{i}_ab")
            if args.mode != "replay":
                match(b, a, f"pair{i}_ba")
                if i == 0:
                    identity, _ = match(a, a, "identity")
                    gy, gx = np.meshgrid(np.arange(512) + .5, np.arange(512) + .5, indexing="ij")
                    error = np.hypot(pixel_center(identity[..., 0], 512) - gx,
                                     pixel_center(identity[..., 1], 512) - gy)
                    records[-1]["self_error_pixels_512"] = {"median": float(np.median(error)), "p90": float(np.quantile(error, .9))}
            if array_digest(a) != a_hash or array_digest(b) != b_hash:
                raise ValueError("RGB arrays modified")
            checked(a_path, pair["target_sha256"]); checked(b_path, pair["source_sha256"])
    replay_exact = None
    if args.mode == "replay":
        old = read_json(args.run_dir / "result.json")
        if old["prepare_sha256"] != sha256(args.prepare_dir / "result.json"):
            raise ValueError("Replay parent differs")
        old_record = next(r for r in old["probes"] if r["name"] == "pair0_ab")
        checked(args.run_dir / old_record["path"], old_record["sha256"])
        for k in ("warp_array_sha256", "overlap_array_sha256"):
            if old_record[k] != records[0][k]:
                raise ValueError("Fresh-process dense replay differs")
        replay_exact = True
    if verify_freeze(ROOT / cfg["freeze_manifest"], args.artifact_root) != freeze:
        raise ValueError("Frozen production changed")
    versions = {n: importlib.metadata.version(n) for n in ("torch", "torchvision", "numpy", "pillow", "einops", "rich", "tqdm")}
    return {"experiment_id": EXP, "status": "passed_new_process_replay" if args.mode == "replay" else "passed_fixed_dense_runtime_preflight",
            "config_sha256": sha256(args.config), "prepare_sha256": sha256(args.prepare_dir / "result.json"),
            "probes": records, "fresh_process_exact": replay_exact, "input_access": input_access,
            "loading_seconds": loading_seconds, "gpu_name": torch.cuda.get_device_name(),
            "total_gpu_mib": total / 2**20, "free_gpu_before_mib": free / 2**20,
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "versions": versions, "python": sys.version, "pillow_module": str(sys.modules["PIL"].__file__),
            "backend": "native_torch_local_corr", "upstream_code_modified": False,
            "frozen_artifacts_verified": freeze["count"], "freeze_after": freeze, "rgb_exact": True,
            "labels_decoded": 0, "real_target_rgb_decoded": 0, "developer_views_decoded": 0,
            "training_steps": 0, "semantic_outputs_created": 0,
            "claim_boundary": "512 resized engineering feasibility only, no matching accuracy or semantic improvement claim"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("prepare", "run", "replay"), required=True)
    for name in ("artifact-root", "data-root", "roma-dir", "dinov3-dir", "weights", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("prepare-dir", "run-dir"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--config", type=Path, default=CONFIG)
    args = p.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path): setattr(args, name, value.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    try:
        if shutil.disk_usage(args.output_dir).free / 2**30 < read_json(args.config)["runtime"]["minimum_free_disk_gib"]:
            raise RuntimeError("Registered disk floor reached")
        result = prepare(args) if args.mode == "prepare" else run(args)
        result["seconds"] = time.perf_counter() - start
        write_json(args.output_dir / "result.json", result)
        print({"status": result["status"], "seconds": round(result["seconds"], 3)}, flush=True)
    except Exception as exc:
        write_json(args.output_dir / "failure.json", {"experiment_id": EXP, "mode": args.mode,
                   "error_type": type(exc).__name__, "error": str(exc), "seconds": time.perf_counter() - start})
        raise


if __name__ == "__main__":
    main()
