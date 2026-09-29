#!/usr/bin/env python3
"""Render the VIS-001 trajectory with three frozen protected-fusion thresholds.

This is an explicitly developer-only visualization.  It never opens labels or
reference RGB: each frame is generated from an official novel camera pose,
frozen Scaffold RGB, the frozen SEM-001 sidecar and frozen SEM-102 teacher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem001 as sem001
import run_sem005_segformer as segformer
import run_sem080_structural_diag as common
from render_external_colmap_camera import make_view, read_pose, read_simple_radial_camera

from src.camera_adapter import Sim3, official_to_model_camera, redistort_semantic_probabilities
from src.scaffold_semantic_sidecar import (
    ScaffoldSettings,
    configure_scaffold_import,
    frozen_rgb_hashes,
    load_frozen_scaffold,
    pipeline,
    render_frozen_rgb,
    render_semantic_probabilities,
)
from src.semantic_protected_fusion import fuse_background_stay_cable

CLASS_COUNT = 5
PALETTE = np.asarray(
    [[0, 0, 0], [230, 86, 37], [255, 210, 0], [38, 139, 210], [133, 153, 0]],
    dtype=np.uint8,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def save_video(frame_dir: Path, output: Path, fps: int) -> None:
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
        "-i", str(frame_dir / "frame%04d.png"), "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p", str(output),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("VIS-004 requires ffmpeg for H.264 output") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sem001-config", type=Path, required=True)
    parser.add_argument("--sem102-config", type=Path, required=True)
    parser.add_argument("--sem120-result", type=Path, required=True)
    parser.add_argument("--sem001-checkpoint", type=Path, required=True)
    parser.add_argument("--sem102-final-result", type=Path, required=True)
    parser.add_argument("--sem102-checkpoint", type=Path, required=True)
    parser.add_argument("--sem102-model-dir", type=Path, required=True)
    parser.add_argument("--rgb-run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Resume only a verified partial VIS-004 output")
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    require(config.get("experiment_id") == "VIS-004-DEV-THRESHOLD-FLYTHROUGH", "VIS-004 received another config")
    require(config.get("status") == "registered_not_run", "VIS-004 may run only from its registered config")
    sem001_config = common.load_json(args.sem001_config)
    sem102_config = common.load_json(args.sem102_config)
    sem120 = common.load_json(args.sem120_result)
    require(sem001_config.get("experiment_id") == "SEM-001", "VIS-004 requires SEM-001 config")
    require(sem102_config.get("experiment_id") == "SEM-102-RENDERED-SEG-FINAL", "VIS-004 requires SEM-102 config")
    require(sha256(args.sem120_result) == config["frozen_inputs"]["sem120_result_sha256"], "VIS-004 SEM-120 result hash differs")
    require(sha256(args.sem001_checkpoint) == config["frozen_inputs"]["sem001_checkpoint_sha256"], "VIS-004 SEM-001 checkpoint hash differs")
    require(sha256(args.sem102_checkpoint) == config["frozen_inputs"]["sem102_final_checkpoint_sha256"], "VIS-004 SEM-102 checkpoint hash differs")
    require(sha256(args.sem102_final_result) == config["frozen_inputs"]["sem102_final_result_sha256"], "VIS-004 SEM-102 final result hash differs")
    require(sem120.get("selected_development_record", {}).get("stay_cable_threshold") == config["comparison"]["sem120_developer_leader_threshold"], "VIS-004 leader threshold differs from SEM-120")
    thresholds = {
        config["comparison"]["sem111_historical_threshold"],
        config["comparison"]["sem120_developer_leader_threshold"],
        config["comparison"]["sem120_pareto_threshold"],
    }
    require(thresholds == {0.2, 0.3, 0.8}, "VIS-004 threshold contract changed")
    require(args.sem001_checkpoint.is_file() and args.sem102_checkpoint.is_file(), "VIS-004 checkpoint is missing")
    result = common.load_json(args.sem102_final_result)
    require(result.get("checkpoint_reload") == "passed", "VIS-004 requires a reloaded SEM-102 checkpoint")
    require(result.get("checkpoint_sha256") == sha256(args.sem102_checkpoint), "VIS-004 SEM-102 result checkpoint hash differs")
    return sem001_config, sem102_config, result


def main() -> int:
    args = parse_args()
    config = common.load_json(args.config)
    sem001_config, sem102_config, sem102_result = validate_inputs(args, config)
    trajectory = common.load_json(args.trajectory_dir / "trajectory_manifest.json")
    require(trajectory.get("trajectory_id") == config["trajectory"]["trajectory_id"], "VIS-004 trajectory differs")
    require(int(trajectory.get("frame_count", -1)) == 144, "VIS-004 requires 144 poses")
    require(trajectory.get("pose_convention") == "CAMERA_FROM_WORLD", "VIS-004 needs official pose convention")
    require(trajectory.get("uses_reference_rgb_or_labels") is False, "VIS-004 trajectory accessed reference data")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        require(args.resume, "VIS-004 refuses to overwrite non-empty output without --resume")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("VIS-004 requires visible CUDA for frozen rendered-domain inference")
    device = torch.device("cuda:0")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    amp_name = "bf16" if amp_dtype == torch.bfloat16 else "fp16"
    frozen = sem102_config["frozen_rgb"]
    settings = ScaffoldSettings(train_camera_count=int(frozen["train_camera_count"]), n_offsets=int(frozen["n_offsets"]))
    point_cloud = args.rgb_run_dir / "point_cloud" / f"iteration_{frozen['iteration']}" / "point_cloud.ply"
    rgb_hashes = frozen_rgb_hashes(args.rgb_run_dir, int(frozen["iteration"]))
    configure_scaffold_import(args.source_root)
    model = load_frozen_scaffold(args.source_root, point_cloud, settings)
    sidecar = sem001.reload_sidecar(args.sem001_checkpoint, int(model.get_anchor.shape[0]), sem001_config, rgb_hashes)
    teacher, teacher_metadata = segformer.reload_checkpoint(args.sem102_checkpoint, args.sem102_model_dir, int(sem102_config["training"]["seed"]), device)
    require(teacher_metadata.get("experiment_id") == "SEM-102-RENDERED-SEG-FINAL", "VIS-004 loaded another SEM-102 checkpoint")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    semantic_roots = {0.8: args.output_dir / "semantic_sem111_080", 0.2: args.output_dir / "semantic_dev_020", 0.3: args.output_dir / "semantic_dev_030"}
    frame_root = args.output_dir / "comparison_frames"
    for directory in (*semantic_roots.values(), frame_root):
        directory.mkdir(parents=True, exist_ok=True)
    cameras_file = args.trajectory_dir / "cameras.txt"
    images_file = args.trajectory_dir / "images.txt"
    camera_id, camera = read_simple_radial_camera(cameras_file, None)
    names = [entry["name"] for entry in trajectory["generated"]]
    require(len(names) == 144 and len(set(names)) == 144, "VIS-004 trajectory names are invalid")
    pipe = pipeline()
    identity = Sim3(1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
    crop = int(sem102_config["inference"]["crop_size"])
    stride = int(sem102_config["inference"]["stride"])
    headers = ["SEM-111 threshold 0.80", "SEM-120 dev leader 0.20", "SEM-120 Pareto 0.30"]
    thresholds = [0.8, 0.2, 0.3]
    started = time.perf_counter()
    frame_records: list[dict[str, Any]] = []
    protected_exact = True
    resumed_frames = 0
    for index, name in enumerate(names):
        stem = Path(name).stem
        existing_paths = {threshold: semantic_roots[threshold] / f"{stem}.png" for threshold in thresholds}
        frame_path = frame_root / f"frame{index:04d}.png"
        if args.resume and frame_path.is_file() and all(path.is_file() for path in existing_paths.values()):
            labels = {threshold: np.asarray(Image.open(path), dtype=np.uint8) for threshold, path in existing_paths.items()}
            require(
                all(value.shape == (camera.height, camera.width) and int(value.max()) < CLASS_COUNT for value in labels.values()),
                f"VIS-004 partial frame has an invalid semantic PNG: {name}",
            )
            paths = {f"threshold_{threshold:.2f}": str(path.relative_to(args.output_dir)) for threshold, path in existing_paths.items()}
            hashes = {f"threshold_{threshold:.2f}": sha256(path) for threshold, path in existing_paths.items()}
            frame_records.append({"index": index, "pose_name": name, "semantic_paths": paths, "semantic_sha256": hashes, "comparison_frame": str(frame_path.relative_to(args.output_dir)), "resumed_existing_frame": True})
            resumed_frames += 1
            continue
        selected_name, pose = read_pose(images_file, name, camera_id)
        view = make_view(camera, official_to_model_camera(pose, identity), int(frozen["appearance_camera_id"]))
        with torch.inference_mode():
            rgb_pinhole = sem001.tensor_to_rgb(render_frozen_rgb(view, model, pipe))
            base_pinhole = render_semantic_probabilities(view, model, sidecar, pipe).cpu().numpy().astype(np.float32)
        teacher_pinhole = segformer.sliding_probabilities(teacher, rgb_pinhole, crop, stride, device, amp_dtype)
        base = redistort_semantic_probabilities(base_pinhole, camera).argmax(axis=0).astype(np.uint8)
        teacher_prob = redistort_semantic_probabilities(teacher_pinhole, camera)
        labels: dict[float, np.ndarray] = {}
        for threshold in thresholds:
            fused, fixed = fuse_background_stay_cable(base, teacher_prob, stay_cable_threshold=threshold)
            protected_exact = protected_exact and bool(np.array_equal(base[fixed], fused[fixed]))
            labels[threshold] = fused.astype(np.uint8)
        stem = Path(selected_name).stem
        paths: dict[str, str] = {}
        hashes: dict[str, str] = {}
        for threshold in thresholds:
            path = semantic_roots[threshold] / f"{stem}.png"
            Image.fromarray(labels[threshold], mode="L").save(path)
            key = f"threshold_{threshold:.2f}"
            paths[key] = str(path.relative_to(args.output_dir))
            hashes[key] = sha256(path)
        header_height = 43
        canvas = Image.new("RGB", (camera.width * 3, camera.height + header_height), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        for column, (threshold, header) in enumerate(zip(thresholds, headers, strict=True)):
            draw.text((column * camera.width + 12, 14), header, fill="white")
            canvas.paste(Image.fromarray(PALETTE[labels[threshold]], mode="RGB"), (column * camera.width, header_height))
        canvas.save(frame_path)
        frame_records.append({"index": index, "pose_name": selected_name, "semantic_paths": paths, "semantic_sha256": hashes, "comparison_frame": str(frame_path.relative_to(args.output_dir))})
        if index == 0 or (index + 1) % 12 == 0 or index + 1 == len(names):
            print(json.dumps({"frame": index + 1, "frame_count": len(names), "pose_name": selected_name}), flush=True)
    del teacher
    torch.cuda.empty_cache()
    video = args.output_dir / config["outputs"]["comparison_video"]
    save_video(frame_root, video, int(config["trajectory"]["fps"]))
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "completed_developer_visualization",
        "developer_only": True,
        "trajectory_manifest_sha256": sha256(args.trajectory_dir / "trajectory_manifest.json"),
        "sem120_result_sha256": sha256(args.sem120_result),
        "sem001_checkpoint_sha256": sha256(args.sem001_checkpoint),
        "sem102_final_checkpoint_sha256": sha256(args.sem102_checkpoint),
        "sem102_final_result_sha256": sha256(args.sem102_final_result),
        "frame_count": len(frame_records),
        "fps": int(config["trajectory"]["fps"]),
        "duration_seconds": len(frame_records) / int(config["trajectory"]["fps"]),
        "render_seconds": time.perf_counter() - started,
        "input_pose_convention": "CAMERA_FROM_WORLD",
        "input_camera_model": "SIMPLE_RADIAL",
        "uses_reference_rgb_or_labels": False,
        "rgb_geometry_color_opacity_checkpoint_unchanged": True,
        "protected_sem001_classes_exact": protected_exact,
        "resumed_existing_frame_count": resumed_frames,
        "protected_sem001_classes_exact_for_newly_rendered_frames": protected_exact,
        "comparison_thresholds": thresholds,
        "amp": amp_name,
        "video": {"path": video.name, "sha256": sha256(video), "codec": "H.264 yuv420p", "resolution": [camera.width * 3, camera.height + 43]},
        "frames": frame_records,
    }
    (args.output_dir / "render_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "frame_count": result["frame_count"], "video_sha256": result["video"]["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
