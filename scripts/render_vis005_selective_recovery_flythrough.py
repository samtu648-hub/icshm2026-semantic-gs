#!/usr/bin/env python3
"""Render the three registered SEM-123 rules on the VIS-001 camera path.

The script is intentionally a developer-only visual check.  It uses only the
novel official camera metadata, frozen Scaffold RGB, SEM-001 and SEM-102; it
never opens a label or a reference image and never tunes a rule.
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
from src.semantic_band_recovery import recover_selective_band
from src.semantic_protected_fusion import CLASS_COUNT, fuse_background_stay_cable

PALETTE = np.asarray(
    [[0, 0, 0], [230, 86, 37], [255, 210, 0], [38, 139, 210], [133, 153, 0]],
    dtype=np.uint8,
)
PROTECTED_CLASSES = np.asarray([1, 3, 4], dtype=np.uint8)


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
        raise RuntimeError("VIS-005 requires ffmpeg for H.264 output") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sem001-config", type=Path, required=True)
    parser.add_argument("--sem102-config", type=Path, required=True)
    parser.add_argument("--sem123-result", type=Path, required=True)
    parser.add_argument("--sem001-checkpoint", type=Path, required=True)
    parser.add_argument("--sem102-final-result", type=Path, required=True)
    parser.add_argument("--sem102-checkpoint", type=Path, required=True)
    parser.add_argument("--sem102-model-dir", type=Path, required=True)
    parser.add_argument("--rgb-run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Resume only a verified partial VIS-005 output")
    return parser.parse_args()


def comparison_columns(config: dict[str, Any]) -> list[dict[str, Any]]:
    columns = config.get("comparison", {}).get("columns")
    require(isinstance(columns, list) and len(columns) == 3, "VIS-005 requires exactly three comparison columns")
    expected_ids = ["hard030", "sem123_leader", "sem123_conservative"]
    require([column.get("id") for column in columns] == expected_ids, "VIS-005 comparison order changed")
    baseline, leader, conservative = columns
    require(baseline.get("rule") == {"kind": "protected_hard_threshold", "stay_cable_threshold": 0.30}, "VIS-005 hard-0.30 baseline changed")
    for column in (leader, conservative):
        rule = column.get("rule", {})
        require(rule.get("kind") == "selective_band_recovery", "VIS-005 requires registered SEM-123 recovery rules")
        require(float(rule.get("lower_threshold")) == 0.20 and float(rule.get("upper_threshold")) == 0.30, "VIS-005 recovery band changed")
        require(float(rule.get("probability_minimum")) == 0.20, "VIS-005 recovery probability floor changed")
    require(leader["rule"].get("maximum_seed_distance") == 2.0 and leader["rule"].get("prior_mode") == "any", "VIS-005 leader changed")
    require(conservative["rule"].get("maximum_seed_distance") == 4.0 and conservative["rule"].get("prior_mode") == "sem001_stay_cable", "VIS-005 conservative candidate changed")
    require(all(isinstance(column.get("header"), str) and column["header"] for column in columns), "VIS-005 headers are invalid")
    return columns


def validate_inputs(args: argparse.Namespace, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    require(config.get("experiment_id") == "VIS-005-SELECTIVE-RECOVERY-FLYTHROUGH", "VIS-005 received another config")
    require(config.get("status") == "registered_not_run", "VIS-005 may run only from its registered config")
    columns = comparison_columns(config)
    sem001_config = common.load_json(args.sem001_config)
    sem102_config = common.load_json(args.sem102_config)
    sem123 = common.load_json(args.sem123_result)
    require(sem001_config.get("experiment_id") == "SEM-001", "VIS-005 requires SEM-001 config")
    require(sem102_config.get("experiment_id") == "SEM-102-RENDERED-SEG-FINAL", "VIS-005 requires SEM-102 config")
    require(sha256(args.sem123_result) == config["frozen_inputs"]["sem123_result_sha256"], "VIS-005 SEM-123 result hash differs")
    require(sha256(args.sem001_checkpoint) == config["frozen_inputs"]["sem001_checkpoint_sha256"], "VIS-005 SEM-001 checkpoint hash differs")
    require(sha256(args.sem102_checkpoint) == config["frozen_inputs"]["sem102_final_checkpoint_sha256"], "VIS-005 SEM-102 checkpoint hash differs")
    require(sha256(args.sem102_final_result) == config["frozen_inputs"]["sem102_final_result_sha256"], "VIS-005 SEM-102 final result hash differs")
    require(sem123.get("experiment_id") == "SEM-123-BAND-RECOVERY-DEV-SWEEP" and sem123.get("status") == "completed_development_sweep", "VIS-005 requires completed SEM-123")
    leader = sem123.get("selected_record", {})
    conservative = next((record for record in sem123.get("candidate_records", []) if record.get("id") == "p20_d04_sem001_stay_cable"), {})
    require(leader.get("id") == "p20_d02_any", "VIS-005 SEM-123 score leader differs")
    require(leader.get("protected_prediction_pixels_exact") is True and conservative.get("protected_prediction_pixels_exact") is True, "VIS-005 requires SEM-123 protected records")
    for record, column in ((leader, columns[1]), (conservative, columns[2])):
        rule = column["rule"]
        require(float(record.get("probability_minimum")) == float(rule["probability_minimum"]), "VIS-005 recovery probability differs from SEM-123")
        require(float(record.get("maximum_seed_distance")) == float(rule["maximum_seed_distance"]), "VIS-005 recovery distance differs from SEM-123")
        require(record.get("sem001_prior_mode") == rule["prior_mode"], "VIS-005 recovery prior differs from SEM-123")
    require(args.sem001_checkpoint.is_file() and args.sem102_checkpoint.is_file(), "VIS-005 checkpoint is missing")
    sem102_result = common.load_json(args.sem102_final_result)
    require(sem102_result.get("checkpoint_reload") == "passed", "VIS-005 requires a reloaded SEM-102 checkpoint")
    require(sem102_result.get("checkpoint_sha256") == sha256(args.sem102_checkpoint), "VIS-005 SEM-102 result checkpoint hash differs")
    return sem001_config, sem102_config, sem102_result


def rendered_labels(base: np.ndarray, teacher: np.ndarray, columns: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    baseline, fixed = fuse_background_stay_cable(base, teacher, stay_cable_threshold=0.30)
    labels: dict[str, np.ndarray] = {columns[0]["id"]: baseline.astype(np.uint8)}
    for column in columns[1:]:
        rule = column["rule"]
        candidate, _selected, candidate_fixed = recover_selective_band(
            base,
            teacher,
            probability_minimum=float(rule["probability_minimum"]),
            maximum_seed_distance=float(rule["maximum_seed_distance"]),
            prior_mode=str(rule["prior_mode"]),
            lower_threshold=float(rule["lower_threshold"]),
            upper_threshold=float(rule["upper_threshold"]),
        )
        require(np.array_equal(candidate_fixed, fixed), "VIS-005 protected mask drifted")
        labels[column["id"]] = candidate.astype(np.uint8)
    return labels, fixed


def audit_frame(labels: dict[str, np.ndarray], *, shape: tuple[int, int]) -> dict[str, Any]:
    baseline = labels["hard030"]
    require(baseline.shape == shape and int(baseline.max()) < CLASS_COUNT, "VIS-005 baseline semantic PNG is invalid")
    protected = np.isin(baseline, PROTECTED_CLASSES)
    audit: dict[str, Any] = {"protected_matches_hard030": {}, "stay_cable_additions_over_hard030": {}}
    for identifier, value in labels.items():
        require(value.shape == shape and int(value.max()) < CLASS_COUNT, f"VIS-005 {identifier} semantic PNG is invalid")
        require(np.array_equal(value[protected], baseline[protected]), f"VIS-005 {identifier} changed a protected class")
        require(not np.any((baseline == 2) & (value != 2)), f"VIS-005 {identifier} deleted a hard-0.30 stay_cable pixel")
        audit["protected_matches_hard030"][identifier] = True
        audit["stay_cable_additions_over_hard030"][identifier] = int(np.count_nonzero((value == 2) & (baseline != 2)))
    return audit


def main() -> int:
    args = parse_args()
    config = common.load_json(args.config)
    sem001_config, sem102_config, _sem102_result = validate_inputs(args, config)
    columns = comparison_columns(config)
    trajectory = common.load_json(args.trajectory_dir / "trajectory_manifest.json")
    require(trajectory.get("trajectory_id") == config["trajectory"]["trajectory_id"], "VIS-005 trajectory differs")
    require(int(trajectory.get("frame_count", -1)) == 144, "VIS-005 requires 144 poses")
    require(trajectory.get("pose_convention") == "CAMERA_FROM_WORLD", "VIS-005 needs official pose convention")
    require(trajectory.get("uses_reference_rgb_or_labels") is False, "VIS-005 trajectory accessed reference data")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        require(args.resume, "VIS-005 refuses to overwrite non-empty output without --resume")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("VIS-005 requires visible CUDA for frozen rendered-domain inference")
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
    require(teacher_metadata.get("experiment_id") == "SEM-102-RENDERED-SEG-FINAL", "VIS-005 loaded another SEM-102 checkpoint")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    semantic_roots = {column["id"]: args.output_dir / f"semantic_{column['id']}" for column in columns}
    frame_root = args.output_dir / "comparison_frames"
    for directory in (*semantic_roots.values(), frame_root):
        directory.mkdir(parents=True, exist_ok=True)
    camera_id, camera = read_simple_radial_camera(args.trajectory_dir / "cameras.txt", None)
    names = [entry["name"] for entry in trajectory["generated"]]
    require(len(names) == 144 and len(set(names)) == 144, "VIS-005 trajectory names are invalid")
    pipe = pipeline()
    identity = Sim3(1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
    crop = int(sem102_config["inference"]["crop_size"])
    stride = int(sem102_config["inference"]["stride"])
    started = time.perf_counter()
    frame_records: list[dict[str, Any]] = []
    resumed_frames = 0
    global_additions = {column["id"]: 0 for column in columns}
    protected_exact = True
    for index, name in enumerate(names):
        stem = Path(name).stem
        paths_by_id = {column["id"]: semantic_roots[column["id"]] / f"{stem}.png" for column in columns}
        frame_path = frame_root / f"frame{index:04d}.png"
        was_resumed = args.resume and frame_path.is_file() and all(path.is_file() for path in paths_by_id.values())
        if was_resumed:
            labels = {identifier: np.asarray(Image.open(path), dtype=np.uint8) for identifier, path in paths_by_id.items()}
            audit = audit_frame(labels, shape=(camera.height, camera.width))
            record_name = name
            resumed_frames += 1
        else:
            selected_name, pose = read_pose(args.trajectory_dir / "images.txt", name, camera_id)
            view = make_view(camera, official_to_model_camera(pose, identity), int(frozen["appearance_camera_id"]))
            with torch.inference_mode():
                rgb_pinhole = sem001.tensor_to_rgb(render_frozen_rgb(view, model, pipe))
                base_pinhole = render_semantic_probabilities(view, model, sidecar, pipe).cpu().numpy().astype(np.float32)
            teacher_pinhole = segformer.sliding_probabilities(teacher, rgb_pinhole, crop, stride, device, amp_dtype)
            base = redistort_semantic_probabilities(base_pinhole, camera).argmax(axis=0).astype(np.uint8)
            teacher_prob = redistort_semantic_probabilities(teacher_pinhole, camera)
            labels, _fixed = rendered_labels(base, teacher_prob, columns)
            audit = audit_frame(labels, shape=(camera.height, camera.width))
            record_name = selected_name
            for identifier, path in paths_by_id.items():
                Image.fromarray(labels[identifier], mode="L").save(path)
            header_height = 43
            canvas = Image.new("RGB", (camera.width * len(columns), camera.height + header_height), (20, 20, 20))
            draw = ImageDraw.Draw(canvas)
            for column_index, column in enumerate(columns):
                draw.text((column_index * camera.width + 12, 14), column["header"], fill="white")
                canvas.paste(Image.fromarray(PALETTE[labels[column["id"]]], mode="RGB"), (column_index * camera.width, header_height))
            canvas.save(frame_path)
        protected_exact = protected_exact and all(audit["protected_matches_hard030"].values())
        for identifier, count in audit["stay_cable_additions_over_hard030"].items():
            global_additions[identifier] += int(count)
        paths = {identifier: str(path.relative_to(args.output_dir)) for identifier, path in paths_by_id.items()}
        hashes = {identifier: sha256(path) for identifier, path in paths_by_id.items()}
        frame_records.append({
            "index": index,
            "pose_name": record_name,
            "semantic_paths": paths,
            "semantic_sha256": hashes,
            "comparison_frame": str(frame_path.relative_to(args.output_dir)),
            "protected_matches_hard030": audit["protected_matches_hard030"],
            "stay_cable_additions_over_hard030": audit["stay_cable_additions_over_hard030"],
            "resumed_existing_frame": was_resumed,
        })
        if index == 0 or (index + 1) % 12 == 0 or index + 1 == len(names):
            print(json.dumps({"frame": index + 1, "frame_count": len(names), "pose_name": record_name}), flush=True)
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
        "sem123_result_sha256": sha256(args.sem123_result),
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
        "comparison_columns": columns,
        "stay_cable_additions_over_hard030": global_additions,
        "resumed_existing_frame_count": resumed_frames,
        "amp": amp_name,
        "video": {"path": video.name, "sha256": sha256(video), "codec": "H.264 yuv420p", "resolution": [camera.width * len(columns), camera.height + 43]},
        "frames": frame_records,
    }
    common.write_json(args.output_dir / "render_manifest.json", result)
    print(json.dumps({"status": result["status"], "frame_count": result["frame_count"], "video_sha256": result["video"]["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
