#!/usr/bin/env python3
"""Render frozen RGB for one externally supplied SEM-102 camera after freeze.

This is the Scaffold-runtime half of the official external-camera gate. It
produces a PINHOLE RGB input for the SegFormer runtime and an official
SIMPLE_RADIAL RGB output, but never loads a reference image, label, or the 2D
model itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem001 as s0
import run_sem080_structural_diag as s80
from render_external_colmap_camera import (
    make_view,
    read_pose,
    read_simple_radial_camera,
)

from src.camera_adapter import Sim3, official_to_model_camera, redistort_rgb
from src.scaffold_semantic_sidecar import (
    ScaffoldSettings,
    configure_scaffold_import,
    frozen_rgb_hashes,
    frozen_rgb_named_parameters,
    load_frozen_scaffold,
    pipeline,
    render_frozen_rgb,
)
from src.semantic_rendered_final import validate_final_training_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--final-result", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--rgb-run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--image-name", required=True)
    parser.add_argument("--camera-id", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def save_rgb(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)
    return s80.sha256(path)


def main() -> int:
    args = parse_args()
    config = s80.load_json(args.config)
    require(
        config.get("experiment_id") == "SEM-102-RENDERED-SEG-FINAL",
        "SEM-102 external RGB renderer received another config",
    )
    require(
        config.get("status") == "registered_not_run",
        "SEM-102 external RGB renderer requires its registered-not-run config",
    )
    require(args.final_checkpoint.is_file(), "SEM-102 final checkpoint is missing")
    final_result = s80.load_json(args.final_result)
    checkpoint_sha256 = s80.sha256(args.final_checkpoint)
    validate_final_training_result(
        final_result,
        checkpoint_sha256=checkpoint_sha256,
        config_sha256=s80.sha256(args.config),
    )
    require(
        not args.output_dir.exists() or not any(args.output_dir.iterdir()),
        "Refusing to overwrite SEM-102 external RGB output",
    )

    import torch

    frozen = config["frozen_rgb"]
    iteration = int(frozen["iteration"])
    settings = ScaffoldSettings(
        train_camera_count=int(frozen["train_camera_count"]),
        n_offsets=int(frozen["n_offsets"]),
    )
    point_cloud = (
        args.rgb_run_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    )
    artifact_before = frozen_rgb_hashes(args.rgb_run_dir, iteration)
    configure_scaffold_import(args.source_root)
    scaffold = load_frozen_scaffold(args.source_root, point_cloud, settings)
    require(
        not any(
            parameter.requires_grad
            for _name, parameter in frozen_rgb_named_parameters(scaffold)
        ),
        "SEM-102 external RGB renderer found a trainable frozen RGB tensor",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("SEM-102 external RGB renderer requires CUDA")
    camera_id, camera = read_simple_radial_camera(args.cameras, args.camera_id)
    selected_name, pose = read_pose(args.images, args.image_name, camera_id)
    identity = Sim3(1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
    view = make_view(
        camera,
        official_to_model_camera(pose, identity),
        int(frozen["appearance_camera_id"]),
    )
    with torch.inference_mode():
        pinhole = s0.tensor_to_rgb(render_frozen_rgb(view, scaffold, pipeline()))
    official = np.rint(
        redistort_rgb(pinhole.astype(np.float32) / 255.0, camera).clip(0.0, 1.0) * 255.0
    ).astype(np.uint8)
    require(
        pinhole.shape == (camera.height, camera.width, 3),
        "SEM-102 external PINHOLE RGB size differs from input camera",
    )
    require(
        official.shape == (camera.height, camera.width, 3),
        "SEM-102 external official RGB size differs from input camera",
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pinhole_sha = save_rgb(args.output_dir / "rgb_pinhole.png", pinhole)
    official_sha = save_rgb(args.output_dir / "rgb.png", official)
    artifact_after = frozen_rgb_hashes(args.rgb_run_dir, iteration)
    require(
        artifact_before == artifact_after,
        "Frozen RGB artifact changed during SEM-102 external RGB render",
    )
    record = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "completed_external_frozen_rgb",
        "final_training_result_sha256": s80.sha256(args.final_result),
        "final_checkpoint_sha256": checkpoint_sha256,
        "selected_pose_name": selected_name,
        "input_camera_model": "SIMPLE_RADIAL",
        "input_pose_convention": "CAMERA_FROM_WORLD",
        "official_to_model_sim3": {
            "scale": 1.0,
            "rotation": np.eye(3).tolist(),
            "translation": [0.0, 0.0, 0.0],
        },
        "camera": {
            "camera_id": camera_id,
            "width": camera.width,
            "height": camera.height,
            "focal": camera.focal,
            "cx": camera.cx,
            "cy": camera.cy,
            "radial_k": camera.radial_k,
        },
        "pinhole_rgb": {
            "path": "rgb_pinhole.png",
            "sha256": pinhole_sha,
            "shape": list(pinhole.shape),
        },
        "official_rgb": {
            "path": "rgb.png",
            "sha256": official_sha,
            "shape": list(official.shape),
        },
        "rgb_artifact_hashes_before": artifact_before,
        "rgb_artifact_hashes_after": artifact_after,
        "rgb_artifact_hashes_equal": artifact_before == artifact_after,
        "uses_reference_rgb_or_labels": False,
        "new_parameters_trained": False,
    }
    s80.write_json(args.output_dir / "external_rgb_record.json", record)
    print(json.dumps(record, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
