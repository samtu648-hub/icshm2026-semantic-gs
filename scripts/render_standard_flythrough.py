#!/usr/bin/env python3
"""Render a frozen Standard-3DGS model over external interpolated COLMAP poses.

The input is only a SIMPLE_RADIAL cameras.txt plus CAMERA_FROM_WORLD poses.
It loads the frozen model once, renders every pose to the official pixel grid,
and records hashes and resource evidence.  No RGB reference image or label is
accepted by this visualization-only entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from render_external_colmap_camera import (  # noqa: E402
    IMAGE_SUFFIXES,
    _data_lines,
    configure_candidate_import_path,
    make_view,
    read_simple_radial_camera,
)
from src.camera_adapter import CameraFromWorld, Sim3, official_to_model_camera, redistort_rgb  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def camera_from_world(fields: list[str]) -> CameraFromWorld:
    qw, qx, qy, qz = (float(value) for value in fields[1:5])
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0.0:
        raise ValueError("Zero-norm COLMAP quaternion")
    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
    rotation = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    return CameraFromWorld(rotation, np.asarray([float(value) for value in fields[5:8]], dtype=np.float64))


def read_poses(path: Path) -> list[tuple[str, int, CameraFromWorld]]:
    poses: list[tuple[str, int, CameraFromWorld]] = []
    seen: set[str] = set()
    for fields in _data_lines(path):
        if len(fields) < 10 or Path(fields[9]).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        name = Path(fields[9]).name
        if name in seen:
            raise ValueError(f"Duplicate pose basename: {name}")
        seen.add(name)
        poses.append((name, int(fields[8]), camera_from_world(fields)))
    if not poses:
        raise ValueError(f"No image poses found in {path}")
    return poses


def render_to_u8(rendered: Any) -> np.ndarray:
    return np.rint(rendered.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    poses = read_poses(args.images)
    camera_ids = {camera_id for _, camera_id, _ in poses}
    if len(camera_ids) != 1:
        raise ValueError("This visualization requires one shared official camera model")
    camera_id = next(iter(camera_ids))
    camera = read_simple_radial_camera(args.cameras, camera_id)[1]
    point_cloud = args.model_path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    if not point_cloud.is_file():
        raise FileNotFoundError(point_cloud)

    configure_candidate_import_path(args.source_root.resolve())
    import torch
    from gaussian_renderer import GaussianModel, render

    pipe = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False,
    )
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    gaussians = GaussianModel(3)
    gaussians.load_ply(str(point_cloud))
    identity = Sim3(1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = args.output_dir / "rgb"
    rgb_dir.mkdir()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    records = []
    for name, pose_camera_id, official_pose in poses:
        if pose_camera_id != camera_id:
            raise RuntimeError("Pose camera changed during render")
        view = make_view(camera, official_to_model_camera(official_pose, identity), appearance_camera_id=0)
        with torch.inference_mode():
            pinhole_rgb = render_to_u8(render(view, gaussians, pipe, background)["render"])
        rgb = np.rint(redistort_rgb(pinhole_rgb.astype(np.float32) / 255.0, camera).clip(0.0, 1.0) * 255.0).astype(np.uint8)
        if rgb.shape != (camera.height, camera.width, 3):
            raise RuntimeError(f"Output dimensions do not match camera for {name}: {rgb.shape}")
        output = rgb_dir / name
        Image.fromarray(rgb, mode="RGB").save(output)
        records.append(
            {
                "pose_name": name,
                "camera_id": camera_id,
                "width": camera.width,
                "height": camera.height,
                "rgb": {"path": output.relative_to(args.output_dir).as_posix(), "sha256": sha256(output)},
            }
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    manifest = {
        "schema_version": 1,
        "visualization_id": "VIS-002-STANDARD-FLYTHROUGH",
        "rgb_experiment": "RGB-011-R2-ALLOC",
        "backbone": "frozen Standard-3DGS",
        "iteration": args.iteration,
        "gaussian_count": int(gaussians.get_xyz.shape[0]),
        "input_camera_model": "SIMPLE_RADIAL",
        "input_pose_convention": "CAMERA_FROM_WORLD",
        "official_to_model_sim3": {"scale": 1.0, "rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0]},
        "output_grid": "official_SIMPLE_RADIAL",
        "uses_reference_rgb_or_labels": False,
        "source_point_cloud_sha256": sha256(point_cloud),
        "cameras_txt_sha256": sha256(args.cameras),
        "images_txt_sha256": sha256(args.images),
        "rendered_views": len(records),
        "elapsed_seconds": elapsed,
        "peak_cuda_allocated_mib": int(torch.cuda.max_memory_allocated() / (1024 * 1024)),
        "outputs": records,
    }
    (args.output_dir / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"elapsed_seconds": elapsed, "rendered_views": len(records), "status": "passed"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
