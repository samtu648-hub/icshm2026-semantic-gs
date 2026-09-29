#!/usr/bin/env python3
"""Create a smooth novel-view COLMAP trajectory from official camera keyframes.

The generated poses remain in the official reference frame and use COLMAP's
CAMERA_FROM_WORLD convention.  Only camera metadata is read; no RGB image or
semantic label is opened.  Half-step sampling keeps every generated frame away
from the exact keyframe poses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_lines(path: Path) -> list[list[str]]:
    return [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("Quaternion must have finite non-zero norm")
    return q / norm


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = normalize_quaternion(q)
    return np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def slerp(q0: np.ndarray, q1: np.ndarray, amount: float) -> np.ndarray:
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return normalize_quaternion(q0 + amount * (q1 - q0))
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    return normalize_quaternion(
        math.sin((1.0 - amount) * theta) / sin_theta * q0 + math.sin(amount * theta) / sin_theta * q1
    )


def catmull_rom(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, amount: float) -> np.ndarray:
    u = amount
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u * u
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u * u * u
    )


def read_poses(path: Path) -> dict[str, dict[str, Any]]:
    poses: dict[str, dict[str, Any]] = {}
    for fields in data_lines(path):
        if len(fields) < 10 or Path(fields[9]).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        name = Path(fields[9]).name
        q = normalize_quaternion(np.asarray([float(value) for value in fields[1:5]], dtype=np.float64))
        translation = np.asarray([float(value) for value in fields[5:8]], dtype=np.float64)
        rotation = quaternion_to_rotation(q)
        center = -rotation.T @ translation
        poses[name] = {"q": q, "translation": translation, "center": center, "camera_id": int(fields[8])}
    if not poses:
        raise ValueError(f"No COLMAP image poses found in {path}")
    return poses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--keyframes", nargs="+", required=True)
    parser.add_argument("--frames-per-segment", type=int, default=24)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.keyframes) < 2:
        raise ValueError("At least two keyframes are required")
    if args.frames_per_segment < 2 or args.fps <= 0:
        raise ValueError("frames-per-segment must be >=2 and fps must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty trajectory directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    poses = read_poses(args.images)
    names = [Path(name).name for name in args.keyframes]
    missing = [name for name in names if name not in poses]
    if missing:
        raise ValueError(f"Unknown keyframes: {missing}")
    camera_ids = {poses[name]["camera_id"] for name in names}
    if len(camera_ids) != 1:
        raise ValueError("All flythrough keyframes must use the same COLMAP camera")
    camera_id = next(iter(camera_ids))

    generated = []
    image_lines = ["# Novel flythrough poses generated from official camera metadata only."]
    centers = [poses[name]["center"] for name in names]
    quaternions = [poses[name]["q"] for name in names]
    all_official_centers = np.stack([entry["center"] for entry in poses.values()])
    frame_index = 0
    previous_center: np.ndarray | None = None
    max_step = 0.0
    min_official_distance = math.inf
    for segment in range(len(names) - 1):
        p0 = centers[max(segment - 1, 0)]
        p1 = centers[segment]
        p2 = centers[segment + 1]
        p3 = centers[min(segment + 2, len(centers) - 1)]
        for local_index in range(args.frames_per_segment):
            # Never sample exactly at an official keyframe.
            u = (local_index + 0.5) / args.frames_per_segment
            center = catmull_rom(p0, p1, p2, p3, u)
            smooth_u = u * u * (3.0 - 2.0 * u)
            q = slerp(quaternions[segment], quaternions[segment + 1], smooth_u)
            rotation = quaternion_to_rotation(q)
            translation = -rotation @ center
            name = f"frame{frame_index:04d}.png"
            image_lines.append(
                f"{frame_index + 1} "
                + " ".join(f"{value:.17g}" for value in q)
                + " "
                + " ".join(f"{value:.17g}" for value in translation)
                + f" {camera_id} {name}"
            )
            image_lines.append("")
            nearest = float(np.linalg.norm(all_official_centers - center[None, :], axis=1).min())
            min_official_distance = min(min_official_distance, nearest)
            if previous_center is not None:
                max_step = max(max_step, float(np.linalg.norm(center - previous_center)))
            previous_center = center
            generated.append(
                {
                    "name": name,
                    "segment": segment,
                    "u": u,
                    "center": center.tolist(),
                    "qvec_camera_from_world": q.tolist(),
                    "translation_camera_from_world": translation.tolist(),
                    "nearest_official_camera_center_distance": nearest,
                }
            )
            frame_index += 1

    shutil.copy2(args.cameras, args.output_dir / "cameras.txt")
    (args.output_dir / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "trajectory_id": "frozen_scaffold_sem001_bridge_flythrough_v1",
        "pose_convention": "CAMERA_FROM_WORLD",
        "coordinate_frame": "official_colmap_reference",
        "camera_model_source": "copied official cameras.txt",
        "keyframes": names,
        "keyframes_are_metadata_only": True,
        "generated_frames_correspond_to_dataset_images": False,
        "sampling": "half-step Catmull-Rom camera centers plus eased quaternion SLERP",
        "frames_per_segment": args.frames_per_segment,
        "frame_count": len(generated),
        "fps": args.fps,
        "duration_seconds": len(generated) / args.fps,
        "minimum_generated_to_official_camera_center_distance": min_official_distance,
        "maximum_generated_center_step": max_step,
        "uses_reference_rgb_or_labels": False,
        "input_cameras_sha256": sha256(args.cameras),
        "input_images_sha256": sha256(args.images),
        "generated": generated,
    }
    (args.output_dir / "trajectory_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in ("trajectory_id", "frame_count", "fps", "duration_seconds")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
