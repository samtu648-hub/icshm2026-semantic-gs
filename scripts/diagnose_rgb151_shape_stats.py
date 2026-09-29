#!/usr/bin/env python3
"""Compare historical artifact and true-cable Gaussian shape proxies."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData  # type: ignore[import-not-found]


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_cameras import read_camera_requests
from src.quality_protocol import sha256, write_new_json


PLY = (
    ROOT
    / "outputs/quality/rgb140/full-r2/model/point_cloud/iteration_30000/point_cloud.ply"
)
SELECTION = ROOT / "outputs/quality/rgb141/diagnostic-r1/selection.json"
CAMERAS = ROOT / "outputs/quality/all_data/RGB-120-r2/scene/sparse/0"


def quaternion_rotation(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion / np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def row_stats(vertices, index: int, camera) -> dict[str, float | int]:
    point = np.array([vertices[index][axis] for axis in ("x", "y", "z")])
    scales = np.exp(
        np.array(
            [vertices[index][f"scale_{axis}"] for axis in range(3)], dtype=np.float64
        )
    )
    rotation = quaternion_rotation(
        np.array(
            [vertices[index][f"rot_{axis}"] for axis in range(4)], dtype=np.float64
        )
    )
    covariance_world = rotation @ np.diag(scales**2) @ rotation.T
    point_camera = camera.pose.rotation @ point + camera.pose.translation
    x, y, z = point_camera
    if z <= 0:
        raise ValueError(f"Gaussian {index} is behind view 159")
    fx, fy = camera.matrix[0, 0], camera.matrix[1, 1]
    jacobian = np.array([[fx / z, 0.0, -fx * x / z**2], [0.0, fy / z, -fy * y / z**2]])
    covariance_camera = camera.pose.rotation @ covariance_world @ camera.pose.rotation.T
    covariance_image = jacobian @ covariance_camera @ jacobian.T
    eigenvalues = np.linalg.eigvalsh(covariance_image)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    sigma_minor, sigma_major = np.sqrt(eigenvalues)
    return {
        "id": index,
        "max_scale_world": float(scales.max()),
        "min_scale_world": float(scales.min()),
        "anisotropy_max_over_min": float(scales.max() / scales.min()),
        "projected_sigma_major_px": float(sigma_major),
        "projected_sigma_minor_px": float(sigma_minor),
        "projected_anisotropy": float(sigma_major / max(sigma_minor, 1e-12)),
        "projected_3sigma_ellipse_area_px2": float(
            9.0 * math.pi * sigma_major * sigma_minor
        ),
    }


def summarize(rows: list[dict[str, float | int]]) -> dict[str, object]:
    keys = [key for key in rows[0] if key != "id"]
    return {
        "count": len(rows),
        "ids": [row["id"] for row in rows],
        "median": {key: float(np.median([row[key] for row in rows])) for key in keys},
        "range": {
            key: [
                float(np.min([row[key] for row in rows])),
                float(np.max([row[key] for row in rows])),
            ]
            for key in keys
        },
        "rows": rows,
    }


def main() -> None:
    output = ROOT / "outputs/quality/RGB-151/shape-stats-r1"
    output.mkdir(parents=True, exist_ok=False)
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    vertices = PlyData.read(PLY)["vertex"].data
    camera = read_camera_requests(CAMERAS)["159.png"]
    cable_ids = set(
        next(row["ids"] for row in selection if row["region"] == "159-stay_cable")
    )
    artifact_ids = {
        index
        for row in selection
        if row["region"] in {"159-shore", "159-far-occlusion", "159-water-ghost"}
        for index in row["ids"]
    }
    artifact_rows = [
        row_stats(vertices, index, camera) for index in sorted(artifact_ids)
    ]
    cable_rows = [row_stats(vertices, index, camera) for index in sorted(cable_ids)]
    result = {
        "schema_version": 1,
        "status": "completed_readonly",
        "experiment_id": "RGB-151-STANDARD-ABSGRAD-LAS-DEV30",
        "view": "159.png",
        "inputs": {
            "rgb140_ply": str(PLY.relative_to(ROOT)),
            "rgb140_ply_sha256": sha256(PLY),
            "rgb141_selection": str(SELECTION.relative_to(ROOT)),
            "rgb141_selection_sha256": sha256(SELECTION),
            "camera_directory": str(CAMERAS.relative_to(ROOT)),
        },
        "artifact_contributors": summarize(artifact_rows),
        "true_stay_cable_contributors": summarize(cable_rows),
        "interpretation": "Eight opacity-derivative-selected contributors are a small diagnostic sample. Projection uses the local PINHOLE covariance Jacobian without rasterizer low-pass/clipping, so values are shape proxies, not exact rendered pixel support or a population-level causal test.",
    }
    write_new_json(output / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
