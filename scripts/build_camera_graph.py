#!/usr/bin/env python3
"""Build a deterministic camera co-visibility graph from official COLMAP text files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geometry_prior.camera_io import read_cameras_txt, read_images_txt  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-shared-points", type=int, default=8)
    parser.add_argument("--max-edges", type=int, default=32)
    return parser.parse_args()


def _image_sort_key(image) -> tuple[int, str, int]:
    stem = Path(image.name).stem
    try:
        return (0, "", int(stem))
    except ValueError:
        return (1, stem, image.image_id)


def build_graph(camera_dir: Path, min_shared_points: int, max_edges: int) -> dict:
    cameras = read_cameras_txt(camera_dir / "cameras.txt")
    images = sorted(read_images_txt(camera_dir / "images.txt").values(), key=_image_sort_key)
    nodes = []
    for image in images:
        camera = cameras[image.camera_id]
        nodes.append(
            {
                "image_id": image.image_id,
                "name": image.name,
                "camera_id": image.camera_id,
                "center": image.center.tolist(),
                "optical_axis": image.optical_axis.tolist(),
                "width": camera.width,
                "height": camera.height,
                "model": camera.model,
                "point3d_count": len(image.point3d_ids),
            }
        )

    edges = []
    for left_index, left in enumerate(images):
        left_ids = left.point3d_ids
        for right in images[left_index + 1 :]:
            shared = len(left_ids.intersection(right.point3d_ids))
            if shared < min_shared_points:
                continue
            center_distance = float(np.linalg.norm(left.center - right.center))
            axis_cosine = float(np.clip(np.dot(left.optical_axis, right.optical_axis), -1.0, 1.0))
            edges.append(
                {
                    "source": left.name,
                    "target": right.name,
                    "shared_point3d": shared,
                    "center_distance": center_distance,
                    "optical_axis_angle_deg": float(np.degrees(np.arccos(axis_cosine))),
                }
            )
    edges.sort(key=lambda edge: (-edge["shared_point3d"], edge["source"], edge["target"]))
    adjacency = {node["name"]: [] for node in nodes}
    for edge in edges:
        adjacency[edge["source"]].append(edge)
        adjacency[edge["target"]].append(
            {
                **edge,
                "source": edge["target"],
                "target": edge["source"],
            }
        )
    for name, values in adjacency.items():
        values.sort(key=lambda edge: (-edge["shared_point3d"], edge["target"]))
        del values[max_edges:]
    payload = {
        "schema_version": 1,
        "source": "official COLMAP images.txt",
        "pose_convention": "CAMERA_FROM_WORLD; centers C=-R^T t; optical axis is c2w +Z",
        "camera_model": sorted({camera.model for camera in cameras.values()}),
        "camera_count": len(cameras),
        "image_count": len(images),
        "min_shared_points": min_shared_points,
        "max_edges_per_node": max_edges,
        "nodes": nodes,
        "edges": edges,
        "adjacency": adjacency,
    }
    return payload


def main() -> int:
    args = _parse_args()
    payload = build_graph(args.camera_dir, args.min_shared_points, args.max_edges)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "nodes": len(payload["nodes"]), "edges": len(payload["edges"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
