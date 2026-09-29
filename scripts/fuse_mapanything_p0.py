#!/usr/bin/env python3
"""Fuse completed training-side MapAnything P0 NPZs into a Scaffold PLY."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusion.mapanything_p0 import (  # noqa: E402
    fuse_mapanything_p0,
    load_best_views,
    median_five_nn_distance,
    read_binary_ply_xyz,
    voxel_downsample,
    write_scaffold_init_ply,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-run", type=Path, required=True)
    parser.add_argument("--camera-graph", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--baseline-init-ply", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--output-points-npz", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--consistency-neighbours", type=int, default=3)
    parser.add_argument("--relative-depth-tolerance", type=float, default=0.05)
    parser.add_argument("--ordinary-min-support", type=int, default=3)
    parser.add_argument("--line-min-support", type=int, default=2)
    parser.add_argument("--maximum-points", type=int, default=100000)
    args = parser.parse_args()
    views = load_best_views(args.map_run)
    graph = json.loads(args.camera_graph.read_text(encoding="utf-8"))
    baseline_xyz = read_binary_ply_xyz(args.baseline_init_ply)
    r0 = median_five_nn_distance(baseline_xyz)
    raw = fuse_mapanything_p0(
        views, graph, args.image_root, stride=args.stride, consistency_neighbours=args.consistency_neighbours,
        relative_depth_tolerance=args.relative_depth_tolerance, ordinary_min_support=args.ordinary_min_support,
        line_min_support=args.line_min_support,
    )
    fused = voxel_downsample(raw, r0, maximum_points=args.maximum_points)
    write_scaffold_init_ply(args.output_ply, fused["xyz"], fused["rgb"])
    args.output_points_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_points_npz, **fused)
    manifest = {
        "schema_version": 1,
        "input_schema": "geometry_prior_npz_v2",
        "coordinate_frame": "official_colmap_reference",
        "map_run_manifest_sha256": sha256(args.map_run / "window_manifest.json"),
        "baseline_init_ply_sha256": sha256(args.baseline_init_ply),
        "r0_median_5nn": r0,
        "raw_candidate_points": int(len(raw["xyz"])),
        "fused_points": int(len(fused["xyz"])),
        "support_views_min_median_max": [int(fused["support_views"].min()), float(np.median(fused["support_views"])), int(fused["support_views"].max())],
        "line_structure_points": int(fused["line_structure"].sum()),
        "output_ply_sha256": sha256(args.output_ply),
        "output_points_npz_sha256": sha256(args.output_points_npz),
        "filters": {"ordinary_min_support": args.ordinary_min_support, "line_min_support": args.line_min_support, "relative_depth_tolerance": args.relative_depth_tolerance, "grid_stride": args.stride},
        "saved_point_attributes": ["xyz", "rgb", "source_image_index", "raw_confidence", "calibrated_confidence", "support_views", "normal", "line_structure"],
        "scaffold_input": "XYZ and original RGB only; no depth, foundation or semantic features",
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
