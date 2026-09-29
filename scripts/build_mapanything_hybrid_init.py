#!/usr/bin/env python3
"""Build a deterministic, additive MapAnything P0 hybrid Scaffold PLY."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusion.mapanything_hybrid import merge_baseline_and_supplement, select_hybrid_map_points  # noqa: E402
from fusion.mapanything_p0 import (  # noqa: E402
    median_five_nn_distance,
    read_binary_ply_xyz_rgb,
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
    parser.add_argument("--baseline-init-ply", type=Path, required=True)
    parser.add_argument("--map-points-npz", type=Path, required=True)
    parser.add_argument("--map-fusion-manifest", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--minimum-confidence", type=float, required=True)
    parser.add_argument("--minimum-support", type=int, required=True)
    parser.add_argument("--coverage-radius-r0", type=float, required=True)
    parser.add_argument("--maximum-added-points", type=int, required=True)
    args = parser.parse_args()
    if args.output_ply.exists() or args.output_manifest.exists():
        raise FileExistsError("Refusing to overwrite an existing hybrid output")

    fusion_manifest = json.loads(args.map_fusion_manifest.read_text(encoding="utf-8"))
    baseline_sha256 = sha256(args.baseline_init_ply)
    map_points_sha256 = sha256(args.map_points_npz)
    if fusion_manifest.get("baseline_init_ply_sha256") != baseline_sha256:
        raise ValueError("Map fusion provenance does not match the requested baseline initialization PLY")
    if fusion_manifest.get("output_points_npz_sha256") != map_points_sha256:
        raise ValueError("Map fusion provenance does not match the requested NPZ")
    if fusion_manifest.get("coordinate_frame") != "official_colmap_reference":
        raise ValueError("Map fusion must be in the official COLMAP reference frame")

    baseline_xyz, baseline_rgb = read_binary_ply_xyz_rgb(args.baseline_init_ply)
    r0 = median_five_nn_distance(baseline_xyz)
    with np.load(args.map_points_npz, allow_pickle=False) as archive:
        map_points = {name: np.asarray(archive[name]) for name in archive.files}
    supplement, selection = select_hybrid_map_points(
        baseline_xyz,
        map_points,
        r0=r0,
        minimum_confidence=args.minimum_confidence,
        minimum_support=args.minimum_support,
        coverage_radius_r0=args.coverage_radius_r0,
        maximum_added_points=args.maximum_added_points,
    )
    hybrid_xyz, hybrid_rgb = merge_baseline_and_supplement(baseline_xyz, baseline_rgb, supplement)
    args.output_ply.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_scaffold_init_ply(args.output_ply, hybrid_xyz, hybrid_rgb)
    output_xyz, output_rgb = read_binary_ply_xyz_rgb(args.output_ply)
    if not np.array_equal(output_xyz[: len(baseline_xyz)], baseline_xyz) or not np.array_equal(output_rgb[: len(baseline_rgb)], baseline_rgb):
        raise AssertionError("Serialized hybrid PLY did not preserve every baseline point and color")
    manifest = {
        "schema_version": 1,
        "experiment_id": "GEO-011-3K",
        "coordinate_frame": "official_colmap_reference",
        "initialization_change": "append_only_high_confidence_MapAnything_P0_low_coverage_points",
        "baseline_init_ply_sha256": baseline_sha256,
        "map_points_npz_sha256": map_points_sha256,
        "map_fusion_manifest_sha256": sha256(args.map_fusion_manifest),
        "r0_median_5nn": r0,
        "selection": selection,
        "hybrid_points": int(len(hybrid_xyz)),
        "hybrid_ply_sha256": sha256(args.output_ply),
        "baseline_prefix_preserved": True,
        "scaffold_input": "XYZ and original RGB only; no depth, confidence, foundation, or semantic feature",
    }
    args.output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
