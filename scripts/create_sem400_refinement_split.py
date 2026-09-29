#!/usr/bin/env python3
"""Create SEM-400's camera-only refinement holdout inside RGB152 training roles."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.semantic_crossview_reliability import build_confirmation_candidate

PARENT = ROOT / "experiments/splits/spatial_confirmation_v1.json"
GRAPH = ROOT / "experiments/geometry_camera_graph.json"
SEGMENTS = ROOT / "experiments/trajectory_segments.json"
OUTPUT = ROOT / "experiments/splits/semantic_direction_refinement_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    segments = json.loads(SEGMENTS.read_text(encoding="utf-8"))
    fold = parent["primary_fold"]
    parent_rgb = set(fold["train_rgb_400"])
    parent_semantic = set(fold["semantic_train_labeled"])
    subgraph = {
        "nodes": [node for node in graph["nodes"] if node["name"] in parent_rgb],
        "edges": [
            edge
            for edge in graph["edges"]
            if edge["source"] in parent_rgb and edge["target"] in parent_rgb
        ],
        "adjacency": {
            name: [
                edge
                for edge in graph["adjacency"][name]
                if edge["source"] in parent_rgb and edge["target"] in parent_rgb
            ]
            for name in parent_rgb
        },
    }
    candidate = build_confirmation_candidate(
        subgraph,
        segments,
        sorted(parent_semantic),
        seed=400,
        region_count=3,
        views_per_region=10,
        guard_radius=0.5220235426020557,
        minimum_region_separation=1.5,
        minimum_rgb_train=260,
        minimum_semantic_train=180,
    )
    confirmation = set(candidate["confirmation_labeled"])
    buffer = set(candidate["buffer_excluded_all_cameras"])
    rgb_train = parent_rgb - confirmation - buffer
    semantic_train = parent_semantic - confirmation - buffer
    if (len(rgb_train), len(semantic_train), len(confirmation), len(buffer)) != (292, 201, 30, 31):
        raise ValueError("Unexpected SEM-400 role counts")
    if confirmation & buffer or confirmation & rgb_train or buffer & rgb_train:
        raise ValueError("SEM-400 roles overlap")
    if confirmation | buffer | rgb_train != parent_rgb:
        raise ValueError("SEM-400 roles do not cover the RGB152 training universe")
    if semantic_train - rgb_train:
        raise ValueError("Semantic training views escape RGB refinement training")
    groups = [
        {
            "group_id": region["region_id"],
            "segment_id": region["segment_id"],
            "description": "Camera-only contiguous refinement confirmation region",
            "images": region["images"],
            "centroid": region["centroid"],
            "maximum_radius": region["maximum_radius"],
        }
        for region in candidate["confirmation_regions"]
    ]
    return {
        "schema_version": 1,
        "split_id": "semantic_direction_refinement_v1",
        "purpose": "Fresh method-level confirmation for deterministic semantic-gradient split direction",
        "claim_scope": "Camera-only holdout from RGB152 refinement inputs; RGB152 parent reconstruction previously saw these RGB images, so this is not reconstruction-blind or official hidden evaluation",
        "parent_split": str(PARENT.relative_to(ROOT)),
        "parent_split_sha256": sha256(PARENT),
        "selection_protocol": {
            "uses": "official camera centres, trajectory segments, and camera graph only",
            "uses_labels_rgb_pixels_model_outputs_or_errors": False,
            "seed": 400,
            "regions": 3,
            "views_per_region": 10,
            "guard_radius_strict_less_than": 0.5220235426020557,
            "minimum_region_separation": 1.5,
        },
        "groups": groups,
        "primary_fold": {
            "holdout_all_cameras": sorted(confirmation),
            "validation_labeled": sorted(confirmation),
            "withheld_unlabeled": [],
            "buffer_excluded_all_cameras": sorted(buffer),
            "buffer_excluded_labeled": sorted(buffer & parent_semantic),
            "buffer_excluded_unlabeled": sorted(buffer - parent_semantic),
            "train_rgb_400": sorted(rgb_train),
            "train_rgb_300": sorted(semantic_train),
            "semantic_train_labeled": sorted(semantic_train),
            "semantic_validation_labeled": sorted(confirmation),
            "checks": {
                "disjoint_roles": True,
                "parent_rgb_universe_count": len(parent_rgb),
                "all_parent_rgb_views_accounted": True,
                "confirmation_count": len(confirmation),
                "buffer_count": len(buffer),
                "rgb_train_count": len(rgb_train),
                "semantic_train_count": len(semantic_train),
                "camera_graph_component_count_not_increased": candidate["checks"]["camera_graph_component_count_not_increased"],
            },
        },
        "forbidden_for_method_development": sorted(confirmation | buffer),
        "historical_warning": "All images were released and the RGB152 parent used the new confirmation RGBs during its original reconstruction. Only SEM-400 refinement/evidence access is isolated.",
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.write_text(json.dumps(build(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
