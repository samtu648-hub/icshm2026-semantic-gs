#!/usr/bin/env python3
"""Build the fixed camera-only second spatial confirmation fold."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.semantic_crossview_reliability import build_confirmation_candidate


OUTPUT = ROOT / "experiments/splits/spatial_confirmation_v2.json"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    graph = read(ROOT / "experiments/geometry_camera_graph.json")
    segments = read(ROOT / "experiments/trajectory_segments.json")
    manifest = read(ROOT / "outputs/quality/all_data/SEM-370-r1/label_manifest.json")
    labeled = sorted(record["image"] for record in manifest["records"])
    candidate = build_confirmation_candidate(
        graph,
        segments,
        labeled,
        seed=402,
        region_count=3,
        views_per_region=10,
        guard_radius=0.5220235426020557,
        minimum_region_separation=1.5,
        minimum_rgb_train=340,
        minimum_semantic_train=250,
    )
    old = read(ROOT / "experiments/splits/spatial_confirmation_v1.json")
    old_holdout = set(old["primary_fold"]["holdout_all_cameras"])
    holdout = candidate["confirmation_labeled"]
    if old_holdout & set(holdout):
        raise ValueError("The second spatial fold overlaps spatial_confirmation_v1")
    groups = [
        {
            "group_id": region["region_id"],
            "segment_id": region["segment_id"],
            "description": "Contiguous ten-view camera-only spatial confirmation region",
            "images": region["images"],
        }
        for region in candidate["confirmation_regions"]
    ]
    fold = {
        "holdout_all_cameras": holdout,
        "validation_labeled": holdout,
        "withheld_unlabeled": [],
        "buffer_excluded_all_cameras": candidate["buffer_excluded_all_cameras"],
        "buffer_excluded_labeled": candidate["buffer_excluded_labeled"],
        "buffer_excluded_unlabeled": candidate["buffer_excluded_unlabeled"],
        "train_rgb_400": candidate["train_rgb_400"],
        "train_rgb_300": candidate["train_rgb_300"],
        "semantic_train_labeled": candidate["train_rgb_300"],
        "semantic_validation_labeled": holdout,
        "checks": {
            "disjoint_roles": candidate["checks"]["disjoint_roles"],
            "all_400_accounted": candidate["checks"]["all_400_accounted"],
            "confirmation_count": candidate["counts"]["confirmation"],
            "buffer_count": candidate["counts"]["buffer_all"],
            "rgb_train_count": candidate["counts"]["rgb_train"],
            "semantic_train_count": candidate["counts"]["semantic_train"],
            "camera_graph_component_count_not_increased": candidate["checks"]["camera_graph_component_count_not_increased"],
            "overlap_with_spatial_confirmation_v1": 0,
        },
    }
    split = {
        "schema_version": 1,
        "split_id": "spatial_confirmation_v2",
        "purpose": "Disjoint spatial replication of the matched AbsGS versus Standard reconstruction and semantic comparison",
        "claim_scope": "Historically exposed spatial development replication; not a blind test or official hidden evaluation",
        "source_coordinate_frame": "official_colmap_reference",
        "pose_convention": "CAMERA_FROM_WORLD",
        "selection_protocol": {
            "source": "Fixed camera-only build_confirmation_candidate procedure",
            "selection_uses_model_outputs_or_label_content": False,
            "seed": 402,
            "guard_radius_strict_less_than": 0.5220235426020557,
            "minimum_region_separation": 1.5,
            "minimum_rgb_train": 340,
            "minimum_semantic_train": 250,
            "overlap_with_spatial_confirmation_v1": 0,
            "history_warning": "All released views participated in earlier project models; this is a disjoint spatial development fold, not historically unseen data.",
        },
        "groups": groups,
        "class_mapping": {
            "0": "background",
            "1": "deck",
            "2": "stay_cable",
            "3": "tower",
            "4": "foundation",
        },
        "primary_fold_id": "spatial-confirmation-replication",
        "primary_fold": fold,
        "global_checks": candidate["checks"] | {"overlap_with_spatial_confirmation_v1": 0},
        "execution_contract": candidate["future_requirement"],
    }
    OUTPUT.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"groups": groups, "counts": candidate["counts"]}, indent=2))


if __name__ == "__main__":
    main()
