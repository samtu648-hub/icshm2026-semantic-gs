#!/usr/bin/env python3
"""Derive the frozen SEM-001 labelled-only split from the R1 spatial fold.

The RGB R1 split deliberately contains allowed unlabelled images in its 400-RGB
training subset.  SEM-001 must not turn those images into supervision.  This
tool makes that exclusion explicit and records the inherited guard checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def is_released_labelled_image(name: str) -> bool:
    try:
        return Path(name).suffix.lower() == ".png" and 1 <= int(Path(name).stem) <= 300
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rgb = json.loads(args.rgb_split.read_text(encoding="utf-8"))
    fold = rgb["primary_fold"]
    train = [name for name in fold["train_rgb_400"] if is_released_labelled_image(name)]
    validation = [name for name in fold["validation_labeled"] if is_released_labelled_image(name)]
    buffer = [name for name in fold["buffer_excluded_labeled"] if is_released_labelled_image(name)]

    if len(train) != 219 or len(validation) != 74:
        raise ValueError(f"Unexpected R1-derived SEM-001 counts: train={len(train)}, validation={len(validation)}")
    if set(train) & set(validation) or set(train) & set(buffer) or set(validation) & set(buffer):
        raise ValueError("SEM-001 train/validation/buffer lists are not disjoint")
    if not set(validation).issubset(set(fold["holdout_all_cameras"])):
        raise ValueError("Validation labels are not an R1 holdout subset")
    if not all(name not in train for name in fold["withheld_unlabeled"]):
        raise ValueError("Unlabelled holdout image leaked into semantic train list")

    payload = {
        "schema_version": 1,
        "split_id": "sem_001_scaffold_static_v1",
        "parent_split_id": rgb["split_id"],
        "parent_primary_fold_id": fold["fold_id"],
        "purpose": "labelled-only static semantic training on the frozen R1 Scaffold-GS RGB scene",
        "source_coordinate_frame": rgb["source_coordinate_frame"],
        "pose_convention": rgb["pose_convention"],
        "seed": 20260730,
        "semantic_train_labeled": train,
        "semantic_validation_labeled": validation,
        "buffer_excluded_labeled": buffer,
        "excluded_unlabeled": sorted(
            set(fold["train_rgb_400"]) - set(train)
            | set(fold["withheld_unlabeled"])
            | set(fold["buffer_excluded_unlabeled"])
        ),
        "challenge_views": fold["challenge_views"],
        "checks": {
            "train_count": len(train),
            "validation_count": len(validation),
            "no_train_validation_filename_overlap": not bool(set(train) & set(validation)),
            "no_train_buffer_filename_overlap": not bool(set(train) & set(buffer)),
            "r1_no_train_camera_within_guard_radius": fold["checks"]["no_train_camera_within_guard_radius"],
            "r1_minimum_train_to_holdout_distance": fold["checks"]["minimum_train_to_holdout_distance"],
            "unlabeled_images_are_not_semantic_supervision": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["checks"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
