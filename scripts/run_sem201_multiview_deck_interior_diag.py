#!/usr/bin/env python3
"""Run SEM-201 with an eroded source-deck core and sealed SEM-200 depth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem200_multiview_deck_diag as sem200

from src.semantic2d import read_json, sha256
from src.semantic_rendered_training import select_rendered_training_views


EXPERIMENT_ID = "SEM-201-MV-DECK-INTERIOR-GEOMETRY-DIAG"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--oof-split", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--depth-source-dir", type=Path, required=True)
    parser.add_argument("--sem001-training-root", type=Path, required=True)
    parser.add_argument("--sem150-root", type=Path, required=True)
    parser.add_argument("--sem160-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate_config(config: dict) -> None:
    require(config.get("experiment_id") == EXPERIMENT_ID, "SEM-201 received another config")
    require(config.get("status") == "registered_not_run", "SEM-201 must start from its registered config")
    geometry = config["geometry"]
    require(int(geometry.get("source_deck_erosion_radius_pixels", -1)) == 8, "SEM-201 source erosion must be 8 pixels")
    common = dict(geometry)
    common.pop("source_deck_erosion_radius_pixels")
    require(
        common
        == {
            "depth_source": "RGB-030-FULL exact dominant generated Gaussian camera-space depth",
            "minimum_foreground_mass": 0.5,
            "source_pixel_stride": 4,
            "source_confirmation_neighbours": 3,
            "minimum_confirming_views": 1,
            "target_source_views": 8,
            "relative_depth_tolerance": 0.05,
            "target_splat_radius_pixels": 2,
            "minimum_target_source_votes": 2,
        },
        "SEM-201 changed a variable besides source-deck interior erosion",
    )


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    validate_config(config)
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()), f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = read_json(args.split)
    oof_split = read_json(args.oof_split)
    names = select_rendered_training_views(split)
    require(sha256(args.split) == config["data"]["semantic_split_sha256"], "SEM-201 semantic split hash differs")
    require(sha256(args.oof_split) == config["data"]["oof_split_sha256"], "SEM-201 OOF split hash differs")
    require(sha256(args.depth_source_dir / "depth_manifest.json") == config["inputs"]["sem200_depth_manifest_sha256"], "SEM-201 sealed depth manifest hash differs")

    # Shared functions use this identifier only for emitted manifest identity.
    sem200.EXPERIMENT_ID = EXPERIMENT_ID
    sem200.generate_oof_evidence(args=args, config=config, names=names, oof_split=oof_split)
    sem200.materialize_candidates(args=args, config=config, names=names)
    result = sem200.evaluate_and_finish(args=args, config=config, names=names)
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "status": result["status"],
                "geometry_representation_qualified": result["geometry_representation_qualified"],
                "result_sha256": sha256(args.output_dir / "result.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
