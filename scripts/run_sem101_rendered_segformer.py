#!/usr/bin/env python3
"""Run one SEM-101 rendered-RGB SegFormer spatial OOF fold.

The implementation delegates the frozen SEM-051 optimizer/inference recipe to
its existing runner after checking that its only RGB input is the complete
training-only rendered dataset created by ``render_sem101_training_rgb.py``.
No validation or unlabelled path is accepted.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem080_structural_diag as s80

from src.semantic_rendered_training import (
    select_rendered_training_views,
    validate_rendered_dataset_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--rendered-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("fold",), required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    args = parse_args()
    config = s80.load_json(args.config)
    require(
        config.get("experiment_id") == "SEM-101-RENDERED-SEG-OOF",
        "SEM-101 fold runner received another config",
    )
    require(
        config.get("status") == "registered_not_run",
        "SEM-101 fold runner must run only from its registered config",
    )
    require(args.mode == "fold", "SEM-101 permits complete OOF folds only")
    require(
        not args.output_dir.exists() or not any(args.output_dir.iterdir()),
        "Refusing to overwrite a SEM-101 fold output",
    )
    split = s80.load_json(REPO_ROOT / config["data"]["semantic_split"])
    training_names = select_rendered_training_views(
        split, expected_count=int(config["data"]["training_images"])
    )
    rendered_manifest_path = args.rendered_root / "rendered_dataset_manifest.json"
    require(
        rendered_manifest_path.is_file(), "SEM-101 rendered dataset manifest is missing"
    )
    rendered_manifest = s80.load_json(rendered_manifest_path)
    validate_rendered_dataset_manifest(rendered_manifest, training_names)
    require(
        (args.rendered_root / "images").is_dir(),
        "SEM-101 rendered images directory is missing",
    )
    require(args.labels_root.is_dir(), "SEM-101 labels directory is missing")
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_sem005_segformer.py"),
        "--config",
        str(args.config),
        "--split",
        str(args.split),
        "--fold",
        str(args.fold),
        "--prepared-scene",
        str(args.rendered_root),
        "--labels-root",
        str(args.labels_root),
        "--model-dir",
        str(args.model_dir),
        "--output-dir",
        str(args.output_dir),
        "--mode",
        args.mode,
    ]
    subprocess.run(command, check=True)
    result_path = args.output_dir / "result.json"
    result = s80.load_json(result_path)
    require(
        result.get("experiment_id") == config["experiment_id"],
        "SEM-101 fold result has another experiment ID",
    )
    require(
        int(result.get("fold", -1)) == args.fold,
        "SEM-101 fold result has the wrong fold",
    )
    require(
        result.get("mode") == "fold" and result.get("status") == "completed_fold_oof",
        "SEM-101 fold did not complete",
    )
    require(
        result.get("fixed_validation_opened") is False,
        "SEM-101 fold opened fixed validation",
    )
    require(
        result.get("unlabeled_rgb_opened") is False,
        "SEM-101 fold opened unlabelled RGB",
    )
    require(
        result.get("scaffold_imported_or_trained") is False,
        "SEM-101 2D trainer imported Scaffold",
    )
    s80.write_json(
        args.output_dir / "sem101_fold_contract.json",
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "fold": args.fold,
            "rendered_dataset_manifest_sha256": s80.sha256(rendered_manifest_path),
            "fold_result_sha256": s80.sha256(result_path),
            "fixed_validation_opened": False,
            "unlabeled_images_opened": False,
            "input_domain": "frozen_Scaffold_RGB_rendering_only",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
