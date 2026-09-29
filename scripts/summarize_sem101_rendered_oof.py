#!/usr/bin/env python3
"""Aggregate SEM-101 rendered-RGB OOF folds and apply its fixed stop gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem080_structural_diag as s80

from src.semantic2d import CLASS_COUNT, confusion_metrics, label_array
from src.semantic_rendered_training import (
    oof_qualification_gate,
    rendered_segmentation_panel,
    select_rendered_training_views,
    validate_oof_prediction_plan,
    validate_rendered_dataset_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--rendered-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    args = parse_args()
    config = s80.load_json(args.config)
    require(
        config.get("experiment_id") == "SEM-101-RENDERED-SEG-OOF",
        "SEM-101 summary received another config",
    )
    require(
        config.get("status") == "registered_not_run",
        "SEM-101 summary must run only from its registered config",
    )
    require(
        not args.output_dir.exists() or not any(args.output_dir.iterdir()),
        "Refusing to overwrite SEM-101 OOF summary",
    )
    split = s80.load_json(REPO_ROOT / config["data"]["semantic_split"])
    oof_split = s80.load_json(args.split)
    training_names = select_rendered_training_views(
        split, expected_count=int(config["data"]["training_images"])
    )
    rendered_manifest_path = args.rendered_root / "rendered_dataset_manifest.json"
    rendered_manifest = s80.load_json(rendered_manifest_path)
    validate_rendered_dataset_manifest(rendered_manifest, training_names)
    fold_records = []
    confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    by_name: dict[str, int] = {}
    fold_results: list[dict[str, object]] = []
    for fold in range(3):
        root = args.fold_root / f"fold-{fold}"
        result_path = root / "result.json"
        prediction_path = root / "prediction_manifest.json"
        contract_path = root / "sem101_fold_contract.json"
        require(
            result_path.is_file()
            and prediction_path.is_file()
            and contract_path.is_file(),
            f"SEM-101 fold {fold} artifacts are incomplete",
        )
        result = s80.load_json(result_path)
        contract = s80.load_json(contract_path)
        require(
            contract.get("rendered_dataset_manifest_sha256")
            == s80.sha256(rendered_manifest_path),
            f"SEM-101 fold {fold} used another rendered dataset",
        )
        require(
            contract.get("fold_result_sha256") == s80.sha256(result_path),
            f"SEM-101 fold {fold} result hash mismatch",
        )
        require(
            result.get("fixed_validation_opened") is False
            and result.get("unlabeled_rgb_opened") is False,
            f"SEM-101 fold {fold} broke data isolation",
        )
        predictions = s80.load_json(prediction_path)
        names = [record["name"] for record in predictions]
        fold_records.append({"fold": fold, "names": names})
        by_name.update({name: fold for name in names})
        confusion += np.asarray(
            result["metrics"]["confusion_matrix_rows_gt_columns_prediction"],
            dtype=np.int64,
        )
        fold_results.append(
            {
                "fold": fold,
                "result_sha256": s80.sha256(result_path),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "prediction_count": len(names),
            }
        )
    validate_oof_prediction_plan(
        training_names=training_names, oof_split=oof_split, fold_records=fold_records
    )
    metrics = confusion_metrics(confusion)
    gate = oof_qualification_gate(metrics, config["oof_qualification_gate"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_names = list(config["outputs"]["training_visual_samples"])
    require(set(sample_names) <= set(by_name), "SEM-101 panels must be OOF predictions")
    panels: list[dict[str, str]] = []
    for name in sample_names:
        fold = by_name[name]
        rendered_rgb = np.asarray(
            Image.open(args.rendered_root / "images" / name).convert("RGB"),
            dtype=np.uint8,
        )
        target = label_array(args.labels_root / name)
        prediction = np.asarray(
            Image.open(args.fold_root / f"fold-{fold}" / "oof_predictions" / name),
            dtype=np.uint8,
        )
        panel = rendered_segmentation_panel(rendered_rgb, target, prediction)
        panel_path = args.output_dir / "training_oof_panels" / name
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(panel).save(panel_path)
        panels.append(
            {
                "name": name,
                "path": str(panel_path.relative_to(args.output_dir)),
                "sha256": s80.sha256(panel_path),
            }
        )
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "completed_oof_gate_passed"
        if gate["passed"]
        else "completed_oof_gate_failed",
        "training_view_count": len(training_names),
        "fold_results": fold_results,
        "metrics": metrics,
        "oof_qualification_gate": gate,
        "training_only_access": {
            "fixed_validation_views_opened": False,
            "unlabeled_images_opened": False,
            "hidden_test_rgb_or_labels_opened": False,
            "external_camera_mode_present": False,
            "labels_opened_for_authorized_train_and_oof_metrics": True,
        },
        "panels": panels,
        "next_step": "Register a separate final rendered-domain model only if the fixed OOF gate passed; this runner never opens the 74-view split.",
    }
    s80.write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
