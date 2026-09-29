#!/usr/bin/env python3
"""Freeze SEM-005 per-class thresholds using only 219 OOF predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_sem005_pseudo_labels import (  # noqa: E402
    freeze_thresholds,
    prediction_records,
    selective_histograms,
    selective_view,
)
from src.semantic2d import CLASS_NAMES, label_array, read_json, sha256, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.fold_dir) != 3:
        raise ValueError("Exactly three fold output directories are required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(args.config)
    records = prediction_records(args.fold_dir)
    bins = int(config["thresholds"]["grid_bins"])
    target_precision = float(config["thresholds"]["target_precision"])
    totals, correct, gt_counts, by_fold = selective_histograms(records, args.labels_root, bins)
    thresholds = freeze_thresholds(totals, correct, gt_counts, target_precision)

    fold_selective = {}
    for fold in range(3):
        fold_result = {}
        for class_id, name in enumerate(CLASS_NAMES):
            threshold_index = min(int(round(float(thresholds[name]["threshold"]) * (bins - 1))), bins - 1)
            accepted = int(by_fold[fold]["totals"][class_id, threshold_index:].sum())
            true = int(by_fold[fold]["correct"][class_id, threshold_index:].sum())
            gt = int(by_fold[fold]["gt"][class_id])
            fold_result[name] = {
                "accepted_pixels": accepted,
                "precision": float(true / accepted) if accepted else 0.0,
                "recall": float(true / gt) if gt else 0.0,
            }
        fold_selective[str(fold)] = fold_result

    accepted_views = 0
    remote = 0
    false = 0
    for directory, record in records:
        target = label_array(args.labels_root / record["name"])
        probability = np.load(directory / record["probabilities"])["probabilities"].astype(np.float32)
        result = selective_view(target, probability, thresholds)
        accepted_views += int(result["stay_cable_accepted_view"])
        remote += int(result["stay_cable_remote_false_positive_pixels"])
        false += int(result["stay_cable_false_positive_pixels"])

    artifact = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "source": "219 spatial OOF predictions only",
        "fixed_validation_opened_when_frozen": False,
        "grid_bins": bins,
        "target_precision": target_precision,
        "thresholds": thresholds,
        "oof": {
            "views": len(records),
            "stay_cable_view_coverage": accepted_views / len(records),
            "stay_cable_remote_false_positive_pixels": remote,
            "stay_cable_false_positive_pixels": false,
            "stay_cable_remote_share_of_false_positive": float(remote / false) if false else 0.0,
            "fold_selective": fold_selective,
        },
        "oof_prediction_manifest_sha256": [sha256(directory / "prediction_manifest.json") for directory in args.fold_dir],
    }
    path = args.output_dir / "frozen_oof_thresholds.json"
    write_json(path, artifact)
    result = {
        "status": "thresholds_frozen_before_fixed_validation",
        "threshold_artifact_sha256": sha256(path),
        "fixed_validation_opened": False,
        "thresholds": thresholds,
        "oof": artifact["oof"],
    }
    write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
