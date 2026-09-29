#!/usr/bin/env python3
"""Combine independently replayed SEM-290 raw-argmax audit chunks."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.semantic2d import confusion_metrics, read_json, sha256, write_json
from src.semantic_protected_fusion import partition_training_names

EXP = "SEM-290-NATIVE960-RAW-FIVECLASS-AUDIT"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/runs/SEM-290-NATIVE960-RAW-FIVECLASS-AUDIT/config.yaml",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cfg = read_json(args.config)
    require(cfg["experiment_id"] == EXP, "SEM-290 config mismatch")
    names = read_json(ROOT / cfg["semantic_split"])["semantic_train_labeled"]
    calibration_names = set(partition_training_names(names, expected_count=219)["calibration"])

    results = [read_json(path / "result.json") for path in args.chunk_dir]
    require(results and all(result["experiment_id"] == EXP for result in results), "Wrong chunk experiment")
    config_sha = sha256(args.config)
    require(all(result["config_sha256"] == config_sha for result in results), "Chunks use different config")
    chunk_count = results[0]["protocol"]["chunk_count"]
    require(len(results) == chunk_count, "Missing or duplicate chunk")
    require({result["protocol"]["chunk_index"] for result in results} == set(range(chunk_count)), "Chunk index mismatch")
    require(all(result["protocol"]["chunk_count"] == chunk_count for result in results), "Chunk count mismatch")
    require(all(result["protocol"]["check_labels_decoded"] == 0 and result["protocol"]["developer_labels_decoded"] == 0 for result in results), "Forbidden label access")

    combined_confusion = np.zeros((5, 5), dtype=np.int64)
    combined_manifest = {}
    combined_access = []
    combined_views = []
    for result in sorted(results, key=lambda row: row["protocol"]["chunk_index"]):
        combined_confusion += np.asarray(result["raw_confusion_rows_gt_columns_prediction"], dtype=np.int64)
        for name, record in result["raw_prediction_manifest"].items():
            require(name not in combined_manifest, f"Duplicate raw prediction: {name}")
            require(sha256(Path(record["path"])) == record["sha256"], f"Raw prediction changed: {name}")
            combined_manifest[name] = record
        combined_access.extend(result["target_access"])
        combined_views.extend(result["per_view_calibration"])
    require(set(combined_manifest) == set(names) and len(combined_manifest) == 219, "Incomplete combined OOF output")
    require({row["name"] for row in combined_access} == calibration_names and len(combined_access) == 108, "Unexpected combined label access")

    raw_metrics = confusion_metrics(combined_confusion)
    reference = results[0]
    result = {
        "experiment_id": EXP,
        "status": "completed_read_only_diagnostic",
        "config_sha256": config_sha,
        "parent": reference["parent"],
        "protocol": {
            "chunks": chunk_count,
            "raw_predictions": 219,
            "calibration_labels_decoded": 108,
            "check_labels_decoded": 0,
            "developer_labels_decoded": 0,
            "training_steps": 0,
            "threshold_search": False,
            "frozen_artifacts_verified": 18,
        },
        "metrics": {
            "raw_five_control_argmax": raw_metrics,
            "sealed_five_control_fusion": reference["metrics"]["sealed_five_control_fusion"],
            "sealed_sem160_baseline": reference["metrics"]["sealed_sem160_baseline"],
        },
        "raw_confusion_rows_gt_columns_prediction": combined_confusion.tolist(),
        "target_access": sorted(combined_access, key=lambda row: row["name"]),
        "per_view_calibration": sorted(combined_views, key=lambda row: row["name"]),
        "raw_prediction_manifest": combined_manifest,
        "chunk_result_sha256": [sha256(path / "result.json") for path in args.chunk_dir],
        "note": "Raw argmax is a reused-calibration mechanism diagnostic only. It cannot promote a candidate or establish hidden-view generalization.",
    }
    write_json(args.output_dir / "result.json", result)
    write_json(args.output_dir / "prediction_manifest.json", combined_manifest)
    print({"experiment_id": EXP, "status": result["status"], "raw_predictions": 219, "calibration_labels_decoded": 108})


if __name__ == "__main__":
    main()
