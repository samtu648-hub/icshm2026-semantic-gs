#!/usr/bin/env python3
"""Independent SEM440 metric, hash and no-GT change recalculation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from audit_mask2former_long_spatial import boundary_counts, boundary_summary, close, confusion, metric_summary
from src.quality_protocol import sha256, verify_hashes, write_new_json


EID = "SEM-440-RGB170-WEAK-STRUCTURE-TAIL14K"
ENDPOINTS = (13000, 14000)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    config_path = ROOT / "experiments/runs" / EID / "config.yaml"
    config = read(config_path)
    batch = ROOT / config["output_path"]
    inference_path = batch / "inference/result.json"
    inference = read(inference_path)
    export = read(ROOT / config["parent_export_result"])
    if inference["status"] != "completed":
        raise ValueError("Completed SEM440 inference required")
    verify_hashes(ROOT / config["rgb_images"], export["image_hashes"])
    labels = ROOT / config["label_root"]
    recomputed = {}
    checks = 0
    for step in ENDPOINTS:
        matrix = np.zeros((5, 5), dtype=np.int64)
        boundary = np.zeros(4, dtype=np.int64)
        per_view = []
        prediction_dir = batch / f"inference/predictions/{step}"
        for name in export["semantic_train"]:
            target = np.asarray(Image.open(labels / name), dtype=np.uint8)
            prediction = np.asarray(Image.open(prediction_dir / name), dtype=np.uint8)
            one = confusion(target, prediction)
            matrix += one
            boundary += boundary_counts(target, prediction)
            per_view.append({"name": name, **metric_summary(one)})
        metrics = metric_summary(matrix)
        boundaries = boundary_summary(boundary)
        saved = inference["endpoints"][str(step)]
        if not close(metrics, saved["metrics"]) or not close(boundaries, saved["boundary_metrics"]):
            raise ValueError(f"Endpoint {step} aggregate recomputation differs")
        if not close({row["name"]: row for row in per_view}, {row["name"]: row for row in saved["per_view"]}):
            raise ValueError(f"Endpoint {step} per-view recomputation differs")
        if {name: sha256(prediction_dir / name) for name in export["image_hashes"]} != saved["prediction_hashes"]:
            raise ValueError(f"Endpoint {step} prediction hash binding differs")
        checks += 303
        recomputed[str(step)] = {"metrics": metrics, "boundary_metrics": boundaries}

    parent = ROOT / "outputs/quality/SEM-437/r2/inference/predictions/12000"
    changes = {}
    for step in ENDPOINTS:
        endpoint = batch / f"inference/predictions/{step}"
        roles = {}
        for role in ("unlabeled", "trajectory"):
            changed = total = 0
            deltas = np.zeros(5, dtype=np.int64)
            for name in export[role]:
                before = np.asarray(Image.open(parent / name), dtype=np.uint8)
                after = np.asarray(Image.open(endpoint / name), dtype=np.uint8)
                changed += int(np.count_nonzero(before != after))
                total += int(before.size)
                deltas += np.bincount(after.ravel(), minlength=5) - np.bincount(before.ravel(), minlength=5)
            roles[role] = {"view_count": len(export[role]), "changed_pixels": changed, "total_pixels": total, "changed_fraction": changed / total, "class_pixel_deltas": deltas.tolist(), "ground_truth_available": False}
        changes[str(step)] = roles
    write_new_json(
        batch / "audit-result.json",
        {
            "schema_version": 1,
            "status": "completed_independent_recalculation",
            "experiment_id": EID,
            "config_sha256": sha256(config_path),
            "inference_result_sha256": sha256(inference_path),
            "checks": checks,
            "endpoints": recomputed,
            "no_gt_changes_vs_sem437_12k": changes,
            "frozen_rgb_hashes_exact": True,
        },
    )


if __name__ == "__main__":
    main()
