#!/usr/bin/env python3
"""Independently recompute SEM438 metrics from sealed PNG predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from scripts.audit_mask2former_long_spatial import (
    boundary_counts,
    boundary_summary,
    close,
    confusion,
    metric_summary,
)
from src.quality_protocol import sha256, verify_hashes, write_new_json


EID = "SEM-438-RGB170-MASK2FORMER-MULTISCALE-AUDIT"
ARMS = ("native", "global_1p5", "global_2p0", "strip_2p0", "fixed_ensemble")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    config_path = ROOT / "experiments/runs" / EID / "config.yaml"
    config = read(config_path)
    batch = ROOT / config["output_path"]
    full_path = batch / "full/result.json"
    full = read(full_path)
    prepared = read(batch / "prepare-result.json")
    export_result = ROOT / config["parent_export_result"]
    export = read(export_result)
    image_root = export_result.parent / "images"
    parent = ROOT / config["parent_prediction_root"]
    labels = ROOT / config["label_root"]
    if full["status"] != "completed":
        raise ValueError("Completed SEM438 full result required")
    verify_hashes(image_root, prepared["image_hashes"])
    train_names = set(export["semantic_train"])
    names = sorted(export["image_hashes"])
    matrices = {arm: np.zeros((5, 5), dtype=np.int64) for arm in ARMS}
    boundaries = {arm: np.zeros(4, dtype=np.int64) for arm in ARMS}
    per_view = {arm: [] for arm in ARMS}
    no_gt = {
        arm: {
            "changed_pixels": 0,
            "total_pixels": 0,
            "deck_pixel_delta": 0,
            "stay_cable_pixel_delta": 0,
        }
        for arm in ARMS[1:]
    }
    native_exact = True
    prediction_hashes_exact = True
    checks = 0
    for name in names:
        predictions = {}
        for arm in ARMS:
            path = batch / "full/predictions" / arm / name
            prediction_hashes_exact &= sha256(path) == full["arms"][arm]["prediction_hashes"][name]
            predictions[arm] = np.asarray(Image.open(path), dtype=np.uint8)
            if predictions[arm].max() > 4:
                raise ValueError(f"Class ID outside 0..4: {arm}/{name}")
            checks += 1
        parent_prediction = np.asarray(Image.open(parent / name), dtype=np.uint8)
        native_exact &= bool(np.array_equal(predictions["native"], parent_prediction))
        checks += 1
        if name in train_names:
            target = np.asarray(Image.open(labels / name), dtype=np.uint8)
            for arm in ARMS:
                one = confusion(target, predictions[arm])
                matrices[arm] += one
                boundaries[arm] += boundary_counts(target, predictions[arm])
                per_view[arm].append({"name": name, **metric_summary(one)})
                checks += 2
        else:
            native = predictions["native"]
            for arm in ARMS[1:]:
                candidate = predictions[arm]
                no_gt[arm]["changed_pixels"] += int(np.count_nonzero(candidate != native))
                no_gt[arm]["total_pixels"] += int(native.size)
                no_gt[arm]["deck_pixel_delta"] += int(np.count_nonzero(candidate == 1) - np.count_nonzero(native == 1))
                no_gt[arm]["stay_cable_pixel_delta"] += int(np.count_nonzero(candidate == 2) - np.count_nonzero(native == 2))
                checks += 1
    recomputed = {}
    for arm in ARMS:
        aggregate = metric_summary(matrices[arm])
        boundary = boundary_summary(boundaries[arm])
        saved = full["arms"][arm]
        if not close(aggregate, saved["metrics"]):
            raise ValueError(f"Aggregate metric mismatch: {arm}")
        if not close(boundary, saved["boundary_metrics"]):
            raise ValueError(f"Boundary metric mismatch: {arm}")
        if not close({row["name"]: row for row in per_view[arm]}, {row["name"]: row for row in saved["per_view"]}):
            raise ValueError(f"Per-view metric mismatch: {arm}")
        recomputed[arm] = {"metrics": aggregate, "boundary_metrics": boundary}
        checks += 3
    for arm in ARMS[1:]:
        no_gt[arm]["changed_fraction"] = no_gt[arm]["changed_pixels"] / no_gt[arm]["total_pixels"]
        no_gt[arm]["ground_truth_available"] = False
        if not close(no_gt[arm], full["no_gt_changes_vs_native"][arm]):
            raise ValueError(f"No-GT change count mismatch: {arm}")
        checks += 1
    checkpoint_exact = sha256(ROOT / config["checkpoint"]) == config["checkpoint_sha256"]
    if not all((native_exact, prediction_hashes_exact, checkpoint_exact)):
        raise ValueError("Frozen artifact identity check failed")
    write_new_json(
        batch / "audit-result.json",
        {
            "schema_version": 1,
            "status": "completed_independent_recalculation",
            "experiment_id": EID,
            "config_sha256": sha256(config_path),
            "full_result_sha256": sha256(full_path),
            "checks": checks,
            "native_exact_parent": native_exact,
            "prediction_hashes_exact": prediction_hashes_exact,
            "checkpoint_unchanged": checkpoint_exact,
            "rgb_unchanged": True,
            "arms": recomputed,
            "no_gt_changes_vs_native": no_gt,
        },
    )


if __name__ == "__main__":
    main()
