#!/usr/bin/env python3
"""Independently recompute SEM436 all300 endpoint metrics and no-GT changes."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_mask2former_long_spatial import boundary_counts, boundary_summary, close, confusion, metric_summary
from src.quality_protocol import sha256, write_new_json


EID = "SEM-436-RGB170-MASK2FORMER-LONG12K-ALL300"
ENDPOINTS = (6000, 8000, 10000, 12000)


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    config_path = ROOT / "experiments/runs" / EID / "config.yaml"
    config = read(config_path)
    batch = ROOT / config["output_path"]
    inference_path = batch / "inference/result.json"
    inference = read(inference_path)
    export = read(batch / "export/result.json")
    if inference["status"] != "completed":
        raise ValueError("Completed inference required")
    labels = ROOT / config["label_root"]
    train_names = list(export["semantic_train"])
    recomputed = {}
    checks = 0
    for step in ENDPOINTS:
        matrix = np.zeros((5, 5), dtype=np.int64)
        boundary = np.zeros(4, dtype=np.int64)
        per_view = []
        prediction_dir = batch / f"inference/predictions/{step}"
        for name in train_names:
            target = np.asarray(Image.open(labels / name), dtype=np.uint8)
            prediction = np.asarray(Image.open(prediction_dir / name), dtype=np.uint8)
            one = confusion(target, prediction)
            matrix += one
            boundary += boundary_counts(target, prediction)
            per_view.append({"name": name, **metric_summary(one)})
        metrics = metric_summary(matrix)
        boundaries = boundary_summary(boundary)
        saved = inference["endpoints"][str(step)]
        if not close(metrics, saved["metrics"]):
            raise ValueError(f"Endpoint {step} metric recomputation differs")
        if not close(boundaries, saved["boundary_metrics"]):
            raise ValueError(f"Endpoint {step} boundary recomputation differs")
        if not close({row["name"]: row for row in per_view}, {row["name"]: row for row in saved["per_view"]}):
            raise ValueError(f"Endpoint {step} per-view recomputation differs")
        checks += 302
        recomputed[str(step)] = {"metrics": metrics, "boundary_metrics": boundaries}

    changes = {}
    control = batch / "inference/predictions/6000"
    for step in (8000, 10000, 12000):
        endpoint = batch / f"inference/predictions/{step}"
        role_changes = {}
        for role in ("unlabeled", "trajectory"):
            names = export[role]
            changed = 0
            cable_delta = 0
            total = 0
            for name in names:
                before = np.asarray(Image.open(control / name), dtype=np.uint8)
                after = np.asarray(Image.open(endpoint / name), dtype=np.uint8)
                changed += int(np.count_nonzero(before != after))
                cable_delta += int(np.count_nonzero(after == 2) - np.count_nonzero(before == 2))
                total += int(before.size)
            role_changes[role] = {
                "view_count": len(names),
                "changed_pixels": changed,
                "total_pixels": total,
                "changed_fraction": changed / total,
                "stay_cable_pixel_delta": cable_delta,
                "ground_truth_available": False,
            }
        changes[str(step)] = role_changes
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
            "no_gt_changes_vs_6000": changes,
        },
    )


if __name__ == "__main__":
    main()
