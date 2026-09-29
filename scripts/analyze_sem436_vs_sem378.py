#!/usr/bin/env python3
"""Compare sealed SEM436 endpoints with the prior full300 SEM378 model."""

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


def subtract(after: Any, before: Any) -> Any:
    if isinstance(after, dict) and isinstance(before, dict):
        return {key: subtract(after[key], before[key]) for key in after if key in before}
    if isinstance(after, (int, float)) and isinstance(before, (int, float)):
        return float(after - before)
    return None


def aggregate(prediction_root: Path, label_root: Path, names: list[str]) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    matrix = np.zeros((5, 5), dtype=np.int64)
    boundary = np.zeros(4, dtype=np.int64)
    per_view: dict[str, dict[str, Any]] = {}
    for name in names:
        target = np.asarray(Image.open(label_root / name), dtype=np.uint8)
        prediction = np.asarray(Image.open(prediction_root / name), dtype=np.uint8)
        one = confusion(target, prediction)
        matrix += one
        boundary += boundary_counts(target, prediction)
        per_view[name] = metric_summary(one)
    return metric_summary(matrix), boundary_summary(boundary), per_view


def no_gt_comparison(old_root: Path, new_root: Path, names: list[str]) -> dict[str, Any]:
    changed = 0
    total = 0
    old_counts = np.zeros(5, dtype=np.int64)
    new_counts = np.zeros(5, dtype=np.int64)
    for name in names:
        old = np.asarray(Image.open(old_root / name), dtype=np.uint8)
        new = np.asarray(Image.open(new_root / name), dtype=np.uint8)
        if old.shape != new.shape:
            raise ValueError(f"prediction shape differs for {name}")
        changed += int(np.count_nonzero(old != new))
        total += int(old.size)
        old_counts += np.bincount(old.ravel(), minlength=5)
        new_counts += np.bincount(new.ravel(), minlength=5)
    return {
        "view_count": len(names),
        "changed_pixels": changed,
        "changed_fraction": changed / total,
        "ground_truth_available": False,
        "class_pixel_delta": {
            name: int(new_counts[index] - old_counts[index])
            for index, name in enumerate(("background", "deck", "stay_cable", "tower", "foundation"))
        },
    }


def main() -> None:
    config_path = ROOT / f"experiments/runs/{EID}/config.yaml"
    config = read(config_path)
    batch = ROOT / config["output_path"]
    inference_path = batch / "inference/result.json"
    inference = read(inference_path)
    audit_path = batch / "audit-result.json"
    audit = read(audit_path)
    export = read(batch / "export/result.json")
    old_result_path = ROOT / "outputs/quality/SEM-378/r1/inference/result.json"
    old_result = read(old_result_path)
    old_predictions = ROOT / "outputs/quality/SEM-378/r1/inference/predictions"
    label_root = ROOT / config["label_root"]
    train_names = list(export["semantic_train"])
    old_metrics, old_boundary, old_per_view = aggregate(old_predictions, label_root, train_names)
    if not close(old_metrics, old_result["metrics"]):
        raise ValueError("SEM378 independent metric recomputation differs")

    endpoints: dict[str, Any] = {}
    for step in ENDPOINTS:
        current = audit["endpoints"][str(step)]
        new_per_view = {row["name"]: row for row in inference["endpoints"][str(step)]["per_view"]}
        foreground_wins = sum(
            new_per_view[name]["mIoU_without_background"] > old_per_view[name]["mIoU_without_background"]
            for name in train_names
        )
        endpoints[str(step)] = {
            "metrics": current["metrics"],
            "boundary_metrics": current["boundary_metrics"],
            "delta_vs_sem378": {
                "metrics": subtract(current["metrics"], old_metrics),
                "boundary_metrics": subtract(current["boundary_metrics"], old_boundary),
                "foreground_miou_view_wins": foreground_wins,
                "foreground_miou_view_losses_or_ties": len(train_names) - foreground_wins,
            },
            "no_gt_vs_sem378": {
                "unlabeled": no_gt_comparison(
                    old_predictions, batch / f"inference/predictions/{step}", list(export["unlabeled"])
                ),
                "trajectory": no_gt_comparison(
                    old_predictions, batch / f"inference/predictions/{step}", list(export["trajectory"])
                ),
            },
        }

    write_new_json(
        batch / "comparison-vs-sem378.json",
        {
            "schema_version": 1,
            "status": "completed_independent_reference_comparison",
            "experiment_id": EID,
            "warning": "001-300 are training-fit comparisons; 301-400 and trajectory have no GT",
            "config_sha256": sha256(config_path),
            "inference_result_sha256": sha256(inference_path),
            "audit_result_sha256": sha256(audit_path),
            "sem378_result_sha256": sha256(old_result_path),
            "sem378": {"metrics": old_metrics, "boundary_metrics": old_boundary},
            "endpoints": endpoints,
        },
    )


if __name__ == "__main__":
    main()
