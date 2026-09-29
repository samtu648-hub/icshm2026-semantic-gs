#!/usr/bin/env python3
"""Independently audit the saved SEM439 class PNGs and trigger records."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_mask2former_long_spatial import boundary_counts, boundary_summary, close, confusion, metric_summary
from src.quality_deck_absence_fallback import apply_deck_absence_fallback
from src.quality_protocol import sha256, write_new_json


EID = "SEM-439-RGB170-DECK-ABSENCE-FALLBACK"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    config_path = ROOT / "experiments/runs" / EID / "config.yaml"
    config = read(config_path)
    output = ROOT / config["output_path"]
    result_path = output / "result.json"
    result = read(result_path)
    export = read(ROOT / config["parent_export_result"])
    train = set(export["semantic_train"])
    matrix = np.zeros((5, 5), dtype=np.int64)
    boundary = np.zeros(4, dtype=np.int64)
    per_view = []
    rows = {}
    hashes_exact = True
    checks = 0
    for name in sorted(export["image_hashes"]):
        native = np.asarray(Image.open(ROOT / config["native_prediction_root"] / name), dtype=np.uint8)
        fallback = np.asarray(Image.open(ROOT / config["fallback_prediction_root"] / name), dtype=np.uint8)
        expected, record = apply_deck_absence_fallback(
            native,
            fallback,
            minimum_fallback_deck_pixels=config["rule"]["fallback_minimum_deck_pixels"],
            allowed_native_classes=tuple(config["rule"]["allowed_overwritten_native_classes"]),
        )
        saved_path = output / "predictions" / name
        saved = np.asarray(Image.open(saved_path), dtype=np.uint8)
        if not np.array_equal(expected, saved):
            raise ValueError(f"Rule replay mismatch: {name}")
        hashes_exact &= sha256(saved_path) == result["prediction_hashes"][name]
        rows[name] = record
        checks += 2
        if name in train:
            target = np.asarray(Image.open(ROOT / config["label_root"] / name), dtype=np.uint8)
            one = confusion(target, saved)
            matrix += one
            boundary += boundary_counts(target, saved)
            per_view.append({"name": name, **metric_summary(one)})
            checks += 2
    if not close(metric_summary(matrix), result["metrics"]):
        raise ValueError("Aggregate metrics differ")
    if not close(boundary_summary(boundary), result["boundary_metrics"]):
        raise ValueError("Boundary metrics differ")
    if not close({r["name"]: r for r in per_view}, {r["name"]: r for r in result["per_view"]}):
        raise ValueError("Per-view metrics differ")
    triggers = [name for name, row in rows.items() if row["triggered"]]
    if triggers != result["triggers"] or not hashes_exact:
        raise ValueError("Trigger or hash audit differs")
    checks += 4
    write_new_json(output / "audit-result.json", {
        "schema_version": 1,
        "status": "completed_independent_recalculation",
        "experiment_id": EID,
        "config_sha256": sha256(config_path),
        "result_sha256": sha256(result_path),
        "checks": checks,
        "prediction_hashes_exact": hashes_exact,
        "trigger_count": len(triggers),
        "triggers": triggers,
        "metrics": metric_summary(matrix),
        "boundary_metrics": boundary_summary(boundary),
    })


if __name__ == "__main__":
    main()
