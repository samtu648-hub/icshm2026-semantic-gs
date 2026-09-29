#!/usr/bin/env python3
"""Apply the preregistered SEM439 zero-deck anomaly fallback."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import run_sem438_multiscale_audit as sem438
from src.quality_deck_absence_fallback import apply_deck_absence_fallback
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_semantic_audit import independent_confusion, metrics


EID = "SEM-439-RGB170-DECK-ABSENCE-FALLBACK"
CONFIG = ROOT / "experiments/runs" / EID / "config.yaml"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    config = read(CONFIG)
    rule = config["rule"]
    if (
        config["experiment_id"] != EID
        or config["semantic_training"]
        or rule["native_deck_pixel_count_equals"] != 0
        or rule["fallback_minimum_deck_pixels"] != 256
        or rule["fallback_class"] != 1
        or rule["allowed_overwritten_native_classes"] != [0, 2]
        or rule["protected_native_classes"] != [3, 4]
    ):
        raise ValueError("Registered SEM439 rule changed")
    output = ROOT / config["output_path"]
    if output.exists():
        raise FileExistsError(output)
    native_root = ROOT / config["native_prediction_root"]
    fallback_root = ROOT / config["fallback_prediction_root"]
    parent_result = read(ROOT / config["sem438_result"])
    if parent_result["status"] != "completed" or not parent_result["native_exact_parent"]:
        raise ValueError("Completed exact SEM438 parent required")
    if sha256(ROOT / config["checkpoint"]) != config["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    export_path = ROOT / config["parent_export_result"]
    export = read(export_path)
    image_root = export_path.parent / "images"
    prepare = read(ROOT / "outputs/quality/SEM-438/r3/prepare-result.json")
    verify_hashes(image_root, prepare["image_hashes"])
    names = sorted(export["image_hashes"])
    train = set(export["semantic_train"])
    unlabeled = set(export["unlabeled"])
    output.mkdir(parents=True)
    prediction_root = output / "predictions"
    prediction_root.mkdir()
    started = time.monotonic()
    rows = {}
    matrix = np.zeros((5, 5), dtype=np.int64)
    boundary = np.zeros(4, dtype=np.int64)
    per_view = []
    prediction_hashes = {}
    for name in names:
        native_path = native_root / name
        fallback_path = fallback_root / name
        if sha256(native_path) != parent_result["arms"]["native"]["prediction_hashes"][name]:
            raise ValueError(f"Native parent changed: {name}")
        if sha256(fallback_path) != parent_result["arms"]["global_1p5"]["prediction_hashes"][name]:
            raise ValueError(f"Fallback parent changed: {name}")
        native = np.asarray(Image.open(native_path), dtype=np.uint8)
        fallback = np.asarray(Image.open(fallback_path), dtype=np.uint8)
        prediction, record = apply_deck_absence_fallback(
            native,
            fallback,
            minimum_fallback_deck_pixels=rule["fallback_minimum_deck_pixels"],
            allowed_native_classes=tuple(rule["allowed_overwritten_native_classes"]),
        )
        record["role"] = "train" if name in train else "unlabeled" if name in unlabeled else "trajectory"
        target = prediction_root / name
        if record["triggered"]:
            Image.fromarray(prediction).save(target)
        else:
            shutil.copy2(native_path, target)
        prediction_hashes[name] = sha256(target)
        rows[name] = record
        if name in train:
            label = np.asarray(Image.open(ROOT / config["label_root"] / name), dtype=np.uint8)
            one = independent_confusion(label, prediction)
            matrix += one
            boundary += sem438.boundary_counts(label, prediction)
            per_view.append({"name": name, **metrics(one)})
    triggers = [name for name in names if rows[name]["triggered"]]
    hard = {name: rows[name] for name in ("355.png", "356.png", "357.png")}
    gate = config["retention_gate"]
    all300_exact = all(prediction_hashes[name] == parent_result["arms"]["native"]["prediction_hashes"][name] for name in train)
    maximum_change_ok = all(rows[name]["changed_fraction"] <= gate["maximum_changed_fraction_per_triggered_view"] for name in triggers)
    hard_ok = hard["355.png"]["added_deck_pixels"] >= 256 and hard["356.png"]["added_deck_pixels"] >= 256 and hard["357.png"]["changed_pixels"] == 0
    verify_hashes(image_root, prepare["image_hashes"])
    result = {
        "schema_version": 1,
        "status": "completed",
        "experiment_id": EID,
        "quality_scope": config["quality_scope"],
        "rows": rows,
        "triggers": triggers,
        "trigger_count": len(triggers),
        "trigger_count_by_role": {role: sum(rows[name]["triggered"] and rows[name]["role"] == role for name in names) for role in ("train", "unlabeled", "trajectory")},
        "hard_views": hard,
        "metrics": metrics(matrix),
        "boundary_metrics": sem438.spatial_base.boundary_metrics(boundary),
        "per_view": per_view,
        "prediction_hashes": prediction_hashes,
        "all300_exact_native": all300_exact,
        "maximum_change_gate": maximum_change_ok,
        "hard_behavior_gate": hard_ok,
        "checkpoint_unchanged": sha256(ROOT / config["checkpoint"]) == config["checkpoint_sha256"],
        "rgb_unchanged": True,
        "retained_as_development_fallback": bool(all300_exact and maximum_change_ok and hard_ok),
        "seconds": time.monotonic() - started,
    }
    write_new_json(output / "result.json", result)


if __name__ == "__main__":
    main()
