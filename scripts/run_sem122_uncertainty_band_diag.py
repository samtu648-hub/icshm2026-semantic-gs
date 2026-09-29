#!/usr/bin/env python3
"""Diagnose, without changing inference, the SEM-120 0.20--0.30 cable band.

This developer-only runner makes no new semantic prediction.  It identifies
the exact mutable pixels admitted by hard fusion at 0.20 but rejected at 0.30,
then measures their ground-truth composition, frozen SEM-001 prior, probability
location, 8-connected attachment to 0.30 seeds, and distance to those seeds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem080_structural_diag as common

from src.semantic_protected_fusion import (
    CLASS_COUNT,
    CLASS_NAMES,
    fuse_background_stay_cable,
    protected_mask,
)
from src.semantic_render_domain import semantic_rgb
from src.semantic_rendered_final import require_exact_view_set
from src.semantic_uncertainty_band import summarise_uncertainty_band


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--sem001-result", type=Path, required=True)
    parser.add_argument("--sem001-predictions", type=Path, required=True)
    parser.add_argument("--sem102-manifest", type=Path, required=True)
    parser.add_argument("--sem102-audit-root", type=Path, required=True)
    parser.add_argument("--sem120-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def class_map(path: Path, *, context: str) -> np.ndarray:
    value = np.asarray(Image.open(path), dtype=np.uint8)
    require(
        value.shape == (989, 1320) and int(value.max()) < CLASS_COUNT,
        f"{context} must be an official five-class 1320x989 PNG: {path}",
    )
    return value


def probabilities(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        require(archive.files == ["probabilities"], "SEM-122 teacher NPZ schema is invalid")
        value = archive["probabilities"]
    require(value.shape == (CLASS_COUNT, 989, 1320), "SEM-122 teacher probability shape is invalid")
    return value


def validate_config(config: dict[str, Any]) -> None:
    require(config.get("experiment_id") == "SEM-122-UNCERTAINTY-BAND-DIAG", "SEM-122 received another config")
    require(config.get("status") == "registered_not_run", "SEM-122 may run only from its registered config")
    rule = config["diagnostic"]
    require(float(rule["lower_threshold"]) == 0.2 and float(rule["upper_threshold"]) == 0.3, "SEM-122 must diagnose exactly 0.20--0.30")
    require(rule["mode"] == "read_only_no_prediction_change", "SEM-122 is a read-only diagnostic")


def validate_sources(
    *, args: argparse.Namespace, config: dict[str, Any], names: list[str]
) -> dict[str, Path]:
    require(sha256(args.sem001_result) == config["inputs"]["sem001_result_sha256"], "SEM-122 SEM-001 result hash differs")
    sem001 = common.load_json(args.sem001_result)
    require(sem001.get("checkpoint_reloaded") is True, "SEM-122 requires reloaded SEM-001")
    require(sha256(args.sem120_result) == config["inputs"]["sem120_result_sha256"], "SEM-122 SEM-120 result hash differs")
    sem120 = common.load_json(args.sem120_result)
    require(sem120.get("status") == "completed_development_sweep", "SEM-122 requires the completed SEM-120 record")
    reported = {float(record["stay_cable_threshold"]) for record in sem120["threshold_records"]}
    require({0.2, 0.3}.issubset(reported), "SEM-122 source lacks the 0.20/0.30 comparison")
    require(sha256(args.sem102_manifest) == config["inputs"]["sem102_prediction_manifest_sha256"], "SEM-122 SEM-102 manifest hash differs")
    manifest = common.load_json(args.sem102_manifest)
    require([record.get("name") for record in manifest] == names, "SEM-122 teacher does not exactly cover 74 developer views")
    result: dict[str, Path] = {}
    for record in manifest:
        path = args.sem102_audit_root / str(record["probabilities"])
        require(path.is_file() and sha256(path) == record.get("probabilities_sha256"), f"SEM-122 teacher hash mismatch: {record.get('name')}")
        result[str(record["name"])] = path
    return result


def aggregate_composition(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {name: sum(int(record["counts"][name]) for record in records) for name in CLASS_NAMES}
    total = sum(counts.values())
    return {
        "pixels": total,
        "counts": counts,
        "fractions": {name: float(count / total) if total else 0.0 for name, count in counts.items()},
    }


def aggregate_three_way(records: list[dict[str, int]]) -> dict[str, int]:
    return {name: sum(int(record[name]) for record in records) for name in ("gt_stay_cable", "gt_background", "gt_other")}


def aggregate_bins(summaries: list[dict[str, Any]], *, key: str) -> list[dict[str, Any]]:
    count = len(summaries[0][key]) if summaries else 0
    output: list[dict[str, Any]] = []
    for index in range(count):
        first = summaries[0][key][index]
        bounds = {name: value for name, value in first.items() if name != "composition"}
        output.append({**bounds, "composition": aggregate_three_way([summary[key][index]["composition"] for summary in summaries])})
    return output


def aggregate_connectivity(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    records = [summary["seed_connectivity"] for summary in summaries]
    return {
        "component_count": sum(int(record["component_count"]) for record in records),
        "touching_seed_component_count": sum(int(record["touching_seed_component_count"]) for record in records),
        "touching_seed": aggregate_three_way([record["touching_seed"] for record in records]),
        "detached": aggregate_three_way([record["detached"] for record in records]),
    }


def band_overlay(band: np.ndarray, target: np.ndarray) -> np.ndarray:
    image = np.zeros((*band.shape, 3), dtype=np.uint8)
    image[band & (target == 2)] = np.asarray([56, 220, 80], dtype=np.uint8)
    image[band & (target == 0)] = np.asarray([240, 56, 56], dtype=np.uint8)
    image[band & (target != 2) & (target != 0)] = np.asarray([255, 172, 35], dtype=np.uint8)
    return image


def seed_and_band_overlay(seed: np.ndarray, band: np.ndarray) -> np.ndarray:
    image = np.zeros((*band.shape, 3), dtype=np.uint8)
    image[seed] = np.asarray([255, 218, 32], dtype=np.uint8)
    image[band] = np.asarray([42, 180, 255], dtype=np.uint8)
    return image


def panel(target: np.ndarray, hard030: np.ndarray, band: np.ndarray, seed: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (semantic_rgb(target), semantic_rgb(hard030), band_overlay(band, target), seed_and_band_overlay(seed, band)),
        axis=1,
    )


def write_per_view_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "view", "band_pixels", "gt_stay_cable_pixels", "gt_background_pixels", "gt_other_pixels",
        "sem001_background_pixels", "sem001_stay_cable_pixels", "touching_seed_pixels",
        "touching_seed_gt_stay_cable", "touching_seed_gt_background", "detached_pixels",
        "detached_gt_stay_cable", "detached_gt_background", "component_count", "touching_seed_component_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    args = parse_args()
    config = common.load_json(args.config)
    split = common.load_json(args.split)
    validate_config(config)
    require(args.labels_root.is_dir(), "SEM-122 developer labels are missing")
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()), "SEM-122 refuses to overwrite diagnostic output")
    names = require_exact_view_set(split["semantic_validation_labeled"], split["semantic_validation_labeled"], context="SEM-122 development view plan")
    require(len(names) == int(config["data"]["development_images"]), "SEM-122 expects exactly 74 developer views")
    teacher_paths = validate_sources(args=args, config=config, names=names)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common.write_json(
        args.output_dir / "input_access_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "inputs_verified_before_labels": {
                "sem001_result_and_reloaded_checkpoint": True,
                "sem102_development_probabilities": True,
                "sem120_020_030_source_record": True,
                "fixed_validation_views_opened": False,
                "real_rgb_input_opened": False,
                "unlabeled_images_opened": False,
                "hidden_test_input_opened": False,
                "external_camera_mode_present": False,
            },
        },
    )
    rule = config["diagnostic"]
    summaries: list[dict[str, Any]] = []
    per_view: list[dict[str, Any]] = []
    panel_root = args.output_dir / "development_panels"
    exact_delta = True
    for name in names:
        base = class_map(args.sem001_predictions / name, context="SEM-122 SEM-001 prediction")
        teacher = probabilities(teacher_paths[name])
        hard020, fixed020 = fuse_background_stay_cable(base, teacher, stay_cable_threshold=float(rule["lower_threshold"]))
        hard030, fixed030 = fuse_background_stay_cable(base, teacher, stay_cable_threshold=float(rule["upper_threshold"]))
        require(np.array_equal(fixed020, fixed030) and np.array_equal(base[fixed020], hard030[fixed030]), "SEM-122 protected fusion drifted")
        target = class_map(args.labels_root / name, context="SEM-122 developer label")
        band, seed, summary = summarise_uncertainty_band(
            base, teacher, target,
            lower=float(rule["lower_threshold"]), upper=float(rule["upper_threshold"]),
            probability_bin_edges=tuple(float(value) for value in rule["probability_bin_edges"]),
            distance_bin_edges=tuple(float(value) for value in rule["distance_bin_edges"]),
        )
        delta = (hard020 == 2) & (hard030 != 2)
        exact_delta = exact_delta and bool(np.array_equal(delta, band))
        require(exact_delta, "SEM-122 band is not exactly hard-0.20 minus hard-0.30")
        summaries.append(summary)
        composition = summary["band"]
        prior = summary["sem001_mutable_prediction"]
        attached = summary["seed_connectivity"]["touching_seed"]
        detached = summary["seed_connectivity"]["detached"]
        per_view.append({
            "view": name,
            "band_pixels": composition["pixels"],
            "gt_stay_cable_pixels": composition["counts"]["stay_cable"],
            "gt_background_pixels": composition["counts"]["background"],
            "gt_other_pixels": composition["pixels"] - composition["counts"]["stay_cable"] - composition["counts"]["background"],
            "sem001_background_pixels": prior["counts"]["background"],
            "sem001_stay_cable_pixels": prior["counts"]["stay_cable"],
            "touching_seed_pixels": sum(attached.values()),
            "touching_seed_gt_stay_cable": attached["gt_stay_cable"],
            "touching_seed_gt_background": attached["gt_background"],
            "detached_pixels": sum(detached.values()),
            "detached_gt_stay_cable": detached["gt_stay_cable"],
            "detached_gt_background": detached["gt_background"],
            "component_count": summary["seed_connectivity"]["component_count"],
            "touching_seed_component_count": summary["seed_connectivity"]["touching_seed_component_count"],
        })
        if name in config["outputs"]["visual_samples"]:
            panel_root.mkdir(parents=True, exist_ok=True)
            Image.fromarray(panel(target, hard030, band, seed)).save(panel_root / name)
    aggregate = {
        "band": aggregate_composition([summary["band"] for summary in summaries]),
        "sem001_mutable_prediction": aggregate_composition([summary["sem001_mutable_prediction"] for summary in summaries]),
        "probability_bins": aggregate_bins(summaries, key="probability_bins"),
        "seed_connectivity": aggregate_connectivity(summaries),
        "seed_distance_bins": aggregate_bins(summaries, key="seed_distance_bins"),
        "hard_030_seed_pixels": sum(int(summary["hard_030_seed_pixels"]) for summary in summaries),
        "mutable_pixels": sum(int(summary["mutable_pixels"]) for summary in summaries),
    }
    write_per_view_csv(args.output_dir / "per_view.csv", per_view)
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "completed_development_read_only_diagnostic",
        "development_data_notice": "The 74 labelled views are an explicitly reused developer set. This result diagnoses fixed SEM-120 outputs only; it is neither a new predictor nor blind/formal evidence.",
        "thresholds": {"lower": rule["lower_threshold"], "upper": rule["upper_threshold"]},
        "band_equals_hard020_minus_hard030_exact": exact_delta,
        "aggregate": aggregate,
        "per_view": per_view,
        "fixed_validation_views_opened": True,
        "real_rgb_input_opened": False,
        "unlabeled_images_opened": False,
        "hidden_test_input_opened": False,
        "external_camera_mode_present": False,
        "rgb_modified": False,
        "outputs": {"per_view_csv": "per_view.csv", "panel_order": ["GT", "hard_0.30", "0.20_only_band: green=true cable/red=background/orange=other", "0.30_seed=yellow/band=blue"], "development_panels": [str((panel_root / name).resolve()) for name in config["outputs"]["visual_samples"]]},
    }
    common.write_json(args.output_dir / "result.json", result)
    print(json.dumps({"status": result["status"], "band_pixels": aggregate["band"]["pixels"], "true_stay_cable_fraction": aggregate["band"]["fractions"]["stay_cable"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
