#!/usr/bin/env python3
"""Qualify a protected conjunction of frozen deck appearance and XYZ evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem200_multiview_deck_diag as sem200

from src.semantic2d import CLASS_NAMES, label_array, read_json, sha256, write_json
from src.semantic_multiview_deck import add_deck_with_protection
from src.semantic_protected_fusion import (
    local_confusion,
    merge_confusions,
    metrics_from_confusion,
    partition_training_names,
)
from src.semantic_rendered_training import select_rendered_training_views


EXPERIMENT_ID = "SEM-211-APPEARANCE-XYZ-JOINT-OOF"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--sem001-training-root", type=Path, required=True)
    parser.add_argument("--sem150-root", type=Path, required=True)
    parser.add_argument("--sem160-root", type=Path, required=True)
    parser.add_argument("--sem160-probability-root", type=Path, required=True)
    parser.add_argument("--sem210-root", type=Path, required=True)
    parser.add_argument("--depth-source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate_config(config: dict[str, Any]) -> None:
    require(config.get("experiment_id") == EXPERIMENT_ID, "SEM-211 received another config")
    require(config.get("status") == "registered_not_run", "SEM-211 must start registered")
    require(
        config["fusion"]["appearance_threshold_grid"] == [0.9, 0.925, 0.95, 0.97, 0.98],
        "SEM-211 appearance grid differs from registration",
    )
    require(
        config["fusion"]["xyz_threshold_grid"] == [0.9, 0.95, 0.97, 0.98, 0.99, 0.995],
        "SEM-211 XYZ grid differs from registration",
    )


def probability_paths_from_sem160(root: Path) -> tuple[dict[str, Path], list[str]]:
    paths: dict[str, Path] = {}
    hashes: list[str] = []
    for fold_index in range(3):
        fold_root = root / f"fold-{fold_index}"
        manifest_path = fold_root / "prediction_manifest.json"
        hashes.append(sha256(manifest_path))
        for record in read_json(manifest_path):
            name = record["name"]
            require(name not in paths, f"SEM-211 duplicate SEM-160 probability: {name}")
            path = fold_root / record["deck_probability"]
            require(path.is_file() and sha256(path) == record["deck_probability_sha256"], f"SEM-211 SEM-160 probability hash mismatch: {name}")
            paths[name] = path
    return paths, hashes


def load_probability(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        require(set(archive.files) == {"deck_probability"}, f"SEM-211 probability schema mismatch: {path}")
        value = np.asarray(archive["deck_probability"], dtype=np.float32)
    require(value.shape == (989, 1320) and np.isfinite(value).all(), f"SEM-211 invalid probability: {path}")
    return value


def evaluate_pairs(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    names: list[str],
    appearance_paths: dict[str, Path],
    xyz_paths: dict[str, Path],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pairs = [
        (float(a), float(x))
        for a in config["fusion"]["appearance_threshold_grid"]
        for x in config["fusion"]["xyz_threshold_grid"]
    ]
    sem160_manifest = read_json(args.sem160_root / "prediction_manifest.json")
    parent = {record["name"]: record for record in sem160_manifest["predictions"]}
    baseline_confusions: list[np.ndarray] = []
    candidate_confusions = {pair: [] for pair in pairs}
    additions_by_target = {pair: np.zeros(5, dtype=np.int64) for pair in pairs}
    protected = {pair: True for pair in pairs}
    for name in names:
        baseline = sem200._read_class_map(args.sem160_root / parent[name]["prediction"], "SEM-211 baseline")
        eligible = sem200._eligibility(
            name=name,
            sem001_root=args.sem001_training_root,
            sem150_root=args.sem150_root,
            baseline=baseline,
        )
        appearance = load_probability(appearance_paths[name])
        xyz = load_probability(xyz_paths[name])
        candidates: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
        for pair in pairs:
            evidence = (appearance >= pair[0]) & (xyz >= pair[1])
            candidates[pair] = add_deck_with_protection(
                baseline, eligible, evidence, minimum_source_votes=1
            )
        # The target is opened only after every registered candidate for this
        # view has been deterministically constructed from frozen predictions.
        target = label_array(args.labels_root / name)
        baseline_confusions.append(local_confusion(target, baseline))
        for pair, (candidate, additions) in candidates.items():
            candidate_confusions[pair].append(local_confusion(target, candidate))
            additions_by_target[pair] += np.bincount(target[additions], minlength=5).astype(np.int64)
            protected[pair] = protected[pair] and bool(np.array_equal(candidate[~additions], baseline[~additions]))
    baseline_metrics = metrics_from_confusion(merge_confusions(baseline_confusions))
    records: list[dict[str, Any]] = []
    for appearance_threshold, xyz_threshold in pairs:
        pair = (appearance_threshold, xyz_threshold)
        metrics = metrics_from_confusion(merge_confusions(candidate_confusions[pair]))
        counts = additions_by_target[pair]
        added = int(counts.sum())
        evaluation = {
            "baseline_metrics": baseline_metrics,
            "candidate_metrics": metrics,
            "added_deck_pixels": added,
            "additions_by_ground_truth": {
                CLASS_NAMES[index]: int(value) for index, value in enumerate(counts)
            },
            "geometry_addition_precision": float(counts[1] / added) if added else 0.0,
            "protected_predictions_exact": protected[pair],
        }
        records.append(
            {
                "appearance_threshold": appearance_threshold,
                "xyz_threshold": xyz_threshold,
                "evaluation": evaluation,
                "gate": sem200._gate(evaluation, config["gates"]["calibration"]),
            }
        )
    return baseline_metrics, records


def materialize_selected(
    *,
    args: argparse.Namespace,
    names: list[str],
    appearance_paths: dict[str, Path],
    xyz_paths: dict[str, Path],
    appearance_threshold: float,
    xyz_threshold: float,
) -> dict[str, Any]:
    sem160_manifest = read_json(args.sem160_root / "prediction_manifest.json")
    parent = {record["name"]: record for record in sem160_manifest["predictions"]}
    output_root = args.output_dir / "oof_candidate"
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for name in names:
        baseline = sem200._read_class_map(args.sem160_root / parent[name]["prediction"], "SEM-211 baseline")
        eligible = sem200._eligibility(
            name=name,
            sem001_root=args.sem001_training_root,
            sem150_root=args.sem150_root,
            baseline=baseline,
        )
        evidence = (load_probability(appearance_paths[name]) >= appearance_threshold) & (
            load_probability(xyz_paths[name]) >= xyz_threshold
        )
        candidate, additions = add_deck_with_protection(
            baseline, eligible, evidence, minimum_source_votes=1
        )
        path = output_root / name
        Image.fromarray(candidate, mode="L").save(path)
        records.append(
            {
                "name": name,
                "path": str(path.relative_to(args.output_dir)),
                "sha256": sha256(path),
                "added_deck_pixels": int(additions.sum()),
                "nonaddition_pixels_exact": bool(np.array_equal(candidate[~additions], baseline[~additions])),
            }
        )
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_label_free_candidate_materialization",
        "selected_appearance_threshold": float(appearance_threshold),
        "selected_xyz_threshold": float(xyz_threshold),
        "target_labels_opened": False,
        "fixed_validation_opened": False,
        "all_nonaddition_pixels_exact": all(record["nonaddition_pixels_exact"] for record in records),
        "predictions": records,
    }
    write_json(args.output_dir / "prediction_manifest.json", manifest)
    return manifest


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    validate_config(config)
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()), f"refusing to overwrite non-empty output: {args.output_dir}")
    require(sha256(args.split) == config["data"]["semantic_split_sha256"], "SEM-211 semantic split hash differs")
    require(sha256(args.sem210_root / "oof_probability_manifest.json") == config["inputs"]["sem210_probability_manifest_sha256"], "SEM-211 SEM-210 manifest hash differs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = select_rendered_training_views(read_json(args.split))
    appearance_paths, appearance_hashes = probability_paths_from_sem160(args.sem160_probability_root)
    require(appearance_hashes == config["inputs"]["sem160_fold_probability_manifest_sha256"], "SEM-211 SEM-160 manifests differ")
    xyz_manifest = read_json(args.sem210_root / "oof_probability_manifest.json")
    xyz_paths = {record["name"]: args.sem210_root / record["path"] for record in xyz_manifest["predictions"]}
    require(set(appearance_paths) == set(names) == set(xyz_paths), "SEM-211 probability coverage differs")
    partition = partition_training_names(names, expected_count=219)
    baseline_metrics, pair_records = evaluate_pairs(
        args=args,
        config=config,
        names=partition["calibration"],
        appearance_paths=appearance_paths,
        xyz_paths=xyz_paths,
    )
    passing = [record for record in pair_records if record["gate"]["passed"]]
    pool = passing if passing else pair_records
    selected = max(
        pool,
        key=lambda record: (
            record["evaluation"]["candidate_metrics"]["per_class"]["deck"]["iou"],
            record["evaluation"]["candidate_metrics"]["mIoU_with_background"],
            record["appearance_threshold"],
            record["xyz_threshold"],
        ),
    )
    materialize_selected(
        args=args,
        names=names,
        appearance_paths=appearance_paths,
        xyz_paths=xyz_paths,
        appearance_threshold=float(selected["appearance_threshold"]),
        xyz_threshold=float(selected["xyz_threshold"]),
    )
    sem200.EXPERIMENT_ID = EXPERIMENT_ID
    final_config = dict(config)
    final_config["visual_samples"] = []
    result = sem200.evaluate_and_finish(args=args, config=final_config, names=names)
    result.update(
        {
            "joint_pair_records": pair_records,
            "calibration_pairs_passing": len(passing),
            "selected_appearance_threshold": float(selected["appearance_threshold"]),
            "selected_xyz_threshold": float(selected["xyz_threshold"]),
            "selection_pool": "passing_only" if passing else "diagnostic_best_no_gate_passed",
            "baseline_calibration_metrics_preselection": baseline_metrics,
            "sem160_fold_probability_manifest_sha256": appearance_hashes,
            "sem210_probability_manifest_sha256": sha256(args.sem210_root / "oof_probability_manifest.json"),
        }
    )
    if not passing:
        result["status"] = "completed_calibration_gate_failed"
        result["geometry_representation_qualified"] = False
        result["next_step"] = "close_geometry_only_deck_recovery_line"
    write_json(args.output_dir / "result.json", result)
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "status": result["status"],
                "calibration_pairs_passing": len(passing),
                "selected_appearance_threshold": result["selected_appearance_threshold"],
                "selected_xyz_threshold": result["selected_xyz_threshold"],
                "check_labels_opened": result["check_labels_opened"],
                "result_sha256": sha256(args.output_dir / "result.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
