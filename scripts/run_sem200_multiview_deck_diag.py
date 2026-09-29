#!/usr/bin/env python3
"""Run the SEM-200 deployable multi-view deck geometry qualification.

The runner has three ordered phases in one fresh output directory:

1. render frozen Scaffold-GS dominant-contributor depth for 219 labelled
   training cameras without opening labels;
2. generate one spatial-OOF multi-view vote map and protected candidate for
   every camera, using only other cameras in that target's OOF training set;
3. open the deterministic calibration labels once and evaluate the single
   pre-registered rule.  The check half is opened only if calibration passes.

The fixed 74-view developer set and hidden RGB/labels are never accepted by
this interface.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem001 as sem001

from src.geometry_prior.camera_io import read_cameras_txt, read_images_txt
from src.scaffold_semantic_sidecar import (
    ScaffoldSettings,
    configure_scaffold_import,
    frozen_rgb_hashes,
    frozen_rgb_named_parameters,
    load_frozen_scaffold,
    pipeline,
)
from src.scaffold_visibility_supervision import (
    load_visibility_extension,
    render_frozen_depth_diagnostics,
)
from src.semantic2d import CLASS_NAMES, label_array, read_json, sha256, write_json
from src.semantic_multiview_deck import (
    DepthObservation,
    accumulate_source_votes,
    add_deck_with_protection,
    confirm_deck_points,
    deck_points_from_view,
    project_points_to_target_mask,
    strong_neighbours,
)
from src.semantic_protected_fusion import (
    local_confusion,
    merge_confusions,
    metrics_from_confusion,
    partition_training_names,
)
from src.semantic_render_domain import semantic_rgb
from src.semantic_rendered_training import select_rendered_training_views


EXPERIMENT_ID = "SEM-200-MV-DECK-GEOMETRY-DIAG"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--oof-split", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--rgb-run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--extension-root", type=Path, required=True)
    parser.add_argument("--sem001-training-root", type=Path, required=True)
    parser.add_argument("--sem150-root", type=Path, required=True)
    parser.add_argument("--sem160-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume-evaluation",
        action="store_true",
        help="Resume only the label evaluation from already sealed depth/evidence/prediction manifests.",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_config(config: dict[str, Any]) -> None:
    require(config.get("experiment_id") == EXPERIMENT_ID, "SEM-200 received another config")
    require(config.get("status") == "registered_not_run", "SEM-200 must start from its registered config")
    geometry = config["geometry"]
    require(
        geometry
        == {
            "depth_source": "RGB-030-FULL exact dominant generated Gaussian camera-space depth",
            "minimum_foreground_mass": 0.5,
            "source_pixel_stride": 4,
            "source_confirmation_neighbours": 3,
            "minimum_confirming_views": 1,
            "target_source_views": 8,
            "relative_depth_tolerance": 0.05,
            "target_splat_radius_pixels": 2,
            "minimum_target_source_votes": 2,
        },
        "SEM-200 geometry recipe differs from registration",
    )


def _depth_path(root: Path, name: str) -> Path:
    return root / "frozen_depth" / f"{Path(name).stem}.npz"


def _votes_path(root: Path, name: str) -> Path:
    return root / "oof_votes" / name


def _candidate_path(root: Path, name: str) -> Path:
    return root / "oof_candidate" / name


def _read_class_map(path: Path, context: str) -> np.ndarray:
    value = np.asarray(Image.open(path), dtype=np.uint8)
    require(value.shape == (989, 1320), f"{context} has wrong shape: {path}")
    require(value.size > 0 and int(value.max()) < 5, f"{context} has invalid class IDs: {path}")
    return value


def _read_votes(path: Path) -> np.ndarray:
    value = np.asarray(Image.open(path), dtype=np.uint8)
    require(value.shape == (989, 1320), f"SEM-200 vote map has wrong shape: {path}")
    return value


def render_depth_maps(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    names: list[str],
) -> dict[str, Any]:
    import torch

    require(torch.cuda.is_available(), "SEM-200 frozen depth rendering requires CUDA")
    frozen = config["frozen_rgb"]
    iteration = int(frozen["iteration"])
    before = frozen_rgb_hashes(args.rgb_run_dir, iteration)
    require(before == frozen["artifact_sha256"], "SEM-200 RGB artifact hashes differ from registration")
    configure_scaffold_import(args.source_root)
    model = load_frozen_scaffold(
        args.source_root,
        args.rgb_run_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply",
        ScaffoldSettings(
            train_camera_count=int(frozen["train_camera_count"]),
            n_offsets=int(frozen["n_offsets"]),
        ),
    )
    require(
        not any(parameter.requires_grad for _name, parameter in frozen_rgb_named_parameters(model)),
        "SEM-200 found trainable RGB parameters",
    )
    extension = load_visibility_extension(args.extension_root)
    pipe = pipeline()
    output_root = args.output_dir / "frozen_depth"
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, name in enumerate(names, start=1):
        _selected, _camera, view = sem001.official_view(
            args.data_root, name, int(frozen["appearance_camera_id"])
        )
        with torch.inference_mode():
            diagnostics = render_frozen_depth_diagnostics(
                view, model, pipe, extension=extension
            )
        depth = diagnostics["depth"].detach().cpu().numpy().astype(np.float32)
        foreground = (
            diagnostics["foreground_mass"].detach().cpu().numpy().astype(np.float32)
        )
        valid = diagnostics["valid"].detach().cpu().numpy().astype(bool)
        require(depth.shape == (989, 1320), f"SEM-200 depth shape mismatch for {name}")
        require(np.isfinite(depth[valid]).all() and bool(np.all(depth[valid] > 0.0)), f"invalid depth for {name}")
        path = _depth_path(args.output_dir, name)
        np.savez_compressed(
            path,
            depth=depth.astype(np.float16),
            foreground_mass=foreground.astype(np.float16),
        )
        records.append(
            {
                "name": name,
                "path": str(path.relative_to(args.output_dir)),
                "sha256": sha256(path),
                "valid_fraction": float(valid.mean()),
                "foreground_mass_mean": float(foreground.mean()),
            }
        )
        if index == 1 or index % 20 == 0 or index == len(names):
            print(json.dumps({"phase": "frozen_depth", "completed": index, "total": len(names)}), flush=True)
        del diagnostics, depth, foreground, valid
    after = frozen_rgb_hashes(args.rgb_run_dir, iteration)
    require(before == after, "SEM-200 modified a frozen RGB artifact")
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_frozen_depth",
        "source": "RGB-030-FULL exact dominant generated Gaussian",
        "depth_convention": "camera_plus_z_on_full_frame_PINHOLE_grid",
        "views": len(records),
        "fixed_validation_opened": False,
        "labels_opened": False,
        "reference_rgb_opened": False,
        "frozen_rgb_artifacts_before": before,
        "frozen_rgb_artifacts_after": after,
        "frozen_rgb_unchanged": before == after,
        "elapsed_seconds": float(time.perf_counter() - started),
        "records": records,
    }
    write_json(args.output_dir / "depth_manifest.json", manifest)
    del model
    torch.cuda.empty_cache()
    return manifest


def generate_oof_evidence(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    names: list[str],
    oof_split: dict[str, Any],
) -> dict[str, Any]:
    geometry = config["geometry"]
    graph = read_json(REPO_ROOT / config["data"]["camera_graph"])
    cameras = read_cameras_txt(args.scene_dir / "sparse" / "0" / "cameras.txt")
    images_by_id = read_images_txt(args.scene_dir / "sparse" / "0" / "images.txt")
    images = {image.name: image for image in images_by_id.values()}
    require(set(names).issubset(images), "SEM-200 prepared PINHOLE scene misses training cameras")
    require(all(images[name].camera_id in cameras for name in names), "SEM-200 camera ID is missing")

    depth_source_dir = getattr(args, "depth_source_dir", args.output_dir)
    depth_manifest_path = depth_source_dir / "depth_manifest.json"
    depth_manifest = read_json(depth_manifest_path)
    require(
        depth_manifest.get("status") == "completed_frozen_depth"
        and depth_manifest.get("labels_opened") is False
        and [record["name"] for record in depth_manifest["records"]] == names,
        "SEM-200 frozen-depth manifest is incomplete",
    )
    for record in depth_manifest["records"]:
        path = depth_source_dir / record["path"]
        require(path.is_file() and sha256(path) == record["sha256"], f"SEM-200 depth hash mismatch: {record['name']}")

    fold_for_target: dict[str, dict[str, Any]] = {}
    for fold in oof_split["folds"]:
        train = set(fold["train"])
        oof = set(fold["oof"])
        guard = set(fold["guard"])
        require(not (train & oof or train & guard or oof & guard), "SEM-200 OOF fold overlaps")
        for name in fold["oof"]:
            require(name not in fold_for_target, f"SEM-200 duplicate OOF target: {name}")
            fold_for_target[name] = fold
    require(set(fold_for_target) == set(names), "SEM-200 OOF split does not cover 219 targets")

    @lru_cache(maxsize=32)
    def observation(name: str) -> DepthObservation:
        with np.load(_depth_path(depth_source_dir, name)) as archive:
            require(set(archive.files) == {"depth", "foreground_mass"}, f"SEM-200 depth schema mismatch: {name}")
            return DepthObservation(
                np.asarray(archive["depth"], dtype=np.float32),
                np.asarray(archive["foreground_mass"], dtype=np.float32),
            )

    @lru_cache(maxsize=32)
    def deck_mask(name: str) -> np.ndarray:
        mask = (label_array(args.labels_root / name) == 1).astype(np.uint8)
        radius = int(geometry.get("source_deck_erosion_radius_pixels", 0))
        if radius < 0:
            raise ValueError("SEM-200/201 source deck erosion radius cannot be negative")
        if radius:
            diameter = 2 * radius + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
            mask = cv2.erode(mask, kernel, iterations=1)
        return mask.astype(bool)

    stable_cache: dict[tuple[int, str], tuple[np.ndarray, dict[str, Any]]] = {}

    def stable_points(fold: dict[str, Any], source: str) -> tuple[np.ndarray, dict[str, Any]]:
        key = (int(fold["fold"]), source)
        if key in stable_cache:
            return stable_cache[key]
        allowed = set(fold["train"])
        require(source in allowed, "SEM-200 source is outside target OOF training set")
        confirmations = strong_neighbours(
            graph,
            source,
            allowed - {source},
            count=int(geometry["source_confirmation_neighbours"]),
        )
        source_image = images[source]
        points = deck_points_from_view(
            deck_mask(source),
            observation(source),
            camera=cameras[source_image.camera_id],
            image=source_image,
            stride=int(geometry["source_pixel_stride"]),
            minimum_foreground_mass=float(geometry["minimum_foreground_mass"]),
        )
        confirmation_inputs = []
        for name in confirmations:
            image = images[name]
            confirmation_inputs.append(
                (deck_mask(name), observation(name), cameras[image.camera_id], image)
            )
        keep, support = confirm_deck_points(
            points,
            confirmation_inputs,
            minimum_confirming_views=int(geometry["minimum_confirming_views"]),
            minimum_foreground_mass=float(geometry["minimum_foreground_mass"]),
            relative_tolerance=float(geometry["relative_depth_tolerance"]),
        )
        stable = points[keep]
        record = {
            "fold": int(fold["fold"]),
            "source": source,
            "confirmation_views": confirmations,
            "raw_deck_points": int(len(points)),
            "stable_deck_points": int(len(stable)),
            "stable_fraction": float(keep.mean()) if len(keep) else 0.0,
            "mean_confirming_views": float(support.mean()) if len(support) else 0.0,
        }
        stable_cache[key] = stable, record
        return stable, record

    votes_root = args.output_dir / "oof_votes"
    votes_root.mkdir(parents=True, exist_ok=True)
    target_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, target in enumerate(names, start=1):
        fold = fold_for_target[target]
        allowed = set(fold["train"])
        sources = strong_neighbours(
            graph,
            target,
            allowed,
            count=int(geometry["target_source_views"]),
        )
        require(target not in sources and all(source in allowed for source in sources), "SEM-200 target/source leakage")
        target_image = images[target]
        source_masks: list[np.ndarray] = []
        projection_records: list[dict[str, Any]] = []
        for source in sources:
            points, stable_record = stable_points(fold, source)
            mask, projection = project_points_to_target_mask(
                points,
                target_observation=observation(target),
                target_camera=cameras[target_image.camera_id],
                target_image=target_image,
                minimum_foreground_mass=float(geometry["minimum_foreground_mass"]),
                relative_tolerance=float(geometry["relative_depth_tolerance"]),
                splat_radius=int(geometry["target_splat_radius_pixels"]),
            )
            source_masks.append(mask)
            projection_records.append({"source": source, **projection, "stable_fraction": stable_record["stable_fraction"]})
        votes = (
            accumulate_source_votes(source_masks)
            if source_masks
            else np.zeros((989, 1320), dtype=np.uint16)
        )
        require(int(votes.max(initial=0)) <= 255, "SEM-200 vote count exceeds PNG range")
        path = _votes_path(args.output_dir, target)
        Image.fromarray(votes.astype(np.uint8), mode="L").save(path)
        target_records.append(
            {
                "name": target,
                "fold": int(fold["fold"]),
                "source_views": sources,
                "source_count": len(sources),
                "target_excluded_from_sources": target not in sources,
                "vote_pixels_ge_1": int(np.count_nonzero(votes >= 1)),
                "vote_pixels_ge_2": int(np.count_nonzero(votes >= 2)),
                "maximum_votes": int(votes.max(initial=0)),
                "path": str(path.relative_to(args.output_dir)),
                "sha256": sha256(path),
                "projections": projection_records,
            }
        )
        if index == 1 or index % 20 == 0 or index == len(names):
            print(json.dumps({"phase": "oof_geometry", "completed": index, "total": len(names)}), flush=True)

    # Validate every label-free vote map before any target evaluation is called.
    for record in target_records:
        path = args.output_dir / record["path"]
        require(path.is_file() and sha256(path) == record["sha256"], f"SEM-200 vote hash mismatch: {record['name']}")
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_oof_geometry_evidence",
        "geometry_recipe": geometry,
        "training_views": len(names),
        "fixed_validation_opened": False,
        "target_label_argument_used_by_prediction": False,
        "source_labels_scope": "only fold.train cameras for each target",
        "all_target_source_sets_disjoint": all(record["target_excluded_from_sources"] for record in target_records),
        "all_vote_maps_materialized_before_target_evaluation": True,
        "depth_manifest_sha256": sha256(depth_manifest_path),
        "depth_source_experiment": depth_manifest.get("experiment_id"),
        "elapsed_seconds": float(time.perf_counter() - started),
        "stable_source_records": [stable_cache[key][1] for key in sorted(stable_cache)],
        "targets": target_records,
    }
    write_json(args.output_dir / "evidence_manifest.json", manifest)
    return manifest


def _eligibility(
    *,
    name: str,
    sem001_root: Path,
    sem150_root: Path,
    baseline: np.ndarray,
) -> np.ndarray:
    sem001_map = _read_class_map(sem001_root / name, "SEM-200 SEM-001 source")
    sem150 = _read_class_map(sem150_root / name, "SEM-200 SEM-150 source")
    pre_foundation = sem150.copy()
    pre_foundation[(sem150 == 0) & (sem001_map == 4)] = 4
    return (baseline == 0) & (sem150 == 0) & (pre_foundation == 0)


def materialize_candidates(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    names: list[str],
) -> dict[str, Any]:
    evidence = read_json(args.output_dir / "evidence_manifest.json")
    require(
        evidence.get("status") == "completed_oof_geometry_evidence"
        and evidence.get("fixed_validation_opened") is False
        and evidence.get("all_vote_maps_materialized_before_target_evaluation") is True,
        "SEM-200 evidence is not sealed",
    )
    sem160_manifest = read_json(args.sem160_root / "prediction_manifest.json")
    require(
        sem160_manifest.get("experiment_id") == "SEM-160-DECK-BINARY-OOF"
        and sem160_manifest.get("selected_threshold") == 0.99
        and sem160_manifest.get("fixed_validation_opened") is False,
        "SEM-200 SEM-160 parent is not frozen",
    )
    expected = {record["name"]: record for record in sem160_manifest["predictions"]}
    require(set(expected) == set(names), "SEM-200 SEM-160 parent does not cover 219 views")
    output_root = args.output_dir / "oof_candidate"
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    minimum_votes = int(config["geometry"]["minimum_target_source_votes"])
    for name in names:
        baseline_path = args.sem160_root / expected[name]["prediction"]
        require(baseline_path.is_file() and sha256(baseline_path) == expected[name]["prediction_sha256"], f"SEM-200 SEM-160 hash mismatch: {name}")
        baseline = _read_class_map(baseline_path, "SEM-200 SEM-160 baseline")
        votes = _read_votes(_votes_path(args.output_dir, name))
        eligible = _eligibility(
            name=name,
            sem001_root=args.sem001_training_root,
            sem150_root=args.sem150_root,
            baseline=baseline,
        )
        candidate, additions = add_deck_with_protection(
            baseline,
            eligible,
            votes,
            minimum_source_votes=minimum_votes,
        )
        path = _candidate_path(args.output_dir, name)
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
        "minimum_target_source_votes": minimum_votes,
        "target_labels_opened": False,
        "fixed_validation_opened": False,
        "sem160_prediction_manifest_sha256": sha256(args.sem160_root / "prediction_manifest.json"),
        "evidence_manifest_sha256": sha256(args.output_dir / "evidence_manifest.json"),
        "all_nonaddition_pixels_exact": all(record["nonaddition_pixels_exact"] for record in records),
        "predictions": records,
    }
    write_json(args.output_dir / "prediction_manifest.json", manifest)
    return manifest


def _evaluate_half(
    *,
    names: list[str],
    labels_root: Path,
    sem160_root: Path,
    candidate_root: Path,
    sem160_manifest: dict[str, Any],
) -> dict[str, Any]:
    parent = {record["name"]: record for record in sem160_manifest["predictions"]}
    baseline_confusions = []
    candidate_confusions = []
    additions_by_target = np.zeros(5, dtype=np.int64)
    protected_exact = True
    per_view = []
    for name in names:
        target = label_array(labels_root / name)
        baseline = _read_class_map(sem160_root / parent[name]["prediction"], "SEM-200 baseline")
        candidate = _read_class_map(candidate_root / name, "SEM-200 candidate")
        changed = candidate != baseline
        protected_exact = protected_exact and bool(np.all((baseline[changed] == 0) & (candidate[changed] == 1)))
        additions_by_target += np.bincount(target[changed], minlength=5).astype(np.int64)
        base_confusion = local_confusion(target, baseline)
        cand_confusion = local_confusion(target, candidate)
        baseline_confusions.append(base_confusion)
        candidate_confusions.append(cand_confusion)
        def deck_iou(confusion: np.ndarray) -> float | None:
            true_positive = int(confusion[1, 1])
            false_positive = int(confusion[:, 1].sum() - true_positive)
            false_negative = int(confusion[1, :].sum() - true_positive)
            denominator = true_positive + false_positive + false_negative
            return float(true_positive / denominator) if denominator else None

        per_view.append(
            {
                "name": name,
                "added_deck_pixels": int(changed.sum()),
                "added_true_deck_pixels": int(np.count_nonzero(changed & (target == 1))),
                "added_background_pixels": int(np.count_nonzero(changed & (target == 0))),
                "baseline_deck_iou": deck_iou(base_confusion),
                "candidate_deck_iou": deck_iou(cand_confusion),
            }
        )
    baseline_confusion = merge_confusions(baseline_confusions)
    candidate_confusion = merge_confusions(candidate_confusions)
    baseline_metrics = metrics_from_confusion(baseline_confusion)
    candidate_metrics = metrics_from_confusion(candidate_confusion)
    evidence_pixels = int(additions_by_target.sum())
    return {
        "view_count": len(names),
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "added_deck_pixels": evidence_pixels,
        "additions_by_ground_truth": {
            CLASS_NAMES[index]: int(value) for index, value in enumerate(additions_by_target)
        },
        "geometry_addition_precision": float(additions_by_target[1] / evidence_pixels) if evidence_pixels else 0.0,
        "protected_predictions_exact": protected_exact,
        "per_view": per_view,
    }


def _gate(evaluation: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    baseline = evaluation["baseline_metrics"]
    candidate = evaluation["candidate_metrics"]
    base_deck = baseline["per_class"]["deck"]
    cand_deck = candidate["per_class"]["deck"]
    base_bg_to_deck = int(baseline["confusion_matrix_rows_gt_columns_prediction"][0][1])
    cand_bg_to_deck = int(candidate["confusion_matrix_rows_gt_columns_prediction"][0][1])
    ratio = float(cand_bg_to_deck / base_bg_to_deck) if base_bg_to_deck else (1.0 if cand_bg_to_deck == 0 else float("inf"))
    checks = {
        "nonempty_geometry_additions": int(evaluation["added_deck_pixels"]) > 0,
        "protected_predictions_exact": evaluation["protected_predictions_exact"] is True,
        "mIoU_non_decreasing": float(candidate["mIoU_with_background"]) >= float(baseline["mIoU_with_background"]) + float(gates["minimum_mIoU_gain"]),
        "deck_iou_gain": float(cand_deck["iou"]) >= float(base_deck["iou"]) + float(gates["minimum_deck_iou_gain"]),
        "deck_recall_gain": float(cand_deck["recall"]) >= float(base_deck["recall"]) + float(gates["minimum_deck_recall_gain"]),
        "deck_precision": float(cand_deck["precision"]) >= float(gates["minimum_deck_precision"]),
        "background_to_deck_ratio": ratio <= float(gates["maximum_background_to_deck_ratio"]),
    }
    return {
        "thresholds": gates,
        "checks": checks,
        "passed": all(checks.values()),
        "delta_mIoU_pp": 100.0 * (float(candidate["mIoU_with_background"]) - float(baseline["mIoU_with_background"])),
        "delta_deck_iou_pp": 100.0 * (float(cand_deck["iou"]) - float(base_deck["iou"])),
        "delta_deck_recall_pp": 100.0 * (float(cand_deck["recall"]) - float(base_deck["recall"])),
        "delta_deck_precision_pp": 100.0 * (float(cand_deck["precision"]) - float(base_deck["precision"])),
        "background_to_deck_ratio": ratio,
    }


def _write_panels(
    *,
    names: list[str],
    labels_root: Path,
    sem160_root: Path,
    candidate_root: Path,
    sem160_manifest: dict[str, Any],
    output_root: Path,
) -> None:
    parent = {record["name"]: record for record in sem160_manifest["predictions"]}
    output_root.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = label_array(labels_root / name)
        baseline = _read_class_map(sem160_root / parent[name]["prediction"], "SEM-200 panel baseline")
        candidate = _read_class_map(candidate_root / name, "SEM-200 panel candidate")
        changed = candidate != baseline
        outcome = np.zeros((*target.shape, 3), dtype=np.uint8)
        outcome[changed & (target == 1)] = np.asarray([48, 220, 96], dtype=np.uint8)
        outcome[changed & (target != 1)] = np.asarray([255, 48, 48], dtype=np.uint8)
        votes = _read_votes(_votes_path(output_root.parent, name))
        vote_rgb = np.zeros((*votes.shape, 3), dtype=np.uint8)
        vote_rgb[..., 0] = np.minimum(votes.astype(np.uint16) * 32, 255).astype(np.uint8)
        vote_rgb[..., 1] = np.minimum(votes.astype(np.uint16) * 16, 255).astype(np.uint8)
        panel = np.concatenate(
            [semantic_rgb(target), semantic_rgb(baseline), vote_rgb, semantic_rgb(candidate), outcome],
            axis=1,
        )
        Image.fromarray(panel, mode="RGB").save(output_root / name)


def evaluate_and_finish(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    names: list[str],
) -> dict[str, Any]:
    prediction_manifest_path = args.output_dir / "prediction_manifest.json"
    prediction_manifest = read_json(prediction_manifest_path)
    require(
        prediction_manifest.get("status") == "completed_label_free_candidate_materialization"
        and prediction_manifest.get("target_labels_opened") is False
        and prediction_manifest.get("fixed_validation_opened") is False,
        "SEM-200 candidate manifest is not sealed before evaluation",
    )
    for record in prediction_manifest["predictions"]:
        path = args.output_dir / record["path"]
        require(path.is_file() and sha256(path) == record["sha256"], f"SEM-200 candidate hash mismatch: {record['name']}")
    sem160_manifest = read_json(args.sem160_root / "prediction_manifest.json")
    partition = partition_training_names(names, expected_count=219)
    calibration = _evaluate_half(
        names=partition["calibration"],
        labels_root=args.labels_root,
        sem160_root=args.sem160_root,
        candidate_root=args.output_dir / "oof_candidate",
        sem160_manifest=sem160_manifest,
    )
    calibration_gate = _gate(calibration, config["gates"]["calibration"])
    check = None
    check_gate = None
    check_opened = False
    if calibration_gate["passed"]:
        check_opened = True
        check = _evaluate_half(
            names=partition["check"],
            labels_root=args.labels_root,
            sem160_root=args.sem160_root,
            candidate_root=args.output_dir / "oof_candidate",
            sem160_manifest=sem160_manifest,
        )
        check_gate = _gate(check, config["gates"]["check"])
    panel_names = [name for name in config["visual_samples"] if name in partition["calibration"]]
    _write_panels(
        names=panel_names,
        labels_root=args.labels_root,
        sem160_root=args.sem160_root,
        candidate_root=args.output_dir / "oof_candidate",
        sem160_manifest=sem160_manifest,
        output_root=args.output_dir / "calibration_panels",
    )
    qualified = bool(calibration_gate["passed"] and check_gate and check_gate["passed"])
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_oof_gate_passed" if qualified else "completed_calibration_gate_failed",
        "geometry_representation_qualified": qualified,
        "next_step": "register_SEM201_final_geometry_fusion" if qualified else "do_not_build_final_geometry_fusion",
        "fixed_validation_opened": False,
        "hidden_inputs_opened": False,
        "target_rgb_opened": False,
        "candidate_materialized_before_target_evaluation": True,
        "prediction_manifest_sha256": sha256(prediction_manifest_path),
        "calibration": calibration,
        "calibration_gate": calibration_gate,
        "check_labels_opened": check_opened,
        "check": check,
        "check_gate": check_gate,
        "frozen_rgb_unchanged": read_json(
            getattr(args, "depth_source_dir", args.output_dir) / "depth_manifest.json"
        )["frozen_rgb_unchanged"],
    }
    write_json(args.output_dir / "result.json", result)
    return result


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    validate_config(config)
    split = read_json(args.split)
    oof_split = read_json(args.oof_split)
    names = select_rendered_training_views(split)
    validation = set(split["semantic_validation_labeled"])
    require(not set(names) & validation and len(validation) == 74, "SEM-200 training/validation boundary failed")
    require(sha256(args.split) == config["data"]["semantic_split_sha256"], "SEM-200 semantic split hash differs")
    require(sha256(args.oof_split) == config["data"]["oof_split_sha256"], "SEM-200 OOF split hash differs")

    if args.resume_evaluation:
        require(args.output_dir.is_dir(), "SEM-200 resume output directory is missing")
        require(not (args.output_dir / "result.json").exists(), "SEM-200 result already exists; refusing to overwrite")
        require((args.output_dir / "depth_manifest.json").is_file(), "SEM-200 sealed depth manifest is missing")
        require((args.output_dir / "evidence_manifest.json").is_file(), "SEM-200 sealed evidence manifest is missing")
        require((args.output_dir / "prediction_manifest.json").is_file(), "SEM-200 sealed prediction manifest is missing")
    else:
        require(not args.output_dir.exists() or not any(args.output_dir.iterdir()), f"refusing to overwrite non-empty output: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        render_depth_maps(args=args, config=config, names=names)
        generate_oof_evidence(args=args, config=config, names=names, oof_split=oof_split)
        materialize_candidates(args=args, config=config, names=names)
    result = evaluate_and_finish(args=args, config=config, names=names)
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "status": result["status"],
                "geometry_representation_qualified": result["geometry_representation_qualified"],
                "result_sha256": sha256(args.output_dir / "result.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
