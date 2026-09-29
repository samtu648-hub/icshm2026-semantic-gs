#!/usr/bin/env python3
"""Train and qualify the SEM-210 continuous world-coordinate deck field."""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem200_multiview_deck_diag as sem200

from src.geometry_prior.camera_io import read_cameras_txt, read_images_txt
from src.semantic2d import CLASS_NAMES, label_array, read_json, sha256, write_json
from src.semantic_protected_fusion import (
    local_confusion,
    merge_confusions,
    metrics_from_confusion,
    partition_training_names,
)
from src.semantic_rendered_training import select_rendered_training_views
from src.semantic_multiview_deck import DepthObservation, add_deck_with_protection
from src.semantic_xyz_deck_field import (
    balanced_batch_indices,
    build_xyz_deck_field,
    coordinate_normalization,
    normalize_coordinates,
    surface_points_from_depth,
    surface_samples_from_view,
)


EXPERIMENT_ID = "SEM-210-XYZ-DECK-FIELD-OOF"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--oof-split", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--depth-source-dir", type=Path, required=True)
    parser.add_argument("--sem001-training-root", type=Path, required=True)
    parser.add_argument("--sem150-root", type=Path, required=True)
    parser.add_argument("--sem160-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate_config(config: dict[str, Any]) -> None:
    require(config.get("experiment_id") == EXPERIMENT_ID, "SEM-210 received another config")
    require(config.get("status") == "registered_not_run", "SEM-210 must start from its registered config")
    require(
        config["model"]
        == {
            "input": "normalized official-world XYZ on RGB-030-FULL visible surface",
            "fourier_levels": 6,
            "width": 64,
            "hidden_layers": 3,
            "output": "binary non_deck versus deck logit",
        },
        "SEM-210 model recipe differs from registration",
    )
    require(
        config["training"]
        == {
            "folds": 3,
            "iterations_per_fold": 2000,
            "surface_stride_pixels": 8,
            "minimum_foreground_mass": 0.5,
            "batch_size": 32768,
            "balanced_half_deck_half_non_deck": True,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 1e-05,
            "checkpoint_selection": "final_2000_only",
        },
        "SEM-210 training recipe differs from registration",
    )


def _load_depth(root: Path, name: str) -> DepthObservation:
    path = root / "frozen_depth" / f"{Path(name).stem}.npz"
    with np.load(path) as archive:
        require(set(archive.files) == {"depth", "foreground_mass"}, f"SEM-210 depth schema mismatch: {name}")
        return DepthObservation(
            np.asarray(archive["depth"], dtype=np.float32),
            np.asarray(archive["foreground_mass"], dtype=np.float32),
        )


def _probability_path(output_dir: Path, fold_index: int, name: str) -> Path:
    return output_dir / f"fold-{fold_index}" / "oof_probabilities" / f"{Path(name).stem}.npz"


def train_and_predict_folds(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    names: list[str],
    oof_split: dict[str, Any],
) -> dict[str, Any]:
    import torch

    require(torch.cuda.is_available(), "SEM-210 requires CUDA")
    cameras = read_cameras_txt(args.scene_dir / "sparse" / "0" / "cameras.txt")
    image_records = read_images_txt(args.scene_dir / "sparse" / "0" / "images.txt")
    images = {record.name: record for record in image_records.values()}
    require(set(names).issubset(images), "SEM-210 prepared scene misses training cameras")
    model_config = config["model"]
    training = config["training"]
    seed = int(config["seed"])
    all_predictions: list[dict[str, Any]] = []
    fold_results: list[dict[str, Any]] = []
    predicted: set[str] = set()
    started_all = time.perf_counter()

    @lru_cache(maxsize=16)
    def observation(name: str) -> DepthObservation:
        return _load_depth(args.depth_source_dir, name)

    for fold in oof_split["folds"]:
        fold_index = int(fold["fold"])
        train_names = list(fold["train"])
        oof_names = list(fold["oof"])
        require(not set(train_names) & set(oof_names), f"SEM-210 fold {fold_index} overlaps")
        sample_points: list[np.ndarray] = []
        sample_targets: list[np.ndarray] = []
        for name in train_names:
            image = images[name]
            points, binary = surface_samples_from_view(
                label_array(args.labels_root / name),
                observation(name),
                camera=cameras[image.camera_id],
                image=image,
                stride=int(training["surface_stride_pixels"]),
                minimum_foreground_mass=float(training["minimum_foreground_mass"]),
            )
            sample_points.append(points)
            sample_targets.append(binary)
        points = np.concatenate(sample_points)
        targets = np.concatenate(sample_targets)
        require(bool((targets == 0).any()) and bool((targets == 1).any()), f"SEM-210 fold {fold_index} misses a binary class")
        center, scale = coordinate_normalization(points)
        normalized = normalize_coordinates(points, center, scale)
        positive_np = normalized[targets == 1]
        negative_np = normalized[targets == 0]
        fold_seed = seed + fold_index
        torch.manual_seed(fold_seed)
        torch.cuda.manual_seed_all(fold_seed)
        model = build_xyz_deck_field(
            fourier_levels=int(model_config["fourier_levels"]),
            width=int(model_config["width"]),
            hidden_layers=int(model_config["hidden_layers"]),
        ).cuda()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        positive = torch.from_numpy(positive_np).cuda()
        negative = torch.from_numpy(negative_np).cuda()
        generator = torch.Generator(device="cuda")
        generator.manual_seed(fold_seed)
        batch_size = int(training["batch_size"])
        iterations = int(training["iterations_per_fold"])
        losses: list[float] = []
        train_started = time.perf_counter()
        peak_before = int(torch.cuda.max_memory_allocated())
        for iteration in range(1, iterations + 1):
            positive_indices, negative_indices = balanced_batch_indices(
                len(positive),
                len(negative),
                batch_size=batch_size,
                generator=generator,
                device=positive.device,
            )
            batch = torch.cat((positive[positive_indices], negative[negative_indices]), dim=0)
            target = torch.cat(
                (
                    torch.ones(batch_size // 2, device=batch.device),
                    torch.zeros(batch_size // 2, device=batch.device),
                )
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))
            if iteration == 1 or iteration % 500 == 0 or iteration == iterations:
                print(json.dumps({"phase": "xyz_field_training", "fold": fold_index, "iteration": iteration, "iterations": iterations, "loss": losses[-1]}), flush=True)
        training_seconds = float(time.perf_counter() - train_started)
        metadata = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "fold": fold_index,
            "seed": fold_seed,
            "model": model_config,
            "training": training,
            "center": center.astype(float).tolist(),
            "scale": float(scale),
        }
        checkpoint = args.output_dir / f"fold-{fold_index}" / "checkpoints" / "xyz_deck_field_final.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"metadata": metadata, "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}},
            checkpoint,
        )
        del model, optimizer
        torch.cuda.empty_cache()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        require(payload["metadata"] == metadata, f"SEM-210 fold {fold_index} checkpoint metadata changed")
        model = build_xyz_deck_field(
            fourier_levels=int(model_config["fourier_levels"]),
            width=int(model_config["width"]),
            hidden_layers=int(model_config["hidden_layers"]),
        ).cuda()
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        del positive, negative, points, targets, normalized, positive_np, negative_np
        torch.cuda.empty_cache()

        inference_started = time.perf_counter()
        for index, name in enumerate(oof_names, start=1):
            image = images[name]
            world, y, x = surface_points_from_depth(
                observation(name),
                camera=cameras[image.camera_id],
                image=image,
                minimum_foreground_mass=float(training["minimum_foreground_mass"]),
            )
            coordinates = normalize_coordinates(world, center, scale)
            probability = np.zeros((989, 1320), dtype=np.float16)
            chunks: list[np.ndarray] = []
            with torch.inference_mode():
                for start in range(0, len(coordinates), 262144):
                    batch = torch.from_numpy(coordinates[start : start + 262144]).cuda()
                    chunks.append(torch.sigmoid(model(batch)).cpu().numpy().astype(np.float16))
            values = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float16)
            probability[y, x] = values
            path = _probability_path(args.output_dir, fold_index, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, deck_probability=probability)
            all_predictions.append(
                {
                    "name": name,
                    "fold": fold_index,
                    "path": str(path.relative_to(args.output_dir)),
                    "sha256": sha256(path),
                    "surface_pixels": int(len(values)),
                }
            )
            predicted.add(name)
            if index == 1 or index % 20 == 0 or index == len(oof_names):
                print(json.dumps({"phase": "xyz_field_oof_inference", "fold": fold_index, "completed": index, "total": len(oof_names)}), flush=True)
        fold_results.append(
            {
                "fold": fold_index,
                "seed": fold_seed,
                "training_views": len(train_names),
                "oof_views": len(oof_names),
                "training_surface_samples": int(len(sample_targets[0]) + sum(len(value) for value in sample_targets[1:])),
                "training_deck_samples": int(sum(int(value.sum()) for value in sample_targets)),
                "normalization_center": center.astype(float).tolist(),
                "normalization_scale": float(scale),
                "loss_first": losses[0],
                "loss_last": losses[-1],
                "loss_last_100_mean": float(np.mean(losses[-100:])),
                "training_seconds": training_seconds,
                "inference_seconds": float(time.perf_counter() - inference_started),
                "peak_cuda_allocated_mib": int(max(peak_before, torch.cuda.max_memory_allocated()) // (1024 * 1024)),
                "checkpoint": str(checkpoint.relative_to(args.output_dir)),
                "checkpoint_sha256": sha256(checkpoint),
                "checkpoint_reload": "passed",
            }
        )
        del model, sample_points, sample_targets
        torch.cuda.empty_cache()

    require(predicted == set(names) and len(all_predictions) == len(names), "SEM-210 OOF predictions are incomplete")
    all_predictions.sort(key=lambda record: names.index(record["name"]))
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_three_fold_oof_probabilities",
        "training_labels_scope": "fold.train_only",
        "target_labels_opened_during_inference": False,
        "target_rgb_opened": False,
        "fixed_validation_opened": False,
        "depth_manifest_sha256": sha256(args.depth_source_dir / "depth_manifest.json"),
        "elapsed_seconds": float(time.perf_counter() - started_all),
        "folds": fold_results,
        "predictions": all_predictions,
    }
    write_json(args.output_dir / "oof_probability_manifest.json", manifest)
    return manifest


def evaluate_thresholds(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    names: list[str],
    probability_manifest: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    probability_paths = {record["name"]: args.output_dir / record["path"] for record in probability_manifest["predictions"]}
    sem160_manifest = read_json(args.sem160_root / "prediction_manifest.json")
    parent = {record["name"]: record for record in sem160_manifest["predictions"]}
    thresholds = [float(value) for value in config["fusion"]["threshold_grid"]]
    baseline_confusions = []
    candidate_confusions = {threshold: [] for threshold in thresholds}
    target_counts = {threshold: np.zeros(5, dtype=np.int64) for threshold in thresholds}
    protected = {threshold: True for threshold in thresholds}
    for name in names:
        target = label_array(args.labels_root / name)
        baseline = sem200._read_class_map(args.sem160_root / parent[name]["prediction"], "SEM-210 baseline")
        eligible = sem200._eligibility(
            name=name,
            sem001_root=args.sem001_training_root,
            sem150_root=args.sem150_root,
            baseline=baseline,
        )
        with np.load(probability_paths[name]) as archive:
            probability = np.asarray(archive["deck_probability"], dtype=np.float32)
        baseline_confusions.append(local_confusion(target, baseline))
        for threshold in thresholds:
            candidate, additions = add_deck_with_protection(
                baseline,
                eligible,
                probability >= threshold,
                minimum_source_votes=1,
            )
            candidate_confusions[threshold].append(local_confusion(target, candidate))
            target_counts[threshold] += np.bincount(target[additions], minlength=5).astype(np.int64)
            protected[threshold] = protected[threshold] and bool(np.array_equal(candidate[~additions], baseline[~additions]))
    baseline_metrics = metrics_from_confusion(merge_confusions(baseline_confusions))
    records = []
    for threshold in thresholds:
        metrics = metrics_from_confusion(merge_confusions(candidate_confusions[threshold]))
        added = int(target_counts[threshold].sum())
        evaluation = {
            "baseline_metrics": baseline_metrics,
            "candidate_metrics": metrics,
            "added_deck_pixels": added,
            "additions_by_ground_truth": {CLASS_NAMES[index]: int(value) for index, value in enumerate(target_counts[threshold])},
            "geometry_addition_precision": float(target_counts[threshold][1] / added) if added else 0.0,
            "protected_predictions_exact": protected[threshold],
        }
        gate = sem200._gate(evaluation, config["gates"]["calibration"])
        records.append({"threshold": threshold, "evaluation": evaluation, "gate": gate})
    return baseline_metrics, records


def materialize_selected(
    *,
    args: argparse.Namespace,
    names: list[str],
    probability_manifest: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    probability_paths = {record["name"]: args.output_dir / record["path"] for record in probability_manifest["predictions"]}
    sem160_manifest = read_json(args.sem160_root / "prediction_manifest.json")
    parent = {record["name"]: record for record in sem160_manifest["predictions"]}
    root = args.output_dir / "oof_candidate"
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for name in names:
        baseline = sem200._read_class_map(args.sem160_root / parent[name]["prediction"], "SEM-210 baseline")
        eligible = sem200._eligibility(
            name=name,
            sem001_root=args.sem001_training_root,
            sem150_root=args.sem150_root,
            baseline=baseline,
        )
        with np.load(probability_paths[name]) as archive:
            probability = np.asarray(archive["deck_probability"], dtype=np.float32)
        candidate, additions = add_deck_with_protection(
            baseline,
            eligible,
            probability >= float(threshold),
            minimum_source_votes=1,
        )
        path = root / name
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
        "selected_threshold": float(threshold),
        "target_labels_opened": False,
        "fixed_validation_opened": False,
        "all_nonaddition_pixels_exact": all(record["nonaddition_pixels_exact"] for record in records),
        "oof_probability_manifest_sha256": sha256(args.output_dir / "oof_probability_manifest.json"),
        "predictions": records,
    }
    write_json(args.output_dir / "prediction_manifest.json", manifest)
    return manifest


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    validate_config(config)
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()), f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = read_json(args.split)
    oof_split = read_json(args.oof_split)
    names = select_rendered_training_views(split)
    require(sha256(args.split) == config["data"]["semantic_split_sha256"], "SEM-210 semantic split hash differs")
    require(sha256(args.oof_split) == config["data"]["oof_split_sha256"], "SEM-210 OOF split hash differs")
    require(sha256(args.depth_source_dir / "depth_manifest.json") == config["inputs"]["sem200_depth_manifest_sha256"], "SEM-210 sealed depth manifest hash differs")

    probability_manifest = train_and_predict_folds(
        args=args, config=config, names=names, oof_split=oof_split
    )
    partition = partition_training_names(names, expected_count=219)
    baseline_metrics, threshold_records = evaluate_thresholds(
        args=args,
        config=config,
        names=partition["calibration"],
        probability_manifest=probability_manifest,
    )
    passing = [record for record in threshold_records if record["gate"]["passed"]]
    pool = passing if passing else threshold_records
    selected = max(
        pool,
        key=lambda record: (
            record["evaluation"]["candidate_metrics"]["per_class"]["deck"]["iou"],
            record["evaluation"]["candidate_metrics"]["mIoU_with_background"],
            record["threshold"],
        ),
    )
    materialize_selected(
        args=args,
        names=names,
        probability_manifest=probability_manifest,
        threshold=float(selected["threshold"]),
    )
    # Reuse the independently checked protected-candidate evaluator.  It opens
    # check labels only if this selected candidate passes the same calibration gate.
    sem200.EXPERIMENT_ID = EXPERIMENT_ID
    config_for_final = dict(config)
    config_for_final["visual_samples"] = []
    result = sem200.evaluate_and_finish(
        args=args, config=config_for_final, names=names
    )
    result.update(
        {
            "threshold_grid": [float(value) for value in config["fusion"]["threshold_grid"]],
            "calibration_threshold_records": threshold_records,
            "calibration_thresholds_passing": len(passing),
            "selected_threshold": float(selected["threshold"]),
            "selection_pool": "passing_only" if passing else "diagnostic_best_no_gate_passed",
            "baseline_calibration_metrics_preselection": baseline_metrics,
        }
    )
    if not passing:
        result["status"] = "completed_calibration_gate_failed"
        result["geometry_representation_qualified"] = False
        result["next_step"] = "do_not_train_final_xyz_deck_field"
    write_json(args.output_dir / "result.json", result)
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "status": result["status"],
                "selected_threshold": result["selected_threshold"],
                "calibration_thresholds_passing": len(passing),
                "check_labels_opened": result["check_labels_opened"],
                "result_sha256": sha256(args.output_dir / "result.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
