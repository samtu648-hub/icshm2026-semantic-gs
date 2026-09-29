#!/usr/bin/env python3
"""Train and qualify the SEM-220 early-fusion RGB plus surface deck branch."""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem200_multiview_deck_diag as sem200

from src.geometry_prior.camera_io import read_cameras_txt, read_images_txt
from src.semantic2d import CLASS_NAMES, fold_for_index, label_array, read_json, sha256, write_json
from src.semantic_deck_binary import binary_class_weights_from_labels, binary_target, read_rgb
from src.semantic_multiview_deck import DepthObservation, add_deck_with_protection
from src.semantic_protected_fusion import local_confusion, merge_confusions, metrics_from_confusion, partition_training_names
from src.semantic_rendered_training import select_rendered_training_views, validate_rendered_dataset_manifest
from src.semantic_rgbd_deck import (
    GEOMETRY_CHANNELS,
    frozen_surface_features,
    load_rgbd_model,
    normalize_rgbd,
    random_native_rgbd_pair,
    sliding_rgbd_deck_probability,
)


EXPERIMENT_ID = "SEM-220-RGBD-SURFACE-DECK-OOF"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--oof-split", type=Path, required=True)
    parser.add_argument("--rendered-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--depth-source-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--sem001-training-root", type=Path, required=True)
    parser.add_argument("--sem150-root", type=Path, required=True)
    parser.add_argument("--sem160-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate_config(config: dict[str, Any]) -> None:
    require(config.get("experiment_id") == EXPERIMENT_ID, "SEM-220 received another config")
    require(config.get("status") == "registered_not_run", "SEM-220 must start registered")
    require(config["model"]["input_channels"] == ["ImageNet RGB", *GEOMETRY_CHANNELS], "SEM-220 input channels differ")
    require(
        config["training"]
        == {
            "seed": 20260805,
            "folds": 3,
            "iterations_per_fold": 3000,
            "crop_size": 960,
            "batch_size": 1,
            "optimizer": "AdamW",
            "learning_rate": 0.00006,
            "weight_decay": 0.01,
            "loss": "two_class_cross_entropy_only",
            "class_weight_rule": "per-fold inverse frequency median_count/count clipped to [0.05,20]",
            "paired_augmentation": "uniform native 960 crop and horizontal flip p=0.5 only",
            "amp": "bf16_if_supported_else_fp16",
            "checkpoint_selection": "final_3000_only_no_fold_selection",
        },
        "SEM-220 training recipe differs from registration",
    )


def load_depth(root: Path, name: str) -> DepthObservation:
    path = root / "frozen_depth" / f"{Path(name).stem}.npz"
    with np.load(path) as archive:
        require(set(archive.files) == {"depth", "foreground_mass"}, f"SEM-220 depth schema mismatch: {name}")
        return DepthObservation(
            np.asarray(archive["depth"], dtype=np.float32),
            np.asarray(archive["foreground_mass"], dtype=np.float32),
        )


def prepare_geometry_features(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    names: list[str],
) -> dict[str, Any]:
    cameras = read_cameras_txt(args.scene_dir / "sparse" / "0" / "cameras.txt")
    image_records = read_images_txt(args.scene_dir / "sparse" / "0" / "images.txt")
    images = {record.name: record for record in image_records.values()}
    root = args.output_dir / "geometry_features"
    records = []
    for index, name in enumerate(names, start=1):
        image = images[name]
        features = frozen_surface_features(
            load_depth(args.depth_source_dir, name),
            cameras[image.camera_id],
            minimum_foreground_mass=float(config["geometry"]["minimum_foreground_mass"]),
        )
        path = root / f"{Path(name).stem}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, geometry=features.astype(np.float16))
        records.append({"name": name, "path": str(path.relative_to(args.output_dir)), "sha256": sha256(path)})
        if index == 1 or index % 30 == 0 or index == len(names):
            print(json.dumps({"phase": "geometry_features", "completed": index, "total": len(names)}), flush=True)
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_label_free_geometry_features",
        "channels": list(GEOMETRY_CHANNELS),
        "source_depth_manifest_sha256": sha256(args.depth_source_dir / "depth_manifest.json"),
        "labels_opened": False,
        "target_rgb_opened": False,
        "fixed_validation_opened": False,
        "records": records,
    }
    write_json(args.output_dir / "geometry_feature_manifest.json", manifest)
    return manifest


def read_geometry(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        require(set(archive.files) == {"geometry"}, f"SEM-220 geometry schema mismatch: {path}")
        value = np.asarray(archive["geometry"], dtype=np.float32)
    require(value.shape == (989, 1320, 4) and np.isfinite(value).all(), f"SEM-220 invalid geometry: {path}")
    return value


def save_checkpoint(path: Path, model: Any, metadata: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": 1, "metadata": metadata, "model_state_dict": model.state_dict()}, path)


def train_oof(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    names: list[str],
    oof_split: dict[str, Any],
    feature_manifest: dict[str, Any],
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    require(torch.cuda.is_available(), "SEM-220 requires CUDA")
    device = torch.device("cuda:0")
    amp_name, amp_dtype = ("bf16", torch.bfloat16) if torch.cuda.is_bf16_supported() else ("fp16", torch.float16)
    training = config["training"]
    crop = int(training["crop_size"])
    image_root = args.rendered_root / "images"
    feature_paths = {record["name"]: args.output_dir / record["path"] for record in feature_manifest["records"]}
    all_records: list[dict[str, Any]] = []
    fold_results: list[dict[str, Any]] = []
    predicted: set[str] = set()
    for fold_index in range(3):
        fold = fold_for_index(oof_split, fold_index)
        train_names = list(fold["train"])
        oof_names = list(fold["oof"])
        class_weights, class_counts = binary_class_weights_from_labels(args.labels_root, train_names)
        geometry_cache = {name: read_geometry(feature_paths[name]) for name in train_names}
        seed = int(training["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        model = load_rgbd_model(args.model_dir, seed).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        weight_tensor = torch.as_tensor(class_weights, device=device)
        rng = random.Random(seed + fold_index * 100003)
        scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
        torch.cuda.reset_peak_memory_stats(device)
        losses: list[float] = []
        started = time.perf_counter()
        model.train()
        for iteration in range(1, int(training["iterations_per_fold"]) + 1):
            name = train_names[rng.randrange(len(train_names))]
            target = binary_target(label_array(args.labels_root / name))
            rgb_crop, geometry_crop, target_crop = random_native_rgbd_pair(
                read_rgb(image_root / name), geometry_cache[name], target, crop, rng
            )
            pixel_values = normalize_rgbd(rgb_crop, geometry_crop).unsqueeze(0).to(device)
            targets = torch.from_numpy(target_crop.astype(np.int64)).unsqueeze(0).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(pixel_values=pixel_values).logits
                logits = functional.interpolate(logits, size=(crop, crop), mode="bilinear", align_corners=False)
                loss = functional.cross_entropy(logits.float(), targets, weight=weight_tensor)
            require(bool(torch.isfinite(loss)), f"SEM-220 non-finite loss at fold {fold_index} iteration {iteration}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            if iteration == 1 or iteration % 250 == 0 or iteration == int(training["iterations_per_fold"]):
                print(json.dumps({"phase": "training", "fold": fold_index, "iteration": iteration, "loss": losses[-1]}), flush=True)
        torch.cuda.synchronize(device)
        training_seconds = float(time.perf_counter() - started)
        metadata = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "fold": fold_index,
            "seed": seed,
            "training": training,
            "model": config["model"],
            "geometry_feature_manifest_sha256": sha256(args.output_dir / "geometry_feature_manifest.json"),
        }
        checkpoint = args.output_dir / f"fold-{fold_index}" / "checkpoints" / "rgbd_deck_final.pth"
        save_checkpoint(checkpoint, model, metadata)
        del model, optimizer, geometry_cache
        torch.cuda.empty_cache()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        require(payload["metadata"] == metadata, f"SEM-220 fold {fold_index} checkpoint metadata changed")
        model = load_rgbd_model(args.model_dir, seed).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        inference_started = time.perf_counter()
        for index, name in enumerate(oof_names, start=1):
            probability = sliding_rgbd_deck_probability(
                model,
                read_rgb(image_root / name),
                read_geometry(feature_paths[name]),
                crop=crop,
                stride=int(config["inference"]["stride"]),
                device=device,
                amp_dtype=amp_dtype,
            )
            probability_path = args.output_dir / f"fold-{fold_index}" / "oof_deck_probabilities" / f"{Path(name).stem}.npz"
            probability_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(probability_path, deck_probability=probability.astype(np.float16))
            all_records.append(
                {
                    "name": name,
                    "fold": fold_index,
                    "path": str(probability_path.relative_to(args.output_dir)),
                    "sha256": sha256(probability_path),
                }
            )
            predicted.add(name)
            if index == 1 or index % 20 == 0 or index == len(oof_names):
                print(json.dumps({"phase": "oof_inference", "fold": fold_index, "completed": index, "total": len(oof_names)}), flush=True)
        torch.cuda.synchronize(device)
        fold_results.append(
            {
                "fold": fold_index,
                "training_views": len(train_names),
                "oof_views": len(oof_names),
                "binary_class_pixel_counts_training": class_counts.astype(int).tolist(),
                "binary_class_weights": class_weights.astype(float).tolist(),
                "loss_first": losses[0],
                "loss_last": losses[-1],
                "loss_last_100_mean": float(np.mean(losses[-100:])),
                "training_seconds": training_seconds,
                "inference_seconds": float(time.perf_counter() - inference_started),
                "peak_cuda_allocated_mib": int(torch.cuda.max_memory_allocated(device) // (1024 * 1024)),
                "amp": amp_name,
                "checkpoint": str(checkpoint.relative_to(args.output_dir)),
                "checkpoint_sha256": sha256(checkpoint),
                "checkpoint_reload": "passed",
            }
        )
        del model
        torch.cuda.empty_cache()
    require(predicted == set(names) and len(all_records) == len(names), "SEM-220 OOF predictions are incomplete")
    all_records.sort(key=lambda record: names.index(record["name"]))
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_three_fold_oof_probabilities",
        "fixed_validation_opened": False,
        "real_rgb_opened": False,
        "training_rendered_rgb_only": True,
        "geometry_feature_manifest_sha256": sha256(args.output_dir / "geometry_feature_manifest.json"),
        "folds": fold_results,
        "predictions": all_records,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    write_json(args.output_dir / "oof_probability_manifest.json", manifest)
    return manifest


def load_probability(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        require(set(archive.files) == {"deck_probability"}, f"SEM-220 probability schema mismatch: {path}")
        value = np.asarray(archive["deck_probability"], dtype=np.float32)
    require(value.shape == (989, 1320) and np.isfinite(value).all(), f"SEM-220 invalid probability: {path}")
    return value


def evaluate_thresholds(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    names: list[str],
    probability_paths: dict[str, Path],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sem160_manifest = read_json(args.sem160_root / "prediction_manifest.json")
    parent = {record["name"]: record for record in sem160_manifest["predictions"]}
    thresholds = [float(value) for value in config["fusion"]["threshold_grid"]]
    baseline_confusions: list[np.ndarray] = []
    candidate_confusions = {threshold: [] for threshold in thresholds}
    counts = {threshold: np.zeros(5, dtype=np.int64) for threshold in thresholds}
    protected = {threshold: True for threshold in thresholds}
    for name in names:
        baseline = sem200._read_class_map(args.sem160_root / parent[name]["prediction"], "SEM-220 baseline")
        eligible = sem200._eligibility(name=name, sem001_root=args.sem001_training_root, sem150_root=args.sem150_root, baseline=baseline)
        probability = load_probability(probability_paths[name])
        candidates = {
            threshold: add_deck_with_protection(baseline, eligible, probability >= threshold, minimum_source_votes=1)
            for threshold in thresholds
        }
        target = label_array(args.labels_root / name)
        baseline_confusions.append(local_confusion(target, baseline))
        for threshold, (candidate, additions) in candidates.items():
            candidate_confusions[threshold].append(local_confusion(target, candidate))
            counts[threshold] += np.bincount(target[additions], minlength=5).astype(np.int64)
            protected[threshold] = protected[threshold] and bool(np.array_equal(candidate[~additions], baseline[~additions]))
    baseline_metrics = metrics_from_confusion(merge_confusions(baseline_confusions))
    records = []
    for threshold in thresholds:
        metrics = metrics_from_confusion(merge_confusions(candidate_confusions[threshold]))
        added = int(counts[threshold].sum())
        evaluation = {
            "baseline_metrics": baseline_metrics,
            "candidate_metrics": metrics,
            "added_deck_pixels": added,
            "additions_by_ground_truth": {CLASS_NAMES[index]: int(value) for index, value in enumerate(counts[threshold])},
            "geometry_addition_precision": float(counts[threshold][1] / added) if added else 0.0,
            "protected_predictions_exact": protected[threshold],
        }
        records.append({"threshold": threshold, "evaluation": evaluation, "gate": sem200._gate(evaluation, config["gates"]["calibration"])})
    return baseline_metrics, records


def materialize_selected(
    *,
    args: argparse.Namespace,
    names: list[str],
    probability_paths: dict[str, Path],
    threshold: float,
) -> None:
    sem160_manifest = read_json(args.sem160_root / "prediction_manifest.json")
    parent = {record["name"]: record for record in sem160_manifest["predictions"]}
    root = args.output_dir / "oof_candidate"
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for name in names:
        baseline = sem200._read_class_map(args.sem160_root / parent[name]["prediction"], "SEM-220 baseline")
        eligible = sem200._eligibility(name=name, sem001_root=args.sem001_training_root, sem150_root=args.sem150_root, baseline=baseline)
        candidate, additions = add_deck_with_protection(
            baseline, eligible, load_probability(probability_paths[name]) >= threshold, minimum_source_votes=1
        )
        path = root / name
        Image.fromarray(candidate, mode="L").save(path)
        records.append({"name": name, "path": str(path.relative_to(args.output_dir)), "sha256": sha256(path), "added_deck_pixels": int(additions.sum()), "nonaddition_pixels_exact": bool(np.array_equal(candidate[~additions], baseline[~additions]))})
    write_json(
        args.output_dir / "prediction_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "status": "completed_label_free_candidate_materialization",
            "selected_threshold": float(threshold),
            "target_labels_opened": False,
            "fixed_validation_opened": False,
            "all_nonaddition_pixels_exact": all(record["nonaddition_pixels_exact"] for record in records),
            "predictions": records,
        },
    )


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    validate_config(config)
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()), f"refusing to overwrite non-empty output: {args.output_dir}")
    require(sha256(args.split) == config["data"]["semantic_split_sha256"], "SEM-220 semantic split hash differs")
    require(sha256(args.oof_split) == config["data"]["oof_split_sha256"], "SEM-220 OOF split hash differs")
    require(sha256(args.depth_source_dir / "depth_manifest.json") == config["inputs"]["sem200_depth_manifest_sha256"], "SEM-220 depth manifest differs")
    rendered_manifest = args.rendered_root / "rendered_dataset_manifest.json"
    require(sha256(rendered_manifest) == config["inputs"]["rendered_training_manifest_sha256"], "SEM-220 rendered RGB manifest differs")
    split = read_json(args.split)
    names = select_rendered_training_views(split, expected_count=219)
    require(not set(names) & set(split["semantic_validation_labeled"]) and len(split["semantic_validation_labeled"]) == 74, "SEM-220 training/validation isolation failed")
    validate_rendered_dataset_manifest(read_json(rendered_manifest), names)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_manifest = prepare_geometry_features(args=args, config=config, names=names)
    oof_manifest = train_oof(args=args, config=config, names=names, oof_split=read_json(args.oof_split), feature_manifest=feature_manifest)
    probability_paths = {record["name"]: args.output_dir / record["path"] for record in oof_manifest["predictions"]}
    partition = partition_training_names(names, expected_count=219)
    baseline_metrics, records = evaluate_thresholds(args=args, config=config, names=partition["calibration"], probability_paths=probability_paths)
    passing = [record for record in records if record["gate"]["passed"]]
    pool = passing if passing else records
    selected = max(pool, key=lambda record: (record["evaluation"]["candidate_metrics"]["per_class"]["deck"]["iou"], record["evaluation"]["candidate_metrics"]["mIoU_with_background"], record["threshold"]))
    materialize_selected(args=args, names=names, probability_paths=probability_paths, threshold=float(selected["threshold"]))
    sem200.EXPERIMENT_ID = EXPERIMENT_ID
    final_config = dict(config)
    final_config["visual_samples"] = []
    result = sem200.evaluate_and_finish(args=args, config=final_config, names=names)
    result.update(
        {
            "calibration_threshold_records": records,
            "calibration_thresholds_passing": len(passing),
            "selected_threshold": float(selected["threshold"]),
            "selection_pool": "passing_only" if passing else "diagnostic_best_no_gate_passed",
            "baseline_calibration_metrics_preselection": baseline_metrics,
            "geometry_feature_manifest_sha256": sha256(args.output_dir / "geometry_feature_manifest.json"),
            "oof_probability_manifest_sha256": sha256(args.output_dir / "oof_probability_manifest.json"),
        }
    )
    if not passing:
        result["status"] = "completed_calibration_gate_failed"
        result["geometry_representation_qualified"] = False
        result["next_step"] = "do_not_train_final_rgbd_deck_model"
    write_json(args.output_dir / "result.json", result)
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "status": result["status"], "calibration_thresholds_passing": len(passing), "selected_threshold": result["selected_threshold"], "check_labels_opened": result["check_labels_opened"], "result_sha256": sha256(args.output_dir / "result.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
