#!/usr/bin/env python3
"""Independent SEM-280 provenance, matched training and native replay audit."""

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_sem260_error_budget import manifest_paths, png
from run_sem250_colmap_track_deck_diag import verify_freeze
from run_sem280_foundation_boundary_oof import array_hash, initialize, state_hash
from src.semantic2d import read_json, sha256, write_json
from src.semantic_foundation_boundary import foundation_inner_boundary_mask, fuse_foundation
from src.submission_sem161 import sliding


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact-root", "run-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/runs/SEM-280-FOUNDATION-BOUNDARY-OOF/config.yaml",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(4)
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    run = read_json(args.run_dir / "result.json")
    seal = read_json(args.run_dir / "prediction_seal.json")
    config = read_json(args.config)
    require(
        run["experiment_id"] == seal["experiment_id"] == config["experiment_id"]
        == "SEM-280-FOUNDATION-BOUNDARY-OOF",
        "Experiment mismatch",
    )
    require(run["config_sha256"] == seal["config_sha256"] == sha256(args.config), "Config changed")
    require(run["prediction_seal_sha256"] == sha256(args.run_dir / "prediction_seal.json"), "Seal changed")
    require(
        seal["prediction_manifest_sha256"] == sha256(args.run_dir / "prediction_manifest.json"),
        "Prediction manifest changed",
    )
    require(run["calibration_sha256"] == sha256(args.run_dir / "calibration_sealed.json"), "Calibration changed")
    for name, digest in seal["source_sha256"].items():
        require(sha256(ROOT / name) == digest, "Current source changed")
        require(sha256(args.run_dir / "source_snapshot" / Path(name).name) == digest, "Snapshot changed")
    require(
        sha256(ROOT / config["parents"]["sem270_result"])
        == config["parents"]["sem270_result_sha256"],
        "SEM-270 parent result changed",
    )
    for key in ("semantic_split", "oof_split"):
        require(sha256(ROOT / config[key]) == config[key + "_sha256"], "Split changed")
    folds = read_json(ROOT / config["oof_split"])
    graph = ROOT / config["camera_graph"]
    require(sha256(graph) == folds["camera_graph_sha256"], "Camera graph changed")
    centers = {row["name"]: np.asarray(row["center"]) for row in read_json(graph)["nodes"]}
    all_names = read_json(ROOT / config["semantic_split"])["semantic_train_labeled"]
    rgb_paths = manifest_paths(
        args.artifact_root / config["parents"]["rgb"],
        "rendered_dataset_manifest.json",
        "rendered_images",
        "path",
        "sha256",
    )
    baseline = manifest_paths(
        args.artifact_root / config["parents"]["sem160"],
        "prediction_manifest.json",
        "predictions",
        "prediction",
        "prediction_sha256",
    )
    label_records = {
        row["image"]: row
        for row in read_json(ROOT / "experiments/runs/SEM-001/label_manifest.json")["records"]
    }
    predictions = read_json(args.run_dir / "prediction_manifest.json")
    old_predictions = read_json(ROOT / "outputs/sem270/full-20260907-r1/prediction_manifest.json")
    rows = []
    prediction_counter = 0
    exact_control_probability_replays = 0
    control_probability_elements = 0
    control_probability_absolute_error_sum = 0.0
    control_probability_max_absolute_error = 0.0
    control_prediction_pixels = 0
    control_prediction_disagreements = 0
    for mapping in (baseline, rgb_paths):
        require(set(mapping) == set(all_names), "Wrong parent universe")
        for path, digest in mapping.values():
            require(sha256(path) == digest, "Parent image changed")
    for arm in config["arm_names"]:
        require(set(predictions[arm]) == set(all_names), "Incomplete OOF arm")

    radius = int(config["training"]["foundation_inner_boundary_radius_pixels"])
    for fold, fold_result in zip(folds["folds"], run["folds"]):
        fold_index = int(fold["fold"])
        require(fold_index == fold_result["fold"], "Fold ordering changed")
        train, held, guard = (set(fold[key]) for key in ("train", "oof", "guard"))
        require(
            not (train & held or train & guard or held & guard)
            and train | held | guard == set(all_names),
            "Fold overlap",
        )
        distance = np.linalg.norm(
            np.asarray([centers[name] for name in train])[:, None]
            - np.asarray([centers[name] for name in held])[None],
            axis=-1,
        )
        require(float(distance.min()) >= folds["guard_policy"]["radius"] - 1e-9, "Guard violation")
        plan_path = args.run_dir / f"fold{fold_index}_sampling_plan.json"
        plan = read_json(plan_path)
        require(len(plan) == config["training"]["iterations"], "Training budget changed")
        rng = random.Random(config["seed"] + fold_index * 100003)
        names = sorted(train)
        crop = int(config["training"]["crop"])
        for actual in plan:
            expected = {
                "name": names[rng.randrange(len(names))],
                "top": rng.randint(0, 989 - crop),
                "left": rng.randint(0, 1320 - crop),
                "flip": rng.random() < 0.5,
            }
            require(actual == expected, "Paired augmentation sequence differs")

        initial_hashes = []
        for arm in config["arm_names"]:
            record = fold_result["arms"][arm]["training"]
            checkpoint = Path(record["path"])
            require(sha256(checkpoint) == record["sha256"], "Checkpoint changed")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            metadata = payload["metadata"]
            require(metadata == record["metadata"], "Checkpoint metadata changed")
            require(
                metadata["experiment_id"] == config["experiment_id"]
                and metadata["arm"] == arm
                and metadata["fold"] == fold_index,
                "Wrong checkpoint identity",
            )
            require(
                metadata["mode"] == "full"
                and metadata["iterations"] == 3000
                and metadata["seed"] == config["seed"],
                "Wrong training budget",
            )
            require(
                set(metadata["train"]) == train
                and set(metadata["outer_train"]) == train
                and set(metadata["oof"]) == held
                and set(metadata["guard"]) == guard,
                "Wrong fitting universe",
            )
            require(
                metadata["config_sha256"] == sha256(args.config)
                and metadata["oof_split_sha256"] == config["oof_split_sha256"],
                "Checkpoint protocol changed",
            )
            access = metadata["actual_label_access"]
            require(len(access) == len(train) and {row["name"] for row in access} == train, "Label access differs")
            require(set(metadata["actual_optimization_names"]) == train, "Training coverage differs")
            require(metadata["sampling_plan_sha256"] == sha256(plan_path), "Plan hash differs")
            boundary_total = 0
            for row in access:
                label_path = args.artifact_root / config["parents"]["labels"] / row["name"]
                require(
                    row["label_sha256"] == label_records[row["name"]]["label_png_sha256"] == sha256(label_path),
                    "Training label changed",
                )
                require(row["rgb_sha256"] == rgb_paths[row["name"]][1], "Training RGB changed")
                boundary_count = int(foundation_inner_boundary_mask(png(label_path), radius=radius).sum())
                require(boundary_count == row["foundation_boundary_pixels"], "Boundary count differs")
                boundary_total += boundary_count
            counts = np.sum(
                [[label_records[name]["pixel_counts"].get(str(class_id), 0) for class_id in range(5)] for name in train],
                axis=0,
            )
            require(counts.tolist() == metadata["class_counts"], "Class counts differ")
            weights = np.clip(np.median(counts.astype(np.float64)) / counts, 0.05, 20).astype(np.float32)
            require(weights.tolist() == metadata["class_weights"], "Class weights differ")
            require(metadata["boundary_radius_pixels"] == radius and boundary_total > 0, "Boundary protocol differs")
            require(
                metadata["boundary_multiplier"]
                == config["training"]["arm_boundary_multipliers"][arm],
                "Boundary multiplier differs",
            )
            for initial in metadata["initialization"]:
                require(sha256(args.artifact_root / initial["path"]) == initial["sha256"], "Initialization changed")
            initial_hashes.append(metadata["initial_state_sha256"])

            for name in sorted(held):
                row = predictions[arm][name]
                probability_path = Path(row["probability_path"])
                prediction_path = Path(row["path"])
                require(
                    row["fold"] == fold_index
                    and sha256(probability_path) == row["probability_sha256"]
                    and sha256(prediction_path) == row["sha256"],
                    "Prediction provenance changed",
                )
                probability = np.load(probability_path, allow_pickle=False)
                base = png(baseline[name][0])
                prediction = png(prediction_path)
                require(
                    probability.dtype == np.float32
                    and probability.shape == base.shape
                    and np.isfinite(probability).all()
                    and np.all((probability >= 0) & (probability <= 1)),
                    "Invalid probability map",
                )
                require(np.array_equal(fuse_foundation(base, probability), prediction), "Fusion mismatch")
                require(
                    all(np.array_equal(prediction == class_id, base == class_id) for class_id in (1, 2, 3)),
                    "Protected class changed",
                )
                if arm == "five_control":
                    previous = np.load(old_predictions["five_control"][name]["probability_path"], allow_pickle=False)
                    difference = np.abs(probability.astype(np.float64) - previous.astype(np.float64))
                    exact_control_probability_replays += int(np.array_equal(probability, previous))
                    control_probability_elements += int(difference.size)
                    control_probability_absolute_error_sum += float(difference.sum())
                    control_probability_max_absolute_error = max(
                        control_probability_max_absolute_error, float(difference.max())
                    )
                    old_prediction = png(old_predictions["five_control"][name]["path"])
                    control_prediction_pixels += int(prediction.size)
                    control_prediction_disagreements += int(np.count_nonzero(prediction != old_prediction))
                prediction_counter += 1

            model = initialize(args.artifact_root / config["parents"]["model_config"], config["seed"])
            require(state_hash(model) == metadata["initial_state_sha256"], "Initial state changed")
            changed_tensors = sum(
                not torch.equal(value, payload["model_state_dict"][key])
                for key, value in model.state_dict().items()
            )
            require(changed_tensors > 0, "Model did not train")
            require(
                sum(parameter.numel() for parameter in model.parameters())
                == metadata["parameters"]
                == metadata["trainable_parameters"],
                "Parameter budget differs",
            )
            model.load_state_dict(payload["model_state_dict"], strict=True)
            model.cuda().eval().requires_grad_(False)
            first = sorted(held)[0]
            rgb = png(rgb_paths[first][0])
            before = array_hash(rgb)
            probability = sliding(model, rgb, crop=960, stride=480, classes=5)[4]
            require(
                np.array_equal(probability, np.load(predictions[arm][first]["probability_path"], allow_pickle=False)),
                "Native probability replay differs",
            )
            require(array_hash(rgb) == before, "RGB input mutated")
            rows.append(
                {
                    "fold": fold_index,
                    "arm": arm,
                    "checkpoint_sha256": record["sha256"],
                    "native_replay_view": first,
                    "probability_replay_exact": True,
                    "changed_parameter_tensors": changed_tensors,
                    "weights": weights.tolist(),
                    "class_counts": counts.tolist(),
                    "foundation_boundary_pixels": boundary_total,
                }
            )
            del model, payload
            torch.cuda.empty_cache()
        require(initial_hashes[0] == initial_hashes[1], "Matched initial weights differ")

    require(len(rows) == 6 and prediction_counter == 438, "Missing models or predictions")
    freeze = verify_freeze(
        ROOT / "experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json",
        args.artifact_root,
    )
    require(freeze == run["freeze_before"] == run["freeze_after"], "Frozen manifest mismatch")
    result = {
        "experiment_id": config["experiment_id"],
        "status": "passed_with_cross_run_control_drift",
        "checkpoint_replays": rows,
        "prediction_files_verified": prediction_counter,
        "exact_sem270_control_probability_replays": exact_control_probability_replays,
        "sem270_control_cross_run_comparison": {
            "views": 219,
            "exact_probability_views": exact_control_probability_replays,
            "mean_absolute_probability_difference": control_probability_absolute_error_sum
            / control_probability_elements,
            "maximum_absolute_probability_difference": control_probability_max_absolute_error,
            "fused_prediction_disagreement_pixels": control_prediction_disagreements,
            "fused_prediction_disagreement_fraction": control_prediction_disagreements
            / control_prediction_pixels,
        },
        "paired_training_steps": 18000,
        "parent_images_verified": 438,
        "frozen_artifacts": 18,
        "target_labels_decoded_for_metrics": [],
        "result_sha256": sha256(args.run_dir / "result.json"),
        "code_sha256": sha256(Path(__file__)),
        "seconds": time.perf_counter() - started,
        "note": "Independent metadata, train-label boundary counts, exact current-checkpoint replay and unlabeled model replay; no new target metric evaluation. The separately trained SEM-270 control is compared diagnostically, not required to be bit-exact across GPU training runs.",
    }
    write_json(args.output_dir / "result.json", result)
    print(json.dumps({"status": result["status"], "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
