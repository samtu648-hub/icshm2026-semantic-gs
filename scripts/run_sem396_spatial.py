#!/usr/bin/env python3
"""Run the SEM378 Mask2Former recipe on spatial_confirmation_v1."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_seed_initialization import tensor_manifest, validate_same_seed_pair


EID = "SEM-396-RGB152-SPATIAL256"
CONFIG = ROOT / "experiments/runs" / EID / "config.yaml"
SEM370_CONFIG = ROOT / "experiments/runs/SEM-370-ALL300/config.yaml"
SEM378_TRAIN = ROOT / "outputs/quality/SEM-378/r1/mask2former"
CODE_PATHS = (
    "scripts/run_sem396_spatial.py",
    "scripts/run_quality_all300.py",
    "src/quality_mask2former.py",
    "src/quality_mask2former_training.py",
    "src/quality_static_training.py",
    "src/quality_semantic_audit.py",
    "src/quality_seed_initialization.py",
)


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sampling_plan(
    names: list[str], seed: int, steps: int = 6000, crop: int = 768
) -> list[dict[str, Any]]:
    if not names or len(names) != len(set(names)) or names != sorted(names):
        raise ValueError(
            "Sampling names must be a sorted, unique, non-empty training role"
        )
    rng = random.Random(seed)
    plan = []
    for step in range(1, steps + 1):
        name = names[rng.randrange(len(names))]
        plan.append(
            {
                "step": step,
                "name": name,
                "top": rng.randint(0, 989 - crop),
                "left": rng.randint(0, 1320 - crop),
                "flip": rng.random() < 0.5,
            }
        )
    return plan


def setup_runtime(artifact_root: Path):
    source = ROOT / "outputs/quality/rgb140/source-r1"
    runtime = ROOT / "outputs/quality/mask2former/SEM-340"
    sys.path[:0] = [
        str(source),
        str(source / "submodules/diff-gaussian-rasterization"),
        str(source / "submodules/simple-knn"),
        str(runtime / "runtime"),
        str(runtime / "vendor/detectron2"),
        str(runtime / "vendor/Mask2Former"),
        str(runtime / "vendor/Mask2Former/mask2former/modeling/pixel_decoder/ops"),
    ]
    cuda = artifact_root / "r1/toolchain/cuda-12.8"
    os.environ.update(
        CUDA_HOME=str(cuda),
        PATH=str(cuda / "bin") + os.pathsep + os.environ.get("PATH", ""),
        LD_LIBRARY_PATH=str(cuda / "lib64")
        + os.pathsep
        + os.environ.get("LD_LIBRARY_PATH", ""),
        TORCH_HOME=str(artifact_root / "r1/weights/torch"),
        WANDB_MODE="disabled",
        HF_HUB_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    import numpy as np
    import torch  # type: ignore[import-not-found]

    return runtime, torch, np


def registered_paths(config: dict[str, Any]) -> tuple[Path, Path, Path, Path, Path]:
    split = ROOT / config["split_path"]
    rgb_run = ROOT / config["rgb_run"]
    rgb_evaluation = ROOT / config["rgb_evaluation"]
    labels = ROOT / config["label_root"]
    batch = ROOT / config["output_path"]
    return split, rgb_run, rgb_evaluation, labels, batch


def training_recipe(config: dict[str, Any], train_names: list[str]) -> dict[str, Any]:
    recipe = read(SEM370_CONFIG)
    recipe["seed"] = config["seed"]
    recipe["diagnostic_views"] = [
        name for name in recipe["diagnostic_views"] if name in set(train_names)
    ]
    return recipe


def prepare(config: dict[str, Any]) -> None:
    split_path, rgb_run, rgb_evaluation, label_root, batch = registered_paths(config)
    if batch.exists():
        raise FileExistsError(f"SEM396 output already exists: {batch}")
    rgb_parent = read(rgb_run / "result.json")
    rgb_result = read(rgb_evaluation / "result.json")
    if (
        rgb_parent["status"] != "completed_spatial_training"
        or rgb_result["status"] != "completed"
    ):
        raise ValueError("Sealed RGB152 training and evaluation are required")
    verify_hashes(rgb_run, rgb_parent["artifacts"])
    split = read(split_path)
    fold = split["primary_fold"]
    train_names = sorted(fold["semantic_train_labeled"])
    holdout_names = sorted(fold["semantic_validation_labeled"])
    if (
        len(train_names) != config["semantic_train_views"]
        or len(holdout_names) != config["semantic_confirmation_views"]
        or set(train_names) & set(holdout_names)
    ):
        raise ValueError("Expected the registered disjoint semantic spatial roles")
    manifest_path = ROOT / config["label_manifest"]
    manifest = read(manifest_path)
    records = {record["image"]: record for record in manifest["records"]}
    label_hashes = {name: sha256(label_root / name) for name in train_names}
    if any(
        label_hashes[name] != records[name]["label_png_sha256"] for name in train_names
    ):
        raise ValueError("A semantic training label differs from the frozen manifest")

    batch.mkdir(parents=True)
    export = batch / "export"
    images = export / "images"
    images.mkdir(parents=True)
    rgb_train_names = sorted(fold["train_rgb_400"])
    training_rgb_hashes = {
        name: rgb_result["image_hashes"][name] for name in rgb_train_names
    }
    verify_hashes(rgb_evaluation / "images", training_rgb_hashes)
    for name in rgb_train_names:
        shutil.copyfile(rgb_evaluation / "images" / name, images / name)
    verify_hashes(images, training_rgb_hashes)
    export_result = {
        "schema_version": 1,
        "status": "completed",
        "experiment_id": EID,
        "rgb_experiment_id": rgb_parent["experiment_id"],
        "split_id": config["split_id"],
        "split_sha256": sha256(split_path),
        "source_rgb_result_sha256": sha256(rgb_run / "result.json"),
        "source_rgb_evaluation_sha256": sha256(rgb_evaluation / "result.json"),
        "image_hashes": training_rgb_hashes,
        "image_count": len(training_rgb_hashes),
        "semantic_train": train_names,
        "semantic_holdout": holdout_names,
        "holdout_rgb_accessed_during_prepare": False,
        "holdout_labels_accessed_during_prepare": False,
    }
    write_new_json(export / "result.json", export_result)

    plan = sampling_plan(
        train_names, config["seed"], config["iterations"], config["crop"]
    )
    counts = Counter(str(row["name"]) for row in plan)
    plan_path = batch / "sampling-plan.json"
    write_new_json(
        plan_path,
        {
            "schema_version": 1,
            "experiment_id": EID,
            "seed": config["seed"],
            "steps": config["iterations"],
            "crop": config["crop"],
            "train_names": train_names,
            "holdout_names_excluded": holdout_names,
            "samples": plan,
            "per_view_counts": dict(sorted(counts.items())),
            "minimum_samples_per_view": min(counts.values()),
            "maximum_samples_per_view": max(counts.values()),
            "all_training_views_sampled": set(counts) == set(train_names),
        },
    )
    write_new_json(
        batch / "prepare-result.json",
        {
            "schema_version": 1,
            "status": "completed",
            "experiment_id": EID,
            "config_sha256": sha256(CONFIG),
            "split_sha256": sha256(split_path),
            "export_result_sha256": sha256(export / "result.json"),
            "sampling_plan_sha256": sha256(plan_path),
            "label_manifest_sha256": sha256(manifest_path),
            "training_label_hashes": label_hashes,
            "semantic_train_count": len(train_names),
            "semantic_holdout_count": len(holdout_names),
            "holdout_label_hashes_read": 0,
            "code_sha256": {path: sha256(ROOT / path) for path in CODE_PATHS},
        },
    )


def train(config: dict[str, Any], artifact_root: Path) -> None:
    split_path, rgb_run, rgb_evaluation, label_root, batch = registered_paths(config)
    output = batch / "mask2former"
    output.mkdir(exist_ok=False)
    split = read(split_path)
    train_names = sorted(split["primary_fold"]["semantic_train_labeled"])
    holdout_names = set(split["primary_fold"]["semantic_validation_labeled"])
    export = read(batch / "export/result.json")
    verify_hashes(batch / "export/images", export["image_hashes"])
    prepare_result = read(batch / "prepare-result.json")
    verify_hashes(label_root, prepare_result["training_label_hashes"])
    manifest = read(ROOT / config["label_manifest"])
    plan_record = read(batch / "sampling-plan.json")
    if sha256(batch / "sampling-plan.json") != prepare_result["sampling_plan_sha256"]:
        raise ValueError("Frozen 6000-step sampling plan changed")
    native_plan = sampling_plan(
        train_names, config["seed"], config["iterations"], config["crop"]
    )
    paired_native_trace = [
        {key: row[key] for key in ("step", "name", "top", "left", "flip")}
        for row in plan_record["samples"]
    ]
    if native_plan != paired_native_trace:
        raise ValueError("Frozen plan does not match the registered training seed")

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "starting",
        "experiment_id": EID,
        "stage": "mask2former",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": sha256(CONFIG),
        "split_sha256": sha256(split_path),
        "seed": config["seed"],
        "trained_steps": 0,
        "access": {
            "train": train_names,
            "evaluation": sorted(holdout_names),
            "labels_allowed_during_fit": train_names,
            "holdout_rgb_allowed_during_fit": False,
            "holdout_labels_allowed_during_fit": False,
        },
        "binding": {
            "config_sha256": sha256(CONFIG),
            "split_sha256": sha256(split_path),
            "label_manifest_sha256": sha256(ROOT / config["label_manifest"]),
            "export_result_sha256": sha256(batch / "export/result.json"),
            "sampling_plan_sha256": sha256(batch / "sampling-plan.json"),
            "rgb_result_sha256": sha256(rgb_run / "result.json"),
            "rgb_evaluation_sha256": sha256(rgb_evaluation / "result.json"),
            "seed": config["seed"],
        },
        "code_sha256": {path: sha256(ROOT / path) for path in CODE_PATHS},
    }
    started = time.monotonic()
    try:
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            queued = time.monotonic()
            fcntl.flock(lock, fcntl.LOCK_EX)
            result["queue_seconds"] = time.monotonic() - queued
            applications = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            result["gpu_processes_before"] = applications
            if any(
                line.strip() and "ToDesk" not in line
                for line in applications.splitlines()
            ):
                raise RuntimeError(
                    "Other GPU compute process active outside the shared lock"
                )
            _, torch, np = setup_runtime(artifact_root)
            from scripts.run_quality_all300 import mask2former
            from src.quality_static_training import TrainingLabels
            import src.quality_mask2former as model_api

            torch.set_num_threads(4)
            random.seed(config["seed"])
            np.random.seed(config["seed"])
            torch.manual_seed(config["seed"])
            torch.cuda.manual_seed_all(config["seed"])
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats()
            labels = TrainingLabels(label_root, train_names, manifest, [1320, 989])
            original_loader = model_api.load_official_model

            def capture_initial(*loader_args, **loader_kwargs):
                model, loaded = original_loader(*loader_args, **loader_kwargs)
                if "initial_model_state_sha256" not in result:
                    digest = hashlib.sha256()
                    for key, value in sorted(model.state_dict().items()):
                        digest.update(key.encode())
                        digest.update(str(value.dtype).encode())
                        digest.update(str(tuple(value.shape)).encode())
                        digest.update(
                            value.detach().cpu().contiguous().numpy().tobytes()
                        )
                    result["initial_model_state_sha256"] = digest.hexdigest()
                    initialization = config.get("seed_aware_initialization")
                    if initialization:
                        current_manifest = tensor_manifest(model.state_dict())
                        reference_path = (
                            ROOT / initialization["training_initialization_reference"]
                        )
                        reference = read(reference_path)
                        reference_manifest = reference.get(
                            "current_seed_state_manifest",
                            reference.get("state_manifest"),
                        )
                        reference_load = reference.get(
                            "current_seed_pretrained_load",
                            reference.get("pretrained_load"),
                        )
                        if reference_manifest is None or reference_load is None:
                            raise ValueError(
                                "Initialization reference lacks state or load evidence"
                            )
                        paired = validate_same_seed_pair(
                            reference_manifest,
                            current_manifest,
                            reference_load,
                            loaded,
                        )
                        write_new_json(
                            output / "initialization-contract.json",
                            {
                                "schema_version": 1,
                                "experiment_id": EID,
                                "seed": config["seed"],
                                "state_sha256": result["initial_model_state_sha256"],
                                "state_manifest": current_manifest,
                                "pretrained_load": loaded,
                                "reference_path": str(reference_path.relative_to(ROOT)),
                                "reference_sha256": sha256(reference_path),
                                "same_seed_validation": paired,
                            },
                        )
                        result["seed_aware_initialization_validated"] = True
                return model, loaded

            model_api.load_official_model = capture_initial
            write_new_json(output / "started.json", result)
            try:
                recipe = training_recipe(config, train_names)
                if recipe["seed"] != config["seed"]:
                    raise ValueError(
                        "Training recipe did not retain the registered seed"
                    )
                mask2former(
                    SimpleNamespace(batch=batch),
                    output,
                    result,
                    labels,
                    recipe,
                )
            finally:
                model_api.load_official_model = original_loader

            actual_samples = [
                json.loads(line)["sample"]
                for line in (output / "training.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            trace_keys = tuple(
                config.get("sampling_trace_keys", ("name", "top", "left", "flip"))
            )
            expected_samples = [
                {key: row[key] for key in trace_keys} for row in plan_record["samples"]
            ]
            if (
                actual_samples != expected_samples
                or len(actual_samples) != config["iterations"]
            ):
                raise ValueError(
                    "Training trace differs from the frozen 6000-step sampling plan"
                )
            sem378 = read(SEM378_TRAIN / "result.json")
            if config.get("seed_aware_initialization"):
                if not result.get("seed_aware_initialization_validated"):
                    raise ValueError(
                        "Seed-aware training initialization was not validated"
                    )
            elif (
                result["initial_model_state_sha256"]
                != sem378["initial_model_state_sha256"]
            ):
                raise ValueError("Fresh ADE pretrained state differs from SEM378")
            if (output / "resolved_model.yaml").read_bytes() != (
                SEM378_TRAIN / "resolved_model.yaml"
            ).read_bytes():
                raise ValueError("Resolved Mask2Former recipe differs from SEM378")
            if labels.accessed != set(train_names) or labels.accessed & holdout_names:
                raise ValueError(
                    "Semantic label access did not remain inside the spatial training role"
                )
            verify_hashes(batch / "export/images", export["image_hashes"])
            result.update(
                status="completed_sealed_6000",
                sampling_plan_exact=True,
                sampling_plan_sha256=sha256(batch / "sampling-plan.json"),
                **(
                    {"seed_aware_initialization_validated": True}
                    if config.get("seed_aware_initialization")
                    else {"initial_state_matches_sem378": True}
                ),
                resolved_recipe_matches_sem378=True,
                training_labels_accessed=sorted(labels.accessed),
                holdout_labels_accessed_during_fit=[],
                frozen_rgb_hashes_exact=True,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                artifacts={
                    str(path.relative_to(output)): sha256(path)
                    for path in output.rglob("*")
                    if path.is_file()
                    and path.name not in {"started.json", "result.json"}
                },
            )
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(output / "result.json", result)


def boundary_metrics(values) -> dict[str, Any]:
    precision = float(values[0] / values[1]) if values[1] else None
    recall = float(values[2] / values[3]) if values[3] else None
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "tolerance_px": 2,
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "counts": values.tolist(),
    }


def inference(config: dict[str, Any], artifact_root: Path) -> None:
    split_path, _, rgb_evaluation, label_root, batch = registered_paths(config)
    output = batch / "inference"
    output.mkdir(exist_ok=False)
    predictions = output / "predictions"
    predictions.mkdir()
    split = read(split_path)
    fold = split["primary_fold"]
    train_names = set(fold["semantic_train_labeled"])
    holdout_names = set(fold["semantic_validation_labeled"])
    group_by_name = {
        name: group["group_id"] for group in split["groups"] for name in group["images"]
    }
    export = read(batch / "export/result.json")
    rgb_result = read(rgb_evaluation / "result.json")
    training = read(batch / "mask2former/result.json")
    if (
        training["status"] != "completed_sealed_6000"
        or training["trained_steps"] != 6000
    ):
        raise ValueError("A sealed 6000-step SEM396 checkpoint is required")
    verify_hashes(batch / "mask2former", training["artifacts"])
    verify_hashes(batch / "export/images", export["image_hashes"])
    verify_hashes(rgb_evaluation / "images", rgb_result["image_hashes"])
    checkpoint = batch / "mask2former/model-6000.pt"
    checkpoint_sha256 = sha256(checkpoint)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "starting",
        "experiment_id": EID,
        "checkpoint_sha256_before_holdout_access": checkpoint_sha256,
        "holdout_access_begins_after_checkpoint_sealed": True,
        "config_sha256": sha256(CONFIG),
        "split_sha256": sha256(split_path),
        "code_sha256": {path: sha256(ROOT / path) for path in CODE_PATHS},
    }
    started = time.monotonic()
    try:
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            queued = time.monotonic()
            fcntl.flock(lock, fcntl.LOCK_EX)
            result["queue_seconds"] = time.monotonic() - queued
            applications = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            if any(
                line.strip() and "ToDesk" not in line
                for line in applications.splitlines()
            ):
                raise RuntimeError(
                    "Other GPU compute process active outside the shared lock"
                )
            runtime, torch, np = setup_runtime(artifact_root)
            from PIL import Image
            from scipy.ndimage import binary_erosion, distance_transform_edt
            from src.quality_mask2former import (
                config_for_model,
                load_official_model,
                native_predict,
            )
            from src.quality_semantic_audit import independent_confusion, metrics
            from src.semantic_cable_binary import read_rgb

            model_config = config_for_model(
                runtime / "vendor/Mask2Former",
                "configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml",
                False,
            )
            model, _ = load_official_model(
                model_config, runtime / "weights/model_final_6b4a3a.pkl"
            )
            model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
            )
            model.cuda()
            model.eval()
            torch.cuda.reset_peak_memory_stats()
            totals = {
                "train": np.zeros((5, 5), dtype=np.int64),
                "holdout": np.zeros((5, 5), dtype=np.int64),
                **{
                    group["group_id"]: np.zeros((5, 5), dtype=np.int64)
                    for group in split["groups"]
                },
            }
            boundary = {key: np.zeros(4, dtype=np.int64) for key in totals}
            per_view = []
            inference_seconds = []
            for index, name in enumerate(sorted(rgb_result["image_hashes"])):
                rgb = read_rgb(rgb_evaluation / "images" / name)
                torch.cuda.synchronize()
                infer_started = time.monotonic()
                probability = native_predict(model, rgb)
                torch.cuda.synchronize()
                inference_seconds.append(time.monotonic() - infer_started)
                prediction = probability.argmax(0).astype(np.uint8)
                Image.fromarray(prediction).save(predictions / name)
                if name in train_names or name in holdout_names:
                    target = np.asarray(Image.open(label_root / name), dtype=np.uint8)
                    confusion = independent_confusion(target, prediction)
                    role = "train" if name in train_names else "holdout"
                    keys = [role]
                    if role == "holdout":
                        keys.append(group_by_name[name])
                    cable = target == 2
                    gt_boundary = cable ^ binary_erosion(cable)
                    gt_distance = np.asarray(distance_transform_edt(~gt_boundary))
                    predicted_cable = prediction == 2
                    predicted_boundary = predicted_cable ^ binary_erosion(
                        predicted_cable
                    )
                    predicted_distance = np.asarray(
                        distance_transform_edt(~predicted_boundary)
                    )
                    counts = np.asarray(
                        [
                            np.sum(predicted_boundary & (gt_distance <= 2)),
                            np.sum(predicted_boundary),
                            np.sum(gt_boundary & (predicted_distance <= 2)),
                            np.sum(gt_boundary),
                        ],
                        dtype=np.int64,
                    )
                    for key in keys:
                        totals[key] += confusion
                        boundary[key] += counts
                    per_view.append(
                        {
                            "name": name,
                            "role": role,
                            "group": group_by_name.get(name),
                            **metrics(confusion),
                        }
                    )
                if index % 25 == 0:
                    print(json.dumps({"inferred": index + 1, "total": len(rgb_result["image_hashes"])}), flush=True)
            if len(per_view) != config["semantic_train_views"] + config["semantic_confirmation_views"]:
                raise ValueError(
                    "Expected metrics for exactly the registered semantic train and confirmation views"
                )
            if sha256(checkpoint) != checkpoint_sha256:
                raise ValueError("Sealed checkpoint changed during inference")
            verify_hashes(batch / "export/images", export["image_hashes"])
            verify_hashes(rgb_evaluation / "images", rgb_result["image_hashes"])
            result.update(
                status="completed",
                checkpoint_sha256_after_holdout_access=sha256(checkpoint),
                prediction_hashes={
                    name: sha256(predictions / name)
                    for name in sorted(rgb_result["image_hashes"])
                },
                inference_seconds=inference_seconds,
                inference_median_seconds=float(np.median(inference_seconds[1:])),
                metrics={key: metrics(matrix) for key, matrix in totals.items()},
                boundary_metrics={
                    key: boundary_metrics(values) for key, values in boundary.items()
                },
                per_view=per_view,
                train_labeled_count=len(train_names),
                holdout_labeled_count=len(holdout_names),
                unlabeled_count=len(rgb_result["image_hashes"]) - len(per_view),
                stay_cable_train_fp=int(
                    metrics(totals["train"])["per_class"]["stay_cable"]["fp"]
                ),
                stay_cable_train_fn=int(
                    metrics(totals["train"])["per_class"]["stay_cable"]["fn"]
                ),
                stay_cable_holdout_fp=int(
                    metrics(totals["holdout"])["per_class"]["stay_cable"]["fp"]
                ),
                stay_cable_holdout_fn=int(
                    metrics(totals["holdout"])["per_class"]["stay_cable"]["fn"]
                ),
                frozen_rgb_hashes_exact=True,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            )
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(output / "result.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("prepare", "train", "inference"), required=True
    )
    args = parser.parse_args()
    config = read(CONFIG)
    if config["experiment_id"] != EID:
        raise ValueError("SEM396 config identity changed")
    if args.stage == "prepare":
        prepare(config)
    elif args.stage == "train":
        train(config, args.artifact_root.resolve())
    else:
        inference(config, args.artifact_root.resolve())


if __name__ == "__main__":
    main()
