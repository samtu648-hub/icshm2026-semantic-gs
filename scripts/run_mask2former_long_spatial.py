#!/usr/bin/env python3
"""Run matched 6k + fixed-low-LR Mask2Former spatial development curves."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_sem396_spatial as spatial_base

from src.quality_protocol import sha256, verify_hashes, write_new_json


SUPPORTED = {
    "SEM-434-RGB152-MASK2FORMER-LONG12K": {
        "historical": "outputs/quality/SEM-396/r1",
        "expected_split": "spatial_confirmation_v1",
    },
    "SEM-435-RGB154-MASK2FORMER-LONG12K": {
        "historical": "outputs/quality/SEM-412/r1",
        "expected_split": "spatial_confirmation_v2",
    },
}
CODE_PATHS = (
    "scripts/run_mask2former_long_spatial.py",
    "scripts/run_sem396_spatial.py",
    "scripts/run_quality_all300.py",
    "src/quality_mask2former.py",
    "src/quality_mask2former_training.py",
    "src/quality_static_training.py",
    "src/quality_semantic_audit.py",
)
ENDPOINTS = (6000, 8000, 10000, 12000)


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def configure_base(experiment_id: str, config_path: Path) -> None:
    spatial_base.EID = experiment_id
    spatial_base.CONFIG = config_path
    spatial_base.CODE_PATHS = CODE_PATHS


def validate_config(config: dict[str, Any]) -> None:
    experiment_id = config["experiment_id"]
    if experiment_id not in SUPPORTED:
        raise ValueError(f"Unsupported long-training experiment: {experiment_id}")
    expected = SUPPORTED[experiment_id]
    if config["split_id"] != expected["expected_split"]:
        raise ValueError("Experiment ID and split do not match")
    if (
        config["iterations"] != 12000
        or config["base_phase_steps"] != 6000
        or config["extension_lr_multiplier"] != 0.1
        or tuple(config["endpoints"]) != ENDPOINTS
        or config["crop"] != 768
        or config["seed"] != 20260805
    ):
        raise ValueError("Long-training schedule differs from the preregistered recipe")
    historical = ROOT / expected["historical"]
    if Path(config["historical_control_run"]) != Path(expected["historical"]):
        raise ValueError("Historical control binding changed")
    if not (historical / "mask2former/model-6000.pt").is_file():
        raise FileNotFoundError("Historical step-6000 control checkpoint is missing")


def phase_lr_factor(step: int, extension_multiplier: float) -> tuple[str, float]:
    from src.quality_mask2former_training import poly_factor

    if not 1 <= step <= 12000:
        raise ValueError("Step outside fixed 12k trajectory")
    if step <= 6000:
        return "base_poly", poly_factor(step - 1, total=6000, power=0.9)
    return "fixed_low_lr", extension_multiplier


def model_state_sha256(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def compare_model_states(model, checkpoint: Path) -> dict[str, Any]:
    import torch

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    current = model.state_dict()
    if set(saved) != set(current):
        raise ValueError("Historical and replayed model keys differ")
    mismatches = []
    for key in sorted(saved):
        if not torch.equal(saved[key], current[key].detach().cpu()):
            mismatches.append(key)
            if len(mismatches) >= 8:
                break
    result = {
        "historical_checkpoint": str(checkpoint.relative_to(ROOT)),
        "historical_checkpoint_sha256": sha256(checkpoint),
        "historical_model_state_sha256": model_state_sha256(saved),
        "current_model_state_sha256": model_state_sha256(current),
        "tensor_count": len(saved),
        "exact": not mismatches,
        "first_mismatches": mismatches,
    }
    del saved
    return result


def save_model_checkpoint(model, binding, access, step: int, path: Path) -> dict[str, Any]:
    import torch

    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    state = model.state_dict()
    copied = OrderedDict((key, value.detach().cpu()) for key, value in state.items())
    copied._metadata = state._metadata
    if any(not torch.isfinite(value).all() for value in copied.values()):
        raise FloatingPointError("Nonfinite model checkpoint state")
    with path.open("xb") as stream:
        torch.save(
            {"model": copied, "binding": binding, "access": access, "steps": step},
            stream,
        )
    if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(
        cuda_rng, torch.cuda.get_rng_state()
    ):
        raise ValueError("Model checkpoint serialization changed torch RNG")
    return {
        "steps": step,
        "sha256": sha256(path),
        "model_state_sha256": model_state_sha256(copied),
        "rng_preserved": True,
    }


def save_resume_checkpoint(
    model,
    optimizer,
    scaler,
    sampler_rng,
    binding,
    access,
    step: int,
    path: Path,
) -> dict[str, Any]:
    import torch

    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    sampler_state = sampler_rng.getstate()
    with path.open("xb") as stream:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "torch_cpu_rng": cpu_rng,
                "torch_cuda_rng": cuda_rng,
                "sampler_rng": sampler_state,
                "binding": binding,
                "access": access,
                "steps": step,
                "next_step": step + 1,
            },
            stream,
        )
    if (
        not torch.equal(cpu_rng, torch.get_rng_state())
        or not torch.equal(cuda_rng, torch.cuda.get_rng_state())
        or sampler_state != sampler_rng.getstate()
    ):
        raise ValueError("Resume checkpoint serialization changed training state")
    return {"steps": step, "sha256": sha256(path), "rng_preserved": True}


def prepare(config: dict[str, Any], config_path: Path) -> None:
    configure_base(config["experiment_id"], config_path)
    spatial_base.prepare(config)
    batch = ROOT / config["output_path"]
    plan = read(batch / "sampling-plan.json")
    historical_plan = read(
        ROOT / config["historical_control_run"] / "sampling-plan.json"
    )
    if plan["samples"][:6000] != historical_plan["samples"]:
        raise ValueError("The new 12k trajectory does not exactly prefix the historical 6k plan")
    result_path = batch / "prepare-result.json"
    result = read(result_path)
    result.update(
        historical_checkpoint_recovery={
            "model": True,
            "optimizer": False,
            "grad_scaler": False,
            "torch_rng": False,
            "sampler_rng": False,
            "direct_resume_reliable": False,
            "decision": "fresh_matched_12k_trajectory",
        },
        first_6000_sampling_exact=True,
        historical_sampling_plan_sha256=sha256(
            ROOT / config["historical_control_run"] / "sampling-plan.json"
        ),
        extension_schedule={
            "steps": [6001, 12000], "type": "fixed_group_lr",
            "multiplier_of_initial_group_lr": config["extension_lr_multiplier"],
        },
    )
    result_path.unlink()
    write_new_json(result_path, result)


def train(config: dict[str, Any], config_path: Path, artifact_root: Path) -> None:
    from src.quality_static_training import TrainingLabels

    split_path, rgb_run, rgb_evaluation, label_root, batch = spatial_base.registered_paths(config)
    output = batch / "mask2former"
    output.mkdir(exist_ok=False)
    split = read(split_path)
    train_names = sorted(split["primary_fold"]["semantic_train_labeled"])
    holdout_names = set(split["primary_fold"]["semantic_validation_labeled"])
    export = read(batch / "export/result.json")
    prepare_result = read(batch / "prepare-result.json")
    plan_record = read(batch / "sampling-plan.json")
    verify_hashes(batch / "export/images", export["image_hashes"])
    verify_hashes(label_root, prepare_result["training_label_hashes"])
    if sha256(batch / "sampling-plan.json") != prepare_result["sampling_plan_sha256"]:
        raise ValueError("Frozen 12k sampling plan changed")
    native_plan = spatial_base.sampling_plan(
        train_names, config["seed"], config["iterations"], config["crop"]
    )
    expected_trace = [
        {key: row[key] for key in ("name", "top", "left", "flip")}
        for row in native_plan
    ]
    registered_trace = [
        {key: row[key] for key in ("name", "top", "left", "flip")}
        for row in plan_record["samples"]
    ]
    if expected_trace != registered_trace:
        raise ValueError("Frozen plan differs from native deterministic sampler")

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "starting",
        "experiment_id": config["experiment_id"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": sha256(config_path),
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
            "config_sha256": sha256(config_path),
            "split_sha256": sha256(split_path),
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
            if any(line.strip() and "ToDesk" not in line for line in applications.splitlines()):
                raise RuntimeError("Other GPU compute process active outside the shared lock")

            runtime, torch, np = spatial_base.setup_runtime(artifact_root)
            from detectron2.structures import Instances
            from src.quality_mask2former import (
                author_optimizer,
                class_masks,
                config_for_model,
                load_official_model,
            )
            from src.quality_mask2former_training import amp_update, sampled_crop
            from src.semantic_cable_binary import read_rgb

            torch.set_num_threads(4)
            random.seed(config["seed"])
            np.random.seed(config["seed"])
            torch.manual_seed(config["seed"])
            torch.cuda.manual_seed_all(config["seed"])
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats()

            manifest = read(ROOT / config["label_manifest"])
            labels = TrainingLabels(label_root, train_names, manifest, [1320, 989])
            model_config = config_for_model(
                runtime / "vendor/Mask2Former",
                "configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml",
                False,
            )
            model_config.defrost()
            model_config.DATASETS.TRAIN = ("sem341_rendered_training",)
            model_config.SOLVER.MAX_ITER = 6000
            model_config.SOLVER.POLY_LR_POWER = 0.9
            model_config.freeze()
            (output / "resolved_model.yaml").write_text(model_config.dump())
            historical_resolved = (
                ROOT / config["historical_control_run"] / "mask2former/resolved_model.yaml"
            )
            if (output / "resolved_model.yaml").read_bytes() != historical_resolved.read_bytes():
                raise ValueError("Resolved model recipe differs from historical control")
            model, loaded = load_official_model(
                model_config, runtime / "weights/model_final_6b4a3a.pkl"
            )
            result["pretrained_load"] = loaded
            result["initial_model_state_sha256"] = model_state_sha256(model.state_dict())
            historical_result = read(
                ROOT / config["historical_control_run"] / "mask2former/result.json"
            )
            if result["initial_model_state_sha256"] != historical_result["initial_model_state_sha256"]:
                raise ValueError("Fresh ADE initialization differs from historical control")
            model.cuda()
            optimizer = author_optimizer(model, model_config)
            initial_rates = [group["lr"] for group in optimizer.param_groups]
            if sorted(set(initial_rates)) != [1e-05, 0.0001]:
                raise ValueError("Unexpected author optimizer learning-rate groups")
            scaler = torch.amp.GradScaler("cuda", init_scale=1024)
            images = {name: read_rgb(batch / "export/images" / name) for name in train_names}
            targets = {name: labels.read(name) for name in train_names}
            sampler_rng = random.Random(config["seed"])
            model.train()
            training_started = time.monotonic()
            result["amp_skips"] = 0
            result["checkpoints"] = {}
            with (output / "training.jsonl").open("x") as journal:
                for step in range(1, config["iterations"] + 1):
                    image_np, label_np, trace = sampled_crop(
                        sampler_rng, train_names, images, targets, config["crop"]
                    )
                    ids, masks = class_masks(label_np)
                    instances = Instances(tuple(label_np.shape))
                    instances.gt_classes = torch.from_numpy(ids)
                    instances.gt_masks = torch.from_numpy(masks)
                    image = torch.from_numpy(
                        image_np.transpose(2, 0, 1).copy()
                    ).float()
                    phase, factor = phase_lr_factor(
                        step, config["extension_lr_multiplier"]
                    )
                    for group, rate in zip(optimizer.param_groups, initial_rates):
                        group["lr"] = rate * factor
                    row = amp_update(
                        model, optimizer, scaler, image, instances, max_attempts=12, clip=0.01
                    )
                    row.update(
                        step=step,
                        sample=trace,
                        schedule_phase=phase,
                        lr_factor=factor,
                        minimum_group_lr=min(group["lr"] for group in optimizer.param_groups),
                        maximum_group_lr=max(group["lr"] for group in optimizer.param_groups),
                        elapsed_seconds=time.monotonic() - training_started,
                    )
                    journal.write(json.dumps(row) + "\n")
                    result["trained_steps"] = step
                    result["amp_skips"] += len(row["skips"])
                    if step % 100 == 0 or step == 1:
                        journal.flush()
                        print(
                            json.dumps(
                                {
                                    "experiment_id": config["experiment_id"],
                                    "step": step,
                                    "phase": phase,
                                    "amp_skips": result["amp_skips"],
                                    "seconds": time.monotonic() - training_started,
                                }
                            ),
                            flush=True,
                        )
                    if step in ENDPOINTS:
                        model_record = save_model_checkpoint(
                            model,
                            result["binding"],
                            result["access"],
                            step,
                            output / f"model-{step}.pt",
                        )
                        result["checkpoints"][str(step)] = {"model": model_record}
                        if step == 6000:
                            result["historical_6000_reproduction"] = compare_model_states(
                                model,
                                ROOT
                                / config["historical_control_run"]
                                / "mask2former/model-6000.pt",
                            )
                        if step in (6000, 12000):
                            result["checkpoints"][str(step)]["resume"] = save_resume_checkpoint(
                                model,
                                optimizer,
                                scaler,
                                sampler_rng,
                                result["binding"],
                                result["access"],
                                step,
                                output / f"resume-{step}.pt",
                            )

            result["training_seconds"] = time.monotonic() - training_started
            actual_samples = [
                json.loads(line)["sample"]
                for line in (output / "training.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            if actual_samples != expected_trace:
                raise ValueError("Actual 12k training trace differs from frozen plan")
            historical_plan = read(
                ROOT / config["historical_control_run"] / "sampling-plan.json"
            )
            if actual_samples[:6000] != [
                {key: row[key] for key in ("name", "top", "left", "flip")}
                for row in historical_plan["samples"]
            ]:
                raise ValueError("First 6000 actual samples differ from historical control")
            if labels.accessed != set(train_names) or labels.accessed & holdout_names:
                raise ValueError("Semantic labels escaped the training role")
            verify_hashes(batch / "export/images", export["image_hashes"])
            historical_exact = result["historical_6000_reproduction"]["exact"]
            result.update(
                status="completed_sealed_12000",
                sampling_plan_exact=True,
                first_6000_sampling_exact=True,
                historical_6000_model_state_exact=historical_exact,
                historical_6000_role=(
                    "exact_reproduction"
                    if historical_exact
                    else "reference_only_new_same_trajectory_6000_is_control"
                ),
                training_labels_accessed=sorted(labels.accessed),
                holdout_labels_accessed_during_fit=[],
                frozen_rgb_hashes_exact=True,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                artifacts={
                    str(path.relative_to(output)): sha256(path)
                    for path in output.rglob("*")
                    if path.is_file() and path.name not in {"started.json", "result.json"}
                },
            )
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(output / "result.json", result)


def inference(config: dict[str, Any], config_path: Path, artifact_root: Path) -> None:
    split_path, _, rgb_evaluation, label_root, batch = spatial_base.registered_paths(config)
    output = batch / "inference"
    output.mkdir(exist_ok=False)
    predictions_root = output / "predictions"
    predictions_root.mkdir()
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
    if training["status"] != "completed_sealed_12000" or training["trained_steps"] != 12000:
        raise ValueError("A sealed matched 12k trajectory is required")
    verify_hashes(batch / "mask2former", training["artifacts"])
    checkpoint_hashes = {
        str(step): sha256(batch / f"mask2former/model-{step}.pt") for step in ENDPOINTS
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "starting",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256(config_path),
        "split_sha256": sha256(split_path),
        "checkpoint_sha256_before_holdout_access": checkpoint_hashes,
        "holdout_access_begins_after_all_endpoints_sealed": True,
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
            if any(line.strip() and "ToDesk" not in line for line in applications.splitlines()):
                raise RuntimeError("Other GPU compute process active outside the shared lock")
            runtime, torch, np = spatial_base.setup_runtime(artifact_root)
            from PIL import Image
            from scipy.ndimage import binary_erosion, distance_transform_edt
            from src.quality_mask2former import config_for_model, load_official_model, native_predict
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
            model.cuda()
            model.eval()
            torch.cuda.reset_peak_memory_stats()
            endpoint_results: dict[str, Any] = {}
            historical_predictions = (
                ROOT / config["historical_control_run"] / "inference/predictions"
            )
            for step in ENDPOINTS:
                checkpoint = batch / f"mask2former/model-{step}.pt"
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                model.load_state_dict(saved["model"])
                del saved
                model.eval()
                endpoint_predictions = predictions_root / str(step)
                endpoint_predictions.mkdir()
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
                    Image.fromarray(prediction).save(endpoint_predictions / name)
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
                        predicted_boundary = predicted_cable ^ binary_erosion(predicted_cable)
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
                        print(
                            json.dumps(
                                {
                                    "experiment_id": config["experiment_id"],
                                    "endpoint": step,
                                    "inferred": index + 1,
                                    "total": len(rgb_result["image_hashes"]),
                                }
                            ),
                            flush=True,
                        )
                endpoint_results[str(step)] = {
                    "checkpoint_sha256": checkpoint_hashes[str(step)],
                    "prediction_hashes": {
                        name: sha256(endpoint_predictions / name)
                        for name in sorted(rgb_result["image_hashes"])
                    },
                    "inference_seconds": inference_seconds,
                    "inference_median_seconds": float(np.median(inference_seconds[1:])),
                    "metrics": {key: metrics(matrix) for key, matrix in totals.items()},
                    "boundary_metrics": {
                        key: spatial_base.boundary_metrics(values)
                        for key, values in boundary.items()
                    },
                    "per_view": per_view,
                }
            historical_hashes = read(
                ROOT / config["historical_control_run"] / "inference/result.json"
            )["prediction_hashes"]
            step6000_hashes = endpoint_results["6000"]["prediction_hashes"]
            historical_prediction_exact = step6000_hashes == historical_hashes
            if checkpoint_hashes != {
                str(step): sha256(batch / f"mask2former/model-{step}.pt")
                for step in ENDPOINTS
            }:
                raise ValueError("A sealed endpoint checkpoint changed during inference")
            verify_hashes(batch / "export/images", export["image_hashes"])
            verify_hashes(rgb_evaluation / "images", rgb_result["image_hashes"])
            result.update(
                status="completed",
                checkpoints_unchanged=True,
                historical_6000_predictions_exact=historical_prediction_exact,
                curve_control_endpoint=6000,
                endpoints=endpoint_results,
                train_labeled_count=len(train_names),
                holdout_labeled_count=len(holdout_names),
                unlabeled_count=len(rgb_result["image_hashes"])
                - len(train_names)
                - len(holdout_names),
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
    parser.add_argument("--experiment-id", choices=sorted(SUPPORTED), required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "train", "inference"), required=True)
    args = parser.parse_args()
    config_path = ROOT / "experiments/runs" / args.experiment_id / "config.yaml"
    config = read(config_path)
    if config["experiment_id"] != args.experiment_id:
        raise ValueError("Config identity differs from command")
    validate_config(config)
    configure_base(args.experiment_id, config_path)
    if args.stage == "prepare":
        prepare(config, config_path)
    elif args.stage == "train":
        train(config, config_path, args.artifact_root.resolve())
    else:
        inference(config, config_path, args.artifact_root.resolve())


if __name__ == "__main__":
    main()
