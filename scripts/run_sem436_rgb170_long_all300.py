#!/usr/bin/env python3
"""Train and infer a fresh 12k all-300 Mask2Former on frozen RGB170 renders."""

from __future__ import annotations

import argparse
import fcntl
import json
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_sem396_spatial as spatial_base
from run_mask2former_long_spatial import (
    ENDPOINTS,
    model_state_sha256,
    phase_lr_factor,
    save_model_checkpoint,
    save_resume_checkpoint,
)
from src.quality_protocol import sha256, verify_hashes, write_new_json


EID = "SEM-436-RGB170-MASK2FORMER-LONG12K-ALL300"
CONFIG = ROOT / "experiments/runs" / EID / "config.yaml"
CODE_PATHS = (
    "scripts/run_sem436_rgb170_long_all300.py",
    "scripts/audit_sem436_rgb170_long_all300.py",
    "scripts/run_mask2former_long_spatial.py",
    "scripts/run_sem396_spatial.py",
    "src/quality_mask2former.py",
    "src/quality_mask2former_training.py",
    "src/quality_static_training.py",
    "src/quality_semantic_audit.py",
)


def build_sampling_plan(
    train_names: list[str], seed: int, iterations: int, crop: int
) -> list[dict[str, Any]]:
    """Build the frozen training plan; wrappers may replace the input policy."""
    return spatial_base.sampling_plan(train_names, seed, iterations, crop)


def sampling_trace_keys(config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(config.get("sampling_trace_keys", ("name", "top", "left", "flip")))


def make_training_sampler(parent_sampler, config: dict[str, Any]):
    """Return the callable used for each update; native crops are the default."""
    return parent_sampler


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> None:
    if config["experiment_id"] != EID:
        raise ValueError("Unexpected experiment ID")
    if (
        config["iterations"] != 12000
        or config["base_phase_steps"] != 6000
        or config["extension_lr_multiplier"] != 0.1
        or tuple(config["endpoints"]) != ENDPOINTS
        or config["crop"] != 768
        or config["seed"] != 20260805
        or config["semantic_train_views"] != 300
    ):
        raise ValueError("Registered all300 long-training recipe changed")


def image_sources(config: dict[str, Any]) -> tuple[dict[str, Path], list[str], list[str], list[str]]:
    rgb_root = ROOT / config["rgb_images"]
    trajectory_root = ROOT / config["trajectory_images"]
    labeled = [f"{index:03d}.png" for index in range(1, 301)]
    unlabeled = [f"{index:03d}.png" for index in range(301, 401)]
    trajectory = [f"frame{index:04d}.png" for index in range(144)]
    sources = {name: rgb_root / name for name in labeled + unlabeled}
    sources.update({name: trajectory_root / name for name in trajectory})
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing RGB170 render inputs: {missing[:8]}")
    return sources, labeled, unlabeled, trajectory


def frozen_parent_hashes(config: dict[str, Any]) -> dict[str, str]:
    paths = {
        "training_result": ROOT / config["rgb_training_result"],
        "model": ROOT / config["rgb_model"],
        "evaluation_result": ROOT / config["rgb_evaluation_result"],
        "trajectory_result": ROOT / config["rgb_trajectory_result"],
    }
    actual = {key: sha256(path) for key, path in paths.items()}
    if actual != config["frozen_rgb170_sha256"]:
        raise ValueError(f"Frozen RGB170 binding changed: {actual}")
    return actual


def prepare(config: dict[str, Any]) -> None:
    batch = ROOT / config["output_path"]
    if batch.exists():
        raise FileExistsError(batch)
    parent_hashes = frozen_parent_hashes(config)
    sources, labeled, unlabeled, trajectory = image_sources(config)
    manifest_path = ROOT / config["label_manifest"]
    manifest = read(manifest_path)
    records = {record["image"]: record for record in manifest["records"]}
    label_root = ROOT / config["label_root"]
    label_hashes = {name: sha256(label_root / name) for name in labeled}
    if any(label_hashes[name] != records[name]["label_png_sha256"] for name in labeled):
        raise ValueError("A frozen all300 semantic label changed")

    batch.mkdir(parents=True)
    images = batch / "export/images"
    images.mkdir(parents=True)
    image_hashes = {}
    for name, source in sources.items():
        image_hashes[name] = sha256(source)
        shutil.copyfile(source, images / name)
    verify_hashes(images, image_hashes)
    write_new_json(
        batch / "export/result.json",
        {
            "schema_version": 1,
            "status": "completed",
            "experiment_id": EID,
            "rgb_experiment_id": "RGB-170-IMPROVEDGS-BUDGET15M-H50K",
            "image_hashes": image_hashes,
            "image_count": len(image_hashes),
            "semantic_train": labeled,
            "unlabeled": unlabeled,
            "trajectory": trajectory,
            "rgb_copies_exact": len(image_hashes),
            "frozen_rgb170_sha256": parent_hashes,
        },
    )
    plan = build_sampling_plan(labeled, config["seed"], config["iterations"], config["crop"])
    counts = Counter(row["name"] for row in plan)
    write_new_json(
        batch / "sampling-plan.json",
        {
            "schema_version": 1,
            "experiment_id": EID,
            "seed": config["seed"],
            "steps": config["iterations"],
            "crop": config["crop"],
            "train_names": labeled,
            "samples": plan,
            "per_view_counts": dict(sorted(counts.items())),
            "minimum_samples_per_view": min(counts.values()),
            "maximum_samples_per_view": max(counts.values()),
            "all_training_views_sampled": set(counts) == set(labeled),
        },
    )
    write_new_json(
        batch / "prepare-result.json",
        {
            "schema_version": 1,
            "status": "completed",
            "experiment_id": EID,
            "config_sha256": sha256(CONFIG),
            "export_result_sha256": sha256(batch / "export/result.json"),
            "sampling_plan_sha256": sha256(batch / "sampling-plan.json"),
            "label_manifest_sha256": sha256(manifest_path),
            "training_label_hashes": label_hashes,
            "semantic_train_count": len(labeled),
            "unlabeled_count": len(unlabeled),
            "trajectory_count": len(trajectory),
            "schedule_frozen_before_metrics": True,
            "code_sha256": {path: sha256(ROOT / path) for path in CODE_PATHS},
        },
    )


def train(config: dict[str, Any], artifact_root: Path) -> None:
    from src.quality_static_training import TrainingLabels

    batch = ROOT / config["output_path"]
    output = batch / "mask2former"
    output.mkdir(exist_ok=False)
    export = read(batch / "export/result.json")
    prepared = read(batch / "prepare-result.json")
    plan_record = read(batch / "sampling-plan.json")
    verify_hashes(batch / "export/images", export["image_hashes"])
    label_root = ROOT / config["label_root"]
    verify_hashes(label_root, prepared["training_label_hashes"])
    train_names = list(export["semantic_train"])
    trace_keys = sampling_trace_keys(config)
    expected_trace = [
        {key: row[key] for key in trace_keys} for row in plan_record["samples"]
    ]
    native_plan = build_sampling_plan(train_names, config["seed"], config["iterations"], config["crop"])
    if expected_trace != [
        {key: row[key] for key in trace_keys} for row in native_plan
    ]:
        raise ValueError("Frozen plan differs from deterministic sampler")
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "starting",
        "experiment_id": EID,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": sha256(CONFIG),
        "seed": config["seed"],
        "trained_steps": 0,
        "access": {
            "train": train_names,
            "labels_allowed_during_fit": train_names,
            "unlabeled_rgb_allowed_during_fit": False,
            "trajectory_rgb_allowed_during_fit": False,
        },
        "binding": {
            "config_sha256": sha256(CONFIG),
            "sampling_plan_sha256": sha256(batch / "sampling-plan.json"),
            "export_result_sha256": sha256(batch / "export/result.json"),
            "frozen_rgb170_sha256": frozen_parent_hashes(config),
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
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True
            )
            result["gpu_processes_before"] = applications
            if any(line.strip() and "ToDesk" not in line for line in applications.splitlines()):
                raise RuntimeError("Other GPU compute process active outside shared lock")
            runtime, torch, np = spatial_base.setup_runtime(artifact_root)
            from detectron2.structures import Instances
            from src.quality_mask2former import author_optimizer, class_masks, config_for_model, load_official_model
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
            labels = TrainingLabels(label_root, train_names, read(ROOT / config["label_manifest"]), [1320, 989])
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
            reference = ROOT / config["recipe_reference_run"] / "mask2former"
            if (output / "resolved_model.yaml").read_bytes() != (reference / "resolved_model.yaml").read_bytes():
                raise ValueError("Mask2Former recipe differs from frozen long-training reference")
            model, loaded = load_official_model(model_config, runtime / "weights/model_final_6b4a3a.pkl")
            result["pretrained_load"] = loaded
            result["initial_model_state_sha256"] = model_state_sha256(model.state_dict())
            if result["initial_model_state_sha256"] != read(reference / "result.json")["initial_model_state_sha256"]:
                raise ValueError("Fresh ADE initialization differs from reference")
            model.cuda()
            optimizer = author_optimizer(model, model_config)
            initial_rates = [group["lr"] for group in optimizer.param_groups]
            if sorted(set(initial_rates)) != [1e-05, 0.0001]:
                raise ValueError("Unexpected optimizer LR groups")
            scaler = torch.amp.GradScaler("cuda", init_scale=1024)
            images = {name: read_rgb(batch / "export/images" / name) for name in train_names}
            targets = {name: labels.read(name) for name in train_names}
            sampler_rng = random.Random(config["seed"])
            sampler = make_training_sampler(sampled_crop, config)
            model.train()
            training_started = time.monotonic()
            result["amp_skips"] = 0
            result["checkpoints"] = {}
            with (output / "training.jsonl").open("x") as journal:
                for step in range(1, config["iterations"] + 1):
                    image_np, label_np, trace = sampler(sampler_rng, train_names, images, targets, config["crop"])
                    ids, masks = class_masks(label_np)
                    instances = Instances(tuple(label_np.shape))
                    instances.gt_classes = torch.from_numpy(ids)
                    instances.gt_masks = torch.from_numpy(masks)
                    image = torch.from_numpy(image_np.transpose(2, 0, 1).copy()).float()
                    phase, factor = phase_lr_factor(step, config["extension_lr_multiplier"])
                    for group, rate in zip(optimizer.param_groups, initial_rates):
                        group["lr"] = rate * factor
                    row = amp_update(model, optimizer, scaler, image, instances, max_attempts=12, clip=0.01)
                    row.update(
                        step=step, sample=trace, schedule_phase=phase, lr_factor=factor,
                        minimum_group_lr=min(group["lr"] for group in optimizer.param_groups),
                        maximum_group_lr=max(group["lr"] for group in optimizer.param_groups),
                        elapsed_seconds=time.monotonic() - training_started,
                    )
                    journal.write(json.dumps(row) + "\n")
                    result["trained_steps"] = step
                    result["amp_skips"] += len(row["skips"])
                    if step % 100 == 0 or step == 1:
                        journal.flush()
                        print(json.dumps({"experiment_id": EID, "step": step, "phase": phase, "amp_skips": result["amp_skips"], "seconds": time.monotonic() - training_started}), flush=True)
                    if step in ENDPOINTS:
                        result["checkpoints"][str(step)] = {
                            "model": save_model_checkpoint(model, result["binding"], result["access"], step, output / f"model-{step}.pt")
                        }
                        if step in (6000, 12000):
                            result["checkpoints"][str(step)]["resume"] = save_resume_checkpoint(
                                model, optimizer, scaler, sampler_rng, result["binding"], result["access"], step, output / f"resume-{step}.pt"
                            )
            result["training_seconds"] = time.monotonic() - training_started
            actual_samples = [json.loads(line)["sample"] for line in (output / "training.jsonl").read_text().splitlines()]
            if actual_samples != expected_trace:
                raise ValueError("Actual training trace differs from frozen plan")
            if labels.accessed != set(train_names):
                raise ValueError("Training did not access exactly the 300 frozen labels")
            verify_hashes(batch / "export/images", export["image_hashes"])
            frozen_parent_hashes(config)
            result.update(
                status="completed_sealed_12000",
                sampling_plan_exact=True,
                curve_control_endpoint=6000,
                training_labels_accessed=sorted(labels.accessed),
                unlabeled_or_trajectory_accessed_during_fit=[],
                frozen_rgb_hashes_exact=True,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                artifacts={
                    str(path.relative_to(output)): sha256(path)
                    for path in output.rglob("*") if path.is_file() and path.name not in {"started.json", "result.json"}
                },
            )
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(output / "result.json", result)


def inference(config: dict[str, Any], artifact_root: Path) -> None:
    batch = ROOT / config["output_path"]
    output = batch / "inference"
    output.mkdir(exist_ok=False)
    predictions_root = output / "predictions"
    predictions_root.mkdir()
    export = read(batch / "export/result.json")
    training = read(batch / "mask2former/result.json")
    if training["status"] != "completed_sealed_12000" or training["trained_steps"] != 12000:
        raise ValueError("Sealed 12k all300 trajectory required")
    verify_hashes(batch / "mask2former", training["artifacts"])
    checkpoint_hashes = {str(step): sha256(batch / f"mask2former/model-{step}.pt") for step in ENDPOINTS}
    train_names = set(export["semantic_train"])
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "starting",
        "experiment_id": EID,
        "config_sha256": sha256(CONFIG),
        "checkpoint_sha256_before_inference": checkpoint_hashes,
        "quality_scope": "all300 training fit; released 301-400 and 144 trajectory have no GT",
    }
    started = time.monotonic()
    try:
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            queued = time.monotonic()
            fcntl.flock(lock, fcntl.LOCK_EX)
            result["queue_seconds"] = time.monotonic() - queued
            applications = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True
            )
            if any(line.strip() and "ToDesk" not in line for line in applications.splitlines()):
                raise RuntimeError("Other GPU compute process active outside shared lock")
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
            model, _ = load_official_model(model_config, runtime / "weights/model_final_6b4a3a.pkl")
            model.cuda()
            model.eval()
            torch.cuda.reset_peak_memory_stats()
            label_root = ROOT / config["label_root"]
            endpoint_results = {}
            for step in ENDPOINTS:
                checkpoint = batch / f"mask2former/model-{step}.pt"
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                model.load_state_dict(saved["model"])
                del saved
                model.eval()
                endpoint_predictions = predictions_root / str(step)
                endpoint_predictions.mkdir()
                total_confusion = np.zeros((5, 5), dtype=np.int64)
                boundary = np.zeros(4, dtype=np.int64)
                per_view = []
                inference_seconds = []
                prediction_hashes = {}
                for index, name in enumerate(sorted(export["image_hashes"])):
                    rgb = read_rgb(batch / "export/images" / name)
                    torch.cuda.synchronize()
                    stamp = time.monotonic()
                    probability = native_predict(model, rgb)
                    torch.cuda.synchronize()
                    inference_seconds.append(time.monotonic() - stamp)
                    prediction = probability.argmax(0).astype(np.uint8)
                    Image.fromarray(prediction).save(endpoint_predictions / name)
                    prediction_hashes[name] = sha256(endpoint_predictions / name)
                    if name in train_names:
                        target = np.asarray(Image.open(label_root / name), dtype=np.uint8)
                        confusion = independent_confusion(target, prediction)
                        total_confusion += confusion
                        cable = target == 2
                        gt_boundary = cable ^ binary_erosion(cable)
                        gt_distance = distance_transform_edt(~gt_boundary)
                        predicted_cable = prediction == 2
                        predicted_boundary = predicted_cable ^ binary_erosion(predicted_cable)
                        predicted_distance = distance_transform_edt(~predicted_boundary)
                        boundary += np.asarray([
                            np.sum(predicted_boundary & (gt_distance <= 2)), np.sum(predicted_boundary),
                            np.sum(gt_boundary & (predicted_distance <= 2)), np.sum(gt_boundary),
                        ], dtype=np.int64)
                        per_view.append({"name": name, **metrics(confusion)})
                    if index % 25 == 0:
                        print(json.dumps({"experiment_id": EID, "endpoint": step, "inferred": index + 1, "total": len(export["image_hashes"])}), flush=True)
                endpoint_results[str(step)] = {
                    "checkpoint_sha256": checkpoint_hashes[str(step)],
                    "prediction_hashes": prediction_hashes,
                    "inference_seconds": inference_seconds,
                    "inference_median_seconds": float(np.median(inference_seconds[1:])),
                    "metrics": metrics(total_confusion),
                    "boundary_metrics": spatial_base.boundary_metrics(boundary),
                    "per_view": per_view,
                }
            verify_hashes(batch / "export/images", export["image_hashes"])
            frozen_parent_hashes(config)
            result.update(
                status="completed",
                checkpoints_unchanged=checkpoint_hashes == {str(step): sha256(batch / f"mask2former/model-{step}.pt") for step in ENDPOINTS},
                endpoints=endpoint_results,
                train_labeled_count=len(train_names),
                unlabeled_count=len(export["unlabeled"]),
                trajectory_count=len(export["trajectory"]),
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
    parser.add_argument("--stage", choices=("prepare", "train", "inference"), required=True)
    args = parser.parse_args()
    config = read(CONFIG)
    validate_config(config)
    if args.stage == "prepare":
        prepare(config)
    elif args.stage == "train":
        train(config, args.artifact_root.resolve())
    else:
        inference(config, args.artifact_root.resolve())


if __name__ == "__main__":
    main()
