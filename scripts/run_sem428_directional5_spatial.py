#!/usr/bin/env python3
"""Train and evaluate the SEM-426 view-conditioned five-class spatial baseline."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_sem425_direct5_preflight import (  # noqa: E402
    class_weights,
    fresh_rgb,
    rgb_state_sha256,
    roles,
    make_crop_view,
    setup_runtime,
)
from src.quality_view_conditioned_semantics import (  # noqa: E402
    build_directional_logit_field,
    directional_logit_sha256,
    render_directional_probabilities,
)
from src.quality_protocol import sha256, verify_hashes  # noqa: E402
from src.quality_semantic_audit import independent_confusion, metrics  # noqa: E402


EID = "SEM-428-RGB152-DIR5-SPATIAL256"
CONFIG = ROOT / "experiments/runs" / EID / "config.yaml"
CODE_PATHS = (
    "scripts/run_sem428_directional5_spatial.py",
    "scripts/run_sem425_direct5_preflight.py",
    "src/quality_view_conditioned_semantics.py",
    "src/quality_feature_splatting.py",
    "src/quality_native_gaussian.py",
    "src/quality_cameras.py",
    "src/quality_protocol.py",
    "src/quality_semantic_audit.py",
)


def semantic_crop(runtime, coefficients, sample):
    probability = render_directional_probabilities(
        runtime.render,
        make_crop_view(runtime, sample),
        runtime.model,
        runtime.pipe,
        coefficients,
    )
    return runtime.torch.flip(probability, dims=(-1,)) if sample["flip"] else probability


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def replace(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalized_plan(record, crop):
    return [
        {
            "step": int(row["step"]),
            "name": str(row["name"]),
            "top": int(row["top"]),
            "left": int(row["left"]),
            "flip": bool(row["flip"]),
            "crop": crop,
        }
        for row in record["samples"]
    ]


def verify_binding(config, binding):
    if sha256(CONFIG) != binding["config_sha256"]:
        raise ValueError("SEM428 config changed after preparation")
    if sha256(ROOT / config["split_path"]) != binding["split_sha256"]:
        raise ValueError("SEM428 split changed after preparation")
    if sha256(ROOT / config["parent_sampling_plan"]) != binding["sampling_plan_sha256"]:
        raise ValueError("SEM396 sampling plan changed after SEM428 preparation")
    verify_hashes(ROOT, binding["code_sha256"])
    verify_hashes(ROOT / config["label_root"], binding["training_label_hashes"])


def prepare(config):
    output = ROOT / config["output_path"]
    if output.exists():
        raise FileExistsError(output)
    audit_path = ROOT / config["parent_audit"]
    audit = read(audit_path)
    if sha256(audit_path) != config["parent_audit_sha256"] or audit["status"] != "passed" or audit["passed"] != audit["total"]:
        raise ValueError("A byte-matched passed independent SEM427 preflight audit is required")
    training, confirmation, buffer_names = roles(config)
    parent_plan_path = ROOT / config["parent_sampling_plan"]
    if sha256(parent_plan_path) != config["parent_sampling_plan_sha256"]:
        raise ValueError("SEM396 sampling-plan hash differs from the registered value")
    plan = normalized_plan(read(parent_plan_path), config["crop"])
    if len(plan) != config["iterations"] or any(row["name"] not in set(training) for row in plan):
        raise ValueError("SEM396 sampling plan is not a 6000-step SEM428 training-only plan")
    label_root = ROOT / config["label_root"]
    manifest = read(ROOT / config["label_manifest"])
    records = {row["image"]: row["label_png_sha256"] for row in manifest["records"]}
    label_hashes = {name: sha256(label_root / name) for name in training}
    if any(label_hashes[name] != records.get(name) for name in training):
        raise ValueError("A SEM428 training label differs from the frozen manifest")
    output.mkdir(parents=True)
    write(output / "sampling-plan.json", {"schema_version": 1, "experiment_id": EID, "samples": plan})
    binding = {
        "schema_version": 1,
        "experiment_id": EID,
        "config_sha256": sha256(CONFIG),
        "split_sha256": sha256(ROOT / config["split_path"]),
        "sampling_plan_sha256": sha256(parent_plan_path),
        "local_sampling_plan_sha256": sha256(output / "sampling-plan.json"),
        "label_manifest_sha256": sha256(ROOT / config["label_manifest"]),
        "training_label_hashes": label_hashes,
        "training_views": training,
        "confirmation_views_sealed": confirmation,
        "buffer_views_sealed": buffer_names,
        "confirmation_rgb_accesses": 0,
        "confirmation_label_accesses": 0,
        "buffer_rgb_accesses": 0,
        "buffer_label_accesses": 0,
        "sem427_audit_sha256": sha256(audit_path),
        "rgb_checkpoint_sha256": sha256(ROOT / config["rgb_model"] / "chkpnt30000.pth"),
        "code_sha256": {path: sha256(ROOT / path) for path in CODE_PATHS},
    }
    write(output / "binding.json", binding)
    write(
        output / "prepare-result.json",
        {
            "schema_version": 1,
            "experiment_id": EID,
            "status": "completed",
            "binding_sha256": sha256(output / "binding.json"),
            "training_views": len(training),
            "confirmation_views": len(confirmation),
            "buffer_views": len(buffer_names),
            "steps": len(plan),
            "quality_access": False,
        },
    )


def save_checkpoint(runtime, path, logits, optimizer, step, binding_sha256):
    runtime.torch.save(
        {
            "schema_version": 1,
            "experiment_id": EID,
            "step": step,
            "binding_sha256": binding_sha256,
            "directional_coefficients": logits.detach().cpu(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def load_logits(runtime, config, path, step, binding_sha256):
    payload = runtime.torch.load(path, map_location="cpu", weights_only=True)
    if payload["experiment_id"] != EID or payload["step"] != step or payload["binding_sha256"] != binding_sha256:
        raise ValueError("SEM428 checkpoint identity mismatch")
    logits = build_directional_logit_field(
        runtime.model.get_xyz.shape[0], device="cuda", dtype=runtime.model.get_xyz.dtype
    )
    if tuple(payload["directional_coefficients"].shape) != tuple(logits.shape):
        raise ValueError("SEM428 checkpoint topology mismatch")
    with runtime.torch.no_grad():
        logits.copy_(payload["directional_coefficients"].cuda())
    return logits


def train(config, artifact_root):
    output = ROOT / config["output_path"]
    training_dir = output / "training"
    training_dir.mkdir(exist_ok=False)
    binding = read(output / "binding.json")
    verify_binding(config, binding)
    result = {
        "schema_version": 1,
        "experiment_id": EID,
        "status": "running",
        "successful_steps": 0,
        "confirmation_rgb_accesses": 0,
        "confirmation_label_accesses": 0,
        "buffer_rgb_accesses": 0,
        "buffer_label_accesses": 0,
    }
    replace(training_dir / "result.json", result)
    started = time.monotonic()
    try:
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            runtime = setup_runtime(config, artifact_root)
            runtime.torch.cuda.reset_peak_memory_stats()
            training_names, _, _ = roles(config)
            label_root = ROOT / config["label_root"]
            counts, weights_np, hashes = class_weights(runtime, training_names, label_root)
            if hashes != binding["training_label_hashes"]:
                raise ValueError("SEM428 training-label binding differs")
            weights = runtime.torch.from_numpy(weights_np).cuda()
            plan = read(output / "sampling-plan.json")["samples"]
            logits = build_directional_logit_field(
                runtime.model.get_xyz.shape[0], device="cuda", dtype=runtime.model.get_xyz.dtype
            )
            initial_hash = directional_logit_sha256(logits)
            optimizer = runtime.torch.optim.Adam(
                [logits],
                lr=config["training"]["learning_rate"],
                weight_decay=config["training"]["weight_decay"],
            )
            before_state = rgb_state_sha256(runtime.model)
            fixed_name = training_names[0]
            before_rgb = fresh_rgb(runtime, fixed_name)
            accessed = set()
            trace_path = training_dir / "training.jsonl"
            with trace_path.open("x", encoding="utf-8") as journal:
                for sample in plan:
                    optimizer.zero_grad(set_to_none=True)
                    probabilities = semantic_crop(runtime, logits, sample)
                    label = runtime.np.asarray(runtime.Image.open(label_root / sample["name"]), dtype=runtime.np.uint8)
                    accessed.add(sample["name"])
                    from src.quality_feature_splatting import crop_and_flip

                    target_np = crop_and_flip(label, sample)
                    target = runtime.torch.from_numpy(target_np.astype("int64")).cuda()
                    loss = runtime.torch.nn.functional.nll_loss(
                        probabilities.clamp_min(runtime.torch.finfo(probabilities.dtype).tiny).log()[None],
                        target[None],
                        weight=weights,
                    )
                    loss.backward()
                    if logits.grad is None or not runtime.torch.isfinite(logits.grad).all():
                        raise FloatingPointError("SEM428 direct-logit gradient is missing or non-finite")
                    gradient_norm = float(logits.grad.detach().norm())
                    if gradient_norm <= 0:
                        raise ValueError("SEM428 direct-logit gradient is zero")
                    optimizer.step()
                    if not runtime.torch.isfinite(logits).all():
                        raise FloatingPointError("SEM428 direct logits became non-finite")
                    step = int(sample["step"])
                    result["successful_steps"] = step
                    row = {
                        "step": step,
                        "sample": sample,
                        "loss": float(loss.detach()),
                        "gradient_norm": gradient_norm,
                        "probability_sum_max_error": float((probabilities.detach().sum(dim=0) - 1).abs().max()),
                        "elapsed_seconds": time.monotonic() - started,
                    }
                    journal.write(json.dumps(row) + "\n")
                    if step == 1 or step % 100 == 0:
                        journal.flush()
                        print(json.dumps(row), flush=True)
                    if step in (config["analysis_checkpoint"], config["primary_checkpoint"]):
                        save_checkpoint(
                            runtime,
                            training_dir / f"directional-logits-{step}.pt",
                            logits,
                            optimizer,
                            step,
                            sha256(output / "binding.json"),
                        )
            checkpoint = training_dir / f"directional-logits-{config['primary_checkpoint']}.pt"
            replay = load_logits(
                runtime,
                config,
                checkpoint,
                config["primary_checkpoint"],
                sha256(output / "binding.json"),
            )
            with runtime.torch.no_grad():
                reference_probability = semantic_crop(runtime, logits, plan[0])
                replay_probability = semantic_crop(runtime, replay, plan[0])
            if directional_logit_sha256(logits) != directional_logit_sha256(replay) or not runtime.torch.equal(reference_probability, replay_probability):
                raise ValueError("SEM428 checkpoint replay differs")
            after_state = rgb_state_sha256(runtime.model)
            after_rgb = fresh_rgb(runtime, fixed_name)
            if before_state != after_state or not runtime.torch.equal(before_rgb, after_rgb):
                raise ValueError("SEM428 frozen RGB isolation failed")
            result.update(
                status="completed_sealed_6000",
                point_count=int(logits.shape[0]),
                trainable_parameter_count=int(logits.numel()),
                initial_directional_logit_sha256=initial_hash,
                final_directional_logit_sha256=directional_logit_sha256(logits),
                training_views_accessed=sorted(accessed),
                class_pixel_counts=counts.tolist(),
                class_weights=weights_np.tolist(),
                checkpoint_replay_exact=True,
                rgb_state_hashes_equal=True,
                fresh_rgb_exact=True,
                frozen_rgb_max_absolute_pixel_difference=0,
                checkpoint_sha256s={
                    str(step): sha256(training_dir / f"directional-logits-{step}.pt")
                    for step in (config["analysis_checkpoint"], config["primary_checkpoint"])
                },
                peak_allocated_mib=runtime.torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=runtime.torch.cuda.max_memory_reserved() / 2**20,
                binding_sha256=sha256(output / "binding.json"),
            )
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        replace(training_dir / "result.json", result)


def boundary_counts(target, prediction):
    from scipy.ndimage import binary_erosion, distance_transform_edt

    cable = target == 2
    gt_boundary = cable ^ binary_erosion(cable)
    gt_distance = np.asarray(distance_transform_edt(~gt_boundary))
    predicted_cable = prediction == 2
    predicted_boundary = predicted_cable ^ binary_erosion(predicted_cable)
    predicted_distance = np.asarray(distance_transform_edt(~predicted_boundary))
    return np.asarray(
        [
            np.sum(predicted_boundary & (gt_distance <= 2)),
            np.sum(predicted_boundary),
            np.sum(gt_boundary & (predicted_distance <= 2)),
            np.sum(gt_boundary),
        ],
        dtype=np.int64,
    )


def boundary_metrics(values):
    precision = float(values[0] / values[1]) if values[1] else None
    recall = float(values[2] / values[3]) if values[3] else None
    f1 = float(2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else None
    return {"tolerance_px": 2, "boundary_precision": precision, "boundary_recall": recall, "boundary_f1": f1, "counts": values.tolist()}


def inference(config, artifact_root):
    output = ROOT / config["output_path"]
    inference_dir = output / "inference"
    inference_dir.mkdir(exist_ok=False)
    predictions = inference_dir / "predictions"
    predictions.mkdir()
    binding = read(output / "binding.json")
    verify_binding(config, binding)
    training_result = read(output / "training/result.json")
    if training_result["status"] != "completed_sealed_6000" or training_result["successful_steps"] != config["iterations"]:
        raise ValueError("A sealed SEM428 step-6000 checkpoint is required")
    checkpoint = output / "training" / f"directional-logits-{config['primary_checkpoint']}.pt"
    checkpoint_hash = sha256(checkpoint)
    result = {
        "schema_version": 1,
        "experiment_id": EID,
        "status": "running",
        "checkpoint_sha256_before_confirmation_access": checkpoint_hash,
        "confirmation_access_begins_after_checkpoint_sealed": True,
    }
    replace(inference_dir / "result.json", result)
    started = time.monotonic()
    try:
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            runtime = setup_runtime(config, artifact_root)
            runtime.torch.cuda.reset_peak_memory_stats()
            logits = load_logits(
                runtime,
                config,
                checkpoint,
                config["primary_checkpoint"],
                sha256(output / "binding.json"),
            )
            logits.requires_grad_(False)
            training_names, confirmation_names, _ = roles(config)
            training_set, confirmation_set = set(training_names), set(confirmation_names)
            split = read(ROOT / config["split_path"])
            group_by_name = {name: group["group_id"] for group in split["groups"] for name in group["images"]}
            totals = {
                "train": np.zeros((5, 5), dtype=np.int64),
                "holdout": np.zeros((5, 5), dtype=np.int64),
                **{group["group_id"]: np.zeros((5, 5), dtype=np.int64) for group in split["groups"]},
            }
            boundaries = {key: np.zeros(4, dtype=np.int64) for key in totals}
            per_view = []
            times = []
            label_root = ROOT / config["label_root"]
            from PIL import Image
            from src.quality_native_gaussian import make_native_view

            ordered = sorted(training_set | confirmation_set)
            for index, name in enumerate(ordered):
                runtime.torch.cuda.synchronize()
                tick = time.monotonic()
                with runtime.torch.inference_mode():
                    probability = render_directional_probabilities(
                        runtime.render,
                        make_native_view(runtime.cameras[name]),
                        runtime.model,
                        runtime.pipe,
                        logits,
                    )
                runtime.torch.cuda.synchronize()
                times.append(time.monotonic() - tick)
                prediction = probability.argmax(dim=0).cpu().numpy().astype(np.uint8)
                Image.fromarray(prediction, mode="L").save(predictions / name)
                target = np.asarray(Image.open(label_root / name), dtype=np.uint8)
                matrix = independent_confusion(target, prediction)
                role = "train" if name in training_set else "holdout"
                keys = [role]
                if role == "holdout":
                    keys.append(group_by_name[name])
                counts = boundary_counts(target, prediction)
                for key in keys:
                    totals[key] += matrix
                    boundaries[key] += counts
                view_metrics = metrics(matrix)
                per_view.append({"name": name, "role": role, "group": group_by_name.get(name), **view_metrics})
                if index % 25 == 0:
                    print(json.dumps({"inferred": index + 1, "total": len(ordered)}), flush=True)
            holdout_views = [row for row in per_view if row["role"] == "holdout"]
            worst_count = max(1, int(np.ceil(0.20 * len(holdout_views))))
            worst = sorted(holdout_views, key=lambda row: row["mIoU_without_background"])[:worst_count]
            sem396 = read(ROOT / "outputs/quality/SEM-396/r1/inference/result.json")
            sem426 = read(ROOT / "outputs/quality/SEM-426/r1/inference/result.json")
            current_metrics = {key: metrics(matrix) for key, matrix in totals.items()}
            current_boundaries = {key: boundary_metrics(value) for key, value in boundaries.items()}
            result.update(
                status="completed",
                checkpoint_sha256_after_confirmation_access=sha256(checkpoint),
                checkpoint_unchanged=sha256(checkpoint) == checkpoint_hash,
                prediction_hashes={name: sha256(predictions / name) for name in ordered},
                metrics=current_metrics,
                boundary_metrics=current_boundaries,
                per_view=per_view,
                worst_20_percent_holdout={
                    "count": worst_count,
                    "views": [row["name"] for row in worst],
                    "mean_foreground_mIoU": float(np.mean([row["mIoU_without_background"] for row in worst])),
                },
                comparison_sem396={
                    "five_class_mIoU_delta_pp": 100 * (current_metrics["holdout"]["mIoU_with_background"] - sem396["metrics"]["holdout"]["mIoU_with_background"]),
                    "foreground_mIoU_delta_pp": 100 * (current_metrics["holdout"]["mIoU_without_background"] - sem396["metrics"]["holdout"]["mIoU_without_background"]),
                    "stay_cable_IoU_delta_pp": 100 * (current_metrics["holdout"]["per_class"]["stay_cable"]["IoU"] - sem396["metrics"]["holdout"]["per_class"]["stay_cable"]["IoU"]),
                    "boundary_f1_delta_pp": 100 * (current_boundaries["holdout"]["boundary_f1"] - sem396["boundary_metrics"]["holdout"]["boundary_f1"]),
                },
                comparison_sem426={
                    "five_class_mIoU_delta_pp": 100 * (current_metrics["holdout"]["mIoU_with_background"] - sem426["metrics"]["holdout"]["mIoU_with_background"]),
                    "foreground_mIoU_delta_pp": 100 * (current_metrics["holdout"]["mIoU_without_background"] - sem426["metrics"]["holdout"]["mIoU_without_background"]),
                    "stay_cable_IoU_delta_pp": 100 * (current_metrics["holdout"]["per_class"]["stay_cable"]["IoU"] - sem426["metrics"]["holdout"]["per_class"]["stay_cable"]["IoU"]),
                    "boundary_f1_delta_pp": 100 * (current_boundaries["holdout"]["boundary_f1"] - sem426["boundary_metrics"]["holdout"]["boundary_f1"]),
                },
                inference_seconds=times,
                inference_median_seconds=float(np.median(times[1:])),
                peak_allocated_mib=runtime.torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=runtime.torch.cuda.max_memory_reserved() / 2**20,
                train_label_accesses=len(training_set),
                confirmation_label_accesses=len(confirmation_set),
                buffer_label_accesses=0,
                unlabeled_label_accesses=0,
            )
            if sha256(checkpoint) != checkpoint_hash:
                raise ValueError("SEM428 checkpoint changed during confirmation inference")
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        replace(inference_dir / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "train", "inference"), required=True)
    args = parser.parse_args()
    config = read(CONFIG)
    if config["experiment_id"] != EID:
        raise ValueError("SEM428 config identity changed")
    if args.stage == "prepare":
        prepare(config)
    elif args.stage == "train":
        train(config, args.artifact_root.resolve())
    else:
        inference(config, args.artifact_root.resolve())


if __name__ == "__main__":
    main()
