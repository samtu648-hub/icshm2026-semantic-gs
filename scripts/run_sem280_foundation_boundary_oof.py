#!/usr/bin/env python3
"""Matched five-class CE versus fixed foundation-boundary completeness loss."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_sem260_error_budget import manifest_paths, png
from run_sem250_colmap_track_deck_diag import verify_freeze
from src.semantic2d import CLASS_NAMES, confusion_metrics, read_json, sha256, write_json
from src.semantic_cable_binary import normalize_rgb
from src.semantic_foundation_boundary import (
    ARMS,
    foundation_boundary_cross_entropy,
    foundation_inner_boundary_mask,
    fuse_foundation,
    quality_gates,
    training_plan,
    validate_fold,
)
from src.semantic_protected_fusion import local_confusion, partition_training_names
from src.semantic_render_domain import semantic_rgb
from src.submission_sem161 import sliding


EXP = "SEM-280-FOUNDATION-BOUNDARY-OOF"


def log(**values):
    print(json.dumps(values), flush=True)


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def initialize(model_dir, seed):
    from transformers import SegformerForSemanticSegmentation

    torch.manual_seed(seed)
    return SegformerForSemanticSegmentation.from_pretrained(
        model_dir,
        local_files_only=True,
        num_labels=5,
        id2label=dict(enumerate(CLASS_NAMES)),
        label2id={name: index for index, name in enumerate(CLASS_NAMES)},
        ignore_mismatched_sizes=True,
    )


def state_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def resource_gate(output_dir, config, started):
    if shutil.disk_usage(output_dir).free < config["minimum_free_disk_gib"] * 2**30:
        raise RuntimeError("Disk floor reached; existing artifacts are retained")
    if time.perf_counter() - started > config["maximum_gpu_hours"] * 3600:
        raise RuntimeError("Registered wall-clock budget reached")


def fit(
    output_dir,
    artifact_root,
    config,
    fold,
    arm,
    plan,
    rgb_paths,
    labels_root,
    smoke,
    started,
):
    folder = output_dir / f"fold{fold['fold']}_{arm}"
    folder.mkdir()
    train_names = sorted({row["name"] for row in plan}) if smoke else sorted(fold["train"])
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

    counts = np.zeros(5, np.int64)
    cache = {}
    access = []
    radius = int(config["training"]["foundation_inner_boundary_radius_pixels"])
    multiplier = float(config["training"]["arm_boundary_multipliers"][arm])
    for name in train_names:
        if name not in fold["train"] or name in fold["oof"] or name in fold["guard"]:
            raise ValueError("Forbidden training label access")
        rgb = png(rgb_paths[name][0])
        target = png(labels_root / name)
        if rgb.shape != (989, 1320, 3) or target.shape != (989, 1320):
            raise ValueError("Native training grid mismatch")
        counts += np.bincount(target.ravel(), minlength=5)
        boundary = foundation_inner_boundary_mask(target, radius=radius)
        cache[name] = (rgb, target, boundary)
        access.append(
            {
                "name": name,
                "label_sha256": sha256(labels_root / name),
                "rgb_sha256": rgb_paths[name][1],
                "foundation_boundary_pixels": int(boundary.sum()),
            }
        )
    if np.any(counts <= 0):
        raise ValueError("Missing training class")
    weights = np.clip(np.median(counts.astype(np.float64)) / counts, 0.05, 20).astype(
        np.float32
    )
    model = initialize(artifact_root / config["parents"]["model_config"], seed).cuda().train()
    initial_state_sha256 = state_hash(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    weight_tensor = torch.from_numpy(weights).cuda()
    crop = int(config["training"]["crop"])
    losses = []
    boundary_fractions = []
    begin = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step, row in enumerate(plan, 1):
        rgb, target, boundary = cache[row["name"]]
        y0, x0 = int(row["top"]), int(row["left"])
        image_crop = rgb[y0 : y0 + crop, x0 : x0 + crop]
        target_crop = target[y0 : y0 + crop, x0 : x0 + crop]
        boundary_crop = boundary[y0 : y0 + crop, x0 : x0 + crop]
        if row["flip"]:
            image_crop = image_crop[:, ::-1]
            target_crop = target_crop[:, ::-1]
            boundary_crop = boundary_crop[:, ::-1]
        pixels = normalize_rgb(np.ascontiguousarray(image_crop)).unsqueeze(0).cuda()
        targets = torch.from_numpy(target_crop.copy().astype(np.int64)).unsqueeze(0).cuda()
        boundary_tensor = torch.from_numpy(boundary_crop.copy()).unsqueeze(0).cuda()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = F.interpolate(
                model(pixel_values=pixels).logits,
                (crop, crop),
                mode="bilinear",
                align_corners=False,
            )
            loss, selected = foundation_boundary_cross_entropy(
                logits,
                targets,
                boundary_tensor,
                class_weights=weight_tensor,
                multiplier=multiplier,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        boundary_fractions.append(float(selected.mean().detach()))
        if step == 1 or step % 250 == 0 or step == len(plan):
            log(
                stage="training",
                fold=fold["fold"],
                arm=arm,
                step=step,
                total=len(plan),
                loss=losses[-1],
                boundary_fraction=boundary_fractions[-1],
                seconds=time.perf_counter() - begin,
            )
            resource_gate(output_dir, config, started)

    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - begin
    peak_mib = torch.cuda.max_memory_allocated() / 2**20
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        replay = model(pixel_values=pixels).logits.cpu()
    metadata = {
        "experiment_id": EXP,
        "fold": fold["fold"],
        "arm": arm,
        "seed": seed,
        "iterations": len(plan),
        "train": train_names,
        "outer_train": fold["train"],
        "oof": fold["oof"],
        "guard": fold["guard"],
        "actual_label_access": access,
        "actual_optimization_names": sorted({row["name"] for row in plan}),
        "sampling_plan_sha256": sha256(output_dir / f"fold{fold['fold']}_sampling_plan.json"),
        "config_sha256": config["_sha256"],
        "oof_split_sha256": config["oof_split_sha256"],
        "initialization": config["_initialization"],
        "initial_state_sha256": initial_state_sha256,
        "class_counts": counts.tolist(),
        "class_weights": weights.tolist(),
        "boundary_radius_pixels": radius,
        "boundary_multiplier": multiplier,
        "mean_sampled_boundary_fraction": float(np.mean(boundary_fractions)),
        "mode": "smoke" if smoke else "full",
        "training_task": "five_class",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    checkpoint = folder / "model.pth"
    torch.save({"metadata": metadata, "model_state_dict": model.state_dict()}, checkpoint)
    del model, optimizer, cache, logits, loss
    torch.cuda.empty_cache()

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    fresh = initialize(artifact_root / config["parents"]["model_config"], seed).cuda().eval()
    fresh.requires_grad_(False)
    fresh.load_state_dict(payload["model_state_dict"], strict=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        actual = fresh(pixel_values=pixels).logits.cpu()
    if not torch.equal(actual, replay) or payload["metadata"] != metadata:
        raise ValueError("Strict checkpoint reload mismatch")
    result = {
        "path": str(checkpoint),
        "sha256": sha256(checkpoint),
        "metadata": metadata,
        "reload_exact": True,
        "seconds": training_seconds,
        "peak_cuda_allocated_mib": peak_mib,
        "loss": {
            "first": losses[0],
            "last": losses[-1],
            "last100": float(np.mean(losses[-100:])),
        },
    }
    write_json(folder / "training_result.json", result)
    del payload, pixels, targets, boundary_tensor
    torch.cuda.empty_cache()
    return fresh, result


def evaluate(names, predictions, baseline, labels_root, config, fold_by_name):
    matrix = {arm: np.zeros((5, 5), np.int64) for arm in ("baseline", *ARMS)}
    fold_matrix = {
        fold: {arm: np.zeros((5, 5), np.int64) for arm in ("baseline", *ARMS)}
        for fold in sorted(set(fold_by_name.values()))
    }
    changes = {
        arm: {"added_foundation": 0, "removed_foundation": 0, "corrected": 0, "harmed": 0}
        for arm in ARMS
    }
    per_view = []
    access = []
    for name in names:
        ground_truth = png(labels_root / name)
        base = png(baseline[name][0])
        fold = fold_by_name[name]
        base_confusion = local_confusion(ground_truth, base)
        matrix["baseline"] += base_confusion
        fold_matrix[fold]["baseline"] += base_confusion
        row = {"name": name, "fold": fold, "baseline": confusion_metrics(base_confusion)}
        for arm in ARMS:
            record = predictions[arm][name]
            if sha256(Path(record["path"])) != record["sha256"]:
                raise ValueError("Sealed prediction changed")
            prediction = png(Path(record["path"]))
            if any(not np.array_equal(prediction == class_id, base == class_id) for class_id in (1, 2, 3)):
                raise ValueError("Protected class mask changed")
            confusion = local_confusion(ground_truth, prediction)
            matrix[arm] += confusion
            fold_matrix[fold][arm] += confusion
            row[arm] = confusion_metrics(confusion)
            changed = prediction != base
            masks = {
                "added_foundation": (base == 0) & (prediction == 4),
                "removed_foundation": (base == 4) & (prediction == 0),
                "corrected": changed & (prediction == ground_truth),
                "harmed": changed & (base == ground_truth),
            }
            for key, mask in masks.items():
                changes[arm][key] += int(mask.sum())
        per_view.append(row)
        access.append({"name": name, "label_sha256": sha256(labels_root / name)})

    metrics = {arm: confusion_metrics(value) for arm, value in matrix.items()}
    checks = quality_gates(metrics["baseline"], metrics[ARMS[1]], metrics[ARMS[0]], config["gates"])
    fold_metrics = {
        str(fold): {arm: confusion_metrics(value) for arm, value in values.items()}
        for fold, values in fold_matrix.items()
    }
    maximum_drop = float(config["gates"]["maximum_per_fold_iou_drop_vs_baseline"])
    fold_checks = {
        str(fold): values[ARMS[1]]["per_class"]["foundation"]["iou"]
        >= values["baseline"]["per_class"]["foundation"]["iou"] - maximum_drop
        for fold, values in ((key, {arm: confusion_metrics(value) for arm, value in val.items()}) for key, val in fold_matrix.items())
    }
    checks["per_fold_foundation_iou_safety"] = all(fold_checks.values())
    checks["protected_exact"] = True
    return {
        "metrics": metrics,
        "checks": checks,
        "fold_checks": fold_checks,
        "fold_metrics": fold_metrics,
        "passed": all(checks.values()),
        "changes": changes,
        "per_view": per_view,
        "target_access": access,
    }


def panels(names, predictions, baseline, rgb_paths, labels_root, output_dir):
    output_dir.mkdir()
    records = []
    titles = [
        "Frozen rendered RGB",
        "GT",
        "SEM-160 OOF",
        "Five-class CE control",
        "Boundary-loss candidate",
        "Green corrected / red harmed",
    ]
    for name in names:
        rgb = png(rgb_paths[name][0])
        ground_truth = png(labels_root / name)
        base = png(baseline[name][0])
        control = png(Path(predictions[ARMS[0]][name]["path"]))
        candidate = png(Path(predictions[ARMS[1]][name]["path"]))
        good = (candidate != base) & (candidate == ground_truth)
        bad = (candidate != base) & (base == ground_truth)
        change = np.zeros_like(rgb)
        change[good] = (40, 220, 80)
        change[bad] = (255, 50, 50)
        columns = [
            rgb,
            semantic_rgb(ground_truth),
            semantic_rgb(base),
            semantic_rgb(control),
            semantic_rgb(candidate),
            change,
        ]
        height, width = base.shape

        def save(tag, box, scale=1):
            x0, y0, x1, y1 = box
            column_width, column_height = (x1 - x0) * scale, (y1 - y0) * scale
            canvas = Image.new("RGB", (6 * column_width, column_height + 24), "white")
            draw = ImageDraw.Draw(canvas)
            for index, (column, title) in enumerate(zip(columns, titles)):
                crop = Image.fromarray(column[y0:y1, x0:x1]).resize(
                    (column_width, column_height), Image.Resampling.NEAREST
                )
                canvas.paste(crop, (index * column_width, 24))
                draw.text((index * column_width + 4, 4), name + " " + title, fill="black")
            path = output_dir / f"{Path(name).stem}_{tag}.png"
            canvas.save(path)
            records.append(
                {"path": str(path), "sha256": sha256(path), "crop_xyxy": list(box), "scale": scale}
            )

        save("panel", (0, 0, width, height))
        ys, xs = np.where((ground_truth == 4) | (base == 4) | (candidate == 4))
        if len(xs):
            save(
                "foundation_zoom",
                (
                    max(0, int(xs.min()) - 16),
                    max(0, int(ys.min()) - 16),
                    min(width, int(xs.max()) + 17),
                    min(height, int(ys.max()) + 17),
                ),
            )
        for tag, mask in (("corrected", good), ("harmed", bad)):
            boxes = [
                (x, y, min(x + 224, width), min(y + 128, height))
                for y in range(0, height - 127, 64)
                for x in range(0, width - 223, 112)
            ]
            save(tag + "_2x", max(boxes, key=lambda box: int(mask[box[1] : box[3], box[0] : box[2]].sum())), 2)
    write_json(output_dir / "manifest.json", {"columns": titles, "files": records})
    return records


def run(args, output_dir, started):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Visible bf16 CUDA is required")
    torch.set_num_threads(4)
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    config = read_json(args.config)
    config["_sha256"] = sha256(args.config)
    if config["experiment_id"] != EXP or tuple(config["arm_names"]) != ARMS:
        raise ValueError("Wrong experiment configuration")
    artifact_root = args.artifact_root.resolve()
    source_paths = [
        Path(__file__),
        ROOT / "src/semantic_foundation_boundary.py",
        ROOT / "src/semantic_foundation_binary.py",
        ROOT / "src/semantic_cable_binary.py",
        ROOT / "src/submission_sem161.py",
        ROOT / "src/semantic2d.py",
        ROOT / "src/semantic_protected_fusion.py",
        ROOT / "scripts/audit_sem260_error_budget.py",
        ROOT / "scripts/run_sem250_colmap_track_deck_diag.py",
    ]
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}
    (output_dir / "source_snapshot").mkdir()
    for path in (*source_paths, args.config):
        shutil.copyfile(path, output_dir / "source_snapshot" / path.name)

    freeze = verify_freeze(
        ROOT / "experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json", artifact_root
    )
    write_json(output_dir / "freeze_before.json", freeze)
    for key in ("semantic_split", "oof_split"):
        if sha256(ROOT / config[key]) != config[key + "_sha256"]:
            raise ValueError("Registered split changed")
    split = read_json(ROOT / config["oof_split"])
    all_names = read_json(ROOT / config["semantic_split"])["semantic_train_labeled"]
    graph = ROOT / config["camera_graph"]
    if sha256(graph) != split["camera_graph_sha256"]:
        raise ValueError("Camera graph changed")
    centers = {row["name"]: row["center"] for row in read_json(graph)["nodes"]}
    held = [name for fold in split["folds"] for name in fold["oof"]]
    if len(held) != len(set(held)) or set(held) != set(all_names):
        raise ValueError("OOF coverage changed")
    for fold in split["folds"]:
        validate_fold(fold, all_names, centers, split["guard_policy"]["radius"])
    fold_by_name = {
        name: int(fold["fold"]) for fold in split["folds"] for name in fold["oof"]
    }

    model_dir = artifact_root / config["parents"]["model_config"]
    config["_initialization"] = [
        {"path": str(path.relative_to(artifact_root)), "sha256": sha256(path)}
        for path in sorted(model_dir.iterdir())
        if path.suffix in (".json", ".bin", ".safetensors")
    ]
    baseline = manifest_paths(
        artifact_root / config["parents"]["sem160"],
        "prediction_manifest.json",
        "predictions",
        "prediction",
        "prediction_sha256",
    )
    rgb_paths = manifest_paths(
        artifact_root / config["parents"]["rgb"],
        "rendered_dataset_manifest.json",
        "rendered_images",
        "path",
        "sha256",
    )
    for mapping in (baseline, rgb_paths):
        if set(mapping) != set(all_names) or any(sha256(path) != digest for path, digest in mapping.values()):
            raise ValueError("Parent image universe/hash changed")
    labels_root = artifact_root / config["parents"]["labels"]
    predictions = {arm: {} for arm in ARMS}
    fold_rows = []
    active_folds = split["folds"][:1] if args.smoke else split["folds"]
    for fold in active_folds:
        fold_index = int(fold["fold"])
        begin = time.perf_counter()
        resource_gate(output_dir, config, started)
        plan = training_plan(
            sorted(fold["train"]),
            int(config["seed"]),
            fold_index,
            2 if args.smoke else int(config["training"]["iterations"]),
            int(config["training"]["crop"]),
        )
        plan_path = output_dir / f"fold{fold_index}_sampling_plan.json"
        write_json(plan_path, plan)
        row = {"fold": fold_index, "arms": {}}
        initial_hash = None
        for arm in ARMS:
            model, training = fit(
                output_dir,
                artifact_root,
                config,
                fold,
                arm,
                plan,
                rgb_paths,
                labels_root,
                args.smoke,
                started,
            )
            current_hash = training["metadata"]["initial_state_sha256"]
            if initial_hash is not None and current_hash != initial_hash:
                raise ValueError("Matched initial states differ")
            initial_hash = current_hash
            inference_started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            names = sorted(fold["oof"])[:1] if args.smoke else sorted(fold["oof"])
            for index, name in enumerate(names):
                rgb = png(rgb_paths[name][0])
                base = png(baseline[name][0])
                before = array_hash(rgb)
                probability = sliding(
                    model,
                    rgb,
                    crop=int(config["inference"]["crop"]),
                    stride=int(config["inference"]["stride"]),
                    classes=5,
                )[4]
                if not np.array_equal(fuse_foundation(base, probability, strength=0), base):
                    raise ValueError("Disabled equivalence failed")
                prediction = fuse_foundation(base, probability)
                if array_hash(rgb) != before or any(
                    not np.array_equal(prediction == class_id, base == class_id)
                    for class_id in (1, 2, 3)
                ):
                    raise ValueError("RGB/protected prediction changed")
                folder = output_dir / arm
                folder.mkdir(exist_ok=True)
                probability_path = folder / (Path(name).stem + ".npy")
                np.save(probability_path, probability.astype(np.float32))
                prediction_path = folder / name
                Image.fromarray(prediction).save(prediction_path)
                predictions[arm][name] = {
                    "path": str(prediction_path),
                    "sha256": sha256(prediction_path),
                    "probability_path": str(probability_path),
                    "probability_sha256": sha256(probability_path),
                    "fold": fold_index,
                }
                if index % 20 == 0 or index + 1 == len(names):
                    log(stage="inference", fold=fold_index, arm=arm, views=index + 1, total=len(names))
            torch.cuda.synchronize()
            row["arms"][arm] = {
                "training": training,
                "inference_seconds": time.perf_counter() - inference_started,
                "inference_peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "disabled_exact_views": len(names),
                "protected_exact": True,
                "rgb_exact": True,
            }
            del model
            torch.cuda.empty_cache()
        row["seconds"] = time.perf_counter() - begin
        fold_rows.append(row)
        write_json(output_dir / f"fold{fold_index}_result.json", row)

    write_json(output_dir / "prediction_manifest.json", predictions)
    seal = {
        "experiment_id": EXP,
        "config_sha256": config["_sha256"],
        "source_sha256": source_hashes,
        "prediction_manifest_sha256": sha256(output_dir / "prediction_manifest.json"),
        "folds": fold_rows,
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "scope": "Incremental OOF; inherited SEM-001 all219; reused calibration/check, not blind data",
    }
    write_json(output_dir / "prediction_seal.json", seal)
    if sha256(args.config) != config["_sha256"] or any(
        sha256(ROOT / path) != digest for path, digest in source_hashes.items()
    ):
        raise ValueError("Source/config changed during run")
    freeze_after = verify_freeze(
        ROOT / "experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json", artifact_root
    )
    if freeze_after != freeze:
        raise ValueError("Frozen artifacts changed")

    result = {
        "experiment_id": EXP,
        "status": "completed_smoke",
        "seed": config["seed"],
        "config_sha256": config["_sha256"],
        "prediction_seal_sha256": sha256(output_dir / "prediction_seal.json"),
        "freeze_before": freeze,
        "freeze_after": freeze_after,
        "folds": fold_rows,
        "check_quality_scored_this_run": False,
        "developer_quality_scored_this_run": False,
    }
    if not args.smoke:
        if any(set(predictions[arm]) != set(all_names) for arm in ARMS):
            raise ValueError("Incomplete OOF predictions")
        partition = partition_training_names(all_names, expected_count=219)
        calibration = evaluate(
            partition["calibration"], predictions, baseline, labels_root, config, fold_by_name
        )
        calibration["checks"]["rgb_exact"] = True
        calibration["passed"] = all(calibration["checks"].values())
        write_json(output_dir / "calibration_sealed.json", calibration)
        result["calibration"] = calibration
        result["calibration_sha256"] = sha256(output_dir / "calibration_sealed.json")
        result["status"] = "completed_calibration_gate_failed"
        if calibration["passed"]:
            check = evaluate(
                partition["check"], predictions, baseline, labels_root, config, fold_by_name
            )
            check["checks"]["rgb_exact"] = True
            check["passed"] = all(check["checks"].values())
            write_json(output_dir / "check_sealed.json", check)
            result["check"] = check
            result["check_sha256"] = sha256(output_dir / "check_sealed.json")
            result["check_quality_scored_this_run"] = True
            result["status"] = (
                "eligible_for_separate_final" if check["passed"] else "completed_check_gate_failed"
            )
        if not set(config["visual_samples"]).issubset(partition["calibration"]):
            raise ValueError("Registered panel view lies outside calibration")
        result["panels"] = panels(
            config["visual_samples"],
            predictions,
            baseline,
            rgb_paths,
            labels_root,
            output_dir / "panels",
        )
    result["seconds"] = time.perf_counter() - started
    result["environment"] = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    write_json(output_dir / "result.json", result)
    log(stage="completed", status=result["status"], seconds=result["seconds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/runs/SEM-280-FOUNDATION-BOUNDARY-OOF/config.yaml",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        run(args, output_dir, started)
    except Exception as error:
        write_json(
            output_dir / "failure.json",
            {
                "experiment_id": EXP,
                "error": repr(error),
                "seconds": time.perf_counter() - started,
                "status": "incomplete_not_quality_result",
            },
        )
        raise


if __name__ == "__main__":
    main()
