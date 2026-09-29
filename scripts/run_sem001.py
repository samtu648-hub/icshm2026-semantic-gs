#!/usr/bin/env python3
"""Train/evaluate the frozen-RGB Scaffold-GS SEM-001 semantic sidecar.

This runner never opens released RGB files.  It reads only frozen Scaffold-GS
artifacts, released COLMAP camera metadata and LabelMe-derived semantic labels.
The upstream RGB model is frozen by a parameter allow-list and independently
verified through artifact hashes and exact RGB render comparisons.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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

from render_external_colmap_camera import make_view, read_pose, read_simple_radial_camera  # noqa: E402
from src.camera_adapter import (  # noqa: E402
    Sim3,
    official_to_model_camera,
    redistort_rgb,
    redistort_semantic_probabilities,
)
from src.scaffold_semantic_sidecar import (  # noqa: E402
    AnchorSemanticSidecar,
    ScaffoldSettings,
    configure_scaffold_import,
    frozen_rgb_hashes,
    frozen_rgb_trainable_parameters,
    load_frozen_scaffold,
    pipeline,
    render_frozen_rgb,
    render_semantic_probabilities,
)


CLASS_NAMES = ("background", "deck", "stay_cable", "tower", "foundation")
CLASS_COUNT = len(CLASS_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--rgb-run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Git-ignored experiment output root")
    parser.add_argument("--mode", choices=("preflight", "full"), required=True)
    parser.add_argument("--iterations", type=int, help="Explicit registered-budget override only")
    parser.add_argument("--eval-limit", type=int, help="Limit validation only for a preflight")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def image_hash(array: np.ndarray, path: Path) -> str:
    image = Image.fromarray(array, mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_label(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path), dtype=np.uint8)
    if array.ndim != 2 or array.shape != (989, 1320):
        raise ValueError(f"{path}: expected single-channel 1320x989 label, got {array.shape}")
    if int(array.max()) >= CLASS_COUNT:
        raise ValueError(f"{path}: contains class ID outside 0..{CLASS_COUNT - 1}")
    return array


def class_weights(label_manifest: dict[str, Any]) -> np.ndarray:
    counts = np.array([float(label_manifest["pixel_counts_train"].get(str(index), 0)) for index in range(CLASS_COUNT)])
    if np.any(counts <= 0):
        raise ValueError(f"Semantic training class has no pixels: {counts.tolist()}")
    # Median-frequency balancing makes rare but official classes audible while
    # retaining some background pressure.  Bounds are part of SEM-001 config.
    weights = np.median(counts) / counts
    return np.clip(weights, 0.05, 20.0).astype(np.float32)


def official_view(data_root: Path, image_name: str, appearance_camera_id: int):
    cameras = data_root / "camera_parameters" / "cameras.txt"
    images = data_root / "camera_parameters" / "images.txt"
    camera_id, camera = read_simple_radial_camera(cameras, None)
    selected_name, pose = read_pose(images, image_name, camera_id)
    identity = Sim3(1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
    return selected_name, camera, make_view(camera, official_to_model_camera(pose, identity), appearance_camera_id)


def tensor_to_rgb(array: Any) -> np.ndarray:
    return np.rint(array.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)


def save_checkpoint(path: Path, sidecar: AnchorSemanticSidecar, config: dict[str, Any], rgb_hashes: dict[str, str]) -> None:
    import torch

    payload = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "class_names": list(CLASS_NAMES),
        "semantic_storage": "per_anchor_logits",
        "frozen_rgb_hashes": rgb_hashes,
        "semantic_anchor_logits": sidecar.logits.detach().cpu(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def reload_sidecar(path: Path, anchor_count: int, config: dict[str, Any], rgb_hashes: dict[str, str]) -> AnchorSemanticSidecar:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("experiment_id") != config["experiment_id"]:
        raise ValueError("Semantic checkpoint belongs to a different experiment")
    if payload.get("class_names") != list(CLASS_NAMES) or payload.get("semantic_storage") != "per_anchor_logits":
        raise ValueError("Semantic checkpoint class/storage schema mismatch")
    if payload.get("frozen_rgb_hashes") != rgb_hashes:
        raise ValueError("Semantic checkpoint frozen-RGB hashes do not match loaded RGB artifact")
    sidecar = AnchorSemanticSidecar(anchor_count)
    sidecar.load_state_dict(payload)
    return sidecar


def render_external(
    output_dir: Path,
    data_root: Path,
    image_name: str,
    appearance_camera_id: int,
    model: Any,
    sidecar: AnchorSemanticSidecar,
    pipe: Any,
) -> dict[str, Any]:
    selected, camera, view = official_view(data_root, image_name, appearance_camera_id)
    rgb = tensor_to_rgb(render_frozen_rgb(view, model, pipe))
    probabilities = render_semantic_probabilities(view, model, sidecar, pipe).detach().cpu().numpy().astype(np.float32)
    official_rgb = redistort_rgb(rgb.astype(np.float32) / 255.0, camera)
    official_probabilities = redistort_semantic_probabilities(probabilities, camera)
    logits = np.log(np.clip(official_probabilities, np.finfo(np.float32).tiny, 1.0))
    labels = logits.argmax(axis=0).astype(np.uint8)
    if labels.shape != (camera.height, camera.width) or labels.max() >= CLASS_COUNT:
        raise RuntimeError("External semantic output is not a valid five-class image")
    root = output_dir / "external_camera"
    root.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(official_rgb.clip(0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB").save(root / "rgb.png")
    Image.fromarray(labels, mode="L").save(root / "semantic.png")
    np.save(root / "semantic_logits.npy", logits)
    return {
        "selected_pose_name": selected,
        "input_pose_convention": "CAMERA_FROM_WORLD",
        "official_to_model_sim3": {"scale": 1.0, "rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0]},
        "input_camera_model": "SIMPLE_RADIAL",
        "output_grid": "official_SIMPLE_RADIAL",
        "rgb_shape": list(official_rgb.shape),
        "semantic_logits_shape": list(logits.shape),
        "semantic_png_shape": list(labels.shape),
        "semantic_png_dtype": str(labels.dtype),
        "semantic_png_min": int(labels.min()),
        "semantic_png_max": int(labels.max()),
        "uses_test_rgb_or_labels": False,
    }


def RGB_isolation_evidence(
    output_dir: Path,
    data_root: Path,
    views: list[str],
    appearance_camera_id: int,
    model: Any,
    pipe: Any,
    rgb_hashes_before: dict[str, str],
    rgb_run_dir: Path,
    iteration: int,
) -> dict[str, Any]:
    before: dict[str, np.ndarray] = {}
    png_hashes_before: dict[str, str] = {}
    for name in views:
        _, _, view = official_view(data_root, name, appearance_camera_id)
        before[name] = tensor_to_rgb(render_frozen_rgb(view, model, pipe))
        png_hashes_before[name] = image_hash(before[name], output_dir / "rgb_isolation" / "before" / name)
    # Called after semantic optimization: model still references only frozen RGB tensors.
    png_hashes_after: dict[str, str] = {}
    max_difference = 0
    for name in views:
        _, _, view = official_view(data_root, name, appearance_camera_id)
        after = tensor_to_rgb(render_frozen_rgb(view, model, pipe))
        max_difference = max(max_difference, int(np.abs(before[name].astype(np.int16) - after.astype(np.int16)).max()))
        png_hashes_after[name] = image_hash(after, output_dir / "rgb_isolation" / "after" / name)
    rgb_hashes_after = frozen_rgb_hashes(rgb_run_dir, iteration)
    evidence = {
        "views": views,
        "rgb_optimizer_parameter_allowlist": ["semantic_anchor_logits"],
        "frozen_model_trainable_parameters": frozen_rgb_trainable_parameters(model),
        "rgb_artifact_hashes_before": rgb_hashes_before,
        "rgb_artifact_hashes_after": rgb_hashes_after,
        "rgb_artifact_hashes_equal": rgb_hashes_before == rgb_hashes_after,
        "rgb_png_sha256_before": png_hashes_before,
        "rgb_png_sha256_after": png_hashes_after,
        "rgb_png_hashes_equal": png_hashes_before == png_hashes_after,
        "rgb_max_absolute_pixel_difference": max_difference,
    }
    if evidence["frozen_model_trainable_parameters"] or not evidence["rgb_artifact_hashes_equal"] or max_difference != 0:
        raise RuntimeError("Frozen RGB isolation gate failed")
    return evidence


def confusion_metrics(confusion: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {"per_class": {}}
    ious: list[float] = []
    for index, name in enumerate(CLASS_NAMES):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        denominator = tp + fp + fn
        iou = float(tp / denominator) if denominator else None
        precision = float(tp / (tp + fp)) if tp + fp else None
        recall = float(tp / (tp + fn)) if tp + fn else None
        result["per_class"][name] = {"tp": tp, "fp": fp, "fn": fn, "IoU": iou, "precision": precision, "recall": recall}
        if iou is not None:
            ious.append(iou)
    result["mIoU_with_background"] = float(np.mean(ious))
    non_background = [result["per_class"][name]["IoU"] for name in CLASS_NAMES[1:] if result["per_class"][name]["IoU"] is not None]
    result["mIoU_without_background"] = float(np.mean(non_background))
    background_total = int(confusion[0, :].sum())
    background_to_stay_cable = int(confusion[0, 2])
    result["background_to_stay_cable_false_positive_pixels"] = background_to_stay_cable
    result["background_to_stay_cable_false_positive_fraction_of_gt_background"] = (
        float(background_to_stay_cable / background_total) if background_total else None
    )
    result["stay_cable"] = result["per_class"]["stay_cable"]
    return result


def evaluate(
    output_dir: Path,
    names: list[str],
    labels_root: Path,
    data_root: Path,
    appearance_camera_id: int,
    model: Any,
    sidecar: AnchorSemanticSidecar,
    pipe: Any,
) -> dict[str, Any]:
    import torch

    combined = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    per_view: list[dict[str, Any]] = []
    render_root = output_dir / "validation_semantic_png"
    render_root.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = read_label(labels_root / name)
        _, _camera, view = official_view(data_root, name, appearance_camera_id)
        with torch.no_grad():
            probabilities = render_semantic_probabilities(view, model, sidecar, pipe)
        prediction = probabilities.argmax(dim=0).cpu().numpy().astype(np.uint8)
        Image.fromarray(prediction, mode="L").save(render_root / name)
        local = np.bincount(CLASS_COUNT * target.reshape(-1).astype(np.int64) + prediction.reshape(-1), minlength=CLASS_COUNT**2)
        local = local.reshape(CLASS_COUNT, CLASS_COUNT)
        combined += local
        metrics = confusion_metrics(local)
        per_view.append({"view": name, **metrics})
    metrics = confusion_metrics(combined)
    worst_count = max(1, int(np.ceil(len(per_view) * 0.10)))
    worst = sorted(per_view, key=lambda record: record["mIoU_without_background"])[:worst_count]
    difficult = [record for record in per_view if 140 <= int(Path(record["view"]).stem) <= 146]
    metrics.update(
        {
            "view_count": len(names),
            "confusion_target_rows_prediction_columns": combined.tolist(),
            "per_view": per_view,
            "worst_10_percent": {"count": worst_count, "views": [item["view"] for item in worst], "mean_mIoU_without_background": float(np.mean([item["mIoU_without_background"] for item in worst]))},
            "challenge_140_146": {
                "views": [item["view"] for item in difficult],
                "mean_mIoU_without_background": float(np.mean([item["mIoU_without_background"] for item in difficult])) if difficult else None,
            },
            "local_protocol_note": "This is a fixed local full-image confusion-matrix protocol. Organizers have not yet released final mIoU aggregation/background rules.",
        }
    )
    with (output_dir / "validation_per_view.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["view", "mIoU_with_background", "mIoU_without_background", "stay_cable_IoU", "stay_cable_precision", "stay_cable_recall", "background_to_stay_cable_false_positive_pixels", "background_to_stay_cable_false_positive_fraction_of_gt_background"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in per_view:
            stay = record["stay_cable"]
            writer.writerow({
                "view": record["view"],
                "mIoU_with_background": record["mIoU_with_background"],
                "mIoU_without_background": record["mIoU_without_background"],
                "stay_cable_IoU": stay["IoU"],
                "stay_cable_precision": stay["precision"],
                "stay_cable_recall": stay["recall"],
                "background_to_stay_cable_false_positive_pixels": record["background_to_stay_cable_false_positive_pixels"],
                "background_to_stay_cable_false_positive_fraction_of_gt_background": record["background_to_stay_cable_false_positive_fraction_of_gt_background"],
            })
    return metrics


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    split = load_json(REPO_ROOT / config["data"]["semantic_split"])
    manifest = load_json(REPO_ROOT / config["data"]["label_manifest"])
    if split["split_id"] != config["data"]["split_id"] or manifest["split_id"] != split["split_id"]:
        raise ValueError("Config, semantic split and label manifest do not refer to the same SEM-001 protocol")
    if manifest["class_mapping"] != {name: index for index, name in enumerate(CLASS_NAMES)}:
        raise ValueError("Label manifest class mapping is not the official SEM-001 mapping")
    if args.mode == "full" and args.eval_limit is not None:
        raise ValueError("A SEM-001 full run must evaluate all 74 validation views")
    budget = config[args.mode]
    iterations = args.iterations if args.iterations is not None else int(budget["iterations"])
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if args.iterations is not None and args.iterations != int(budget["iterations"]):
        raise ValueError("Iteration override differs from the registered budget; create a new experiment config instead")
    if not args.labels_root.is_dir():
        raise FileNotFoundError(f"Missing external SEM-001 labels: {args.labels_root}")

    import torch

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.manual_seed(int(config["training"]["seed"]))
    np.random.seed(int(config["training"]["seed"]))
    random.seed(int(config["training"]["seed"]))
    torch.cuda.reset_peak_memory_stats()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "invocation.json", {"config": str(args.config), "mode": args.mode, "iterations": iterations, "git_status_recorded_in_report": True})

    settings = ScaffoldSettings(train_camera_count=int(config["frozen_rgb"]["train_camera_count"]), n_offsets=int(config["frozen_rgb"]["n_offsets"]))
    point_cloud = args.rgb_run_dir / "point_cloud" / f"iteration_{config['frozen_rgb']['iteration']}" / "point_cloud.ply"
    rgb_hashes_before = frozen_rgb_hashes(args.rgb_run_dir, int(config["frozen_rgb"]["iteration"]))
    configure_scaffold_import(args.source_root)
    model = load_frozen_scaffold(args.source_root, point_cloud, settings)
    sidecar = AnchorSemanticSidecar(int(model.get_anchor.shape[0]), initial_background_logit=float(config["training"]["initial_background_logit"]))
    optimizer = torch.optim.AdamW([sidecar.logits], lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    weights = torch.tensor(class_weights(manifest), device="cuda")
    pipe = pipeline()
    train_names = list(split["semantic_train_labeled"])
    sampler = np.random.default_rng(int(config["training"]["seed"]))
    losses: list[float] = []
    started = time.perf_counter()
    for step in range(1, iterations + 1):
        name = train_names[int(sampler.integers(len(train_names)))]
        target = torch.from_numpy(read_label(args.labels_root / name).astype(np.int64, copy=False)).to(device="cuda", non_blocking=True)
        _, _camera, view = official_view(args.data_root, name, int(config["frozen_rgb"]["appearance_camera_id"]))
        probabilities = render_semantic_probabilities(view, model, sidecar, pipe)
        loss = torch.nn.functional.nll_loss(torch.log(probabilities.clamp_min(1e-8)).unsqueeze(0), target.unsqueeze(0), weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step == 1 or step % int(budget["log_every"]) == 0 or step == iterations:
            print(json.dumps({"step": step, "loss": losses[-1], "mean_recent_loss": float(np.mean(losses[-min(len(losses), int(budget["log_every"])):])), "max_memory_allocated_mib": int(torch.cuda.max_memory_allocated() / 1024**2)}, sort_keys=True), flush=True)

    training_seconds = time.perf_counter() - started
    isolation = RGB_isolation_evidence(
        args.output_dir,
        args.data_root,
        list(config["isolation"]["views"]),
        int(config["frozen_rgb"]["appearance_camera_id"]),
        model,
        pipe,
        rgb_hashes_before,
        args.rgb_run_dir,
        int(config["frozen_rgb"]["iteration"]),
    )
    checkpoint = args.output_dir / "checkpoints" / "semantic_sidecar.pth"
    save_checkpoint(checkpoint, sidecar, config, rgb_hashes_before)
    reloaded = reload_sidecar(checkpoint, int(model.get_anchor.shape[0]), config, rgb_hashes_before)
    with torch.no_grad():
        if not torch.equal(sidecar.logits, reloaded.logits):
            raise RuntimeError("Saved/reloaded semantic logits are not bit-identical")
    external = render_external(
        args.output_dir,
        args.data_root,
        str(config["external_camera"]["image_name"]),
        int(config["frozen_rgb"]["appearance_camera_id"]),
        model,
        reloaded,
        pipe,
    )
    validation = list(split["semantic_validation_labeled"])
    if args.eval_limit is not None:
        validation = validation[: args.eval_limit]
    evaluation = evaluate(
        args.output_dir,
        validation,
        args.labels_root,
        args.data_root,
        int(config["frozen_rgb"]["appearance_camera_id"]),
        model,
        reloaded,
        pipe,
    )
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "mode": args.mode,
        "status": "completed" if args.mode == "full" else "completed_preflight_not_a_semantic_baseline",
        "iterations": iterations,
        "seed": config["training"]["seed"],
        "train_views": len(train_names),
        "validation_views": len(validation),
        "class_weights": {name: float(value) for name, value in zip(CLASS_NAMES, weights.cpu().numpy())},
        "loss": {"first": losses[0], "last": losses[-1], "mean": float(np.mean(losses))},
        "training_seconds": training_seconds,
        "peak_cuda_allocated_mib": int(torch.cuda.max_memory_allocated() / 1024**2),
        "checkpoint_reloaded": True,
        "rgb_isolation": isolation,
        "external_camera": external,
        "evaluation": evaluation,
    }
    write_json(args.output_dir / "result.json", result)
    print(json.dumps({"status": result["status"], "training_seconds": training_seconds, "peak_cuda_allocated_mib": result["peak_cuda_allocated_mib"], "mIoU_without_background": evaluation["mIoU_without_background"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
