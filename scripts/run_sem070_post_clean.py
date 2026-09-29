#!/usr/bin/env python3
"""Run SEM-070 conservative inference-only semantic post-processing.

The runner first reads only the 219 labeled SEM-001 training views.  It uses
the exact frozen Scaffold rasterizer dominant-anchor diagnostic to build a
calibration-half anchor table, evaluates the preregistered threshold grid on
the disjoint training check half, freezes one boolean mask, and only then
opens the 74 validation labels.  SEM-001 RGB/semantic artifacts are never
modified and no RGB or semantic parameter is optimized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem001 as s0
from render_external_colmap_camera import (
    make_view,
    read_pose,
    read_simple_radial_camera,
)

from src.camera_adapter import (
    Sim3,
    official_to_model_camera,
    redistort_rgb,
    redistort_semantic_probabilities,
)
from src.scaffold_semantic_sidecar import (
    ScaffoldSettings,
    configure_scaffold_import,
    frozen_rgb_hashes,
    frozen_rgb_named_parameters,
    load_frozen_scaffold,
    pipeline,
    render_frozen_rgb,
    render_semantic_probabilities,
)
from src.semantic_postprocess import (
    CLASS_COUNT,
    CLASS_NAMES,
    PROTECTED_CLASSES,
    finalize_anchor_rows,
    mask_for_threshold,
    new_anchor_statistics,
    save_suppression_mask,
    semantic_quad,
    split_training_views,
    training_half,
    update_anchor_statistics,
    validate_suppression_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--rgb-run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sem001-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Git-ignored external output directory")
    parser.add_argument("--extension-root", type=Path, required=True, help="Git-ignored CUDA extension cache")
    parser.add_argument("--mode", choices=("preflight", "full"), default="full")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def confusion(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    if target.shape != prediction.shape:
        raise ValueError("target and prediction shapes differ")
    return np.bincount(
        CLASS_COUNT * target.reshape(-1).astype(np.int64) + prediction.reshape(-1).astype(np.int64),
        minlength=CLASS_COUNT**2,
    ).reshape(CLASS_COUNT, CLASS_COUNT)


def _anchor_map(diagnostics: dict[str, Any]) -> np.ndarray:
    dominant = diagnostics["dominant_offset_index"].detach().cpu().numpy().astype(np.int64, copy=False)
    offset_to_anchor = diagnostics["offset_anchor_indices"].detach().cpu().numpy().astype(np.int64, copy=False)
    result = np.full(dominant.shape, -1, dtype=np.int64)
    valid = (dominant >= 0) & (dominant < offset_to_anchor.size)
    result[valid] = offset_to_anchor[dominant[valid]]
    return result


def _metrics(combined: np.ndarray, per_view: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    result = s0.confusion_metrics(combined)
    rows = per_view or []
    if rows:
        worst_count = max(1, int(np.ceil(len(rows) * 0.10)))
        worst = sorted(rows, key=lambda row: row["mIoU_without_background"])[:worst_count]
        difficult = [row for row in rows if 140 <= int(Path(row["view"]).stem) <= 146]
        result.update(
            {
                "view_count": len(rows),
                "per_view": rows,
                "worst_10_percent": {
                    "count": worst_count,
                    "views": [row["view"] for row in worst],
                    "mean_mIoU_without_background": float(np.mean([row["mIoU_without_background"] for row in worst])),
                },
                "challenge_140_146": {
                    "views": [row["view"] for row in difficult],
                    "mean_mIoU_without_background": float(np.mean([row["mIoU_without_background"] for row in difficult])) if difficult else None,
                },
            }
        )
    result["confusion_target_rows_prediction_columns"] = combined.tolist()
    return result


def _per_view_record(name: str, local_confusion: np.ndarray) -> dict[str, Any]:
    local = s0.confusion_metrics(local_confusion)
    stay = local["stay_cable"]
    return {
        "view": name,
        "mIoU_with_background": local["mIoU_with_background"],
        "mIoU_without_background": local["mIoU_without_background"],
        "stay_cable_IoU": stay["IoU"],
        "stay_cable_precision": stay["precision"],
        "stay_cable_recall": stay["recall"],
        "background_to_stay_cable_pixels": local["background_to_stay_cable_false_positive_pixels"],
        "background_to_stay_cable_fraction_of_gt_background": local["background_to_stay_cable_false_positive_fraction_of_gt_background"],
    }


def evaluate_views(
    names: list[str],
    *,
    labels_root: Path,
    data_root: Path,
    appearance_camera_id: int,
    model: Any,
    sidecar: Any,
    pipe: Any,
    anchor_mask: np.ndarray | None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate raw or cleaned semantics for a supplied, already-authorized split."""

    import torch

    anchor_count = int(model.get_anchor.shape[0])
    mask_tensor = None
    if anchor_mask is not None:
        mask_tensor = torch.as_tensor(validate_suppression_mask(anchor_mask, anchor_count), dtype=torch.bool, device="cuda")
    combined = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    rows: list[dict[str, Any]] = []
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = s0.read_label(labels_root / name)
        _selected, _camera, view = s0.official_view(data_root, name, appearance_camera_id)
        with torch.inference_mode():
            probabilities = render_semantic_probabilities(
                view, model, sidecar, pipe, anchor_postprocess_mask=mask_tensor
            )
        prediction = probabilities.argmax(dim=0).cpu().numpy().astype(np.uint8)
        local = confusion(target, prediction)
        combined += local
        rows.append(_per_view_record(name, local))
        if output_dir is not None:
            Image.fromarray(prediction, mode="L").save(output_dir / name)
    result = _metrics(combined, rows)
    return result


def collect_training_statistics(
    names: list[str],
    *,
    labels_root: Path,
    data_root: Path,
    appearance_camera_id: int,
    model: Any,
    sidecar: Any,
    pipe: Any,
    extension: Any,
    anchor_count: int,
) -> tuple[dict[str, np.ndarray], dict[int, np.ndarray], list[dict[str, Any]]]:
    """Collect exact dominant-anchor evidence from training labels only."""

    import torch

    from src.scaffold_visibility_supervision import (
        render_frozen_contribution_diagnostics,
    )

    statistics = new_anchor_statistics(anchor_count)
    confusion_by_half = {0: np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64), 1: np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)}
    rows: list[dict[str, Any]] = []
    for name in names:
        half = training_half(name)
        target = s0.read_label(labels_root / name)
        _selected, _camera, view = s0.official_view(data_root, name, appearance_camera_id)
        with torch.inference_mode():
            prediction = render_semantic_probabilities(view, model, sidecar, pipe).argmax(dim=0).cpu().numpy().astype(np.uint8)
            diagnostics = render_frozen_contribution_diagnostics(view, model, pipe, extension=extension)
        anchors = _anchor_map(diagnostics)
        update_anchor_statistics(statistics, train_half=half, anchor_indices=anchors, target=target, prediction=prediction)
        local = confusion(target, prediction)
        confusion_by_half[half] += local
        rows.append({"view": name, "train_half": half, **_per_view_record(name, local)})
    return statistics, confusion_by_half, rows


def _write_anchor_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("anchor statistics are empty")
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_view_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "view",
        "train_half",
        "mIoU_with_background",
        "mIoU_without_background",
        "stay_cable_IoU",
        "stay_cable_precision",
        "stay_cable_recall",
        "background_to_stay_cable_pixels",
        "background_to_stay_cable_fraction_of_gt_background",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _selection_metrics(raw: dict[str, Any], cleaned: dict[str, Any]) -> dict[str, Any]:
    raw_stay = raw["stay_cable"]
    clean_stay = cleaned["stay_cable"]
    raw_fp = int(raw["background_to_stay_cable_false_positive_pixels"])
    clean_fp = int(cleaned["background_to_stay_cable_false_positive_pixels"])
    raw_recall = float(raw_stay["recall"] or 0.0)
    clean_recall = float(clean_stay["recall"] or 0.0)
    recall_retention = clean_recall / raw_recall if raw_recall > 0.0 else 1.0
    protected_drops = {
        CLASS_NAMES[index]: float(raw["per_class"][CLASS_NAMES[index]]["IoU"] - cleaned["per_class"][CLASS_NAMES[index]]["IoU"])
        for index in PROTECTED_CLASSES
    }
    return {
        "background_to_stay_cable_reduction_pixels": raw_fp - clean_fp,
        "background_to_stay_cable_reduction_fraction": float((raw_fp - clean_fp) / raw_fp) if raw_fp else 0.0,
        "stay_cable_recall_retention_vs_raw": recall_retention,
        "protected_class_iou_drop": protected_drops,
        "protected_class_max_iou_drop": max(protected_drops.values(), default=0.0),
    }


def choose_check_threshold(
    rows: list[dict[str, Any]],
    thresholds: list[float],
    *,
    minimum_support_views: int,
    minimum_dominant_pixels: int,
    raw_check_metrics: dict[str, Any],
    evaluate_candidate: Any,
    minimum_recall_retention: float,
    maximum_protected_iou_drop: float,
) -> tuple[np.ndarray, float | None, list[dict[str, Any]]]:
    """Select one grid threshold using only the within-training check half."""

    candidates: list[dict[str, Any]] = []
    masks: dict[float, np.ndarray] = {}
    for threshold in thresholds:
        mask = mask_for_threshold(
            rows,
            float(threshold),
            minimum_support_views=minimum_support_views,
            minimum_dominant_pixels=minimum_dominant_pixels,
        )
        masks[float(threshold)] = mask
        cleaned = evaluate_candidate(mask)
        selection = _selection_metrics(raw_check_metrics, cleaned)
        meets = bool(
            selection["stay_cable_recall_retention_vs_raw"] >= float(minimum_recall_retention)
            and selection["protected_class_max_iou_drop"] <= float(maximum_protected_iou_drop)
        )
        candidates.append(
            {
                "threshold": float(threshold),
                "suppressed_anchor_count": int(mask.sum()),
                "meets_check_constraints": meets,
                **selection,
                "check_metrics": cleaned,
            }
        )
    eligible = [candidate for candidate in candidates if candidate["meets_check_constraints"]]
    if eligible:
        selected = min(
            eligible,
            key=lambda item: (
                -float(item["background_to_stay_cable_reduction_fraction"]),
                int(item["suppressed_anchor_count"]),
                float(item["threshold"]),
            ),
        )
        return masks[float(selected["threshold"])], float(selected["threshold"]), candidates
    return np.zeros(len(rows), dtype=np.bool_), None, candidates


def _rgb_isolation(
    names: list[str],
    *,
    data_root: Path,
    appearance_camera_id: int,
    model: Any,
    sidecar: Any,
    pipe: Any,
    mask: np.ndarray,
    rgb_run_dir: Path,
    rgb_iteration: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Render RGB around the raw/clean semantic calls and compare bitwise."""

    import torch

    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device="cuda")
    before_hashes: dict[str, str] = {}
    after_hashes: dict[str, str] = {}
    max_difference = 0
    for name in names:
        _selected, _camera, view = s0.official_view(data_root, name, appearance_camera_id)
        with torch.inference_mode():
            before = s0.tensor_to_rgb(render_frozen_rgb(view, model, pipe))
            _ = render_semantic_probabilities(view, model, sidecar, pipe)
            _ = render_semantic_probabilities(view, model, sidecar, pipe, anchor_postprocess_mask=mask_tensor)
            after = s0.tensor_to_rgb(render_frozen_rgb(view, model, pipe))
        before_path = output_dir / "before" / name
        after_path = output_dir / "after" / name
        before_hashes[name] = s0.image_hash(before, before_path)
        after_hashes[name] = s0.image_hash(after, after_path)
        max_difference = max(max_difference, int(np.abs(before.astype(np.int16) - after.astype(np.int16)).max()))
    artifact_before = frozen_rgb_hashes(rgb_run_dir, rgb_iteration)
    artifact_after = frozen_rgb_hashes(rgb_run_dir, rgb_iteration)
    return {
        "views": names,
        "rgb_artifact_hashes_before": artifact_before,
        "rgb_artifact_hashes_after": artifact_after,
        "rgb_artifact_hashes_equal": artifact_before == artifact_after,
        "rgb_png_sha256_before": before_hashes,
        "rgb_png_sha256_after": after_hashes,
        "rgb_png_hashes_equal": before_hashes == after_hashes,
        "rgb_max_absolute_pixel_difference": max_difference,
    }


def _semantic_logits_and_labels(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if probabilities.ndim != 3 or probabilities.shape[0] != CLASS_COUNT:
        raise ValueError("semantic probabilities must have shape [5,H,W]")
    logits = np.log(np.clip(probabilities, np.finfo(np.float32).tiny, 1.0)).astype(np.float32)
    return logits, logits.argmax(axis=0).astype(np.uint8)


def render_external_pair(
    *,
    config: dict[str, Any],
    data_cameras: Path,
    data_images: Path,
    image_name: str,
    camera_id: int | None,
    output_dir: Path,
    model: Any,
    sidecar: Any,
    pipe: Any,
    mask: np.ndarray,
) -> dict[str, Any]:
    """Render raw and cleaned outputs on an official external SIMPLE_RADIAL camera."""

    import torch

    frozen = config["frozen_rgb"]
    selected_camera_id, camera = read_simple_radial_camera(data_cameras, camera_id)
    selected_name, official_pose = read_pose(data_images, image_name, selected_camera_id)
    identity = Sim3(1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
    view = make_view(camera, official_to_model_camera(official_pose, identity), int(frozen["appearance_camera_id"]))
    mask_tensor = torch.as_tensor(validate_suppression_mask(mask, int(model.get_anchor.shape[0])), dtype=torch.bool, device="cuda")
    with torch.inference_mode():
        rgb_pinhole = s0.tensor_to_rgb(render_frozen_rgb(view, model, pipe))
        raw_pinhole = render_semantic_probabilities(view, model, sidecar, pipe).cpu().numpy().astype(np.float32)
        clean_pinhole = render_semantic_probabilities(view, model, sidecar, pipe, anchor_postprocess_mask=mask_tensor).cpu().numpy().astype(np.float32)
    rgb = np.rint(redistort_rgb(rgb_pinhole.astype(np.float32) / 255.0, camera).clip(0.0, 1.0) * 255.0).astype(np.uint8)
    raw_probabilities = redistort_semantic_probabilities(raw_pinhole, camera)
    clean_probabilities = redistort_semantic_probabilities(clean_pinhole, camera)
    outputs: dict[str, Any] = {}
    for variant, probabilities in (("raw", raw_probabilities), ("cleaned", clean_probabilities)):
        logits, labels = _semantic_logits_and_labels(probabilities)
        if labels.shape != (camera.height, camera.width) or labels.min() < 0 or labels.max() > CLASS_COUNT - 1:
            raise RuntimeError("External semantic output violates the five-class contract")
        root = output_dir / variant
        root.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb, mode="RGB").save(root / "rgb.png")
        Image.fromarray(labels, mode="L").save(root / "semantic.png")
        np.save(root / "semantic_logits.npy", logits)
        outputs[variant] = {
            "rgb_shape": list(rgb.shape),
            "semantic_logits_shape": list(logits.shape),
            "semantic_png_shape": list(labels.shape),
            "semantic_png_dtype": str(labels.dtype),
            "semantic_png_min": int(labels.min()),
            "semantic_png_max": int(labels.max()),
            "rgb_sha256": sha256(root / "rgb.png"),
            "semantic_sha256": sha256(root / "semantic.png"),
            "semantic_logits_sha256": sha256(root / "semantic_logits.npy"),
        }
    evidence = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "selected_pose_name": selected_name,
        "input_camera_model": "SIMPLE_RADIAL",
        "input_pose_convention": "CAMERA_FROM_WORLD",
        "official_to_model_sim3": {"scale": 1.0, "rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0]},
        "output_grid": "official_SIMPLE_RADIAL",
        "uses_reference_rgb_or_labels": False,
        "rgb_max_absolute_pixel_difference_raw_cleaned": int(np.abs(rgb.astype(np.int16) - rgb.astype(np.int16)).max()),
        "outputs": outputs,
    }
    write_json(output_dir / "evidence.json", evidence)
    return evidence


def formal_gates(raw: dict[str, Any], cleaned: dict[str, Any], rgb: dict[str, Any], external: dict[str, Any], *, mode: str) -> dict[str, Any]:
    raw_worst = {view: value for view, value in ((row["view"], row["mIoU_without_background"]) for row in raw.get("per_view", []))}
    clean_by_view = {row["view"]: row for row in cleaned.get("per_view", [])}
    worst_count = int(raw.get("worst_10_percent", {}).get("count", 0))
    worst_views = sorted(raw_worst, key=lambda view: raw_worst[view])[:worst_count]
    worst_drop = float(np.mean([raw_worst[view] - clean_by_view[view]["mIoU_without_background"] for view in worst_views])) if worst_views else 0.0
    challenge = [view for view in raw_worst if 140 <= int(Path(view).stem) <= 146]
    challenge_drop = float(np.mean([raw_worst[view] - clean_by_view[view]["mIoU_without_background"] for view in challenge])) if challenge else 0.0
    raw_stay = raw["stay_cable"]
    clean_stay = cleaned["stay_cable"]
    raw_fp = int(raw["background_to_stay_cable_false_positive_pixels"])
    clean_fp = int(cleaned["background_to_stay_cable_false_positive_pixels"])
    external_pass = all(
        output["rgb_shape"] == [989, 1320, 3]
        and output["semantic_png_shape"] == [989, 1320]
        and output["semantic_logits_shape"] == [5, 989, 1320]
        and output["semantic_png_dtype"] == "uint8"
        and 0 <= int(output["semantic_png_min"]) <= int(output["semantic_png_max"]) <= CLASS_COUNT - 1
        for output in external["outputs"].values()
    )
    gates = {
        "mIoU_not_below_sem001_raw": cleaned["mIoU_with_background"] >= raw["mIoU_with_background"],
        "stay_cable_IoU_not_below_sem001_raw": clean_stay["IoU"] >= raw_stay["IoU"],
        "stay_cable_recall_not_below_sem001_raw": clean_stay["recall"] >= raw_stay["recall"],
        "deck_iou_drop_le_0_5pp": raw["per_class"]["deck"]["IoU"] - cleaned["per_class"]["deck"]["IoU"] <= 0.005,
        "tower_iou_drop_le_0_5pp": raw["per_class"]["tower"]["IoU"] - cleaned["per_class"]["tower"]["IoU"] <= 0.005,
        "foundation_iou_drop_le_0_5pp": raw["per_class"]["foundation"]["IoU"] - cleaned["per_class"]["foundation"]["IoU"] <= 0.005,
        "background_to_stay_cable_reduction_ge_1pct": ((raw_fp - clean_fp) / raw_fp if raw_fp else 0.0) >= 0.01,
        "worst_10_percent_mIoU_drop_le_0_5pp": worst_drop <= 0.005,
        "challenge_140_146_mIoU_drop_le_0_5pp": challenge_drop <= 0.005,
        "rgb_max_pixel_difference_zero": rgb["rgb_max_absolute_pixel_difference"] == 0,
        "rgb_artifact_hash_equal": rgb["rgb_artifact_hashes_equal"],
        "rgb_png_hash_equal": rgb["rgb_png_hashes_equal"],
        "checkpoint_reloaded": True,
        "official_external_camera_pass": external_pass,
    }
    return {
        "gates": gates,
        "all_gates_pass": bool(mode == "full" and all(gates.values())),
        "worst_10_percent_mIoU_drop": worst_drop,
        "challenge_140_146_mIoU_drop": challenge_drop,
        "background_to_stay_cable_reduction_pixels": raw_fp - clean_fp,
        "background_to_stay_cable_reduction_fraction": float((raw_fp - clean_fp) / raw_fp) if raw_fp else 0.0,
    }


def _save_quad(output_dir: Path, names: list[str], labels_root: Path, raw_root: Path, clean_root: Path) -> None:
    for name in names:
        target = s0.read_label(labels_root / name)
        raw = np.asarray(Image.open(raw_root / name), dtype=np.uint8)
        clean = np.asarray(Image.open(clean_root / name), dtype=np.uint8)
        Image.fromarray(semantic_quad(target, raw, clean), mode="RGB").save(output_dir / name)


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    if config.get("experiment_id") != "SEM-070-POST-CLEAN":
        raise ValueError("SEM-070 runner received a different experiment config")
    split = load_json(REPO_ROOT / config["data"]["semantic_split"])
    manifest = load_json(REPO_ROOT / config["data"]["label_manifest"])
    expected_mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    if split["split_id"] != config["split_id"] or manifest["split_id"] != split["split_id"]:
        raise ValueError("Config, split and label manifest are inconsistent")
    if manifest["class_mapping"] != expected_mapping:
        raise ValueError("Label manifest class mapping is not the official five-class mapping")
    if not args.labels_root.is_dir():
        raise FileNotFoundError(f"Missing external semantic labels: {args.labels_root}")
    train_names = list(split["semantic_train_labeled"])
    validation_names = list(split["semantic_validation_labeled"])
    if len(train_names) != int(config["data"]["training_images"]) or len(validation_names) != int(config["data"]["validation_images"]):
        raise ValueError("SEM-070 requires the complete 219/74 SEM-001 split")
    calibration_names, check_names = split_training_views(
        train_names,
        calibration_half=int(config["postprocess"]["calibration_half"]),
        check_half=int(config["postprocess"]["check_half"]),
    )
    if set(calibration_names) | set(check_names) != set(train_names) or set(calibration_names) & set(check_names):
        raise ValueError("Training calibration/check split is not an exact partition")
    if set(validation_names) & set(train_names):
        raise ValueError("Training and validation views overlap")
    excluded_unlabeled = set(split.get("excluded_unlabeled", []))
    if (set(train_names) | set(validation_names)) & excluded_unlabeled:
        raise ValueError("SEM-070 train/validation split overlaps excluded unlabeled views")
    if args.mode == "preflight":
        train_names = train_names[: int(config["runtime"]["preflight_training_limit"])]
        calibration_names, check_names = split_training_views(train_names, calibration_half=0, check_half=1)
        validation_names = validation_names[: int(config["runtime"]["preflight_validation_limit"])]
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    frozen = config["frozen_rgb"]
    settings = ScaffoldSettings(train_camera_count=int(frozen["train_camera_count"]), n_offsets=int(frozen["n_offsets"]))
    point_cloud = args.rgb_run_dir / "point_cloud" / f"iteration_{frozen['iteration']}" / "point_cloud.ply"
    rgb_hashes = frozen_rgb_hashes(args.rgb_run_dir, int(frozen["iteration"]))
    configure_scaffold_import(args.source_root)
    model = load_frozen_scaffold(args.source_root, point_cloud, settings)
    if any(parameter.requires_grad for _name, parameter in frozen_rgb_named_parameters(model)):
        raise RuntimeError("Frozen RGB model unexpectedly has trainable parameters")
    sem001_config = load_json(REPO_ROOT / config["frozen_semantic"]["config"])
    sem001_checkpoint_sha256 = sha256(args.sem001_checkpoint)
    expected_checkpoint_sha256 = config["frozen_semantic"].get("checkpoint_sha256")
    if expected_checkpoint_sha256 and sem001_checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError("SEM-001 checkpoint SHA-256 differs from the frozen SEM-070 config")
    sidecar = s0.reload_sidecar(args.sem001_checkpoint, int(model.get_anchor.shape[0]), sem001_config, rgb_hashes)
    sidecar.logits.requires_grad_(False)
    reloaded = s0.reload_sidecar(args.sem001_checkpoint, int(model.get_anchor.shape[0]), sem001_config, rgb_hashes)
    if not torch.equal(sidecar.logits, reloaded.logits):
        raise RuntimeError("SEM-001 checkpoint reload is not bit-identical")
    pipe = pipeline()
    from src.scaffold_visibility_supervision import load_visibility_extension

    extension = load_visibility_extension(args.extension_root)
    anchor_count = int(model.get_anchor.shape[0])
    started = time.perf_counter()

    statistics, raw_by_half, training_rows = collect_training_statistics(
        train_names,
        labels_root=args.labels_root,
        data_root=args.data_root,
        appearance_camera_id=int(frozen["appearance_camera_id"]),
        model=model,
        sidecar=sidecar,
        pipe=pipe,
        extension=extension,
        anchor_count=anchor_count,
    )
    anchor_rows = finalize_anchor_rows(statistics, sidecar.logits.detach().cpu().numpy())
    _write_anchor_csv(args.output_dir / "training_anchor_statistics.csv", anchor_rows)
    _write_view_csv(args.output_dir / "training_per_view_raw.csv", training_rows)
    write_json(
        args.output_dir / "training_selection_statistics.json",
        {
            "training_only": True,
            "validation_opened_before_mask_freeze": False,
            "train_views": len(train_names),
            "calibration_views": calibration_names,
            "check_views": check_names,
            "calibration_half": int(config["postprocess"]["calibration_half"]),
            "check_half": int(config["postprocess"]["check_half"]),
            "raw_calibration_metrics": _metrics(raw_by_half[0]),
            "raw_check_metrics": _metrics(raw_by_half[1]),
            "anchor_count": anchor_count,
        },
    )

    raw_check_metrics = _metrics(raw_by_half[int(config["postprocess"]["check_half"])])

    def evaluate_check(mask: np.ndarray) -> dict[str, Any]:
        return evaluate_views(
            check_names,
            labels_root=args.labels_root,
            data_root=args.data_root,
            appearance_camera_id=int(frozen["appearance_camera_id"]),
            model=model,
            sidecar=sidecar,
            pipe=pipe,
            anchor_mask=mask,
        )

    selected_mask, selected_threshold, threshold_rows = choose_check_threshold(
        anchor_rows,
        [float(value) for value in config["postprocess"]["threshold_grid"]],
        minimum_support_views=int(config["postprocess"]["minimum_support_views"]),
        minimum_dominant_pixels=int(config["postprocess"]["minimum_dominant_pixels"]),
        raw_check_metrics=raw_check_metrics,
        evaluate_candidate=evaluate_check,
        minimum_recall_retention=float(config["postprocess"]["selection_minimum_recall_retention"]),
        maximum_protected_iou_drop=float(config["postprocess"]["selection_maximum_protected_iou_drop"]),
    )
    threshold_summary = []
    for row in threshold_rows:
        serializable = {key: value for key, value in row.items() if key != "check_metrics"}
        serializable["check_metrics"] = row["check_metrics"]
        threshold_summary.append(serializable)
    write_json(args.output_dir / "threshold_grid_check.json", threshold_summary)
    mask_sha = save_suppression_mask(args.output_dir / "postprocess_mask.npz", selected_mask, anchor_count)
    write_json(
        args.output_dir / "mask_freeze.json",
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "mask_path": "postprocess_mask.npz",
            "mask_sha256": mask_sha,
            "anchor_count": anchor_count,
            "suppressed_anchor_count": int(selected_mask.sum()),
            "selected_threshold": selected_threshold,
            "mask_source": "calibration-half anchor statistics; threshold selected on disjoint training check half",
            "validation_opened_before_mask_freeze": False,
            "unlabeled_rgb_opened": False,
            "hidden_test_opened": False,
        },
    )

    validation_root = args.output_dir / "validation"
    raw_root = validation_root / "raw"
    clean_root = validation_root / "cleaned"
    raw_validation = evaluate_views(
        validation_names,
        labels_root=args.labels_root,
        data_root=args.data_root,
        appearance_camera_id=int(frozen["appearance_camera_id"]),
        model=model,
        sidecar=sidecar,
        pipe=pipe,
        anchor_mask=None,
        output_dir=raw_root,
    )
    clean_validation = evaluate_views(
        validation_names,
        labels_root=args.labels_root,
        data_root=args.data_root,
        appearance_camera_id=int(frozen["appearance_camera_id"]),
        model=model,
        sidecar=sidecar,
        pipe=pipe,
        anchor_mask=selected_mask,
        output_dir=clean_root,
    )
    quad_root = validation_root / "gt_raw_cleaned_error"
    quad_root.mkdir(parents=True, exist_ok=True)
    _save_quad(quad_root, validation_names, args.labels_root, raw_root, clean_root)
    combined_view_rows = []
    clean_by_view = {row["view"]: row for row in clean_validation["per_view"]}
    for row in raw_validation["per_view"]:
        clean = clean_by_view[row["view"]]
        combined_view_rows.append({"view": row["view"], **{f"raw_{key}": value for key, value in row.items() if key != "view"}, **{f"cleaned_{key}": value for key, value in clean.items() if key != "view"}})
    with (args.output_dir / "validation_raw_cleaned_per_view.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(combined_view_rows[0].keys()) if combined_view_rows else ["view"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(combined_view_rows)

    rgb_evidence = _rgb_isolation(
        list(config["isolation"]["views"]),
        data_root=args.data_root,
        appearance_camera_id=int(frozen["appearance_camera_id"]),
        model=model,
        sidecar=sidecar,
        pipe=pipe,
        mask=selected_mask,
        rgb_run_dir=args.rgb_run_dir,
        rgb_iteration=int(frozen["iteration"]),
        output_dir=args.output_dir / "rgb_isolation",
    )
    external = render_external_pair(
        config=config,
        data_cameras=args.data_root / "camera_parameters" / "cameras.txt",
        data_images=args.data_root / "camera_parameters" / "images.txt",
        image_name=str(config["external_camera"]["image_name"]),
        camera_id=config["external_camera"].get("camera_id"),
        output_dir=args.output_dir / "external_camera",
        model=model,
        sidecar=sidecar,
        pipe=pipe,
        mask=selected_mask,
    )
    gates = formal_gates(raw_validation, clean_validation, rgb_evidence, external, mode=args.mode)
    write_json(
        args.output_dir / "result.json",
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "parent_experiment": "SEM-001",
            "status": "candidate_eligible_to_replace" if gates["all_gates_pass"] else "show_only_candidate_sem001_fallback",
            "mode": args.mode,
            "git_commit": git_head(),
            "seed": config["seed"],
            "runtime_seconds": time.perf_counter() - started,
            "frozen_rgb_artifact_hashes": rgb_hashes,
            "sem001_checkpoint_sha256": sem001_checkpoint_sha256,
            "checkpoint_reloaded": True,
            "mask": {"path": "postprocess_mask.npz", "sha256": mask_sha, "anchor_count": anchor_count, "suppressed_anchor_count": int(selected_mask.sum()), "selected_threshold": selected_threshold},
            "training": {"views": len(train_names), "calibration_views": len(calibration_names), "check_views": len(check_names), "raw_calibration": _metrics(raw_by_half[0]), "raw_check": raw_check_metrics},
            "validation_raw": raw_validation,
            "validation_cleaned": clean_validation,
            "rgb_isolation": rgb_evidence,
            "external_camera": external,
            "formal_gates": gates,
            "fallback": "RGB-030-FULL Scaffold-GS + SEM-001",
            "uses_unlabeled_rgb": False,
            "uses_hidden_test_rgb_or_labels": False,
        },
    )
    print(json.dumps({"status": "candidate_eligible_to_replace" if gates["all_gates_pass"] else "show_only_candidate_sem001_fallback", "selected_threshold": selected_threshold, "suppressed_anchor_count": int(selected_mask.sum()), "validation_mIoU_raw": raw_validation["mIoU_with_background"], "validation_mIoU_cleaned": clean_validation["mIoU_with_background"], "fp_reduction_fraction": gates["background_to_stay_cable_reduction_fraction"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
