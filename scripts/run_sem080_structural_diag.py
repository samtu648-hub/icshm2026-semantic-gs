#!/usr/bin/env python3
"""Run the training-only SEM-080 tower/deck topology cleanup diagnostic.

The rule is applied to frozen SEM-001 probabilities after semantic
rasterization.  It never changes RGB geometry, color, opacity, or SEM-001
anchor logits.  The runner opens exactly the 219 labelled semantic-training
views and deliberately has no validation, unlabeled-image, or external-camera
mode: this is a feasibility diagnostic, not a replacement evaluation.
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
from src.semantic_postprocess import CLASS_COUNT, CLASS_NAMES, split_training_views
from src.semantic_structural_cleanup import (
    apply_structural_probability_transfer,
    structural_triptych,
    validate_structural_rule,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--rgb-run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sem001-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Git-ignored external output directory",
    )
    parser.add_argument("--mode", choices=("preflight", "full"), default="full")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def confusion(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    if target.shape != prediction.shape:
        raise ValueError("target and prediction shapes differ")
    return np.bincount(
        CLASS_COUNT * target.reshape(-1).astype(np.int64)
        + prediction.reshape(-1).astype(np.int64),
        minlength=CLASS_COUNT**2,
    ).reshape(CLASS_COUNT, CLASS_COUNT)


def comparative_metrics(raw: dict[str, Any], cleaned: dict[str, Any]) -> dict[str, Any]:
    raw_stay = raw["stay_cable"]
    clean_stay = cleaned["stay_cable"]
    raw_fp = int(raw["background_to_stay_cable_false_positive_pixels"])
    clean_fp = int(cleaned["background_to_stay_cable_false_positive_pixels"])
    raw_recall = float(raw_stay["recall"] or 0.0)
    clean_recall = float(clean_stay["recall"] or 0.0)
    return {
        "mIoU_with_background_delta": float(
            cleaned["mIoU_with_background"] - raw["mIoU_with_background"]
        ),
        "mIoU_without_background_delta": float(
            cleaned["mIoU_without_background"] - raw["mIoU_without_background"]
        ),
        "stay_cable_IoU_delta": float(clean_stay["IoU"] - raw_stay["IoU"]),
        "stay_cable_recall_retention_vs_raw": clean_recall / raw_recall
        if raw_recall
        else 1.0,
        "background_to_stay_cable_reduction_pixels": raw_fp - clean_fp,
        "background_to_stay_cable_reduction_fraction": float(
            (raw_fp - clean_fp) / raw_fp
        )
        if raw_fp
        else 0.0,
        "protected_class_iou_delta": {
            name: float(
                cleaned["per_class"][name]["IoU"] - raw["per_class"][name]["IoU"]
            )
            for name in ("deck", "tower", "foundation")
        },
    }


def _rgb_isolation(
    names: list[str],
    *,
    data_root: Path,
    appearance_camera_id: int,
    model: Any,
    pipe: Any,
    rgb_run_dir: Path,
    iteration: int,
    output_dir: Path,
) -> dict[str, Any]:
    import torch

    before: dict[str, np.ndarray] = {}
    before_hashes: dict[str, str] = {}
    for name in names:
        _selected, _camera, view = s0.official_view(
            data_root, name, appearance_camera_id
        )
        with torch.inference_mode():
            before[name] = s0.tensor_to_rgb(render_frozen_rgb(view, model, pipe))
        before_hashes[name] = s0.image_hash(before[name], output_dir / "before" / name)
    after_hashes: dict[str, str] = {}
    max_difference = 0
    for name in names:
        _selected, _camera, view = s0.official_view(
            data_root, name, appearance_camera_id
        )
        with torch.inference_mode():
            after = s0.tensor_to_rgb(render_frozen_rgb(view, model, pipe))
        max_difference = max(
            max_difference,
            int(np.abs(before[name].astype(np.int16) - after.astype(np.int16)).max()),
        )
        after_hashes[name] = s0.image_hash(after, output_dir / "after" / name)
    artifact_before = frozen_rgb_hashes(rgb_run_dir, iteration)
    artifact_after = frozen_rgb_hashes(rgb_run_dir, iteration)
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


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("diagnostic emitted no rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    if config.get("experiment_id") != "SEM-080-STRUCTURAL-DIAG":
        raise ValueError("SEM-080 runner received a different experiment config")
    rule = validate_structural_rule(config["structural_rule"])
    split = load_json(REPO_ROOT / config["data"]["semantic_split"])
    manifest = load_json(REPO_ROOT / config["data"]["label_manifest"])
    expected_mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    if (
        split["split_id"] != config["split_id"]
        or manifest["split_id"] != split["split_id"]
    ):
        raise ValueError("Config, split and label manifest are inconsistent")
    if manifest["class_mapping"] != expected_mapping:
        raise ValueError(
            "Label manifest class mapping is not the official five-class mapping"
        )
    train_names = list(split["semantic_train_labeled"])
    validation_names = set(split["semantic_validation_labeled"])
    if (
        len(train_names) != int(config["data"]["training_images"])
        or set(train_names) & validation_names
    ):
        raise ValueError(
            "SEM-080 requires exactly the complete, disjoint 219-view semantic training split"
        )
    if set(train_names) & set(split.get("excluded_unlabeled", [])):
        raise ValueError("SEM-080 training views overlap the excluded unlabeled split")
    if not args.labels_root.is_dir():
        raise FileNotFoundError(f"Missing external semantic labels: {args.labels_root}")
    if args.mode == "preflight":
        train_names = train_names[: int(config["runtime"]["preflight_training_limit"])]
    calibration_names, check_names = split_training_views(train_names)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "training_only_access.json",
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "mode": args.mode,
            "semantic_training_views": train_names,
            "calibration_views": calibration_names,
            "check_views": check_names,
            "validation_views_opened": False,
            "unlabeled_images_opened": False,
            "hidden_test_rgb_or_labels_opened": False,
            "rule_uses_only_frozen_semantic_probabilities": True,
            "labels_are_used_only_after_cleanup_for_metrics": True,
        },
    )

    import torch

    frozen = config["frozen_rgb"]
    settings = ScaffoldSettings(
        train_camera_count=int(frozen["train_camera_count"]),
        n_offsets=int(frozen["n_offsets"]),
    )
    point_cloud = (
        args.rgb_run_dir
        / "point_cloud"
        / f"iteration_{frozen['iteration']}"
        / "point_cloud.ply"
    )
    rgb_hashes = frozen_rgb_hashes(args.rgb_run_dir, int(frozen["iteration"]))
    configure_scaffold_import(args.source_root)
    model = load_frozen_scaffold(args.source_root, point_cloud, settings)
    if any(
        parameter.requires_grad
        for _name, parameter in frozen_rgb_named_parameters(model)
    ):
        raise RuntimeError("Frozen RGB model unexpectedly has trainable parameters")
    checkpoint_sha = sha256(args.sem001_checkpoint)
    if checkpoint_sha != config["frozen_semantic"]["checkpoint_sha256"]:
        raise ValueError(
            "SEM-001 checkpoint SHA-256 differs from the frozen SEM-080 config"
        )
    sem001_config = load_json(REPO_ROOT / config["frozen_semantic"]["config"])
    sidecar = s0.reload_sidecar(
        args.sem001_checkpoint,
        int(model.get_anchor.shape[0]),
        sem001_config,
        rgb_hashes,
    )
    sidecar.logits.requires_grad_(False)
    reloaded = s0.reload_sidecar(
        args.sem001_checkpoint,
        int(model.get_anchor.shape[0]),
        sem001_config,
        rgb_hashes,
    )
    if not torch.equal(sidecar.logits, reloaded.logits):
        raise RuntimeError("SEM-001 checkpoint reload is not bit-identical")
    pipe = pipeline()
    started = time.perf_counter()

    raw_confusions = {
        0: np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64),
        1: np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64),
    }
    clean_confusions = {
        0: np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64),
        1: np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64),
    }
    aggregate_diagnostics: dict[str, int | float] = {}
    rows: list[dict[str, Any]] = []
    configured_sample_names = set(config["outputs"]["training_visual_samples"])
    calibration_set = set(calibration_names)
    if args.mode == "full" and not configured_sample_names <= set(train_names):
        raise ValueError(
            "Every SEM-080 visual sample must belong to the authorised training views"
        )
    sample_names = configured_sample_names & set(train_names)
    for index, name in enumerate(train_names, start=1):
        train_half = 0 if name in calibration_set else 1
        _selected, _camera, view = s0.official_view(
            args.data_root, name, int(frozen["appearance_camera_id"])
        )
        with torch.inference_mode():
            raw_probabilities = (
                render_semantic_probabilities(view, model, sidecar, pipe)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        cleaned_probabilities, retained, diagnostics = (
            apply_structural_probability_transfer(raw_probabilities, rule)
        )
        raw_prediction = raw_probabilities.argmax(axis=0).astype(np.uint8)
        cleaned_prediction = cleaned_probabilities.argmax(axis=0).astype(np.uint8)
        if name in sample_names:
            (args.output_dir / "training_structural_triptych").mkdir(
                parents=True, exist_ok=True
            )
            Image.fromarray(
                structural_triptych(raw_prediction, cleaned_prediction, retained),
                mode="RGB",
            ).save(args.output_dir / "training_structural_triptych" / name)
        # The topology predicate has already completed.  Labels are loaded only
        # here, solely to score the frozen output on the authorised training split.
        target = s0.read_label(args.labels_root / name)
        raw_confusions[train_half] += confusion(target, raw_prediction)
        clean_confusions[train_half] += confusion(target, cleaned_prediction)
        for key, value in diagnostics.items():
            if key != "retained_fraction_of_raw_stay_cable":
                aggregate_diagnostics[key] = aggregate_diagnostics.get(key, 0) + value
        rows.append({"view": name, "train_half": train_half, **diagnostics})
        if index % 25 == 0 or index == len(train_names):
            write_json(
                args.output_dir / "progress.json",
                {
                    "processed_training_views": index,
                    "total_training_views": len(train_names),
                    "last_view": name,
                    "validation_views_opened": False,
                    "unlabeled_images_opened": False,
                },
            )

    raw_metrics = {
        str(half): s0.confusion_metrics(value) for half, value in raw_confusions.items()
    }
    cleaned_metrics = {
        str(half): s0.confusion_metrics(value)
        for half, value in clean_confusions.items()
    }
    comparisons = {
        str(half): comparative_metrics(
            raw_metrics[str(half)], cleaned_metrics[str(half)]
        )
        for half in (0, 1)
    }
    aggregate_diagnostics["retained_fraction_of_raw_stay_cable"] = (
        float(
            aggregate_diagnostics["retained_stay_cable_pixels"]
            / aggregate_diagnostics["raw_stay_cable_pixels"]
        )
        if aggregate_diagnostics["raw_stay_cable_pixels"]
        else 1.0
    )
    rgb = _rgb_isolation(
        list(config["isolation"]["training_views"]),
        data_root=args.data_root,
        appearance_camera_id=int(frozen["appearance_camera_id"]),
        model=model,
        pipe=pipe,
        rgb_run_dir=args.rgb_run_dir,
        iteration=int(frozen["iteration"]),
        output_dir=args.output_dir / "rgb_isolation",
    )
    if (
        not rgb["rgb_artifact_hashes_equal"]
        or not rgb["rgb_png_hashes_equal"]
        or rgb["rgb_max_absolute_pixel_difference"] != 0
    ):
        raise RuntimeError("Frozen RGB isolation failed")
    _write_rows(args.output_dir / "training_per_view_structural_diagnostics.csv", rows)
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "completed_training_only_diagnostic"
        if args.mode == "full"
        else "completed_preflight_not_quality_evidence",
        "mode": args.mode,
        "git_head": git_head(),
        "rule": rule,
        "frozen_sem001_checkpoint_sha256": checkpoint_sha,
        "checkpoint_reload_bit_identical": True,
        "training_only_access": load_json(
            args.output_dir / "training_only_access.json"
        ),
        "raw_metrics_by_training_half": raw_metrics,
        "cleaned_metrics_by_training_half": cleaned_metrics,
        "raw_to_cleaned_comparison_by_training_half": comparisons,
        "aggregate_structural_diagnostics": aggregate_diagnostics,
        "rgb_isolation": rgb,
        "validation_metrics": None,
        "external_camera_evidence": None,
        "rule_selected_for_formal_candidate": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
