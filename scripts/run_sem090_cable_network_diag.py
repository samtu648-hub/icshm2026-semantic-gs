#!/usr/bin/env python3
"""Run the training-only SEM-090 frozen cable-network cleanup diagnostic."""

from __future__ import annotations

import argparse
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
import run_sem080_structural_diag as s80

from src.scaffold_semantic_sidecar import (
    ScaffoldSettings,
    configure_scaffold_import,
    frozen_rgb_hashes,
    frozen_rgb_named_parameters,
    load_frozen_scaffold,
    pipeline,
    render_semantic_probabilities,
)
from src.semantic_cable_network_cleanup import (
    apply_cable_network_probability_transfer,
    cable_network_triptych,
    validate_cable_network_rule,
)
from src.semantic_postprocess import CLASS_COUNT, CLASS_NAMES, split_training_views


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--rgb-run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sem001-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "full"), default="full")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = s80.load_json(args.config)
    if config.get("experiment_id") != "SEM-090-CABLE-NETWORK-DIAG":
        raise ValueError("SEM-090 runner received a different experiment config")
    rule = validate_cable_network_rule(config["cable_network_rule"])
    split = s80.load_json(REPO_ROOT / config["data"]["semantic_split"])
    manifest = s80.load_json(REPO_ROOT / config["data"]["label_manifest"])
    if (
        split["split_id"] != config["split_id"]
        or manifest["split_id"] != split["split_id"]
    ):
        raise ValueError("Config, split and label manifest are inconsistent")
    if manifest["class_mapping"] != {
        name: index for index, name in enumerate(CLASS_NAMES)
    }:
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
            "SEM-090 requires exactly the complete, disjoint 219-view training split"
        )
    if set(train_names) & set(split.get("excluded_unlabeled", [])):
        raise ValueError("SEM-090 training views overlap excluded unlabeled images")
    if not args.labels_root.is_dir():
        raise FileNotFoundError(f"Missing semantic labels: {args.labels_root}")
    if args.mode == "preflight":
        train_names = train_names[: int(config["runtime"]["preflight_training_limit"])]
    calibration_names, check_names = split_training_views(train_names)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    s80.write_json(
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
    checkpoint_sha = s80.sha256(args.sem001_checkpoint)
    if checkpoint_sha != config["frozen_semantic"]["checkpoint_sha256"]:
        raise ValueError(
            "SEM-001 checkpoint SHA-256 differs from the frozen SEM-090 config"
        )
    sem001_config = s80.load_json(REPO_ROOT / config["frozen_semantic"]["config"])
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
    started = time.perf_counter()
    pipe = pipeline()
    raw_confusions = {
        half: np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64) for half in (0, 1)
    }
    clean_confusions = {
        half: np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64) for half in (0, 1)
    }
    aggregate: dict[str, int | float] = {}
    rows: list[dict[str, Any]] = []
    configured_samples = set(config["outputs"]["training_visual_samples"])
    if args.mode == "full" and not configured_samples <= set(train_names):
        raise ValueError(
            "Every SEM-090 visual sample must be an authorised training view"
        )
    samples = configured_samples & set(train_names)
    calibration_set = set(calibration_names)
    for index, name in enumerate(train_names, start=1):
        half = 0 if name in calibration_set else 1
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
        cleaned_probabilities, families, diagnostics = (
            apply_cable_network_probability_transfer(raw_probabilities, rule)
        )
        raw_prediction = raw_probabilities.argmax(axis=0).astype(np.uint8)
        cleaned_prediction = cleaned_probabilities.argmax(axis=0).astype(np.uint8)
        if name in samples:
            destination = args.output_dir / "training_cable_network_triptych"
            destination.mkdir(parents=True, exist_ok=True)
            Image.fromarray(
                cable_network_triptych(raw_prediction, cleaned_prediction, families),
                mode="RGB",
            ).save(destination / name)
        target = s0.read_label(args.labels_root / name)
        raw_confusions[half] += s80.confusion(target, raw_prediction)
        clean_confusions[half] += s80.confusion(target, cleaned_prediction)
        for key, value in diagnostics.items():
            if key != "retained_fraction_of_raw_stay_cable":
                aggregate[key] = aggregate.get(key, 0) + value
        rows.append({"view": name, "train_half": half, **diagnostics})
        if index % 25 == 0 or index == len(train_names):
            s80.write_json(
                args.output_dir / "progress.json",
                {
                    "processed_training_views": index,
                    "total_training_views": len(train_names),
                    "last_view": name,
                    "validation_views_opened": False,
                    "unlabeled_images_opened": False,
                },
            )
    aggregate["retained_fraction_of_raw_stay_cable"] = (
        float(
            aggregate["retained_stay_cable_pixels"] / aggregate["raw_stay_cable_pixels"]
        )
        if aggregate["raw_stay_cable_pixels"]
        else 1.0
    )
    raw_metrics = {
        str(half): s0.confusion_metrics(value) for half, value in raw_confusions.items()
    }
    cleaned_metrics = {
        str(half): s0.confusion_metrics(value)
        for half, value in clean_confusions.items()
    }
    comparisons = {
        str(half): s80.comparative_metrics(
            raw_metrics[str(half)], cleaned_metrics[str(half)]
        )
        for half in (0, 1)
    }
    rgb = s80._rgb_isolation(
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
    s80._write_rows(
        args.output_dir / "training_per_view_cable_network_diagnostics.csv", rows
    )
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "completed_training_only_diagnostic"
        if args.mode == "full"
        else "completed_preflight_not_quality_evidence",
        "mode": args.mode,
        "git_head": s80.git_head(),
        "rule": rule,
        "frozen_sem001_checkpoint_sha256": checkpoint_sha,
        "checkpoint_reload_bit_identical": True,
        "training_only_access": s80.load_json(
            args.output_dir / "training_only_access.json"
        ),
        "raw_metrics_by_training_half": raw_metrics,
        "cleaned_metrics_by_training_half": cleaned_metrics,
        "raw_to_cleaned_comparison_by_training_half": comparisons,
        "aggregate_cable_network_diagnostics": aggregate,
        "rgb_isolation": rgb,
        "validation_metrics": None,
        "external_camera_evidence": None,
        "rule_selected_for_formal_candidate": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    s80.write_json(args.output_dir / "result.json", result)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
