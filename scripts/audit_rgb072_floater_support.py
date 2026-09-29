#!/usr/bin/env python3
"""Audit training-view support, residual and scale of frozen Scaffold anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem001 as sem001  # noqa: E402
import run_sem080_structural_diag as common  # noqa: E402

from src.scaffold_appearance import partition_scaffold_camera_records  # noqa: E402
from src.scaffold_semantic_sidecar import (  # noqa: E402
    ScaffoldSettings,
    configure_scaffold_import,
    frozen_rgb_hashes,
    frozen_rgb_named_parameters,
    load_frozen_scaffold,
    pipeline,
    render_frozen_rgb,
)
from src.scaffold_visibility_supervision import (  # noqa: E402
    load_visibility_extension,
    render_frozen_contribution_diagnostics,
)
from src.semantic_crossview_consensus import dominant_anchor_map  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--rgb-run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--extension-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: np.ndarray) -> dict[str, float] | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    return {
        "minimum": float(values.min()),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "maximum": float(values.max()),
    }


def group_summary(
    selected: np.ndarray,
    *,
    dominant_pixels: np.ndarray,
    dominant_views: np.ndarray,
    mean_mae: np.ndarray,
    mean_contribution: np.ndarray,
    max_scale: np.ndarray,
    max_anisotropy: np.ndarray,
) -> dict[str, Any]:
    ids = np.flatnonzero(selected)
    return {
        "anchor_count": int(ids.size),
        "dominant_pixel_count_sum": int(dominant_pixels[ids].sum(dtype=np.int64)),
        "dominant_views": quantiles(dominant_views[ids]),
        "mean_RGB_MAE": quantiles(mean_mae[ids]),
        "mean_dominant_contribution": quantiles(mean_contribution[ids]),
        "maximum_generated_scale": quantiles(max_scale[ids]),
        "maximum_generated_anisotropy": quantiles(max_anisotropy[ids]),
    }


def main() -> int:
    args = parse_args()
    config = common.load_json(args.config)
    require(
        config.get("experiment_id") == "RGB-072-FLOATER-SUPPORT-DIAG",
        "RGB-072 received another experiment config",
    )
    require(config.get("status") == "registered_not_run", "RGB-072 must run from its registered config")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split = common.load_json(REPO_ROOT / config["data"]["split"])
    validation_names = list(split["primary_fold"]["validation_labeled"])
    camera_records = common.load_json(args.rgb_run_dir / "cameras.json")
    _test_records, train_records = partition_scaffold_camera_records(camera_records, validation_names)
    require(len(train_records) == int(config["data"]["training_count"]), "training count differs from registration")
    train_names = [str(record["img_name"]) + ".png" for record in train_records]
    require(not set(train_names) & set(validation_names), "RGB-072 training and validation names overlap")
    missing_references = [name for name in train_names if not (args.reference_dir / name).is_file()]
    require(not missing_references, f"missing training references: {missing_references[:5]}")

    import torch

    require(torch.cuda.is_available(), "RGB-072 requires the frozen CUDA renderer")
    frozen = config["frozen_rgb"]
    iteration = int(frozen["iteration"])
    settings = ScaffoldSettings(
        train_camera_count=int(frozen["train_camera_count"]),
        n_offsets=int(frozen["n_offsets"]),
    )
    point_cloud = args.rgb_run_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    artifact_before = frozen_rgb_hashes(args.rgb_run_dir, iteration)
    configure_scaffold_import(args.source_root)
    scaffold = load_frozen_scaffold(args.source_root, point_cloud, settings)
    require(
        not any(parameter.requires_grad for _name, parameter in frozen_rgb_named_parameters(scaffold)),
        "RGB-072 found trainable frozen RGB parameters",
    )
    extension = load_visibility_extension(args.extension_root)
    pipe = pipeline()
    anchor_count = int(scaffold.get_anchor.shape[0])
    minimum_view_pixels = int(config["support_statistics"]["minimum_dominant_pixels_per_view"])
    high_error_threshold = float(config["support_statistics"]["high_error_RGB_MAE_threshold"])

    dominant_pixels = np.zeros(anchor_count, dtype=np.int64)
    dominant_views = np.zeros(anchor_count, dtype=np.uint16)
    active_views = np.zeros(anchor_count, dtype=np.uint16)
    max_pixels_in_view = np.zeros(anchor_count, dtype=np.int32)
    error_sum = np.zeros(anchor_count, dtype=np.float64)
    contribution_sum = np.zeros(anchor_count, dtype=np.float64)
    high_error_pixels = np.zeros(anchor_count, dtype=np.int64)
    max_generated_scale = np.zeros(anchor_count, dtype=np.float32)
    max_generated_anisotropy = np.zeros(anchor_count, dtype=np.float32)
    max_generated_opacity = np.zeros(anchor_count, dtype=np.float32)
    per_view: list[dict[str, Any]] = []
    total_squared_error = 0.0
    total_values = 0
    started = time.perf_counter()

    for appearance_id, name in enumerate(train_names):
        _selected, _camera, view = sem001.official_view(args.data_root, name, appearance_id)
        with torch.inference_mode():
            rendered_uint8 = sem001.tensor_to_rgb(render_frozen_rgb(view, scaffold, pipe))
            diagnostics = render_frozen_contribution_diagnostics(view, scaffold, pipe, extension=extension)
        with Image.open(args.reference_dir / name) as image:
            reference_uint8 = np.asarray(image.convert("RGB"), dtype=np.uint8)
        require(reference_uint8.shape == rendered_uint8.shape, f"RGB shape mismatch for {name}")
        difference = reference_uint8.astype(np.float32) / 255.0 - rendered_uint8.astype(np.float32) / 255.0
        mae = np.abs(difference).mean(axis=2)
        total_squared_error += float(np.square(difference, dtype=np.float64).sum())
        total_values += int(difference.size)

        anchors = dominant_anchor_map(
            diagnostics["dominant_offset_index"].detach().cpu().numpy(),
            diagnostics["offset_anchor_indices"].detach().cpu().numpy(),
            anchor_count=anchor_count,
        )
        maximum = diagnostics["max_contribution"].detach().cpu().numpy()
        valid = anchors >= 0
        valid_anchors = anchors[valid].astype(np.int64, copy=False)
        counts = np.bincount(valid_anchors, minlength=anchor_count)
        dominant_pixels += counts
        dominant_views += (counts >= minimum_view_pixels).astype(np.uint16)
        max_pixels_in_view = np.maximum(max_pixels_in_view, counts.astype(np.int32))
        error_sum += np.bincount(valid_anchors, weights=mae[valid], minlength=anchor_count)
        contribution_sum += np.bincount(valid_anchors, weights=maximum[valid], minlength=anchor_count)
        high_error_pixels += np.bincount(
            valid_anchors,
            weights=(mae[valid] >= high_error_threshold).astype(np.int64),
            minlength=anchor_count,
        ).astype(np.int64)

        offset_anchors = diagnostics["offset_anchor_indices"].detach().cpu().numpy().astype(np.int64, copy=False)
        scales = diagnostics["generated_scaling"].detach().cpu().numpy()
        opacity = diagnostics["generated_opacity"].detach().cpu().numpy().reshape(-1)
        active_views[np.unique(offset_anchors)] += 1
        scale_max = scales.max(axis=1)
        anisotropy = scale_max / np.maximum(scales.min(axis=1), np.finfo(np.float32).tiny)
        np.maximum.at(max_generated_scale, offset_anchors, scale_max)
        np.maximum.at(max_generated_anisotropy, offset_anchors, anisotropy)
        np.maximum.at(max_generated_opacity, offset_anchors, opacity)

        per_view.append(
            {
                "view": name,
                "appearance_id": appearance_id,
                "dominant_anchor_pixels": int(valid.sum()),
                "dominant_anchor_count_at_least_minimum_pixels": int(np.count_nonzero(counts >= minimum_view_pixels)),
                "RGB_MAE": float(mae.mean()),
            }
        )
        completed = appearance_id + 1
        if completed == 1 or completed % 20 == 0 or completed == len(train_names):
            print(json.dumps({"phase": "training_support", "completed": completed, "total": len(train_names)}), flush=True)
        del diagnostics, rendered_uint8, reference_uint8, difference, mae, anchors

    artifact_after = frozen_rgb_hashes(args.rgb_run_dir, iteration)
    require(artifact_before == artifact_after, "RGB-072 changed a frozen RGB artifact")
    has_pixels = dominant_pixels > 0
    mean_mae = np.full(anchor_count, np.nan, dtype=np.float64)
    mean_contribution = np.full(anchor_count, np.nan, dtype=np.float64)
    high_error_fraction = np.full(anchor_count, np.nan, dtype=np.float64)
    mean_mae[has_pixels] = error_sum[has_pixels] / dominant_pixels[has_pixels]
    mean_contribution[has_pixels] = contribution_sum[has_pixels] / dominant_pixels[has_pixels]
    high_error_fraction[has_pixels] = high_error_pixels[has_pixels] / dominant_pixels[has_pixels]

    groups = {
        "any_active": active_views > 0,
        "any_dominant_pixel": has_pixels,
        "dominant_at_least_256_pixels": dominant_pixels >= 256,
        "low_support_le2_views_and_at_least_256_pixels": (dominant_views <= 2) & (dominant_pixels >= 256),
        "low_support_le4_views_and_at_least_256_pixels": (dominant_views <= 4) & (dominant_pixels >= 256),
    }
    group_summaries = {
        name: group_summary(
            selected,
            dominant_pixels=dominant_pixels,
            dominant_views=dominant_views,
            mean_mae=mean_mae,
            mean_contribution=mean_contribution,
            max_scale=max_generated_scale,
            max_anisotropy=max_generated_anisotropy,
        )
        for name, selected in groups.items()
    }
    eligible = dominant_pixels >= 256
    ranked = np.flatnonzero(eligible)
    ranked = ranked[np.argsort(mean_mae[ranked], kind="mergesort")[::-1]][:100]
    top_residual_anchors = [
        {
            "anchor_id": int(anchor_id),
            "dominant_pixels": int(dominant_pixels[anchor_id]),
            "dominant_views": int(dominant_views[anchor_id]),
            "active_views": int(active_views[anchor_id]),
            "max_pixels_in_one_view": int(max_pixels_in_view[anchor_id]),
            "mean_RGB_MAE": float(mean_mae[anchor_id]),
            "high_error_pixel_fraction": float(high_error_fraction[anchor_id]),
            "mean_dominant_contribution": float(mean_contribution[anchor_id]),
            "maximum_generated_scale": float(max_generated_scale[anchor_id]),
            "maximum_generated_anisotropy": float(max_generated_anisotropy[anchor_id]),
            "maximum_generated_opacity": float(max_generated_opacity[anchor_id]),
        }
        for anchor_id in ranked
    ]

    arrays_path = args.output_dir / "anchor_statistics.npz"
    np.savez_compressed(
        arrays_path,
        dominant_pixels=dominant_pixels,
        dominant_views=dominant_views,
        active_views=active_views,
        max_pixels_in_view=max_pixels_in_view,
        mean_RGB_MAE=mean_mae,
        high_error_pixel_fraction=high_error_fraction,
        mean_dominant_contribution=mean_contribution,
        maximum_generated_scale=max_generated_scale,
        maximum_generated_anisotropy=max_generated_anisotropy,
        maximum_generated_opacity=max_generated_opacity,
    )
    mse = total_squared_error / total_values
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "completed_training_only_diagnostic",
        "training_views": len(train_names),
        "validation_or_semantic_data_opened": False,
        "anchor_count": anchor_count,
        "support_definition": {"minimum_dominant_pixels_per_view": minimum_view_pixels},
        "high_error_RGB_MAE_threshold": high_error_threshold,
        "aggregate_training_RGB": {
            "PSNR_dB": float(-10.0 * math.log10(mse)),
            "per_view_mean_RGB_MAE": float(np.mean([record["RGB_MAE"] for record in per_view])),
        },
        "groups": group_summaries,
        "top_100_residual_anchors_with_at_least_256_dominant_pixels": top_residual_anchors,
        "per_view": per_view,
        "runtime_seconds": time.perf_counter() - started,
        "frozen_artifact_hashes_before": artifact_before,
        "frozen_artifact_hashes_after": artifact_after,
        "frozen_artifact_hashes_equal": artifact_before == artifact_after,
        "anchor_statistics_npz": str(arrays_path),
        "anchor_statistics_npz_sha256": sha256(arrays_path),
    }
    result_path = args.output_dir / "result.json"
    common.write_json(result_path, result)
    print(json.dumps({"result": str(result_path), "training_views": len(train_names), "anchor_count": anchor_count}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

