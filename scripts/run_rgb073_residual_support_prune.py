#!/usr/bin/env python3
"""Render the one registered RGB-073 low-support/high-residual anchor mask."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem001 as sem001  # noqa: E402
import run_sem080_structural_diag as common  # noqa: E402

from src.scaffold_floater_pruning import residual_support_prune_mask  # noqa: E402
from src.scaffold_semantic_sidecar import (  # noqa: E402
    ScaffoldSettings,
    configure_scaffold_import,
    frozen_rgb_hashes,
    frozen_rgb_named_parameters,
    load_frozen_scaffold,
    pipeline,
    render_frozen_rgb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--rgb-run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--anchor-statistics", type=Path, required=True)
    parser.add_argument("--baseline-render-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def render_with_anchor_keep_mask(view, model, pipe, keep_mask):
    import torch
    from gaussian_renderer import prefilter_voxel, render

    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    with torch.inference_mode():
        visible = prefilter_voxel(view, model, pipe, background)
        require(visible.shape == keep_mask.shape, "RGB-073 keep mask does not match visible anchors")
        visible = visible & keep_mask
        return render(view, model, pipe, background, visible_mask=visible)["render"].detach()


def main() -> int:
    args = parse_args()
    config = common.load_json(args.config)
    experiment_id = config.get("experiment_id")
    require(
        experiment_id in {
            "RGB-073-RESIDUAL-SUPPORT-PRUNE-PF",
            "RGB-074-RESIDUAL-SUPPORT-PRUNE-TOL-PF",
        },
        "residual-support renderer received another experiment config",
    )
    require(config.get("status") == "registered_not_run", f"{experiment_id} must run from its registered config")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = config["mask_source"]
    require(common.sha256(args.anchor_statistics) == source["anchor_statistics_npz_sha256"], "RGB-072 statistics hash differs from registration")
    with np.load(args.anchor_statistics) as arrays:
        mask = residual_support_prune_mask(
            arrays,
            minimum_dominant_pixels=int(source["minimum_dominant_pixels"]),
            maximum_dominant_views=int(source["maximum_dominant_views"]),
            minimum_mean_rgb_mae=float(source["minimum_mean_RGB_MAE"]),
            minimum_high_error_fraction=float(source["minimum_high_error_pixel_fraction"]),
        )
    selected_ids = np.flatnonzero(mask)
    require(selected_ids.size == int(source["expected_selected_anchor_count"]), "RGB-073 selected-anchor count differs from registration")
    np.save(args.output_dir / "pruned_anchor_ids.npy", selected_ids.astype(np.int64))

    import torch

    require(torch.cuda.is_available(), "RGB-073 requires the frozen CUDA renderer")
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
        "RGB-073 found trainable frozen RGB parameters",
    )
    require(mask.shape == (int(scaffold.get_anchor.shape[0]),), "RGB-073 mask shape differs from checkpoint")
    keep_mask = torch.from_numpy(~mask).to(device="cuda", dtype=torch.bool)
    pipe = pipeline()
    split = common.load_json(REPO_ROOT / config["data"]["split"])
    names = sorted(split["primary_fold"]["validation_labeled"])
    require(len(names) == int(config["data"]["validation_count"]), "RGB-073 validation count differs from registration")

    per_view = []
    total_changed_pixels = 0
    maximum_absolute_difference = 0
    total_reproduction_different_pixels = 0
    maximum_reproduction_absolute_difference = 0
    maximum_reproduction_mean_absolute_difference = 0.0
    reproduction_gate = config["baseline_reproduction"]
    started = time.perf_counter()
    for appearance_id, name in enumerate(names):
        _selected, _camera, view = sem001.official_view(args.data_root, name, appearance_id)
        with torch.inference_mode():
            reproduced = sem001.tensor_to_rgb(render_frozen_rgb(view, scaffold, pipe))
            candidate = sem001.tensor_to_rgb(render_with_anchor_keep_mask(view, scaffold, pipe, keep_mask))
        with Image.open(args.baseline_render_dir / name) as image:
            recorded = np.asarray(image.convert("RGB"), dtype=np.uint8)
        reproduction_absolute = np.abs(reproduced.astype(np.int16) - recorded.astype(np.int16))
        reproduction_different_pixels = int(np.count_nonzero(np.any(reproduction_absolute != 0, axis=2)))
        reproduction_mean_absolute = float(reproduction_absolute.mean())
        reproduction_maximum_absolute = int(reproduction_absolute.max())
        reproduction_different_fraction = float(reproduction_different_pixels / (recorded.shape[0] * recorded.shape[1]))
        reproduction_passed = (
            reproduction_maximum_absolute <= int(reproduction_gate["maximum_absolute_difference_uint8"])
            and reproduction_mean_absolute <= float(reproduction_gate["maximum_mean_absolute_difference_uint8"])
            and reproduction_different_fraction <= float(reproduction_gate["maximum_different_pixel_fraction"])
        )
        if not reproduction_passed:
            common.write_json(
                args.output_dir / "baseline_reproduction_failure.json",
                {
                    "view": name,
                    "different_channel_values": int(np.count_nonzero(reproduction_absolute)),
                    "different_pixels": reproduction_different_pixels,
                    "different_pixel_fraction": reproduction_different_fraction,
                    "mean_absolute_difference_uint8": reproduction_mean_absolute,
                    "maximum_absolute_difference_uint8": reproduction_maximum_absolute,
                    "registered_gate": reproduction_gate,
                },
            )
            require(False, f"{experiment_id} could not reproduce frozen baseline for {name}")
        total_reproduction_different_pixels += reproduction_different_pixels
        maximum_reproduction_absolute_difference = max(
            maximum_reproduction_absolute_difference, reproduction_maximum_absolute
        )
        maximum_reproduction_mean_absolute_difference = max(
            maximum_reproduction_mean_absolute_difference, reproduction_mean_absolute
        )
        if bool(reproduction_gate.get("save_replayed_baseline", False)):
            reproduced_output = args.output_dir / "replayed_baseline" / name
            reproduced_output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(reproduced, mode="RGB").save(reproduced_output)
        absolute = np.abs(candidate.astype(np.int16) - recorded.astype(np.int16))
        changed = np.any(absolute != 0, axis=2)
        changed_pixels = int(changed.sum())
        total_changed_pixels += changed_pixels
        maximum_absolute_difference = max(maximum_absolute_difference, int(absolute.max()))
        output = args.output_dir / "renders" / name
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(candidate, mode="RGB").save(output)
        per_view.append(
            {
                "view": name,
                "baseline_reproduction_different_pixels": reproduction_different_pixels,
                "baseline_reproduction_mean_absolute_difference_uint8": reproduction_mean_absolute,
                "changed_pixels": changed_pixels,
                "mean_absolute_difference_from_baseline_uint8": float(absolute.mean()),
                "maximum_absolute_difference_from_baseline_uint8": int(absolute.max()),
                "render_sha256": common.sha256(output),
            }
        )
        completed = appearance_id + 1
        if completed == 1 or completed % 10 == 0 or completed == len(names):
            print(json.dumps({"phase": "paired_render", "completed": completed, "total": len(names)}), flush=True)

    artifact_after = frozen_rgb_hashes(args.rgb_run_dir, iteration)
    require(artifact_before == artifact_after, "RGB-073 changed a frozen RGB artifact")
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "completed_render_pending_fixed_metrics",
        "candidate_variable": "exclude_the_29_pre_registered_training_low_support_high_residual_anchors_at_inference",
        "selected_anchor_count": int(selected_ids.size),
        "selected_anchor_ids": selected_ids.tolist(),
        "selection_reads_validation_RGB_or_semantic_labels": False,
        "baseline_reproduction_gate": reproduction_gate,
        "baseline_reproduction_gate_passed_for_all_views": True,
        "total_baseline_reproduction_different_pixels": total_reproduction_different_pixels,
        "maximum_baseline_reproduction_absolute_difference_uint8": maximum_reproduction_absolute_difference,
        "maximum_per_view_baseline_reproduction_mean_absolute_difference_uint8": maximum_reproduction_mean_absolute_difference,
        "validation_views": len(names),
        "total_changed_pixels": total_changed_pixels,
        "maximum_absolute_difference_from_baseline_uint8": maximum_absolute_difference,
        "render_seconds_including_baseline_reproduction": time.perf_counter() - started,
        "frozen_artifact_hashes_before": artifact_before,
        "frozen_artifact_hashes_after": artifact_after,
        "frozen_artifact_hashes_equal": artifact_before == artifact_after,
        "pruned_anchor_ids_sha256": common.sha256(args.output_dir / "pruned_anchor_ids.npy"),
        "per_view": per_view,
    }
    common.write_json(args.output_dir / "render_result.json", result)
    print(json.dumps({key: result[key] for key in ("status", "selected_anchor_count", "total_changed_pixels")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
