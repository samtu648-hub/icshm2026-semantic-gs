#!/usr/bin/env python3
"""Render SEM-101's complete training-only frozen Scaffold RGB dataset.

This entry point only renders the 219 labelled SEM-001 training cameras.  It
cannot accept a validation list, image RGB, label, optimizer, checkpoint, or
external camera input.  Its output becomes the sole RGB input to SEM-101's
rendered-domain SegFormer folds.
"""

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

import run_sem001 as s0
import run_sem080_structural_diag as s80

from src.scaffold_semantic_sidecar import (
    ScaffoldSettings,
    configure_scaffold_import,
    frozen_rgb_hashes,
    frozen_rgb_named_parameters,
    load_frozen_scaffold,
    pipeline,
    render_frozen_rgb,
)
from src.semantic_rendered_training import select_rendered_training_views


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--rgb-run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("training-render-shard", "finalize-training-render"),
        required=True,
    )
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--end-index", type=int)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def save_rgb(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    return s80.sha256(path)


def main() -> int:
    args = parse_args()
    config = s80.load_json(args.config)
    require(
        config.get("experiment_id") == "SEM-101-RENDERED-SEG-OOF",
        "SEM-101 training renderer received another config",
    )
    require(
        config.get("status") == "registered_not_run",
        "SEM-101 training renderer must run only from its registered config",
    )
    split = s80.load_json(REPO_ROOT / config["data"]["semantic_split"])
    names = select_rendered_training_views(
        split, expected_count=int(config["data"]["training_images"])
    )
    if args.mode == "training-render-shard":
        require(
            args.start_index is not None and args.end_index is not None,
            "SEM-101 render shards require start/end indexes",
        )
        require(
            0 <= args.start_index < args.end_index <= len(names),
            "SEM-101 render shard indexes are outside the frozen training plan",
        )
        require(
            not (args.output_dir / "rendered_dataset_manifest.json").exists(),
            "SEM-101 rendered dataset has already been finalized",
        )
        selected_names = names[args.start_index : args.end_index]
    else:
        require(
            args.start_index is None and args.end_index is None,
            "SEM-101 finalization does not accept shard indexes",
        )
        selected_names = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    access = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "mode": "training-render-shards",
        "rendered_training_views": names,
        "fixed_validation_views_opened": False,
        "unlabeled_images_opened": False,
        "hidden_test_rgb_or_labels_opened": False,
        "labels_opened": False,
        "external_camera_mode_present": False,
        "new_parameters_trained": False,
        "new_checkpoint_written": False,
    }
    access_path = args.output_dir / "training_only_access.json"
    if access_path.exists():
        require(
            s80.load_json(access_path) == access,
            "SEM-101 rendered dataset access contract changed between shards",
        )
    else:
        s80.write_json(access_path, access)

    import torch

    frozen = config["frozen_rgb"]
    settings = ScaffoldSettings(
        train_camera_count=int(frozen["train_camera_count"]),
        n_offsets=int(frozen["n_offsets"]),
    )
    iteration = int(frozen["iteration"])
    point_cloud = (
        args.rgb_run_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    )
    artifact_before = frozen_rgb_hashes(args.rgb_run_dir, iteration)
    configure_scaffold_import(args.source_root)
    scaffold = load_frozen_scaffold(args.source_root, point_cloud, settings)
    require(
        not any(
            parameter.requires_grad
            for _name, parameter in frozen_rgb_named_parameters(scaffold)
        ),
        "Frozen RGB model unexpectedly has trainable tensors",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("SEM-101 training renderer requires CUDA")
    pipe = pipeline()
    visual_samples = list(config["outputs"]["rgb_isolation_samples"])
    require(
        set(visual_samples) <= set(names),
        "RGB isolation samples must be training views",
    )
    started = time.perf_counter()
    if args.mode == "training-render-shard":
        shard_records: list[dict[str, str]] = []
        for index, name in enumerate(selected_names, start=args.start_index + 1):
            output_path = args.output_dir / "images" / name
            require(not output_path.exists(), f"SEM-101 shard would overwrite {name}")
            _selected, _camera, view = s0.official_view(
                args.data_root, name, int(frozen["appearance_camera_id"])
            )
            with torch.inference_mode():
                image = s0.tensor_to_rgb(render_frozen_rgb(view, scaffold, pipe))
            shard_records.append(
                {
                    "name": name,
                    "path": str(output_path.relative_to(args.output_dir)),
                    "sha256": save_rgb(output_path, image),
                }
            )
            if (
                index == args.start_index + 1
                or index % 20 == 0
                or index == args.end_index
            ):
                print(
                    json.dumps(
                        {
                            "training_render": index,
                            "training_total": len(names),
                            "name": name,
                        }
                    ),
                    flush=True,
                )
        artifact_after = frozen_rgb_hashes(args.rgb_run_dir, iteration)
        require(
            artifact_before == artifact_after,
            "Frozen RGB artifact changed during SEM-101 render shard",
        )
        s80.write_json(
            args.output_dir
            / "shards"
            / f"{args.start_index:03d}-{args.end_index:03d}.json",
            {
                "schema_version": 1,
                "experiment_id": config["experiment_id"],
                "status": "completed_training_rgb_render_shard",
                "start_index": args.start_index,
                "end_index": args.end_index,
                "rendered_images": shard_records,
                "training_only_access": access,
                "rgb_artifact_hashes_before": artifact_before,
                "rgb_artifact_hashes_after": artifact_after,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        return 0

    image_paths = {
        path.name: path for path in (args.output_dir / "images").glob("*.png")
    }
    require(
        set(image_paths) == set(names) and len(image_paths) == len(names),
        "SEM-101 finalization requires exactly the 219 frozen training render PNGs",
    )
    rendered = [
        {
            "name": name,
            "path": str(image_paths[name].relative_to(args.output_dir)),
            "sha256": s80.sha256(image_paths[name]),
        }
        for name in names
    ]
    before_arrays = {
        name: np.asarray(Image.open(image_paths[name]).convert("RGB"), dtype=np.uint8)
        for name in visual_samples
    }
    isolation_before = {name: s80.sha256(image_paths[name]) for name in visual_samples}
    isolation_after: dict[str, str] = {}
    max_difference = 0
    for name in visual_samples:
        _selected, _camera, view = s0.official_view(
            args.data_root, name, int(frozen["appearance_camera_id"])
        )
        with torch.inference_mode():
            repeated = s0.tensor_to_rgb(render_frozen_rgb(view, scaffold, pipe))
        max_difference = max(
            max_difference,
            int(
                np.abs(
                    before_arrays[name].astype(np.int16) - repeated.astype(np.int16)
                ).max()
            ),
        )
        isolation_after[name] = save_rgb(
            args.output_dir / "rgb_isolation" / "after" / name, repeated
        )
    artifact_after = frozen_rgb_hashes(args.rgb_run_dir, iteration)
    require(
        artifact_before == artifact_after, "Frozen RGB artifact changed during SEM-101"
    )
    require(
        isolation_before == isolation_after,
        "Frozen RGB repeated PNG hashes differ during SEM-101",
    )
    require(max_difference == 0, "Frozen RGB repeated render differs during SEM-101")
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "completed_training_rgb_render",
        "mode": args.mode,
        "training_view_count": len(names),
        "rendered_images": rendered,
        "training_only_access": access,
        "rgb_isolation": {
            "views": visual_samples,
            "rgb_artifact_hashes_before": artifact_before,
            "rgb_artifact_hashes_after": artifact_after,
            "rgb_artifact_hashes_equal": artifact_before == artifact_after,
            "rgb_png_sha256_before": isolation_before,
            "rgb_png_sha256_after": isolation_after,
            "rgb_png_hashes_equal": isolation_before == isolation_after,
            "rgb_max_absolute_pixel_difference": max_difference,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "new_checkpoint_written": False,
    }
    s80.write_json(args.output_dir / "rendered_dataset_manifest.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
