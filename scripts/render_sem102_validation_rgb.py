#!/usr/bin/env python3
"""Render frozen RGB for SEM-102's once-only validation audit.

The checkpoint contract must already exist. This renderer consumes only the
frozen Scaffold artifact and official camera metadata; it never accepts labels,
released RGB references, unlabelled images, or segmentation model parameters.
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
from src.semantic_rendered_final import validate_final_training_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--final-result", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--rgb-run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("validation-render-shard", "finalize-validation-render"),
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
    Image.fromarray(image, mode="RGB").save(path)
    return s80.sha256(path)


def final_checkpoint_evidence(
    config: dict[str, object], config_path: Path, result_path: Path, checkpoint: Path
) -> dict[str, object]:
    require(checkpoint.is_file(), "SEM-102 final checkpoint is missing")
    require(result_path.is_file(), "SEM-102 final training result is missing")
    result = s80.load_json(result_path)
    validate_final_training_result(
        result,
        checkpoint_sha256=s80.sha256(checkpoint),
        config_sha256=s80.sha256(config_path),
    )
    return {
        "final_training_result_sha256": s80.sha256(result_path),
        "final_checkpoint_sha256": s80.sha256(checkpoint),
    }


def main() -> int:
    args = parse_args()
    config = s80.load_json(args.config)
    require(
        config.get("experiment_id") == "SEM-102-RENDERED-SEG-FINAL",
        "SEM-102 validation renderer received another config",
    )
    require(
        config.get("status") == "registered_not_run",
        "SEM-102 validation renderer requires its registered-not-run config",
    )
    checkpoint_evidence = final_checkpoint_evidence(
        config, args.config, args.final_result, args.final_checkpoint
    )
    split_path = REPO_ROOT / config["data"]["semantic_split"]
    require(
        s80.sha256(split_path) == config["data"]["semantic_split_sha256"],
        "SEM-102 semantic split differs from its registered hash",
    )
    split = s80.load_json(split_path)
    validation_names = list(split["semantic_validation_labeled"])
    require(
        len(validation_names) == int(config["data"]["validation_images"])
        and len(validation_names) == len(set(validation_names)),
        "SEM-102 validation render view count differs from its registered plan",
    )
    training_names = set(split["semantic_train_labeled"])
    require(
        not (set(validation_names) & training_names),
        "SEM-102 validation render includes a training view",
    )
    if args.mode == "validation-render-shard":
        require(
            args.start_index is not None and args.end_index is not None,
            "SEM-102 validation render shards require start/end indexes",
        )
        require(
            0 <= args.start_index < args.end_index <= len(validation_names),
            "SEM-102 validation shard indexes are outside the frozen plan",
        )
        require(
            not (args.output_dir / "rendered_validation_manifest.json").exists(),
            "SEM-102 validation RGB has already been finalized",
        )
        selected_names = validation_names[args.start_index : args.end_index]
    else:
        require(
            args.start_index is None and args.end_index is None,
            "SEM-102 validation finalizer does not accept shard indexes",
        )
        selected_names = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    access = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "mode": "validation_render_after_final_checkpoint",
        "fixed_validation_views_rendered": validation_names,
        "validation_labels_opened": False,
        "released_or_reference_rgb_opened": False,
        "unlabeled_images_opened": False,
        "hidden_test_rgb_or_labels_opened": False,
        "external_camera_mode_present": False,
        "final_checkpoint": checkpoint_evidence,
        "new_parameters_trained": False,
        "new_checkpoint_written": False,
    }
    access_path = args.output_dir / "validation_render_access.json"
    if access_path.exists():
        require(
            s80.load_json(access_path) == access,
            "SEM-102 validation render access contract changed between shards",
        )
    else:
        s80.write_json(access_path, access)

    import torch

    frozen = config["frozen_rgb"]
    iteration = int(frozen["iteration"])
    settings = ScaffoldSettings(
        train_camera_count=int(frozen["train_camera_count"]),
        n_offsets=int(frozen["n_offsets"]),
    )
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
        "SEM-102 validation renderer found a trainable frozen RGB tensor",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("SEM-102 validation renderer requires CUDA")
    pipe = pipeline()
    isolation_names = list(config["outputs"]["validation_rgb_isolation_samples"])
    require(
        set(isolation_names) <= set(validation_names),
        "SEM-102 validation RGB isolation samples must be validation views",
    )
    started = time.perf_counter()
    if args.mode == "validation-render-shard":
        records: list[dict[str, str]] = []
        for index, name in enumerate(selected_names, start=args.start_index + 1):
            output = args.output_dir / "images" / name
            require(
                not output.exists(), f"SEM-102 validation shard would overwrite {name}"
            )
            _selected, _camera, view = s0.official_view(
                args.data_root, name, int(frozen["appearance_camera_id"])
            )
            with torch.inference_mode():
                image = s0.tensor_to_rgb(render_frozen_rgb(view, scaffold, pipe))
            records.append(
                {
                    "name": name,
                    "path": str(output.relative_to(args.output_dir)),
                    "sha256": save_rgb(output, image),
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
                            "validation_render": index,
                            "validation_total": len(validation_names),
                            "name": name,
                        }
                    ),
                    flush=True,
                )
        artifact_after = frozen_rgb_hashes(args.rgb_run_dir, iteration)
        require(
            artifact_before == artifact_after,
            "Frozen RGB artifact changed during SEM-102 validation render",
        )
        s80.write_json(
            args.output_dir
            / "shards"
            / f"{args.start_index:03d}-{args.end_index:03d}.json",
            {
                "schema_version": 1,
                "experiment_id": config["experiment_id"],
                "status": "completed_validation_rgb_render_shard",
                "start_index": args.start_index,
                "end_index": args.end_index,
                "rendered_images": records,
                "validation_render_access": access,
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
        set(image_paths) == set(validation_names)
        and len(image_paths) == len(validation_names),
        "SEM-102 finalization requires exactly the 74 frozen validation render PNGs",
    )
    png_before = {name: s80.sha256(image_paths[name]) for name in isolation_names}
    reference = {
        name: np.asarray(Image.open(image_paths[name]).convert("RGB"), dtype=np.uint8)
        for name in isolation_names
    }
    png_after: dict[str, str] = {}
    max_difference = 0
    for name in isolation_names:
        _selected, _camera, view = s0.official_view(
            args.data_root, name, int(frozen["appearance_camera_id"])
        )
        with torch.inference_mode():
            repeated = s0.tensor_to_rgb(render_frozen_rgb(view, scaffold, pipe))
        max_difference = max(
            max_difference,
            int(
                np.abs(
                    reference[name].astype(np.int16) - repeated.astype(np.int16)
                ).max()
            ),
        )
        png_after[name] = save_rgb(
            args.output_dir / "rgb_isolation" / "after" / name, repeated
        )
    artifact_after = frozen_rgb_hashes(args.rgb_run_dir, iteration)
    require(
        artifact_before == artifact_after, "Frozen RGB artifact changed during SEM-102"
    )
    require(
        png_before == png_after, "SEM-102 repeated validation RGB PNG hashes differ"
    )
    require(max_difference == 0, "SEM-102 repeated validation RGB render differs")
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "completed_frozen_validation_rgb_render",
        "validation_view_count": len(validation_names),
        "rendered_images": [
            {
                "name": name,
                "path": str(image_paths[name].relative_to(args.output_dir)),
                "sha256": s80.sha256(image_paths[name]),
            }
            for name in validation_names
        ],
        "validation_render_access": access,
        "rgb_isolation": {
            "views": isolation_names,
            "rgb_artifact_hashes_before": artifact_before,
            "rgb_artifact_hashes_after": artifact_after,
            "rgb_artifact_hashes_equal": artifact_before == artifact_after,
            "rgb_png_sha256_before": png_before,
            "rgb_png_sha256_after": png_after,
            "rgb_png_hashes_equal": png_before == png_after,
            "rgb_max_absolute_pixel_difference": max_difference,
        },
        "new_checkpoint_written": False,
    }
    s80.write_json(args.output_dir / "rendered_validation_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
