#!/usr/bin/env python3
"""Render RGB-070 with nearest-training-camera Scaffold appearance IDs.

The frozen RGB checkpoint, anchors, offsets, opacity/covariance/color MLPs and
camera poses are unchanged.  The only candidate variable is the deterministic
appearance embedding selected from official camera-center metadata.
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
from src.scaffold_appearance import (
    nearest_training_appearance,
    partition_scaffold_camera_records,
)
from src.scaffold_semantic_sidecar import (
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
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    args = parse_args()
    config = s80.load_json(args.config)
    require(
        config.get("experiment_id") == "RGB-070-APPEARANCE-PF",
        "RGB-070 renderer received another experiment config",
    )
    require(
        config.get("status") == "registered_not_run",
        "RGB-070 must start from its registered-not-run config",
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split = s80.load_json(REPO_ROOT / config["data"]["split"])
    validation_names = list(split["primary_fold"]["validation_labeled"])
    require(
        len(validation_names) == int(config["data"]["validation_count"]),
        "RGB-070 validation count differs from the registered plan",
    )
    camera_records = s80.load_json(args.rgb_run_dir / "cameras.json")
    test_records, train_records = partition_scaffold_camera_records(
        camera_records, validation_names
    )
    require(
        len(train_records) == int(config["frozen_rgb"]["train_camera_count"]),
        "RGB-070 training-camera count differs from the embedding table",
    )
    mapping = [nearest_training_appearance(target, train_records) for target in test_records]
    require(
        [record["target_name"] for record in mapping] == sorted(validation_names),
        "RGB-070 camera mapping differs from the validation view order",
    )

    import torch

    require(torch.cuda.is_available(), "RGB-070 requires the frozen CUDA renderer")
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
        not any(parameter.requires_grad for _name, parameter in frozen_rgb_named_parameters(scaffold)),
        "RGB-070 found trainable frozen RGB parameters",
    )
    pipe = pipeline()

    started = time.perf_counter()
    rendered = []
    sample_names = set(config["outputs"]["visual_samples"])
    for index, record in enumerate(mapping, start=1):
        name = record["target_name"]
        _selected, _camera, view = s0.official_view(
            args.data_root, name, int(record["appearance_id"])
        )
        with torch.inference_mode():
            rgb = s0.tensor_to_rgb(render_frozen_rgb(view, scaffold, pipe))
        output = args.output_dir / "renders" / name
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb, mode="RGB").save(output)
        rendered.append(
            {
                **record,
                "render_path": str(output.relative_to(args.output_dir)),
                "render_sha256": s80.sha256(output),
            }
        )
        if name in sample_names or index == len(mapping):
            print(
                json.dumps(
                    {
                        "rendered": index,
                        "total": len(mapping),
                        "target": name,
                        "appearance_source": record["source_name"],
                        "center_distance": record["center_distance"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    artifact_after = frozen_rgb_hashes(args.rgb_run_dir, iteration)
    require(artifact_before == artifact_after, "RGB-070 changed a frozen RGB artifact")
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "completed_render_pending_metric_summary",
        "candidate_variable": "nearest_training_camera_center_appearance_embedding",
        "selection_uses_reference_rgb_or_labels": False,
        "validation_views": len(rendered),
        "training_embeddings": len(train_records),
        "render_seconds": time.perf_counter() - started,
        "center_distance": {
            "minimum": float(min(record["center_distance"] for record in mapping)),
            "median": float(np.median([record["center_distance"] for record in mapping])),
            "maximum": float(max(record["center_distance"] for record in mapping)),
        },
        "rgb_artifact_hashes_before": artifact_before,
        "rgb_artifact_hashes_after": artifact_after,
        "rgb_artifact_hashes_equal": artifact_before == artifact_after,
        "renders": rendered,
    }
    s80.write_json(args.output_dir / "render_result.json", result)
    print(json.dumps({key: result[key] for key in ("status", "validation_views", "render_seconds")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
