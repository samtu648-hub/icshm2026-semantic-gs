#!/usr/bin/env python3
"""Render frozen SEM-001 class maps for SEM-110's authorised training views.

This is an inference-only, sharded extractor.  It never accepts labels,
validation cameras, unlabelled images, real RGB or an external-camera mode.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem001 as sem001
import run_sem080_structural_diag as common

from src.scaffold_semantic_sidecar import (
    ScaffoldSettings,
    configure_scaffold_import,
    frozen_rgb_hashes,
    load_frozen_scaffold,
    pipeline,
    render_semantic_probabilities,
)
from src.semantic_rendered_training import select_rendered_training_views


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sem001-config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--sem001-checkpoint", type=Path, required=True)
    parser.add_argument("--rgb-run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("shard", "finalize"), required=True)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
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


def validate_config(config: dict[str, Any]) -> None:
    require(
        config.get("experiment_id") == "SEM-110-PROTECTED-FUSION-OOF",
        "SEM-110 renderer received another experiment config",
    )
    require(
        config.get("status") == "registered_not_run",
        "SEM-110 renderer may only run from its registered configuration",
    )


def render_shard(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    sem001_config: dict[str, Any],
    names: list[str],
) -> None:
    require(
        args.start is not None and args.end is not None,
        "SEM-110 shard requires --start and --end",
    )
    require(
        0 <= args.start < args.end <= len(names),
        "SEM-110 shard bounds are outside the training plan",
    )
    require(args.sem001_checkpoint.is_file(), "SEM-110 SEM-001 checkpoint is missing")
    expected_checkpoint = config["inputs"]["sem001_checkpoint_sha256"]
    require(
        sha256(args.sem001_checkpoint) == expected_checkpoint,
        "SEM-110 SEM-001 checkpoint hash differs from config",
    )
    selected = names[args.start : args.end]
    image_root = args.output_dir / "semantic_png"
    shard_path = args.output_dir / "shards" / f"{args.start:03d}-{args.end:03d}.json"
    require(
        not shard_path.exists(), "SEM-110 refuses to overwrite a completed render shard"
    )
    require(
        not any((image_root / name).exists() for name in selected),
        "SEM-110 render shard overlaps existing outputs",
    )

    settings = ScaffoldSettings(
        train_camera_count=int(sem001_config["frozen_rgb"]["train_camera_count"]),
        n_offsets=int(sem001_config["frozen_rgb"]["n_offsets"]),
    )
    point_cloud = (
        args.rgb_run_dir
        / "point_cloud"
        / f"iteration_{sem001_config['frozen_rgb']['iteration']}"
        / "point_cloud.ply"
    )
    rgb_hashes = frozen_rgb_hashes(
        args.rgb_run_dir, int(sem001_config["frozen_rgb"]["iteration"])
    )
    configure_scaffold_import(args.source_root)
    model = load_frozen_scaffold(args.source_root, point_cloud, settings)
    sidecar = sem001.reload_sidecar(
        args.sem001_checkpoint,
        int(model.get_anchor.shape[0]),
        sem001_config,
        rgb_hashes,
    )
    pipe = pipeline()
    records: list[dict[str, str]] = []
    import torch

    for name in selected:
        _, _, view = sem001.official_view(
            args.data_root,
            name,
            int(sem001_config["frozen_rgb"]["appearance_camera_id"]),
        )
        with torch.no_grad():
            prediction = (
                render_semantic_probabilities(view, model, sidecar, pipe)
                .argmax(dim=0)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
        if prediction.shape != (989, 1320) or int(prediction.max()) >= 5:
            raise RuntimeError(f"SEM-110 rendered invalid class map for {name}")
        output = image_root / name
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(prediction).save(output)
        records.append(
            {
                "name": name,
                "semantic_png": str(output.relative_to(args.output_dir)),
                "sha256": sha256(output),
            }
        )
    common.write_json(
        shard_path,
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "mode": "training_sem001_render_shard",
            "range": [args.start, args.end],
            "rendered": records,
            "training_only_access": {
                "labels_opened": False,
                "fixed_validation_views_opened": False,
                "unlabeled_images_opened": False,
                "real_rgb_input_opened": False,
                "external_camera_mode_present": False,
            },
        },
    )


def finalize(
    *, args: argparse.Namespace, config: dict[str, Any], names: list[str]
) -> None:
    require(
        args.start is None and args.end is None,
        "SEM-110 finalization does not accept shard bounds",
    )
    rendered: list[dict[str, str]] = []
    image_root = args.output_dir / "semantic_png"
    for name in names:
        output = image_root / name
        require(output.is_file(), f"SEM-110 training render is missing {name}")
        prediction = np.asarray(Image.open(output), dtype=np.uint8)
        require(
            prediction.shape == (989, 1320), f"SEM-110 render has a wrong shape: {name}"
        )
        require(
            int(prediction.max()) < 5, f"SEM-110 render has invalid class IDs: {name}"
        )
        rendered.append(
            {
                "name": name,
                "semantic_png": str(output.relative_to(args.output_dir)),
                "sha256": sha256(output),
            }
        )
    common.write_json(
        args.output_dir / "training_sem001_render_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "status": "completed_training_sem001_render",
            "sem001_checkpoint_sha256": config["inputs"]["sem001_checkpoint_sha256"],
            "rendered": rendered,
            "training_only_access": {
                "labels_opened": False,
                "fixed_validation_views_opened": False,
                "unlabeled_images_opened": False,
                "real_rgb_input_opened": False,
                "external_camera_mode_present": False,
            },
        },
    )


def main() -> int:
    args = parse_args()
    config = common.load_json(args.config)
    sem001_config = common.load_json(args.sem001_config)
    split = common.load_json(args.split)
    validate_config(config)
    require(
        sem001_config.get("experiment_id") == "SEM-001",
        "SEM-110 requires the SEM-001 config",
    )
    names = select_rendered_training_views(
        split, expected_count=int(config["data"]["training_images"])
    )
    if args.mode == "shard":
        render_shard(args=args, config=config, sem001_config=sem001_config, names=names)
    else:
        finalize(args=args, config=config, names=names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
