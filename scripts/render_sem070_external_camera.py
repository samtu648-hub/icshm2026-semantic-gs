#!/usr/bin/env python3
"""Render SEM-001 raw and SEM-070-cleaned semantics from official cameras.

This interface accepts only frozen model artifacts, the SEM-001 checkpoint,
the independently saved SEM-070 boolean anchor mask, and official COLMAP
camera metadata.  It supports official ``CAMERA_FROM_WORLD`` poses with
``SIMPLE_RADIAL`` cameras and never reads a target RGB image or semantic label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem001 as s0
from run_sem070_post_clean import render_external_pair

from src.scaffold_semantic_sidecar import (
    ScaffoldSettings,
    configure_scaffold_import,
    frozen_rgb_hashes,
    load_frozen_scaffold,
    pipeline,
)
from src.semantic_postprocess import load_suppression_mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="SEM-070 config")
    parser.add_argument("--sem001-config", type=Path, default=REPO_ROOT / "experiments/runs/SEM-001/config.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Frozen SEM-001 semantic_sidecar.pth")
    parser.add_argument("--mask", type=Path, required=True, help="SEM-070 postprocess_mask.npz")
    parser.add_argument("--rgb-run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True, help="Official COLMAP cameras.txt")
    parser.add_argument("--images", type=Path, required=True, help="Official COLMAP images.txt")
    parser.add_argument("--image-name", required=True)
    parser.add_argument("--camera-id", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    sem001_config = json.loads(args.sem001_config.read_text(encoding="utf-8"))
    expected_checkpoint_sha256 = config.get("frozen_semantic", {}).get("checkpoint_sha256")
    if expected_checkpoint_sha256 and hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != expected_checkpoint_sha256:
        raise ValueError("SEM-001 checkpoint SHA-256 differs from the frozen SEM-070 config")
    frozen = config["frozen_rgb"]
    settings = ScaffoldSettings(train_camera_count=int(frozen["train_camera_count"]), n_offsets=int(frozen["n_offsets"]))
    point_cloud = args.rgb_run_dir / "point_cloud" / f"iteration_{frozen['iteration']}" / "point_cloud.ply"
    rgb_hashes = frozen_rgb_hashes(args.rgb_run_dir, int(frozen["iteration"]))
    configure_scaffold_import(args.source_root)
    model = load_frozen_scaffold(args.source_root, point_cloud, settings)
    sidecar = s0.reload_sidecar(args.checkpoint, int(model.get_anchor.shape[0]), sem001_config, rgb_hashes)
    mask = load_suppression_mask(args.mask, int(model.get_anchor.shape[0]))
    evidence = render_external_pair(
        config=config,
        data_cameras=args.cameras,
        data_images=args.images,
        image_name=args.image_name,
        camera_id=args.camera_id,
        output_dir=args.output_dir,
        model=model,
        sidecar=sidecar,
        pipe=pipeline(),
        mask=mask,
    )
    evidence["mask_path"] = str(args.mask)
    evidence["mask_sha256"] = hashlib.sha256(args.mask.read_bytes()).hexdigest()
    (args.output_dir / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
