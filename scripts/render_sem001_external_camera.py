#!/usr/bin/env python3
"""Render a SEM-001 checkpoint from arbitrary externally supplied COLMAP cameras.

This is a submission-interface gate, not an evaluator: it accepts only model
artifacts and the external camera files, never a reference RGB or label.  The
input pose is COLMAP CAMERA_FROM_WORLD.  RGB and semantic probabilities are
rendered in the frozen R1 PINHOLE space, then mapped to the external
SIMPLE_RADIAL grid whose width and height come directly from ``cameras.txt``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from render_external_colmap_camera import make_view, read_pose, read_simple_radial_camera  # noqa: E402
from run_sem001 import reload_sidecar, tensor_to_rgb  # noqa: E402
from src.camera_adapter import Sim3, official_to_model_camera, redistort_rgb, redistort_semantic_probabilities  # noqa: E402
from src.scaffold_semantic_sidecar import (  # noqa: E402
    ScaffoldSettings,
    configure_scaffold_import,
    frozen_rgb_hashes,
    load_frozen_scaffold,
    pipeline,
    render_frozen_rgb,
    render_semantic_probabilities,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Git-ignored semantic_sidecar.pth")
    parser.add_argument("--rgb-run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True, help="Externally supplied official COLMAP cameras.txt")
    parser.add_argument("--images", type=Path, required=True, help="Externally supplied official COLMAP images.txt")
    parser.add_argument("--image-name", required=True, help="Pose name selected from --images")
    parser.add_argument("--camera-id", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    frozen = config["frozen_rgb"]
    settings = ScaffoldSettings(train_camera_count=int(frozen["train_camera_count"]), n_offsets=int(frozen["n_offsets"]))
    point_cloud = args.rgb_run_dir / "point_cloud" / f"iteration_{frozen['iteration']}" / "point_cloud.ply"
    rgb_hashes = frozen_rgb_hashes(args.rgb_run_dir, int(frozen["iteration"]))
    configure_scaffold_import(args.source_root)
    model = load_frozen_scaffold(args.source_root, point_cloud, settings)
    sidecar = reload_sidecar(args.checkpoint, int(model.get_anchor.shape[0]), config, rgb_hashes)
    pipe = pipeline()

    camera_id, camera = read_simple_radial_camera(args.cameras, args.camera_id)
    selected_name, official_pose = read_pose(args.images, args.image_name, camera_id)
    identity = Sim3(1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
    view = make_view(camera, official_to_model_camera(official_pose, identity), int(frozen["appearance_camera_id"]))
    rgb_pinhole = tensor_to_rgb(render_frozen_rgb(view, model, pipe))
    semantic_pinhole = render_semantic_probabilities(view, model, sidecar, pipe).detach().cpu().numpy().astype(np.float32)
    rgb = redistort_rgb(rgb_pinhole.astype(np.float32) / 255.0, camera)
    probabilities = redistort_semantic_probabilities(semantic_pinhole, camera)
    logits = np.log(np.clip(probabilities, np.finfo(np.float32).tiny, 1.0))
    labels = logits.argmax(axis=0).astype(np.uint8)
    if labels.shape != (camera.height, camera.width) or labels.min() < 0 or labels.max() > 4:
        raise RuntimeError("Semantic PNG violates the five-class output contract")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(rgb.clip(0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB").save(args.output_dir / "rgb.png")
    Image.fromarray(labels, mode="L").save(args.output_dir / "semantic.png")
    np.save(args.output_dir / "semantic_logits.npy", logits)
    evidence = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "selected_pose_name": selected_name,
        "input_camera_model": "SIMPLE_RADIAL",
        "input_pose_convention": "CAMERA_FROM_WORLD",
        "official_to_model_sim3": {"scale": 1.0, "rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0]},
        "output_grid": "official_SIMPLE_RADIAL",
        "rgb_shape": list(rgb.shape),
        "semantic_logits_shape": list(logits.shape),
        "semantic_png_shape": list(labels.shape),
        "semantic_png_dtype": str(labels.dtype),
        "semantic_png_min": int(labels.min()),
        "semantic_png_max": int(labels.max()),
        "uses_reference_rgb_or_labels": False,
    }
    (args.output_dir / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
