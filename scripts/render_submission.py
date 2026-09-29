#!/usr/bin/env python3
"""Batch-render the frozen Scaffold-GS + SEM-001 bundle from external COLMAP cameras.

Only camera metadata and frozen model artifacts are accepted.  No RGB image or
semantic label path exists in this interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from render_external_colmap_camera import IMAGE_SUFFIXES, _data_lines, make_view, read_pose, read_simple_radial_camera  # noqa: E402
from src.camera_adapter import Sim3, official_to_model_camera, redistort_rgb, redistort_semantic_probabilities  # noqa: E402
from src.scaffold_semantic_sidecar import (  # noqa: E402
    AnchorSemanticSidecar,
    ScaffoldSettings,
    configure_scaffold_import,
    frozen_rgb_hashes,
    load_frozen_scaffold,
    pipeline,
    render_frozen_rgb,
    render_semantic_probabilities,
)


CLASS_NAMES = ("background", "deck", "stay_cable", "tower", "foundation")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def list_external_poses(images: Path) -> list[tuple[str, int]]:
    poses: list[tuple[str, int]] = []
    names: set[str] = set()
    for fields in _data_lines(images):
        if len(fields) < 10 or Path(fields[9]).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        name = Path(fields[9]).name
        if name in names:
            raise ValueError(f"Duplicate external pose basename: {name}")
        names.add(name)
        poses.append((name, int(fields[8])))
    if not poses:
        raise ValueError(f"No image poses found in {images}")
    return poses


def select_external_poses(poses: list[tuple[str, int]], requested: list[str] | None) -> list[tuple[str, int]]:
    if not requested:
        return poses
    requested_names = [Path(name).name for name in requested]
    if len(set(requested_names)) != len(requested_names):
        raise ValueError("Requested pose names must be unique")
    by_name = {name: (name, camera_id) for name, camera_id in poses}
    missing = [name for name in requested_names if name not in by_name]
    if missing:
        raise ValueError(f"Requested poses absent from images.txt: {missing}")
    return [by_name[name] for name in requested_names]


def _load_sidecar(path: Path, anchor_count: int, config: dict[str, Any], rgb_hashes: dict[str, str]) -> AnchorSemanticSidecar:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("experiment_id") != config["experiment_id"]:
        raise ValueError("Semantic checkpoint experiment mismatch")
    if payload.get("class_names") != list(CLASS_NAMES) or payload.get("semantic_storage") != "per_anchor_logits":
        raise ValueError("Semantic checkpoint class/storage schema mismatch")
    if payload.get("frozen_rgb_hashes") != rgb_hashes:
        raise ValueError("Semantic checkpoint does not match the bundled frozen RGB artifacts")
    sidecar = AnchorSemanticSidecar(anchor_count)
    sidecar.load_state_dict(payload)
    sidecar.logits.requires_grad_(False)
    return sidecar


def _tensor_rgb_u8(array: Any) -> np.ndarray:
    return np.rint(array.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)


def _verify_locked_artifacts(bundle: Path, lock: dict[str, Any]) -> None:
    for entry in lock["artifacts"]:
        path = bundle / entry["bundle_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise ValueError(f"Bundle artifact hash mismatch for {entry['bundle_path']}: {actual}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, help="Override vendored Scaffold source for a controlled audit")
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--image-name", action="append", dest="image_names", help="Repeat to render a subset; default renders all poses")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--write-logits", action="store_true")
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty render directory: {args.output_dir}")
    lock = json.loads((bundle / "code/configs/submission/sem001_baseline_v1.json").read_text(encoding="utf-8"))
    config = json.loads((bundle / "metadata/sem001_config.json").read_text(encoding="utf-8"))
    _verify_locked_artifacts(bundle, lock)
    if config["semantic"]["class_mapping"].get("stay_cable") != 2:
        raise ValueError("Frozen class mapping does not assign stay_cable to ID 2")

    rgb_run_dir = bundle / "model/rgb"
    checkpoint = bundle / "model/semantic/semantic_sidecar.pth"
    source_root = (args.source_root or (bundle / "third_party/scaffold-gs")).resolve()
    frozen = config["frozen_rgb"]
    settings = ScaffoldSettings(train_camera_count=int(frozen["train_camera_count"]), n_offsets=int(frozen["n_offsets"]))
    point_cloud = rgb_run_dir / "point_cloud" / f"iteration_{frozen['iteration']}" / "point_cloud.ply"
    rgb_hashes = frozen_rgb_hashes(rgb_run_dir, int(frozen["iteration"]))
    configure_scaffold_import(source_root)
    model = load_frozen_scaffold(source_root, point_cloud, settings)
    sidecar = _load_sidecar(checkpoint, int(model.get_anchor.shape[0]), config, rgb_hashes)
    pipe = pipeline()

    poses = select_external_poses(list_external_poses(args.images), args.image_names)
    cameras: dict[int, Any] = {}
    for camera_id in sorted({camera_id for _, camera_id in poses}):
        cameras[camera_id] = read_simple_radial_camera(args.cameras, camera_id)[1]

    import torch

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    outputs = []
    identity = Sim3(1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
    for requested_name, camera_id in poses:
        camera = cameras[camera_id]
        selected_name, official_pose = read_pose(args.images, requested_name, camera_id)
        view = make_view(camera, official_to_model_camera(official_pose, identity), int(frozen["appearance_camera_id"]))
        with torch.inference_mode():
            rgb_pinhole = _tensor_rgb_u8(render_frozen_rgb(view, model, pipe))
            semantic_pinhole = render_semantic_probabilities(view, model, sidecar, pipe).cpu().numpy().astype(np.float32)
        rgb = np.rint(redistort_rgb(rgb_pinhole.astype(np.float32) / 255.0, camera).clip(0.0, 1.0) * 255.0).astype(np.uint8)
        probabilities = redistort_semantic_probabilities(semantic_pinhole, camera)
        logits = np.log(np.clip(probabilities, np.finfo(np.float32).tiny, 1.0)).astype(np.float32)
        labels = logits.argmax(axis=0).astype(np.uint8)
        if rgb.shape != (camera.height, camera.width, 3) or labels.shape != (camera.height, camera.width):
            raise RuntimeError(f"Output dimensions do not match camera for {selected_name}")
        if labels.min() < 0 or labels.max() > 4:
            raise RuntimeError(f"Semantic output contains an invalid class ID for {selected_name}")

        stem = Path(selected_name).stem
        rgb_path = args.output_dir / "rgb" / f"{stem}.png"
        semantic_path = args.output_dir / "semantic" / f"{stem}.png"
        rgb_path.parent.mkdir(parents=True, exist_ok=True)
        semantic_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb, mode="RGB").save(rgb_path)
        Image.fromarray(labels, mode="L").save(semantic_path)
        record = {
            "pose_name": selected_name,
            "camera_id": camera_id,
            "width": camera.width,
            "height": camera.height,
            "rgb": {"path": rgb_path.relative_to(args.output_dir).as_posix(), "sha256": sha256(rgb_path)},
            "semantic": {
                "path": semantic_path.relative_to(args.output_dir).as_posix(),
                "sha256": sha256(semantic_path),
                "dtype": "uint8",
                "min": int(labels.min()),
                "max": int(labels.max()),
            },
        }
        if args.write_logits:
            logits_path = args.output_dir / "logits" / f"{stem}.npy"
            logits_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(logits_path, logits)
            record["logits"] = {
                "path": logits_path.relative_to(args.output_dir).as_posix(),
                "sha256": sha256(logits_path),
                "shape": list(logits.shape),
                "dtype": str(logits.dtype),
            }
        outputs.append(record)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    manifest = {
        "schema_version": 1,
        "submission_id": lock["submission_id"],
        "rgb_experiment": lock["rgb_experiment"],
        "semantic_experiment": lock["semantic_experiment"],
        "input_camera_model": "SIMPLE_RADIAL",
        "input_pose_convention": "CAMERA_FROM_WORLD",
        "official_to_model_sim3": {"scale": 1.0, "rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0]},
        "output_grid": "official_SIMPLE_RADIAL",
        "uses_reference_rgb_or_labels": False,
        "appearance_fallback": {"embedding_index": int(frozen["appearance_camera_id"]), "training_embeddings": int(frozen["train_camera_count"])},
        "cameras_txt_sha256": sha256(args.cameras),
        "images_txt_sha256": sha256(args.images),
        "rendered_views": len(outputs),
        "elapsed_seconds": elapsed,
        "peak_cuda_allocated_mib": int(torch.cuda.max_memory_allocated() / (1024 * 1024)),
        "outputs": outputs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "submission_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rendered_views": len(outputs), "elapsed_seconds": elapsed, "status": "passed"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
