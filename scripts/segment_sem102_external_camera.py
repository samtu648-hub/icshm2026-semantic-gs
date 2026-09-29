#!/usr/bin/env python3
"""Apply a frozen SEM-102 SegFormer checkpoint to external frozen RGB output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem005_segformer as segformer
import run_sem080_structural_diag as s80

from src.camera_adapter import (
    SimpleRadialCamera,
    redistort_rgb,
    redistort_semantic_probabilities,
)
from src.semantic_rendered_final import validate_final_training_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--final-result", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--external-rgb-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_external_rgb(path: Path, *, crop: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not read external frozen RGB image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < crop:
        raise ValueError("External frozen RGB must be HxWx3 and at least one crop wide")
    return image


def save_rgb(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)
    return s80.sha256(path)


def main() -> int:
    args = parse_args()
    config = s80.load_json(args.config)
    require(
        config.get("experiment_id") == "SEM-102-RENDERED-SEG-FINAL",
        "SEM-102 external segmenter received another config",
    )
    require(
        config.get("status") == "registered_not_run",
        "SEM-102 external segmenter requires its registered-not-run config",
    )
    require(args.final_checkpoint.is_file(), "SEM-102 final checkpoint is missing")
    final_result = s80.load_json(args.final_result)
    checkpoint_sha256 = s80.sha256(args.final_checkpoint)
    validate_final_training_result(
        final_result,
        checkpoint_sha256=checkpoint_sha256,
        config_sha256=s80.sha256(args.config),
    )
    record_path = args.external_rgb_root / "external_rgb_record.json"
    pinhole_path = args.external_rgb_root / "rgb_pinhole.png"
    require(
        record_path.is_file() and pinhole_path.is_file(),
        "SEM-102 external RGB evidence is incomplete",
    )
    record = s80.load_json(record_path)
    require(
        record.get("experiment_id") == config["experiment_id"]
        and record.get("status") == "completed_external_frozen_rgb",
        "SEM-102 external RGB record is invalid",
    )
    require(
        record.get("final_checkpoint_sha256") == checkpoint_sha256,
        "SEM-102 external RGB uses another checkpoint",
    )
    require(
        record.get("pinhole_rgb", {}).get("sha256") == s80.sha256(pinhole_path),
        "SEM-102 external PINHOLE RGB hash changed",
    )
    require(
        record.get("uses_reference_rgb_or_labels") is False,
        "SEM-102 external RGB record accessed reference data",
    )
    require(
        not args.output_dir.exists() or not any(args.output_dir.iterdir()),
        "Refusing to overwrite SEM-102 external semantic output",
    )
    camera_data = record["camera"]
    camera = SimpleRadialCamera(
        width=int(camera_data["width"]),
        height=int(camera_data["height"]),
        focal=float(camera_data["focal"]),
        cx=float(camera_data["cx"]),
        cy=float(camera_data["cy"]),
        radial_k=float(camera_data["radial_k"]),
    )
    crop = int(config["inference"]["crop_size"])
    pinhole = read_external_rgb(pinhole_path, crop=crop)
    require(
        pinhole.shape == tuple(record["pinhole_rgb"]["shape"]),
        "SEM-102 external PINHOLE RGB shape changed",
    )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("SEM-102 external segmenter requires a visible CUDA GPU")
    device = torch.device("cuda:0")
    if torch.cuda.is_bf16_supported():
        amp_name, amp_dtype = "bf16", torch.bfloat16
    else:
        amp_name, amp_dtype = "fp16", torch.float16
    model, metadata = segformer.reload_checkpoint(
        args.final_checkpoint,
        args.model_dir,
        int(config["training"]["seed"]),
        device,
    )
    require(
        metadata.get("experiment_id") == config["experiment_id"],
        "SEM-102 external segmenter loaded another checkpoint",
    )
    probability = segformer.sliding_probabilities(
        model,
        pinhole,
        crop,
        int(config["inference"]["stride"]),
        device,
        amp_dtype,
    )
    del model
    torch.cuda.empty_cache()
    require(
        np.all(np.isfinite(probability))
        and np.all(probability >= 0.0)
        and np.allclose(probability.sum(axis=0), 1.0, atol=2e-5),
        "SEM-102 external probabilities violate the five-class simplex",
    )
    official_rgb = np.rint(
        redistort_rgb(pinhole.astype(np.float32) / 255.0, camera).clip(0.0, 1.0) * 255.0
    ).astype(np.uint8)
    official_probability = redistort_semantic_probabilities(probability, camera)
    logits = np.log(np.clip(official_probability, np.finfo(np.float32).tiny, 1.0))
    labels = logits.argmax(axis=0).astype(np.uint8)
    require(
        official_rgb.shape == (camera.height, camera.width, 3)
        and logits.shape == (5, camera.height, camera.width)
        and labels.shape == (camera.height, camera.width)
        and labels.min() >= 0
        and labels.max() < 5,
        "SEM-102 external output violates the RGB/logit/class-ID contract",
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rgb_sha = save_rgb(args.output_dir / "rgb.png", official_rgb)
    Image.fromarray(labels, mode="L").save(args.output_dir / "semantic.png")
    np.save(args.output_dir / "semantic_logits.npy", logits.astype(np.float32))
    evidence = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "completed_official_external_camera_semantic",
        "final_training_result_sha256": s80.sha256(args.final_result),
        "final_checkpoint_sha256": checkpoint_sha256,
        "external_rgb_record_sha256": s80.sha256(record_path),
        "selected_pose_name": record["selected_pose_name"],
        "input_camera_model": "SIMPLE_RADIAL",
        "input_pose_convention": "CAMERA_FROM_WORLD",
        "official_to_model_sim3": record["official_to_model_sim3"],
        "output_grid": "official_SIMPLE_RADIAL",
        "rgb_sha256": rgb_sha,
        "rgb_shape": list(official_rgb.shape),
        "semantic_logits_shape": list(logits.shape),
        "semantic_logits_dtype": str(logits.dtype),
        "semantic_png_shape": list(labels.shape),
        "semantic_png_dtype": str(labels.dtype),
        "semantic_png_min": int(labels.min()),
        "semantic_png_max": int(labels.max()),
        "probability_simplex": True,
        "amp": amp_name,
        "uses_reference_rgb_or_labels": False,
    }
    s80.write_json(args.output_dir / "evidence.json", evidence)
    print(json.dumps(evidence, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
