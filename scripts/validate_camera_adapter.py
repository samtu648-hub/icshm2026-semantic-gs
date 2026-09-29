#!/usr/bin/env python3
"""Check that the R1 PINHOLE preprocessing round-trips to the released pixel grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from camera_adapter import SimpleRadialCamera, redistort_rgb  # noqa: E402
from prepare_rgb_colmap_scene import parse_single_simple_radial_camera  # noqa: E402


def raw_image_path(dataset_root: Path, name: str) -> Path:
    for directory in ("images", "unlabeled_Images"):
        candidate = dataset_root / directory / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Released image not found: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prepared-scene", type=Path, required=True)
    parser.add_argument("--max-mae", type=float, default=4.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads((args.prepared_scene / "preparation_manifest.json").read_text(encoding="utf-8"))
    _, width, height, focal, cx, cy, radial = parse_single_simple_radial_camera(
        args.dataset_root / "camera_parameters" / "cameras.txt"
    )
    camera = SimpleRadialCamera(width, height, focal, cx, cy, radial)
    image_names = sorted(path.name for path in (args.prepared_scene / "images").iterdir() if path.is_file())
    if not image_names:
        raise ValueError("Prepared scene contains no images")
    name = image_names[0]
    raw = cv2.imread(str(raw_image_path(args.dataset_root, name)), cv2.IMREAD_COLOR)
    prepared = cv2.imread(str(args.prepared_scene / "images" / name), cv2.IMREAD_COLOR)
    if raw is None or prepared is None:
        raise ValueError("Could not read raw or prepared image")
    reconstructed = redistort_rgb(prepared, camera)
    mae = float(np.mean(np.abs(raw.astype(np.float32) - reconstructed.astype(np.float32))))
    payload = {
        "split_id": manifest["split_id"],
        "image": name,
        "mae_raw_vs_undistort_redistort": mae,
        "max_mae": args.max_mae,
        "passes": mae <= args.max_mae,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
