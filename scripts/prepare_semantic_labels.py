#!/usr/bin/env python3
"""Convert released LabelMe polygons into SEM-001 full-frame PINHOLE labels.

R1 trained on full-frame RGB images obtained by OpenCV undistortion with the
released SIMPLE_RADIAL K retained.  LabelMe vertices start in that released,
distorted grid; they are transformed with the identical camera model before
being rasterized.  No RGB value is inspected while labels are built.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


CLASS_IDS = {"background": 0, "deck": 1, "stay_cable": 2, "tower": 3, "foundation": 4}
IMAGE_SIZE = (1320, 989)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_camera(path: Path) -> tuple[np.ndarray, np.ndarray]:
    records = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    if len(records) != 1 or len(records[0]) != 8 or records[0][1] != "SIMPLE_RADIAL":
        raise ValueError("SEM-001 expects exactly one released SIMPLE_RADIAL camera")
    _, _, width, height, focal, cx, cy, radial = records[0]
    if (int(width), int(height)) != IMAGE_SIZE:
        raise ValueError(f"Unexpected released camera size {(width, height)}")
    matrix = np.array([[float(focal), 0.0, float(cx)], [0.0, float(focal), float(cy)], [0.0, 0.0, 1.0]])
    distortion = np.array([float(radial), 0.0, 0.0, 0.0, 0.0])
    return matrix, distortion


def polygon_points(shape: dict[str, object], matrix: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    if shape.get("shape_type", "polygon") != "polygon":
        raise ValueError(f"Unsupported LabelMe shape_type={shape.get('shape_type')!r}; SEM-001 only admits polygons")
    raw = np.asarray(shape.get("points"), dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] < 3 or raw.shape[1] != 2 or not np.isfinite(raw).all():
        raise ValueError("Invalid LabelMe polygon points")
    width, height = IMAGE_SIZE
    # LabelMe contains two vertices at about -2e-16 px.  They are numerical
    # round-off at the left image boundary, not geometric outliers.  Permit a
    # tiny closed-boundary tolerance and make the resulting rasterization
    # deterministic by clipping to the valid pixel coordinate range.
    tolerance = 1e-6
    if (raw[:, 0] < -tolerance).any() or (raw[:, 0] > width + tolerance).any() or (raw[:, 1] < -tolerance).any() or (raw[:, 1] > height + tolerance).any():
        raise ValueError("LabelMe polygon has vertices outside the released image grid")
    raw = raw.copy()
    raw[:, 0] = np.clip(raw[:, 0], 0.0, width - 1.0)
    raw[:, 1] = np.clip(raw[:, 1], 0.0, height - 1.0)
    converted = cv2.undistortPoints(raw.reshape(-1, 1, 2), matrix, distortion, P=matrix).reshape(-1, 2)
    return np.rint(converted).astype(np.int32)


def make_label(json_path: Path, matrix: np.ndarray, distortion: np.ndarray) -> tuple[np.ndarray, Counter[str]]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if (int(payload.get("imageWidth", -1)), int(payload.get("imageHeight", -1))) != IMAGE_SIZE:
        raise ValueError(f"{json_path.name}: unexpected LabelMe image size")
    label = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=np.uint8)
    labels: Counter[str] = Counter()
    for shape in payload.get("shapes", []):
        if not isinstance(shape, dict):
            raise ValueError(f"{json_path.name}: non-object LabelMe shape")
        name = shape.get("label")
        if name not in CLASS_IDS or name == "background":
            raise ValueError(f"{json_path.name}: unsupported official label {name!r}")
        cv2.fillPoly(label, [polygon_points(shape, matrix, distortion)], color=CLASS_IDS[name])
        labels[str(name)] += 1
    return label, labels


def overlay(rgb_path: Path, label: np.ndarray, output: Path) -> None:
    palette = np.array([[0, 0, 0], [230, 86, 37], [255, 210, 0], [38, 139, 210], [133, 153, 0]], dtype=np.uint8)
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    if rgb.shape[:2] != label.shape:
        raise ValueError(f"QC image {rgb_path} has wrong size {rgb.shape[:2]}")
    color = palette[label]
    mask = label != 0
    result = rgb.copy()
    result[mask] = np.rint(0.45 * rgb[mask] + 0.55 * color[mask]).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result, mode="RGB").save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Git-ignored label PNG directory")
    parser.add_argument("--manifest", type=Path, required=True, help="Small repository-tracked manifest")
    parser.add_argument("--prepared-image-dir", type=Path, help="R1 PINHOLE images for QC overlays")
    parser.add_argument("--qc-dir", type=Path, help="Git-ignored QC overlay directory")
    args = parser.parse_args()

    split = json.loads(args.split.read_text(encoding="utf-8"))
    selected = sorted(set(split["semantic_train_labeled"]) | set(split["semantic_validation_labeled"]))
    if len(selected) != 293:
        raise ValueError("SEM-001 must materialize exactly 219 train + 74 validation labels")
    camera, distortion = read_camera(args.data_root / "camera_parameters" / "cameras.txt")
    source_dir = args.data_root / "json"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    all_pixels: Counter[str] = Counter()
    all_shapes: Counter[str] = Counter()
    for name in selected:
        source = source_dir / f"{Path(name).stem}.json"
        if not source.is_file():
            raise FileNotFoundError(f"Missing LabelMe annotation: {source}")
        label, shape_counts = make_label(source, camera, distortion)
        destination = args.output_dir / name
        Image.fromarray(label, mode="L").save(destination)
        unique, counts = np.unique(label, return_counts=True)
        pixels = {str(int(key)): int(value) for key, value in zip(unique, counts)}
        for key, value in pixels.items():
            all_pixels[key] += value
        all_shapes.update(shape_counts)
        records.append(
            {
                "image": name,
                "source_json": f"json/{source.name}",
                "source_json_sha256": sha256(source),
                "label_png": f"labels/{name}",
                "label_png_sha256": sha256(destination),
                "shape_counts": dict(sorted(shape_counts.items())),
                "pixel_counts": dict(sorted(pixels.items())),
            }
        )
        if args.prepared_image_dir is not None and args.qc_dir is not None and name in {"001.png", "054.png", "140.png"}:
            overlay(args.prepared_image_dir / name, label, args.qc_dir / f"{Path(name).stem}_overlay.png")

    train_set = set(split["semantic_train_labeled"])
    train_pixels: Counter[str] = Counter()
    validation_pixels: Counter[str] = Counter()
    for record in records:
        target = train_pixels if record["image"] in train_set else validation_pixels
        target.update({key: int(value) for key, value in dict(record["pixel_counts"]).items()})
    manifest = {
        "schema_version": 1,
        "label_manifest_id": "sem_001_label_manifest_v1",
        "split_id": split["split_id"],
        "image_count": len(records),
        "class_mapping": CLASS_IDS,
        "source_camera": "released camera_parameters/cameras.txt SIMPLE_RADIAL",
        "output_grid": "R1 full-frame PINHOLE, 1320x989, same K, no crop",
        "coordinate_alignment": "Each LabelMe polygon vertex is first accepted within a 1e-6 px closed-boundary tolerance, clipped to the valid source grid, mapped distorted-to-PINHOLE by cv2.undistortPoints(K, [k1,0,0,0,0], P=K), rounded to nearest pixel, then rasterized with cv2.fillPoly.",
        "overlap_rule": "LabelMe source-list z-order: each later polygon overwrites already-filled pixels.",
        "ignore_policy": "No released ignore/void labels; every pixel is evaluated as one of official IDs 0..4.",
        "source_rgb_used": False,
        "shape_counts": dict(sorted(all_shapes.items())),
        "pixel_counts_all": dict(sorted(all_pixels.items())),
        "pixel_counts_train": dict(sorted(train_pixels.items())),
        "pixel_counts_validation": dict(sorted(validation_pixels.items())),
        "records": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "train_pixels": dict(train_pixels), "validation_pixels": dict(validation_pixels)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
