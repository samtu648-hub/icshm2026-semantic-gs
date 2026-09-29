#!/usr/bin/env python3
"""Read-only audit for the released IC-SHM Project 2 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
OFFICIAL_LABELS = {"deck", "stay_cable", "tower", "foundation"}


def image_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
        if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
            width, height = struct.unpack(">II", header[16:24])
            return width, height

        if header[:2] == b"\xff\xd8":
            handle.seek(2)
            while True:
                byte = handle.read(1)
                if not byte:
                    break
                if byte != b"\xff":
                    continue
                marker = handle.read(1)
                while marker == b"\xff":
                    marker = handle.read(1)
                if marker in {
                    b"\xc0",
                    b"\xc1",
                    b"\xc2",
                    b"\xc3",
                    b"\xc5",
                    b"\xc6",
                    b"\xc7",
                    b"\xc9",
                    b"\xca",
                    b"\xcb",
                    b"\xcd",
                    b"\xce",
                    b"\xcf",
                }:
                    length = struct.unpack(">H", handle.read(2))[0]
                    payload = handle.read(length - 2)
                    height, width = struct.unpack(">HH", payload[1:5])
                    return width, height
                length_bytes = handle.read(2)
                if len(length_bytes) != 2:
                    break
                length = struct.unpack(">H", length_bytes)[0]
                handle.seek(length - 2, 1)
    raise ValueError(f"Unsupported or corrupt image: {path}")


def find_subdir(root: Path, candidates: tuple[str, ...]) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.is_dir():
            return path
    raise FileNotFoundError(f"None of these directories exist under {root}: {candidates}")


def list_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_fingerprint(root: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def camera_summary(camera_dir: Path) -> dict[str, object]:
    cameras_path = camera_dir / "cameras.txt"
    images_path = camera_dir / "images.txt"
    model = "not_available"
    width = None
    height = None
    if cameras_path.exists():
        for line in cameras_path.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                fields = line.split()
                if len(fields) >= 4:
                    model = fields[1]
                    width = int(fields[2])
                    height = int(fields[3])
                break

    image_count = None
    if images_path.exists():
        pattern = re.compile(r"^# Number of images:\s*(\d+)")
        with images_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                match = pattern.match(line)
                if match:
                    image_count = int(match.group(1))
                    break

    files = {}
    for name in ("cameras.txt", "images.txt", "rigs.txt", "frames.txt"):
        path = camera_dir / name
        files[name] = {"present": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0}

    return {
        "model": model,
        "width": width,
        "height": height,
        "pose_count": image_count,
        "files": files,
        "points3D_txt_present": (camera_dir / "points3D.txt").exists(),
    }


def build_manifest(root: Path, fingerprint: bool = False) -> tuple[dict[str, object], list[str]]:
    root = root.resolve()
    labeled_dir = find_subdir(root, ("images", "images_labeled"))
    annotations_dir = find_subdir(root, ("json", "annotations_labelme"))
    unlabeled_dir = find_subdir(root, ("unlabeled_Images", "unlabeled_images", "images_unlabeled"))
    camera_dir = find_subdir(root, ("camera_parameters", "colmap"))

    labeled = list_images(labeled_dir)
    unlabeled = list_images(unlabeled_dir)
    annotations = sorted(annotations_dir.glob("*.json"))
    all_images = labeled + unlabeled

    resolutions = Counter(f"{width}x{height}" for width, height in map(image_dimensions, all_images))
    label_shapes: Counter[str] = Counter()
    invalid_labels: Counter[str] = Counter()
    annotation_sizes: Counter[str] = Counter()
    for annotation in annotations:
        payload = json.loads(annotation.read_text(encoding="utf-8"))
        width = payload.get("imageWidth")
        height = payload.get("imageHeight")
        annotation_sizes[f"{width}x{height}"] += 1
        for shape in payload.get("shapes", []):
            label = str(shape.get("label", "")).strip()
            label_shapes[label] += 1
            if label not in OFFICIAL_LABELS:
                invalid_labels[label] += 1

    labeled_stems = {path.stem for path in labeled}
    annotation_stems = {path.stem for path in annotations}
    missing_annotations = sorted(labeled_stems - annotation_stems)
    orphan_annotations = sorted(annotation_stems - labeled_stems)

    cameras = camera_summary(camera_dir)
    errors: list[str] = []
    if len(labeled) != 300:
        errors.append(f"expected 300 labeled images, found {len(labeled)}")
    if len(annotations) != 300:
        errors.append(f"expected 300 annotations, found {len(annotations)}")
    if len(unlabeled) != 100:
        errors.append(f"expected 100 unlabeled images, found {len(unlabeled)}")
    if missing_annotations:
        errors.append(f"missing annotations for {len(missing_annotations)} labeled images")
    if orphan_annotations:
        errors.append(f"orphan annotations: {len(orphan_annotations)}")
    if invalid_labels:
        errors.append(f"unexpected labels: {dict(invalid_labels)}")
    if resolutions != Counter({"1320x989": 400}):
        errors.append(f"unexpected image resolutions: {dict(resolutions)}")
    if cameras.get("pose_count") != 400:
        errors.append(f"expected 400 camera poses, found {cameras.get('pose_count')}")
    if cameras.get("model") != "SIMPLE_RADIAL":
        errors.append(f"expected SIMPLE_RADIAL camera, found {cameras.get('model')}")

    all_files = labeled + unlabeled + annotations
    all_files += [path for path in camera_dir.iterdir() if path.is_file()]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset": "IC-SHM 2026 Project 2 released dataset",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_path_policy": "repository-relative layout only; no absolute path recorded",
        "layout": {
            "labeled_images": labeled_dir.relative_to(root).as_posix(),
            "annotations": annotations_dir.relative_to(root).as_posix(),
            "unlabeled_images": unlabeled_dir.relative_to(root).as_posix(),
            "camera_parameters": camera_dir.relative_to(root).as_posix(),
        },
        "counts": {
            "labeled_images": len(labeled),
            "annotations": len(annotations),
            "unlabeled_images": len(unlabeled),
        },
        "image_resolutions": dict(sorted(resolutions.items())),
        "annotation_declared_resolutions": dict(sorted(annotation_sizes.items())),
        "shape_label_counts": dict(sorted(label_shapes.items())),
        "official_class_mapping": {
            "0": "background",
            "1": "deck",
            "2": "stay_cable",
            "3": "tower",
            "4": "foundation",
        },
        "camera": cameras,
        "pairing": {
            "missing_annotations": missing_annotations,
            "orphan_annotations": orphan_annotations,
        },
        "validation_errors": errors,
    }
    if fingerprint:
        manifest["aggregate_sha256"] = aggregate_fingerprint(root, all_files)
        manifest["fingerprinted_file_count"] = len(all_files)
    else:
        manifest["aggregate_sha256"] = "not_computed"
        manifest["fingerprinted_file_count"] = 0
    return manifest, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("ICSHM_DATA_ROOT", "data/Contest Dataset")),
    )
    parser.add_argument("--write-manifest", type=Path)
    parser.add_argument("--fingerprint", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest, errors = build_manifest(args.root, fingerprint=args.fingerprint)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"data check failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(manifest["counts"], ensure_ascii=False))
    print(json.dumps(manifest["image_resolutions"], ensure_ascii=False))
    print(json.dumps(manifest["shape_label_counts"], ensure_ascii=False))
    print(json.dumps(manifest["camera"], ensure_ascii=False))

    if args.write_manifest:
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote manifest: {args.write_manifest}")

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("Dataset check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

