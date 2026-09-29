#!/usr/bin/env python3
"""Verify that an R1 upstream adapter honors sparse/0/test.txt exactly.

The three upstream projects otherwise choose validation cameras by their own index-based rules.
This check is deliberately CPU-only and operates on the prepared COLMAP scene before a costly run.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--candidate", choices=("standard-3dgs", "mip-splatting", "scaffold-gs"), required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--expected-train", type=int, required=True)
    parser.add_argument("--expected-test", type=int, required=True)
    return parser.parse_args()


def load_scene_info(candidate: str, scene: Path):
    reader = importlib.import_module("scene.dataset_readers")
    if candidate == "standard-3dgs":
        return reader.readColmapSceneInfo(str(scene), None, "", True, False)
    if candidate == "mip-splatting":
        return reader.readColmapSceneInfo(str(scene), None, True)
    return reader.readColmapSceneInfo(str(scene), None, True, 0)


def camera_file_name(camera: object) -> str:
    name = Path(str(camera.image_name)).name
    return name if Path(name).suffix else f"{name}.png"


def main() -> int:
    args = parse_args()
    test_path = args.scene / "sparse" / "0" / "test.txt"
    if not test_path.is_file():
        raise FileNotFoundError(f"Explicit evaluation list missing: {test_path}")
    expected_names = {line.strip() for line in test_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not expected_names:
        raise ValueError("Explicit evaluation list is empty")
    sys.path.insert(0, str(args.candidate_root.resolve()))
    scene_info = load_scene_info(args.candidate, args.scene)
    train_names = {camera_file_name(camera) for camera in scene_info.train_cameras}
    test_names = {camera_file_name(camera) for camera in scene_info.test_cameras}
    payload = {
        "candidate": args.candidate,
        "scene": args.scene.name,
        "train_count": len(train_names),
        "test_count": len(test_names),
        "expected_train": args.expected_train,
        "expected_test": args.expected_test,
        "test_names_match_explicit_file": test_names == expected_names,
        "missing_from_test": sorted(expected_names - test_names)[:10],
        "unexpected_in_test": sorted(test_names - expected_names)[:10],
        "disjoint": not (train_names & test_names),
        "covered": len(train_names | test_names) == args.expected_train + args.expected_test,
    }
    payload["passes"] = (
        payload["train_count"] == args.expected_train
        and payload["test_count"] == args.expected_test
        and payload["test_names_match_explicit_file"]
        and payload["disjoint"]
        and payload["covered"]
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
