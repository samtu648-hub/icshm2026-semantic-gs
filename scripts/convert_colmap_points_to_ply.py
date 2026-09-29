#!/usr/bin/env python3
"""Convert the project COLMAP points3D.txt subset to native 3DGS binary PLY."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def read_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xyz: list[tuple[float, float, float]] = []
    rgb: list[tuple[int, int, int]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 7:
            raise ValueError(f"Malformed COLMAP point line: {line[:120]}")
        xyz.append(tuple(float(value) for value in fields[1:4]))
        rgb.append(tuple(int(value) for value in fields[4:7]))
    if not xyz:
        raise ValueError("No points found")
    return np.asarray(xyz, dtype=np.float32), np.asarray(rgb, dtype=np.uint8)


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(path)
    if xyz.shape != (len(rgb), 3) or rgb.shape[1:] != (3,):
        raise ValueError("Expected matching Nx3 xyz/rgb arrays")
    dtype = np.dtype(
        [
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ]
    )
    values = np.zeros(len(xyz), dtype=dtype)
    values["x"], values["y"], values["z"] = xyz.T
    values["red"], values["green"], values["blue"] = rgb.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(values)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    with path.open("xb") as handle:
        handle.write(header)
        values.tofile(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    xyz, rgb = read_points(args.input)
    write_ply(args.output, xyz, rgb)
    print(f"Wrote {len(xyz)} points: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
